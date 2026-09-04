"""Tasks T1-T4 — Operational TSFM Evaluation on AmLight Telemetry.

Data resolution: 60-minute SNMP polling (MIB-II ifHCInOctets/ifHCOutOctets).
  - Context window L : 336 steps = 14 days (two full diurnal + two full weekly cycles)
  - Horizon H        : 48 steps  = 48 h
  - Eval sub-window  : steps [24:48] — the harder, second half of the horizon;
    persistence predictors do well on the first 24 h simply by repeating recent
    values, so evaluation is restricted to the 24-48 h lookahead.

T1 - Trajectory Forecasting Accuracy
    Input:  336-step context of per-link utilization (util_norm in [0, 1]).
    Output: the full 48-step utilization trajectory.
    Metric: Mean Absolute Scaled Error (MASE) vs. the seasonal-naive predictor
            (repeat the last 24 h cycle), reported as Skill_SN = 1 - MASE.
    Answers: HOW will link utilization evolve over the next 48 h?

T2 - Peak Utilization Exceedance Detection
    Input:  336-step context of per-link utilization (same as T1).
    Output: P(max(util, eval steps [24:48]) > tau), tau = peak_threshold (10%,
            i.e. 10 Gbps on a 100GE link - a sustained elephant flow).
    Target: binary label - did any active link exceed tau in the 24-48 h window?
    Metric: AUROC + Brier Skill Score (BSS) vs. the climatological reference.
    Answers: WILL a link exceed the threshold in the next 24-48 h?
    Probabilistic models (Chronos, Moirai) return calibrated P-hat from sample
    ensembles; point-forecast models (TimesFM, PatchTST, TTM) use max(forecast) > tau.
    Reference: B1 - persistence exceedance: score = max(last eval-window context steps).

T3 - Peak Timing Prediction
    Input:  336-step context of per-link utilization (same as T1).
    Output: the step index (within the eval sub-window) at which each link peaks.
    Metric: MAE (hours) + Skill Score vs. a random-guess baseline
            (MAE_random = eval_len / 3 = 8 h for a 24-step sub-window).
    Answers: WHEN will the daily peak arrive?
    Reference: B3 - persistence timing: argmax of the last eval-window context steps.

T4 - Traffic Asymmetry Forecasting
    Input:  336-step context of per-link asymmetry, a = (out - in) / (|out| + |in| + eps).
    Output: mean asymmetry over the eval sub-window.
    Metric: MAE vs. actual mean asymmetry, with Skill vs. persistence and vs.
            seasonal-naive references.
    Answers: IS outbound traffic dominant in the next 24-48 h?
    Reference: B4 - persistence: last observed asymmetry value.

Evaluation is restricted to active links (mean utilization >= 0.5%) to exclude
the majority of near-idle, provisioned-but-unused links that would otherwise
dilute performance estimates.

Also provides FrozenBackboneAdapter, a feature-extraction helper for the
few-shot task heads used in Experiment 4 (ZS+MLP@n) and the LP-FT protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.loaders import get_util_matrix, get_asym_matrix
from models.base import TSFMBase


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def make_trajectory_windows(
    telemetry: pd.DataFrame,
    context_len: int,
    prediction_len: int,
    capacity_gbps: float = 100.0,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (context, trajectory) windows for T1 and T3.

    Returns
    -------
    contexts     : (N, context_len, n_links)    — util fraction [0,1]
    trajectories : (N, prediction_len, n_links) — full horizon util
    """
    util = get_util_matrix(telemetry, capacity_gbps)
    n_steps = len(util)
    total = context_len + prediction_len

    contexts, trajectories = [], []
    for i in range(0, n_steps - total, stride):
        contexts.append(util[i : i + context_len])
        trajectories.append(util[i + context_len : i + total])

    return np.array(contexts, dtype=np.float32), np.array(trajectories, dtype=np.float32)


def make_peak_timing_windows(
    telemetry: pd.DataFrame,
    context_len: int,
    prediction_len: int,
    capacity_gbps: float = 100.0,
    stride: int = 4,
    eval_start: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (context, timing) windows for T3.

    Timing target: 0-based step index (within the eval sub-window) of max util.
    eval_start : first step of the eval sub-window [eval_start:pred_len].

    Returns
    -------
    contexts : (N, context_len, n_links) — util fraction [0,1]
    timing   : (N, n_links) — argmax step within eval sub-window [0, pred_len-eval_start-1]
    """
    contexts, trajectories = make_trajectory_windows(
        telemetry, context_len, prediction_len, capacity_gbps, stride
    )
    timing = trajectories[:, eval_start:, :].argmax(axis=1).astype(np.float32)
    return contexts, timing


# ---------------------------------------------------------------------------
# Skill score metrics — shared by T1, T3, T4
# ---------------------------------------------------------------------------

def compute_skill(mae_model: float, mae_reference: float) -> float:
    """Skill Score = 1 − MAE_model / MAE_reference.

    Returns 1 for perfect prediction, 0 for equal to reference,
    negative when worse than reference.
    """
    if mae_reference < 1e-10:
        return 0.0
    return float(1.0 - mae_model / mae_reference)


def _compute_peak_skill_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    link_indices: list[int] | None,
) -> tuple[float, float, float]:
    """Return (skill, mae_model, mae_persist) averaged over active links."""
    if link_indices is not None:
        y_true    = y_true[:, link_indices]
        y_pred    = y_pred[:, link_indices]
        y_persist = y_persist[:, link_indices]
    mae_m = float(np.mean(np.abs(y_true - y_pred)))
    mae_p = float(np.mean(np.abs(y_true - y_persist)))
    return compute_skill(mae_m, mae_p), mae_m, mae_p


def _bootstrap_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    link_indices: list[int] | None,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """95% bootstrap CI for a skill score."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    skills = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sk, _, _ = _compute_peak_skill_arrays(
            y_true[idx], y_pred[idx], y_persist[idx], link_indices
        )
        skills.append(sk)
    return float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))


# ---------------------------------------------------------------------------
# T2 — Peak Utilization Exceedance Detection (AUROC + Brier Skill Score)
# ---------------------------------------------------------------------------

def _compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via Wilcoxon-Mann-Whitney U statistic (midrank AUC).

    Handles tied scores correctly: each tied (pos, neg) pair contributes 0.5
    to U, matching sklearn's roc_auc_score.
    """
    y = y_true.flatten().astype(float)
    s = scores.flatten().astype(float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order    = np.argsort(s, kind="mergesort")           # ascending, stable
    s_sorted = s[order]
    y_sorted = y[order]
    # Midranks: each tied group gets the average of its 1-indexed ranks.
    # For group spanning [start, end) (0-indexed), midrank = (start + end + 1) / 2.
    _, grp_inv, grp_cnt = np.unique(s_sorted, return_inverse=True, return_counts=True)
    cum      = np.concatenate([[0], grp_cnt.cumsum()])
    midranks = (cum[:-1] + cum[1:] + 1) / 2.0
    ranks    = midranks[grp_inv]
    pos_rank_sum = float((ranks * y_sorted).sum())
    U = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


def _brier_skill_score(
    y_true: np.ndarray, p_hat: np.ndarray, p_clim: float
) -> float:
    """BSS = 1 - BS_model / BS_climatological. Range (-∞, 1]; 1=perfect, 0=clim."""
    bs_model = float(np.mean((p_hat - y_true) ** 2))
    bs_ref = float(p_clim * (1.0 - p_clim))
    if bs_ref < 1e-10:
        return 0.0
    return float(1.0 - bs_model / bs_ref)


def _bootstrap_auroc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(_compute_auroc(y_true[idx], scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _bootstrap_bss(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    p_clim: float,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(_brier_skill_score(y_true[idx], p_hat[idx], p_clim))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _exceedance_scores_and_probs(
    model: "TSFMBase",
    contexts: np.ndarray,
    prediction_len: int,
    link_indices: list[int] | None,
    threshold: float,
    reduction: str = "max",
    eval_start: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, probs) each of shape (N, n_active) for T2 exceedance.

    reduction  : 'max' — P(max(trajectory[eval_start:]) > threshold).
    eval_start : first step of the evaluation sub-window (0 = full horizon).

    scores : for AUROC — calibrated P̂ for probabilistic models,
             aggregated point forecast as rank score for deterministic models.
    probs  : for BSS — same as scores for probabilistic; hard 0/1 for deterministic.
    """
    n, _, n_links = contexts.shape
    active = list(range(n_links)) if link_indices is None else link_indices
    n_active = len(active)
    scores = np.zeros((n, n_active), dtype=np.float32)
    probs  = np.zeros((n, n_active), dtype=np.float32)

    _agg = np.max if reduction == "max" else np.mean

    for i in range(n):
        link_ctxs = [contexts[i, :, l] for l in active]
        samples_list = model.forecast_samples_many(link_ctxs)
        for j, samps in enumerate(samples_list):
            samps = np.asarray(samps)                      # (n_samples, pred_len)
            agg   = _agg(samps[:, eval_start:], axis=-1)  # (n_samples,)
            if samps.shape[0] > 1:
                # Probabilistic: calibrated P̂ = fraction of samples exceeding τ
                p = float(np.mean(agg > threshold))
                scores[i, j] = p
                probs[i, j]  = p
            else:
                # Point-forecast: aggregate as continuous rank score; hard 0/1 for BSS
                m = float(agg[0])
                scores[i, j] = m
                probs[i, j]  = 1.0 if m > threshold else 0.0
    return scores, probs


def evaluate_exceedance_persistence(
    contexts: np.ndarray,
    trajectories: np.ndarray,
    prediction_len: int,
    threshold: float = 0.10,
    link_indices: list[int] | None = None,
    max_windows: int = 500,
    n_boot: int = 500,
    eval_start: int = 0,
) -> dict:
    """B1 — Persistence exceedance: score = max(last eval-window context steps)."""
    n = min(len(contexts), max_windows)
    act = list(range(contexts.shape[2])) if link_indices is None else link_indices
    y_true = (trajectories[:n, eval_start:, :].max(axis=1)[:, act] > threshold).astype(np.float32)
    scores = contexts[:n, -96:, :][:, :, act].max(axis=1)
    probs  = (scores > threshold).astype(np.float32)
    y_flat, s_flat, p_flat = y_true.flatten(), scores.flatten(), probs.flatten()
    p_clim = float(y_flat.mean())
    auroc = _compute_auroc(y_flat, s_flat)
    bss   = _brier_skill_score(y_flat, p_flat, p_clim)
    au_lo, au_hi = _bootstrap_auroc(y_flat, s_flat, n_boot)
    bs_lo, bs_hi = _bootstrap_bss(y_flat, p_flat, p_clim, n_boot)
    return {
        "t2_auroc": auroc, "t2_auroc_ci": [au_lo, au_hi],
        "t2_bss": bss, "t2_bss_ci": [bs_lo, bs_hi],
        "t2_event_rate": p_clim, "t2_n_pos": int(y_flat.sum()),
        "t2_threshold": threshold, "baseline": "persistence_exceedance",
    }


def evaluate_exceedance_tsfm(
    model: "TSFMBase",
    contexts: np.ndarray,
    trajectories: np.ndarray,
    prediction_len: int,
    threshold: float = 0.10,
    link_indices: list[int] | None = None,
    max_windows: int = 200,
    n_boot: int = 500,
    eval_start: int = 0,
) -> dict:
    """T2: P(max(util, eval sub-window) > threshold) evaluated with AUROC + Brier Skill Score.

    eval_start : first step of the eval sub-window (0 = full horizon; 24 = second 24 h).
    Probabilistic models (Chronos, Moirai) return calibrated P̂ from sample ensembles.
    Point-forecast models (TimesFM, PatchTST, TTM) use max(forecast[eval_start:]) as rank score.
    """
    n = min(len(contexts), max_windows)
    act = list(range(contexts.shape[2])) if link_indices is None else link_indices
    y_true = (trajectories[:n, eval_start:, :].max(axis=1)[:, act] > threshold).astype(np.float32)
    scores, probs = _exceedance_scores_and_probs(
        model, contexts[:n], prediction_len, link_indices, threshold,
        reduction="max", eval_start=eval_start,
    )
    y_flat, s_flat, p_flat = y_true.flatten(), scores.flatten(), probs.flatten()
    p_clim = float(y_flat.mean())
    auroc = _compute_auroc(y_flat, s_flat)
    bss   = _brier_skill_score(y_flat, p_flat, p_clim)
    au_lo, au_hi = _bootstrap_auroc(y_flat, s_flat, n_boot)
    bs_lo, bs_hi = _bootstrap_bss(y_flat, p_flat, p_clim, n_boot)
    return {
        "t2_auroc": auroc, "t2_auroc_ci": [au_lo, au_hi],
        "t2_bss": bss, "t2_bss_ci": [bs_lo, bs_hi],
        "t2_event_rate": p_clim, "t2_n_pos": int(y_flat.sum()),
        "t2_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# T3 — Peak Timing Prediction (Skill Score vs random baseline)
# ---------------------------------------------------------------------------

def compute_timing_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours_per_step: float = 1.0,      # 60-min resolution: 1 h/step
) -> float:
    """MAE on peak step index × hours_per_step → MAE in hours."""
    return float(np.mean(np.abs(y_true - y_pred)) * hours_per_step)


def compute_timing_skill(
    mae_h: float,
    eval_len: int,
    hours_per_step: float = 1.0,
) -> float:
    """T3 skill score: 1 - MAE_h / MAE_random.

    MAE_random = eval_len × hours_per_step / 3, where eval_len = pred_len - eval_start.
    At eval_len=24 steps × 1.0 h/step (60-min, pred_len=48, eval_start=24):
      MAE_random = 24 × 1.0 / 3 = 8 h.

    Range: (−∞, 1]. 1=perfect; 0=random baseline; <0=worse than random.
    """
    random_mae = eval_len * hours_per_step / 3.0
    return float(1.0 - mae_h / random_mae)


def _bootstrap_timing_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 500,
    hours_per_step: float = 1.0,
    seed: int = 0,
) -> tuple[float, float]:
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    maes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        maes.append(compute_timing_mae(y_true[idx], y_pred[idx], hours_per_step))
    return float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))


def evaluate_timing_persistence(
    contexts: np.ndarray,
    timing: np.ndarray,
    prediction_len: int,
    link_indices: list[int] | None = None,
    hours_per_step: float = 1.0,
    max_windows: int = 500,
    n_boot: int = 500,
    eval_start: int = 0,
) -> dict:
    """B3 — Persistence timing: argmax of last (pred_len-eval_start) context steps per link.

    timing must already be sliced to the eval sub-window by make_peak_timing_windows(eval_start=...).
    Skill is computed vs random baseline for eval sub-window (prediction_len - eval_start steps).
    """
    n = min(len(contexts), max_windows)
    eval_len = prediction_len - eval_start
    ctx = contexts[:n, :, link_indices] if link_indices is not None else contexts[:n]
    tim = timing[:n, link_indices]     if link_indices is not None else timing[:n]
    y_pred   = ctx[:, -eval_len:, :].argmax(axis=1).astype(np.float32)
    mae      = compute_timing_mae(tim, y_pred, hours_per_step)
    lo, hi   = _bootstrap_timing_mae(tim, y_pred, n_boot=n_boot, hours_per_step=hours_per_step)
    skill    = compute_timing_skill(mae, eval_len, hours_per_step)
    return {
        "t3_skill":         skill,   # vs random baseline (8 h)
        "t3_skill_persist": 0.0,     # by definition — this IS the persistence predictor
        "mae_h":            mae,
        "mae_persist_h":    mae,
        "mae_ci":           [lo, hi],
        "baseline":         "persistence_timing",
    }


def evaluate_timing_tsfm(
    model: TSFMBase,
    contexts: np.ndarray,
    timing: np.ndarray,
    prediction_len: int,
    link_indices: list[int] | None = None,
    hours_per_step: float = 1.0,
    max_windows: int = 200,
    n_boot: int = 500,
    eval_start: int = 0,
) -> dict:
    """T3 TSFM: forecast trajectory → argmax of eval sub-window → skill vs random and persistence.

    eval_start : first step of the eval sub-window [eval_start:pred_len].
                 timing must already reflect this sub-window (from make_peak_timing_windows).

    t3_skill        : Skill vs random baseline (MAE_random = 8 h for a 24-step sub-window).
    t3_skill_persist: Skill vs persistence timing (argmax of last eval-window context steps).
    """
    n = min(len(contexts), max_windows)
    raw    = _raw_predictions(model, contexts[:n], prediction_len, link_indices)
    y_pred = raw[:, eval_start:, :].argmax(axis=1).astype(np.float32)
    act    = list(range(contexts.shape[2])) if link_indices is None else link_indices
    tim    = timing[:n, act]

    eval_len = prediction_len - eval_start
    mae    = compute_timing_mae(tim, y_pred, hours_per_step)
    lo, hi = _bootstrap_timing_mae(tim, y_pred, n_boot=n_boot, hours_per_step=hours_per_step)
    skill  = compute_timing_skill(mae, eval_len, hours_per_step)

    # Persistence timing: argmax of the last eval_len context steps (same reference as B3).
    y_persist     = contexts[:n, -eval_len:, :][:, :, act].argmax(axis=1).astype(np.float32)
    mae_persist   = compute_timing_mae(tim, y_persist, hours_per_step)
    skill_persist = compute_skill(mae, mae_persist)

    return {
        "t3_skill":         skill,
        "t3_skill_persist": skill_persist,
        "mae_h":            mae,
        "mae_persist_h":    mae_persist,
        "mae_ci":           [lo, hi],
    }


# ---------------------------------------------------------------------------
# TSFM inference helper
# ---------------------------------------------------------------------------

def _raw_predictions(
    model: TSFMBase,
    contexts: np.ndarray,
    pred_len: int,
    link_indices: list[int] | None = None,
) -> np.ndarray:
    """Run model channel-independently on contexts. Returns (N, pred_len, n_out_links).

    When link_indices is given, only those link columns are forecasted — saves
    significant inference time when evaluating a small active-link subset.
    """
    n, _, n_links = contexts.shape
    active = list(range(n_links)) if link_indices is None else link_indices
    preds = []
    for i in range(n):
        link_ctxs = [contexts[i, :, l] for l in active]
        forecasts = model.forecast_many(link_ctxs)
        fc = np.stack(
            [np.array(f).flatten()[:pred_len] for f in forecasts], axis=1
        )
        preds.append(fc)
    return np.array(preds, dtype=np.float32)   # (N, pred_len, n_active)


# ---------------------------------------------------------------------------
# Frozen-backbone feature extraction — few-shot task heads (Experiment 4, LP-FT)
# ---------------------------------------------------------------------------

class FrozenBackboneAdapter:
    """Wraps a frozen TSFM backbone to produce per-link features for a task head.

    Used by the few-shot task heads (Experiment 4's ZS+MLP@n and LP-FT, paper
    §3.2-3.3): the backbone is never updated, and a small task-specific head
    (see `experiments/exp3_lp_ft.py`'s `_MLP`) is trained on top of these
    features using auto-derived labels — no human annotation is needed.

    Workflow:
        adapter = FrozenBackboneAdapter(model, pred_len, link_indices)
        raw     = adapter.get_raw_predictions(contexts)          # T1/T3/T4 features
        scores  = adapter.get_exceedance_scores(contexts, tau)   # T2 features
    """

    def __init__(
        self,
        model: TSFMBase,
        pred_len: int,
        link_indices: list[int] | None = None,
    ) -> None:
        self.model       = model
        self.pred_len    = pred_len
        self.link_indices = link_indices

    def get_raw_predictions(self, contexts: np.ndarray) -> np.ndarray:
        """Return (N, pred_len, n_active) raw TSFM predictions."""
        return _raw_predictions(self.model, contexts, self.pred_len, self.link_indices)

    def get_exceedance_scores(
        self,
        contexts: np.ndarray,
        threshold: float,
        reduction: str = "max",
        eval_start: int = 0,
    ) -> np.ndarray:
        """Return (N, n_active) exceedance scores for the T2 task head.

        reduction  = 'max' — P(max(trajectory[eval_start:]) > threshold).
        eval_start — first step of eval sub-window (0 = full horizon).

        Probabilistic models: calibrated P̂ from sample fraction.
        Point-forecast models: aggregated point forecast as continuous rank score.
        """
        n, _, n_links = contexts.shape
        active = self.link_indices if self.link_indices is not None else list(range(n_links))
        n_active = len(active)
        all_ctxs = [contexts[i, :, l] for i in range(n) for l in active]
        samples_list = self.model.forecast_samples_many(all_ctxs)
        _agg = np.max if reduction == "max" else np.mean
        scores = np.zeros((n, n_active), dtype=np.float32)
        for flat_idx, samps in enumerate(samples_list):
            i, j = divmod(flat_idx, n_active)
            samps = np.asarray(samps)                      # (n_samples, pred_len)
            agg   = _agg(samps[:, eval_start:], axis=-1)  # (n_samples,)
            if samps.shape[0] > 1:
                scores[i, j] = float(np.mean(agg > threshold))
            else:
                scores[i, j] = float(agg[0])
        return scores


# ---------------------------------------------------------------------------
# Kept for backward compatibility with tasks/baselines_classical.py
# ---------------------------------------------------------------------------

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def compute_peak_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return compute_r2(y_true[:, None, :], y_pred[:, None, :])


def _bootstrap_peak_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    r2s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r2s.append(compute_peak_r2(y_true[idx], y_pred[idx]))
    return float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))


# ---------------------------------------------------------------------------
# T1 — Trajectory Forecasting Accuracy (MASE + CRPS)
# ---------------------------------------------------------------------------

def make_asym_windows(
    telemetry: pd.DataFrame,
    context_len: int,
    prediction_len: int,
    stride: int = 4,
    eval_start: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """T4: (asym_context, asym_target) windows.

    Returns
    -------
    contexts : (N, context_len, n_links)  — asym ∈ [-1, 1]
    targets  : (N, n_links)               — mean(asym[eval_start:pred_len])
    """
    asym = get_asym_matrix(telemetry)
    n_steps = len(asym)
    total = context_len + prediction_len
    contexts, targets = [], []
    for i in range(0, n_steps - total, stride):
        ctx  = asym[i : i + context_len]
        traj = asym[i + context_len : i + total]
        contexts.append(ctx)
        targets.append(traj[eval_start:].mean(axis=0))
    return np.array(contexts, dtype=np.float32), np.array(targets, dtype=np.float32)


def _seasonal_naive(contexts: np.ndarray, eval_len: int, seasonal_period: int = 24) -> np.ndarray:
    """(N, eval_len, n_links) — repeat last seasonal_period from context."""
    tail = contexts[:, -seasonal_period:, :]          # (N, sp, n_links)
    reps = (eval_len + seasonal_period - 1) // seasonal_period
    return np.tile(tail, (1, reps, 1))[:, :eval_len, :]


def _crps_from_samples(samples: np.ndarray, y_true: float, n_q: int = 19) -> float:
    """CRPS via Weighted Quantile Loss approximation from ensemble samples."""
    qs = np.linspace(0.05, 0.95, n_q)
    quantiles = np.quantile(samples, qs)
    crps = sum(
        ql * max(y_true - q, 0.0) + (1.0 - ql) * max(q - y_true, 0.0)
        for q, ql in zip(quantiles, qs)
    )
    return 2.0 * crps / n_q


def _bootstrap_mase(
    mae_model_per_sample: np.ndarray,
    mae_sn_per_sample: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(mae_model_per_sample)
    mases = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mases.append(float(mae_model_per_sample[idx].mean() /
                           (mae_sn_per_sample[idx].mean() + 1e-10)))
    return float(np.percentile(mases, 2.5)), float(np.percentile(mases, 97.5))


def evaluate_trajectory_mase_tsfm(
    model: "TSFMBase",
    contexts: np.ndarray,
    trajectories: np.ndarray,
    prediction_len: int,
    link_indices: list[int] | None = None,
    max_windows: int = 200,
    n_boot: int = 500,
    seasonal_period: int = 24,
    eval_start: int = 0,
) -> dict:
    """T1: trajectory MASE vs seasonal-naive for a TSFM (point forecast).

    MASE < 1 → better than seasonal naive.
    Skill_SN = 1 − MASE (>0 means model beats seasonal naive).
    """
    n = min(len(contexts), max_windows)
    act = list(range(contexts.shape[2])) if link_indices is None else link_indices
    raw    = _raw_predictions(model, contexts[:n], prediction_len, link_indices)
    eval_len = prediction_len - eval_start
    y_pred = raw[:, eval_start:, :]                                  # (N, eval_len, n_active)
    y_true = trajectories[:n, eval_start:, :][:, :, act]            # (N, eval_len, n_active)
    sn     = _seasonal_naive(contexts[:n, :, act], eval_len, seasonal_period)

    ae_model = np.abs(y_true - y_pred).mean(axis=(1, 2))            # (N,)
    ae_sn    = np.abs(y_true - sn).mean(axis=(1, 2))                # (N,)
    mase     = float(ae_model.mean() / (ae_sn.mean() + 1e-10))
    skill_sn = 1.0 - mase
    lo, hi   = _bootstrap_mase(ae_model, ae_sn, n_boot)

    return {
        "t1_mase":     mase,
        "t1_skill_sn": skill_sn,
        "t1_mae":      float(ae_model.mean()),
        "t1_mae_sn":   float(ae_sn.mean()),
        "t1_mase_ci":  [lo, hi],
    }


def evaluate_trajectory_crps_tsfm(
    model: "TSFMBase",
    contexts: np.ndarray,
    trajectories: np.ndarray,
    prediction_len: int,
    link_indices: list[int] | None = None,
    max_windows: int = 100,
    seasonal_period: int = 24,
    eval_start: int = 0,
) -> dict:
    """T1b: CRPS for probabilistic TSFMs (Chronos, Moirai).

    Uses model.forecast_samples_many() to get sample ensembles and computes
    CRPS via WQL. Returns NaN if model does not support sampling.
    """
    if not hasattr(model, "forecast_samples_many"):
        return {"t1_crps": float("nan"), "t1_crps_skill_sn": float("nan")}

    n = min(len(contexts), max_windows)
    act      = list(range(contexts.shape[2])) if link_indices is None else link_indices
    eval_len = prediction_len - eval_start
    y_true   = trajectories[:n, eval_start:, :][:, :, act]   # (N, eval_len, n_active)
    sn       = _seasonal_naive(contexts[:n, :, act], eval_len, seasonal_period)
    mae_sn   = float(np.abs(y_true - sn).mean())

    crps_vals = []
    for i in range(n):
        link_ctxs = [contexts[i, :, l] for l in act]
        samples_list = model.forecast_samples_many(link_ctxs)  # list of (n_samp, pred_len)
        for j, samps in enumerate(samples_list):
            samps = np.asarray(samps)[:, eval_start:]          # (n_samp, eval_len)
            for t in range(eval_len):
                crps_vals.append(_crps_from_samples(samps[:, t], float(y_true[i, t, j])))

    crps = float(np.mean(crps_vals)) if crps_vals else float("nan")
    return {
        "t1_crps":          crps,
        "t1_crps_skill_sn": float(1.0 - crps / (mae_sn + 1e-10)),
        "t1_mae_sn":        mae_sn,
    }


# ---------------------------------------------------------------------------
# T4 — Traffic Asymmetry Forecasting (MAE + Skill vs persistence)
# ---------------------------------------------------------------------------

def evaluate_asym_persistence(
    asym_contexts: np.ndarray,
    asym_targets: np.ndarray,
    link_indices: list[int] | None = None,
    max_windows: int = 500,
    n_boot: int = 500,
) -> dict:
    """B4 — Persistence: last observed asym value as forecast."""
    n    = min(len(asym_contexts), max_windows)
    act  = list(range(asym_contexts.shape[2])) if link_indices is None else link_indices
    y_true    = asym_targets[:n, :][:, act]          # (N, n_active)
    y_persist = asym_contexts[:n, -1, :][:, act]     # (N, n_active)
    mae = float(np.abs(y_true - y_persist).mean())

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(np.abs(y_true[idx] - y_persist[idx]).mean()))
    return {
        "t4_mae":          mae,
        "t4_mae_ci":       [float(np.percentile(boots, 2.5)),
                            float(np.percentile(boots, 97.5))],
        "t4_skill_sn":     0.0,
        "baseline":        "persistence_asym",
    }


def evaluate_asym_tsfm(
    model: "TSFMBase",
    asym_contexts: np.ndarray,
    asym_targets: np.ndarray,
    prediction_len: int,
    link_indices: list[int] | None = None,
    max_windows: int = 200,
    n_boot: int = 500,
    seasonal_period: int = 24,
    eval_start: int = 0,
) -> dict:
    """T4: TSFM forecasts asym series → mean(asym[eval_start:]) vs actual mean.

    Skill vs persistence and vs seasonal naive (daily periodicity).
    """
    n   = min(len(asym_contexts), max_windows)
    act = list(range(asym_contexts.shape[2])) if link_indices is None else link_indices

    raw    = _raw_predictions(model, asym_contexts[:n], prediction_len, link_indices)
    y_pred = raw[:, eval_start:, :].mean(axis=1)                # (N, n_active)
    y_true = asym_targets[:n, :][:, act]                        # (N, n_active)
    y_per  = asym_contexts[:n, -1, :][:, act]                   # (N, n_active) persistence
    # seasonal naive: mean of the same eval window 24h before
    sn_traj = _seasonal_naive(asym_contexts[:n, :, act],
                              prediction_len - eval_start, seasonal_period)
    y_sn   = sn_traj.mean(axis=1)                               # (N, n_active)

    mae_m  = float(np.abs(y_true - y_pred).mean())
    mae_p  = float(np.abs(y_true - y_per).mean())
    mae_sn = float(np.abs(y_true - y_sn).mean())

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(np.abs(y_true[idx] - y_pred[idx]).mean()))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return {
        "t4_mae":          mae_m,
        "t4_mae_ci":       [lo, hi],
        "t4_mae_persist":  mae_p,
        "t4_mae_sn":       mae_sn,
        "t4_skill_persist": compute_skill(mae_m, mae_p),
        "t4_skill_sn":     compute_skill(mae_m, mae_sn),
    }
