"""Classical time series baselines — T1, T2, T3, T4.

ARIMA, SARIMA, Prophet, and Holt-Winters (Exponential Smoothing) evaluated
channel-independently (one model fit per link per window). This is the
Experiment 1 performance floor (paper §5.1, Table 2).

Data resolution: 60-minute SNMP polling.
  - Seasonal period m=24 (daily cycle at 60-min resolution)
  - Context L=336 steps = 14 days, horizon H=48 steps = 48 h

These are slow: each fit costs 50–500 ms, so evaluation is limited to
max_windows windows by default.

Tasks:
    T1 — Trajectory Forecasting       : evaluate_classical_trajectory_mase → MASE / Skill_SN
    T2 — Peak Utilization Exceedance  : evaluate_classical_exceedance      → AUROC + BSS
    T3 — Peak Timing Prediction       : evaluate_classical_timing          → MAE (h) + Skill
    T4 — Traffic Asymmetry            : evaluate_classical_asym            → MAE + Skill

Dependencies (optional — functions raise ImportError with install hint):
    pip install statsmodels prophet
"""

from __future__ import annotations

import multiprocessing
import os
import warnings

# Suppress statsmodels warnings in parent and all joblib worker processes.
# Force-set (not setdefault) so any pre-existing empty value is overridden.
# Workers inherit this env var and Python reads it at interpreter startup,
# suppressing warnings before any module-level code runs.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
# Explicit category filters as a belt-and-suspenders fallback
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
except ImportError:
    pass

from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from .forecasting import (
    compute_skill,
    compute_timing_mae,
    compute_timing_skill,
    _bootstrap_timing_mae,
    _compute_auroc,
    _brier_skill_score,
    _bootstrap_auroc,
    _bootstrap_bss,
    _bootstrap_skill,
    _seasonal_naive,
    _bootstrap_mase,
)


def _exceedance_metrics(y_flat, s_flat, p_flat, n_boot, prefix):
    """Shared AUROC+BSS computation. prefix is 't2' (exceedance)."""
    p_clim = float(y_flat.mean())
    auroc = _compute_auroc(y_flat, s_flat)
    bss   = _brier_skill_score(y_flat, p_flat, p_clim)
    au_lo, au_hi = _bootstrap_auroc(y_flat, s_flat, n_boot)
    bs_lo, bs_hi = _bootstrap_bss(y_flat, p_flat, p_clim, n_boot)
    return {
        f"{prefix}_auroc": auroc, f"{prefix}_auroc_ci": [au_lo, au_hi],
        f"{prefix}_bss": bss,     f"{prefix}_bss_ci":   [bs_lo, bs_hi],
        f"{prefix}_event_rate": p_clim,
    }


# ---------------------------------------------------------------------------
# Package helpers
# ---------------------------------------------------------------------------

def _require_statsmodels():
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        return ARIMA, SARIMAX, ExponentialSmoothing
    except ImportError:
        raise ImportError("Run: pip install statsmodels")


def _n_jobs(method: str) -> int:
    """Worker count. Prophet spawns Stan — cap at 4 to avoid OOM."""
    n = max(1, multiprocessing.cpu_count() - 1)
    return min(n, 4) if method == "prophet" else n


def _require_prophet():
    for pkg in ("prophet", "fbprophet"):
        try:
            import importlib
            mod = importlib.import_module(pkg)
            return mod.Prophet
        except ImportError:
            continue
    raise ImportError("Run: pip install prophet")


# ---------------------------------------------------------------------------
# Core forecasting helper
# ---------------------------------------------------------------------------

Method = Literal["arima", "sarima", "prophet", "holtwinters"]


def _forecast_series(
    series: np.ndarray,
    pred_len: int,
    method: Method,
    seasonal_period: int = 24,
) -> np.ndarray:
    """Fit a classical model on `series` and forecast `pred_len` steps.

    Returns array of shape (pred_len,). On failure or non-finite result returns
    the series mean. Data is assumed stationary (d=0 throughout).
    Seasonal period is capped at 60 (hourly cycle) so SARIMA/HW have ≥24 full
    cycles available in a 1-day (1440-point) context window.
    """
    series = np.asarray(series, dtype=np.float64)
    fallback = np.full(pred_len, float(np.mean(series)))

    if np.std(series) < 1e-6:
        return fallback

    sp = min(seasonal_period, 60, len(series) // 3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if method == "arima":
                ARIMA, _, _ = _require_statsmodels()
                fit = ARIMA(series, order=(1, 0, 1)).fit()
                mle = getattr(fit, 'mle_retvals', None)
                if mle is not None:
                    conv = mle.get('converged', True) if hasattr(mle, 'get') else getattr(mle, 'converged', True)
                    if not conv:
                        return fallback
                result = np.asarray(fit.forecast(steps=pred_len), dtype=np.float64)

            elif method == "sarima":
                _, SARIMAX, _ = _require_statsmodels()
                fit = SARIMAX(
                    series,
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 0, sp),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                mle = getattr(fit, 'mle_retvals', None)
                if mle is not None:
                    conv = mle.get('converged', True) if hasattr(mle, 'get') else getattr(mle, 'converged', True)
                    if not conv:
                        return fallback
                result = np.asarray(fit.forecast(steps=pred_len), dtype=np.float64)

            elif method == "prophet":
                Prophet = _require_prophet()
                import pandas as pd
                ds = pd.date_range("2025-06-16", periods=len(series), freq="15min")
                df = pd.DataFrame({"ds": ds, "y": series})
                # weekly_seasonality disabled: max context is 2880 min = 2 days,
                # far below the 7-day minimum needed for reliable weekly estimation.
                # Fitting weekly on 2 days produces spurious components that corrupt
                # the forecast mean, especially for asymmetry (T2) with weak daily signal.
                m = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_mode="additive",
                ).fit(df, algo="Newton")
                future = m.make_future_dataframe(periods=pred_len, freq="1min")
                fc = m.predict(future)
                result = np.asarray(fc["yhat"].iloc[-pred_len:].values, dtype=np.float64)

            elif method == "holtwinters":
                _, _, ExponentialSmoothing = _require_statsmodels()
                try:
                    fit = ExponentialSmoothing(
                        series,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=sp,
                        initialization_method="estimated",
                    ).fit(optimized=True)
                except Exception:
                    fit = ExponentialSmoothing(
                        series, trend="add", seasonal=None,
                        initialization_method="estimated",
                    ).fit(optimized=True)
                result = np.asarray(fit.forecast(pred_len), dtype=np.float64)

            else:
                raise ValueError(f"Unknown method: {method!r}")

            # Guard against NaN/inf or values wildly outside input range
            if not np.all(np.isfinite(result)):
                return fallback
            safe_range = 5.0 * max(float(np.abs(series).max()), 1.0)
            if float(np.abs(result).max()) > safe_range:
                return fallback
            return result

        except ImportError:
            raise
        except Exception:
            return fallback


# ---------------------------------------------------------------------------
# Per-(window, link) helpers — module-level so joblib can pickle them
# ---------------------------------------------------------------------------

def _peak_one(ctx: np.ndarray, pred_len: int, method: Method, sp: int, eval_start: int = 0) -> float:
    warnings.filterwarnings("ignore")
    return float(np.max(_forecast_series(ctx, pred_len, method, sp)[eval_start:]))

def _argmax_one(ctx: np.ndarray, pred_len: int, method: Method, sp: int, eval_start: int = 0) -> float:
    warnings.filterwarnings("ignore")
    return float(np.argmax(_forecast_series(ctx, pred_len, method, sp)[eval_start:]))


# ---------------------------------------------------------------------------
# T2 — Peak Utilization Exceedance Detection
# ---------------------------------------------------------------------------

def evaluate_classical_exceedance(
    contexts: np.ndarray,
    trajectories: np.ndarray,
    prediction_len: int,
    method: Method = "arima",
    max_windows: int = 50,
    seasonal_period: int = 96,
    threshold: float = 0.10,
    link_indices: list[int] | None = None,
    n_boot: int = 200,
    eval_start: int = 0,
) -> dict:
    """T2 classical exceedance: forecast trajectory[eval_start:] → max score → AUROC + BSS.

    eval_start : first step of the eval sub-window (0 = full horizon; 24 = second 24 h).
    Score for AUROC = max(forecast[eval_start:]) — continuous ranking signal.
    """
    N = min(len(contexts), max_windows)
    lnks = list(range(contexts.shape[2])) if link_indices is None else link_indices
    n_jobs = _n_jobs(method)
    print(f"      [{method}] {N}w × {len(lnks)}l = {N*len(lnks)} fits  ({n_jobs} workers)", flush=True)

    flat = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_peak_one)(contexts[i, :, l], prediction_len, method, seasonal_period, eval_start)
        for i in range(N) for l in lnks
    )
    pred_max  = np.array(flat, dtype=np.float32).reshape(N, len(lnks))
    y_true    = (trajectories[:N, eval_start:, :].max(axis=1)[:, lnks] > threshold).astype(np.float32)
    pred_prob = (pred_max > threshold).astype(np.float32)
    result = _exceedance_metrics(y_true.flatten(), pred_max.flatten(), pred_prob.flatten(), n_boot, "t2")
    result["t2_threshold"] = threshold
    return result


# ---------------------------------------------------------------------------
# T3 — Peak Timing Prediction
# ---------------------------------------------------------------------------

def evaluate_classical_timing(
    contexts: np.ndarray,
    timing: np.ndarray,
    prediction_len: int,
    method: Method = "arima",
    max_windows: int = 50,
    seasonal_period: int = 24,
    link_indices: list[int] | None = None,
    hours_per_step: float = 1.0,
    n_boot: int = 200,
    eval_start: int = 0,
) -> dict:
    """T3 classical: fit model → argmax(forecast[eval_start:]) → Skill / MAE.

    timing must already be sliced to the eval sub-window by make_peak_timing_windows(eval_start=...).
    Skill is computed vs random baseline for the sub-window size (prediction_len - eval_start).
    """
    N = min(len(contexts), max_windows)
    lnks = list(range(contexts.shape[2])) if link_indices is None else link_indices
    n_jobs = _n_jobs(method)
    print(f"      [{method}] {N}w × {len(lnks)}l = {N*len(lnks)} fits  ({n_jobs} workers)", flush=True)

    flat = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_argmax_one)(contexts[i, :, l], prediction_len, method, seasonal_period, eval_start)
        for i in range(N) for l in lnks
    )
    pred_timing = np.array(flat, dtype=np.float32).reshape(N, len(lnks))
    y_true  = timing[:N][:, lnks]
    eval_len = prediction_len - eval_start
    mae     = compute_timing_mae(y_true, pred_timing, hours_per_step)
    lo, hi  = _bootstrap_timing_mae(y_true, pred_timing, n_boot=n_boot, hours_per_step=hours_per_step)
    skill   = compute_timing_skill(mae, eval_len, hours_per_step)
    return {"mae_h": mae, "mae_ci": [lo, hi], "t3_skill": skill}


# ---------------------------------------------------------------------------
# Convenience: run all four methods at once
# ---------------------------------------------------------------------------

def evaluate_all_classical(
    contexts: np.ndarray,
    targets: np.ndarray,
    prediction_len: int,
    task: Literal["t2_exceedance", "t3_timing"] = "t2_exceedance",
    max_windows: int = 50,
    seasonal_period: int = 24,
    link_indices: list[int] | None = None,
    hours_per_step: float = 1.0,
    eval_start: int = 0,
    threshold: float = 0.10,
) -> dict[str, dict]:
    """Run ARIMA, SARIMA, Prophet, and Holt-Winters for the given task.

    task:
        "t2_exceedance" → evaluate_classical_exceedance  (AUROC + BSS)
        "t3_timing"     → evaluate_classical_timing      (MAE(h) + Skill)
    """
    results: dict[str, dict] = {}
    for method in ("arima", "sarima", "prophet", "holtwinters"):
        try:
            if task == "t2_exceedance":
                res = evaluate_classical_exceedance(
                    contexts, targets, prediction_len,
                    method=method, max_windows=max_windows,
                    seasonal_period=seasonal_period, link_indices=link_indices,
                    threshold=threshold, eval_start=eval_start,
                )
            elif task == "t3_timing":
                res = evaluate_classical_timing(
                    contexts, targets, prediction_len,
                    method=method, max_windows=max_windows,
                    seasonal_period=seasonal_period, link_indices=link_indices,
                    hours_per_step=hours_per_step, eval_start=eval_start,
                )
            else:
                raise ValueError(f"Unknown task: {task!r}")
            results[method] = res
        except ImportError as exc:
            results[method] = {"error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# T1 — Trajectory MASE (classical point forecasts vs seasonal naive)
# ---------------------------------------------------------------------------

def _traj_mase_one(ctx: np.ndarray, traj: np.ndarray, method: Method,
                   sp: int, eval_start: int) -> tuple[float, float]:
    """Return (ae_model, ae_sn) averaged over the eval window for one (window, link)."""
    warnings.filterwarnings("ignore")
    fc  = _forecast_series(ctx, len(traj), method, sp)
    y   = traj[eval_start:]
    yfc = fc[eval_start:]
    ysn = np.tile(ctx[-sp:], (len(y) // sp + 1))[:len(y)]
    return float(np.mean(np.abs(y - yfc))), float(np.mean(np.abs(y - ysn)))


def evaluate_classical_trajectory_mase(
    contexts: np.ndarray,
    trajectories: np.ndarray,
    prediction_len: int,
    method: Method = "holtwinters",
    max_windows: int = 100,
    seasonal_period: int = 24,
    link_indices: list[int] | None = None,
    n_boot: int = 200,
    eval_start: int = 0,
) -> dict:
    """T0: trajectory MASE vs seasonal naive for a classical model.

    MASE < 1 → better than seasonal naive.  Skill_SN = 1 − MASE.
    """
    N    = min(len(contexts), max_windows)
    lnks = list(range(contexts.shape[2])) if link_indices is None else link_indices
    n_jobs = _n_jobs(method)
    print(f"      [{method}] T1 MASE: {N}w × {len(lnks)}l = {N*len(lnks)} fits  ({n_jobs} workers)",
          flush=True)

    pairs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_traj_mase_one)(
            contexts[i, :, l], trajectories[i, :, l], method, seasonal_period, eval_start
        )
        for i in range(N) for l in lnks
    )
    ae_m  = np.array([p[0] for p in pairs], dtype=np.float32)
    ae_sn = np.array([p[1] for p in pairs], dtype=np.float32)
    mase     = float(ae_m.mean() / (ae_sn.mean() + 1e-10))
    skill_sn = 1.0 - mase
    lo, hi   = _bootstrap_mase(ae_m, ae_sn, n_boot)
    return {
        "t1_mase":     mase,
        "t1_skill_sn": skill_sn,
        "t1_mae":      float(ae_m.mean()),
        "t1_mae_sn":   float(ae_sn.mean()),
        "t1_mase_ci":  [lo, hi],
    }


# ---------------------------------------------------------------------------
# T4 — Traffic Asymmetry Forecasting (classical baseline)
# ---------------------------------------------------------------------------

def _asym_mean_one(ctx: np.ndarray, pred_len: int, method: Method,
                   sp: int, eval_start: int) -> float:
    warnings.filterwarnings("ignore")
    return float(np.mean(_forecast_series(ctx, pred_len, method, sp)[eval_start:]))


def evaluate_classical_asym(
    asym_contexts: np.ndarray,
    asym_targets: np.ndarray,
    prediction_len: int,
    method: Method = "holtwinters",
    max_windows: int = 100,
    seasonal_period: int = 24,
    link_indices: list[int] | None = None,
    n_boot: int = 200,
    eval_start: int = 0,
) -> dict:
    """T4: classical model forecasts asym series → mean(asym[eval_start:]) vs actual.

    Skill vs persistence (last observed asym) and vs seasonal naive.
    """
    N    = min(len(asym_contexts), max_windows)
    lnks = list(range(asym_contexts.shape[2])) if link_indices is None else link_indices
    n_jobs = _n_jobs(method)
    print(f"      [{method}] T4 asym: {N}w × {len(lnks)}l = {N*len(lnks)} fits  ({n_jobs} workers)",
          flush=True)

    flat = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_asym_mean_one)(
            asym_contexts[i, :, l], prediction_len, method, seasonal_period, eval_start
        )
        for i in range(N) for l in lnks
    )
    y_pred    = np.array(flat, dtype=np.float32).reshape(N, len(lnks))
    y_true    = asym_targets[:N, :][:, lnks]
    y_persist = asym_contexts[:N, -1, :][:, lnks]
    # seasonal naive: mean of same eval window 24h earlier in context
    eval_len  = prediction_len - eval_start
    sn_traj   = _seasonal_naive(asym_contexts[:N, :, :][:, :, lnks], eval_len, seasonal_period)
    y_sn      = sn_traj.mean(axis=1)

    mae_m  = float(np.abs(y_true - y_pred).mean())
    mae_p  = float(np.abs(y_true - y_persist).mean())
    mae_sn = float(np.abs(y_true - y_sn).mean())

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
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
