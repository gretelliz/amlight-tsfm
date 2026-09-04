from .forecasting import (
    FrozenBackboneAdapter,
    compute_skill,
    compute_timing_mae,
    compute_timing_skill,
    # T1 — Trajectory Forecasting (MASE + CRPS)
    evaluate_trajectory_mase_tsfm,
    evaluate_trajectory_crps_tsfm,
    # T2 — Peak Utilization Exceedance Detection (AUROC + BSS)
    evaluate_exceedance_persistence,
    evaluate_exceedance_tsfm,
    # T3 — Peak Timing Prediction (MAE + Skill)
    evaluate_timing_persistence,
    evaluate_timing_tsfm,
    # T4 — Traffic Asymmetry Forecasting (MAE + Skill)
    make_asym_windows,
    evaluate_asym_persistence,
    evaluate_asym_tsfm,
    # Window builders
    make_trajectory_windows,
    make_peak_timing_windows,
    _compute_auroc,
    _brier_skill_score,
    # backward compat
    compute_r2,
    compute_peak_r2,
)
from .baselines_classical import (
    evaluate_all_classical,
    evaluate_classical_exceedance,
    evaluate_classical_timing,
    evaluate_classical_trajectory_mase,
    evaluate_classical_asym,
)

__all__ = [
    # T1 — Trajectory Forecasting Accuracy (MASE + CRPS)
    "make_trajectory_windows",
    "evaluate_trajectory_mase_tsfm",
    "evaluate_trajectory_crps_tsfm",
    # T2 — Peak Utilization Exceedance Detection (AUROC + BSS)
    "evaluate_exceedance_persistence",
    "evaluate_exceedance_tsfm",
    # T3 — Peak Timing Prediction (MAE + Skill)
    "make_peak_timing_windows",
    "compute_timing_mae",
    "compute_timing_skill",
    "evaluate_timing_persistence",
    "evaluate_timing_tsfm",
    # T4 — Traffic Asymmetry Forecasting (MAE + Skill)
    "make_asym_windows",
    "evaluate_asym_persistence",
    "evaluate_asym_tsfm",
    # Classical baselines
    "evaluate_classical_exceedance",
    "evaluate_classical_timing",
    "evaluate_classical_trajectory_mase",
    "evaluate_classical_asym",
    "evaluate_all_classical",
    # Frozen-backbone feature extraction (Experiment 4, LP-FT)
    "FrozenBackboneAdapter",
    "_compute_auroc",
    "_brier_skill_score",
    "compute_skill",
    # backward compat
    "compute_r2",
    "compute_peak_r2",
]
