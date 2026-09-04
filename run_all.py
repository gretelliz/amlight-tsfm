"""Master runner: runs the benchmark experiments and saves results/all_results.json.

Usage (from amlight-tsfm/ root):
    python run_all.py                                 # all experiments, all models
    python run_all.py --exp 1                         # classical baselines only
    python run_all.py --exp 2 --models chronos         # zero-shot, one model
    python run_all.py --exp 3                          # Full DA (all backbone weights)
    python run_all.py --exp 3-lora                     # LoRA domain adaptation
    python run_all.py --exp 3-elora                    # Extended LoRA (+ FFN, output layer)
    python run_all.py --exp 3-lpft                     # LP-FT (frozen backbone + MLP head)
    python run_all.py --exp 3-reducedlr                # TimesFM reduced-LR ablation (~1h)
    python run_all.py --exp 4                          # ZS+MLP@n few-shot learning curves
    python run_all.py --exp 5                          # RevIN ablation (~1h)
    python run_all.py --exp 6                          # Long-context ablation (~1.5h)
    python run_all.py --exp 2 --models ttm              # TTM zero-shot only
    python run_all.py --exp 3 --models ttm              # TTM Full DA only

After completion run:
    python compute_significance.py

NOTE: exp3 (Full DA, Stage 1 domain adaptation) ALWAYS runs when selected.
      exp4 (few-shot) uses the zero-shot backbone directly — it does not
      require exp3 to have run first.
      exp3-lora / exp3-elora require LoRA-capable models (chronos, timesfm, moirai).
      exp3-lpft works independently of exp3.
      exp3-reducedlr runs TimesFM only (the catastrophic-forgetting ablation).
      exp5 (RevIN): TimesFM already has RevIN internally — double normalization.
      exp6 (long-context): TTM crops 672h to its native 512h; results may
      match the exp2 baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml

os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")


def _load_cached(path: Path) -> dict | None:
    if path.exists():
        print(f"  Loading cached results from {path}")
        return json.loads(path.read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TSFM benchmark experiments")
    ap.add_argument("--exp", type=str,
                    choices=["1", "2", "3", "3-lora", "3-elora", "3-lpft",
                             "3-reducedlr", "4", "5", "6"],
                    help="Run only this experiment")
    ap.add_argument("--models", nargs="+",
                    choices=["chronos", "moirai", "timesfm", "patchtst", "ttm"],
                    help="Restrict experiments 2/3/4/5/6 to these models")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip all experiments; just merge cached JSONs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config.yaml").read_text())

    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        args.exp = "-1"

    run_all = args.exp is None
    exp1 = exp2 = exp3 = exp4 = None
    exp3_lora = exp3_elora = exp3_lpft = exp3_reducedlr = None
    exp5 = exp6 = None

    # ── Experiment 1: Classical baselines ─────────────────────────────────────
    if run_all or args.exp == "1":
        from experiments.exp1_baseline import run as _run
        exp1 = _run(cfg, out_dir)
    else:
        exp1 = _load_cached(out_dir / "exp1_results.json") or {}

    # ── Experiment 2: Zero-shot TSFM evaluation ───────────────────────────────
    if run_all or args.exp == "2":
        from experiments.exp2_zeroshot import run as _run
        exp2 = _run(cfg, out_dir, model_names=args.models)
    else:
        exp2 = _load_cached(out_dir / "exp2_results.json") or {}

    # ── Experiment 3: Full DA (all backbone weights updated) ─────────────────
    if run_all or args.exp == "3":
        from experiments.exp3_full_da import run as _run
        exp3 = _run(cfg, out_dir, model_names=args.models, zeroshot_results=exp2)
    else:
        exp3 = _load_cached(out_dir / "exp3_fulldA_results.json") or {}

    # ── Experiment 3 (LoRA) ────────────────────────────────────────────────────
    if run_all or args.exp == "3-lora":
        from experiments.exp3_lora import run as _run
        exp3_lora = _run(cfg, out_dir, model_names=args.models, zeroshot_results=exp2)
    else:
        exp3_lora = _load_cached(out_dir / "exp3_lora_results.json") or {}

    # ── Experiment 3 (ELoRA) ───────────────────────────────────────────────────
    if run_all or args.exp == "3-elora":
        from experiments.exp3_elora import run as _run
        exp3_elora = _run(cfg, out_dir, model_names=args.models)
    else:
        exp3_elora = _load_cached(out_dir / "exp3_elora_results.json") or {}

    # ── Experiment 3 (LP-FT) ───────────────────────────────────────────────────
    if run_all or args.exp == "3-lpft":
        from experiments.exp3_lp_ft import run as _run
        exp3_lpft = _run(cfg, out_dir, model_names=args.models)
    else:
        exp3_lpft = _load_cached(out_dir / "exp3_lpft_results.json") or {}

    # ── Experiment 3 (reduced-LR ablation, TimesFM only) ──────────────────────
    if run_all or args.exp == "3-reducedlr":
        from experiments.exp3_reduced_lr_ablation import run as _run
        exp3_reducedlr = _run(cfg, out_dir)
    else:
        exp3_reducedlr = _load_cached(out_dir / "exp3_reducedlr_results.json") or {}

    # ── Experiment 5 (RevIN ablation) ─────────────────────────────────────────
    if run_all or args.exp == "5":
        from experiments.exp5_revin_ablation import run as _run
        exp5 = _run(cfg, out_dir, model_names=args.models)
    else:
        exp5 = _load_cached(out_dir / "exp5_revin_results.json") or {}

    # ── Experiment 6 (long-context ablation) ──────────────────────────────────
    if run_all or args.exp == "6":
        from experiments.exp6_longcontext_ablation import run as _run
        exp6 = _run(cfg, out_dir, model_names=args.models)
    else:
        exp6 = _load_cached(out_dir / "exp6_longcontext_results.json") or {}

    # ── Experiment 4: Few-shot learning curves (ZS+MLP@n) ─────────────────────
    if run_all or args.exp == "4":
        from experiments.exp4_fewshot import run as _run
        exp4 = _run(cfg, out_dir, model_names=args.models)
    else:
        exp4 = _load_cached(out_dir / "exp4_results.json") or {}

    # ── Merge ─────────────────────────────────────────────────────────────────
    all_results = {
        "exp1":          exp1          or {},
        "exp2":          exp2          or {},
        "exp3":          exp3          or {},
        "exp3_lora":     exp3_lora     or {},
        "exp3_elora":    exp3_elora    or {},
        "exp3_lpft":     exp3_lpft     or {},
        "exp3_reducedlr": exp3_reducedlr or {},
        "exp4":          exp4          or {},
        "exp5":          exp5          or {},
        "exp6":          exp6          or {},
    }
    out_path = out_dir / "all_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"  all_results.json → {out_path}")
    print(f"  Next: python compute_significance.py")
    print(f"{'='*60}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    nan = float("nan")

    def _fmt(v: float) -> str:
        return f"{v:+.3f}" if v == v else "  nan "

    def _fmth(v: float) -> str:
        return f"{v:.2f}" if v == v else " nan"

    # Summary table: T1 Skill_SN / T2 AUROC / T3 Skill / T4 MAE
    print(f"\n  {'Model':<28}  {'T1 Sk_SN':>9} {'T2 AUROC':>9} {'T3 Skill':>9} {'T4 MAE':>8}")
    print("  " + "-" * 72)

    # Classical baselines from exp1
    if exp1:
        for label, t1key, t2key, t3key, t4key in [
            ("Persistence", None, "b1_persistence", "b3_persistence_timing", "b4_persistence"),
            ("ARIMA", "classical_arima", "classical_arima", "classical_arima", "classical_arima"),
            ("Prophet", "classical_prophet", "classical_prophet", "classical_prophet", "classical_prophet"),
        ]:
            t1s = exp1.get("t1_trajectory", {}).get(t1key, {}).get("t1_skill_sn", nan) if t1key else nan
            t2s = exp1.get("t2_exceedance", {}).get(t2key, {}).get("t2_auroc", nan)
            t3d = exp1.get("t3_timing",     {}).get(t3key, {})
            t3s = 0.0 if t3key == "b3_persistence_timing" else t3d.get("t3_skill", nan)
            t4s = exp1.get("t4_asym",       {}).get(t4key, {}).get("t4_mae", nan)
            print(f"  {label:<28}  {_fmt(t1s):>9} {_fmt(t2s):>9} {_fmt(t3s):>9} {_fmth(t4s):>8}")

    print("  " + "-" * 72)

    # TSFM: zero-shot, Full DA, LoRA, ELoRA, LP-FT, ablations
    stages = [
        ("ZS",       exp2          or {}),
        ("FullDA",   exp3          or {}),
        ("LoRA",     exp3_lora     or {}),
        ("ELoRA",    exp3_elora    or {}),
        ("LPFT",     exp3_lpft     or {}),
        ("RevIN",    exp5          or {}),
        ("LongCtx",  exp6          or {}),
    ]
    for stage, res in stages:
        for mname, r in res.items():
            t1s = r.get("t1_trajectory", {}).get("t1_skill_sn", nan)
            t2s = r.get("t2_exceedance", {}).get("t2_auroc", nan)
            t3d = r.get("t3_timing",     {})
            t3s = t3d.get("t3_skill_persist", t3d.get("t3_skill", nan))
            t4s = r.get("t4_asym",       {}).get("t4_mae", nan)
            label = f"{mname} ({stage})"
            print(f"  {label:<28}  {_fmt(t1s):>9} {_fmt(t2s):>9} {_fmt(t3s):>9} {_fmth(t4s):>8}")

    # Few-shot: best N per model (ZS+MLP@n)
    if exp4:
        print("  " + "-" * 72)
        for mname, curves in exp4.items():
            def _best_auroc(key: str) -> dict:
                pts = curves.get(key, [])
                return max(pts, key=lambda p: p.get("auroc_mean", -999), default={})

            def _best_skill(key: str) -> dict:
                pts = curves.get(key, [])
                return max(pts, key=lambda p: p.get("skill_mean", -999), default={})

            def _best_asym(key: str) -> dict:
                pts = curves.get(key, [])
                return min(pts, key=lambda p: p.get("mae_mean", 999), default={})

            t2b = _best_auroc("t2_exceedance_mlp")
            t3b = _best_skill("t3_timing_mlp")
            t4b = _best_asym("t4_asym_mlp")
            t2s = t2b.get("auroc_mean", nan); n = t2b.get("n", "?")
            t3s = t3b.get("skill_mean", nan)
            t4s = t4b.get("mae_mean", nan)
            label = f"{mname} (ZS+MLP@{n})"
            print(f"  {label:<28}  {'  nan ':>9} {_fmt(t2s):>9} {_fmt(t3s):>9} {_fmth(t4s):>8}")


if __name__ == "__main__":
    main()
