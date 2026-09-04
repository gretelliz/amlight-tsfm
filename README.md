# A Multi-Task Benchmark for Time Series Foundation Models on R&E Backbone Telemetry

Code accompanying the paper **"A Multi-Task Benchmark for Time Series Foundation Models
on R&E Backbone Telemetry"**, accepted at **INDIS @ SC26**.

**CIARA — Florida International University**

---

## Overview

Research and Education (R&E) backbone networks carry petabyte-scale scientific
data transfers, but their operational monitoring remains largely reactive
(threshold alerts, dashboards) rather than predictive. This repository
evaluates whether pre-trained **Time Series Foundation Models (TSFMs)** —
TimesFM, Moirai, Chronos, PatchTST, and TTM/Granite — can support predictive
monitoring of R&E backbone telemetry, using 26 months of SNMP data from the
AmLight backbone with **no human-annotated labels**.

We benchmark five TSFMs across four adaptation strategies (zero-shot, full
domain adaptation, LoRA/ELoRA, linear-probe fine-tuning) and a label-efficient
few-shot protocol (ZS+MLP@n), on four auto-labeled operational tasks:

| Task | NOC question | Metric |
|---|---|---|
| **T1** — Trajectory Forecasting | How will link utilization evolve over 48 h? | Skill_SN (MASE vs. seasonal-naive) ↑ |
| **T2** — Peak Exceedance Detection | Will a link exceed 10 Gbps in the next 24-48 h? | AUROC ↑ |
| **T3** — Peak Timing Prediction | When will the daily peak arrive? | Skill (vs. random) ↑ |
| **T4** — Traffic Asymmetry Forecasting | Is outbound traffic dominant? | MAE ↓ |

All labels are derived automatically from SNMP MIB-II counters
(`ifHCInOctets`, `ifHCOutOctets`) — no operator annotation is required.

**Key finding:** TSFM performance is task-structured, not uniformly superior.
Zero-shot models beat classical baselines (ARIMA, SARIMA, Prophet,
Holt-Winters) on trajectory and timing tasks, where broad pre-training
captures diurnal patterns, but not on tasks that reward high-autocorrelation
exploitation. Full-parameter fine-tuning risks catastrophic forgetting;
linear-probe fine-tuning avoids it but is task-selective. A frozen backbone
with a small, task-specific MLP head (ZS+MLP@n) resolves both failure modes
and is the recommended deployment strategy — see the paper for the full
analysis.

## Dataset

The AmLight-SNMP-TS dataset consists of 26 months of 60-minute SNMP telemetry from 195 physical 100GE backbone interfaces (32 active links, mean utilization ≥ 0.5%), with pre-defined chronological splits (domain-adapt / train / val / test).

**[DATASET_URL / DOI — TODO]**

Download it and place it at `data/AmLight-SNMP-TS/` (or point `data.raw_dir`
in `config.yaml` at wherever you extracted it), so that
`data/AmLight-SNMP-TS/telemetry.parquet` and `split_info.parquet` exist.
`data/loaders.py` reads the dataset from that path; no other file in this
repository builds, cleans, or publishes it.

## Requirements

- Python ≥ 3.11 (required by the `ttm` extra's `granite-tsfm` dependency).
- `numpy<2` and `transformers<5` are pinned explicitly — several TSFM
  libraries used here are not yet compatible with NumPy 2.x or Transformers
  5.x, and an unconstrained `pip install` can silently resolve to versions
  that break at import or inference time.
- `granite-tsfm` (the `ttm` extra) currently requires `torch>=2.10,<2.12`.
  On platforms without a prebuilt torch wheel in that range (e.g. Intel
  macOS, where the last available wheel is 2.2.x), installing `.[ttm]`
  will fail to resolve — install the other four models' extras
  (`chronos,moirai,timesfm,classical`) in that case and run TTM separately
  on a supported platform (Linux + CUDA, as used for the paper's results).

## Quick Start

```bash
# 1. Install dependencies (pick the extras for the models you need)
pip install -e ".[chronos,moirai,timesfm,ttm,classical]"
# If the command above fails to resolve (see Requirements above), drop ttm:
#   pip install -e ".[chronos,moirai,timesfm,classical]"

# 2. Place the dataset (see Dataset section above)
#    data/AmLight-SNMP-TS/telemetry.parquet
#    data/AmLight-SNMP-TS/split_info.parquet

# 3. Run the full experiment suite
python run_all.py

# 4. Run a single experiment
python run_all.py --exp 1                      # classical baselines (Table 2)
python run_all.py --exp 2 --models chronos      # zero-shot, one model (Table 3)
python run_all.py --exp 3                       # Full DA (Table 4)
python run_all.py --exp 3-lpft                  # LP-FT (Table 4)
python run_all.py --exp 4                       # ZS+MLP@n few-shot curves (Figs. 1-3)

# 5. Statistical significance (Wilcoxon signed-rank, Bonferroni-corrected)
python compute_significance.py

# 6. Reproduce the learning-curve figures
python plot_learning_curves.py
```

Results are saved to `results/`; figures to `results/figures/` and `figures/`.

## Models

| Model | Source | Pre-training | Install extra |
|---|---|---|---|
| TimesFM | Google | Decoder-only, large heterogeneous corpus | `timesfm` |
| Moirai 1.1 | Salesforce | Masked-patch mixture-of-distributions | `moirai` |
| Chronos | Amazon | Discretized-token language-model objective | `chronos` |
| TTM/Granite | IBM | Multi-scale patch MSE, compact (≈1-5M params) | `ttm` |
| PatchTST | Academic | None — trained from scratch on AmLight | (bundled) |

## Adaptation strategies

| Strategy | Script | Backbone update | Notes |
|---|---|---|---|
| Zero-shot (ZS) | `exp2_zeroshot.py` | none | Direct inference, no gradient updates |
| Full DA | `exp3_full_da.py` | all parameters | Native self-supervised objective per model; risk of catastrophic forgetting |
| LoRA | `exp3_lora.py` | rank-8 attention adapters (~0.6%) | Frozen base, no forgetting |
| ELoRA | `exp3_elora.py` | LoRA + FFN/output layers (~1.2%) | Wider adapter set |
| LP-FT | `exp3_lp_ft.py` | none (frozen) + MLP head | Task-selective — see paper §5.3 |
| ZS+MLP@n | `exp4_fewshot.py` | none (frozen) + per-task MLP head | Recommended: correct loss per task, N ≤ 500 labeled windows |

## Hardware

All experiments were run on a single NVIDIA A30 (24 GB). Stage 1 domain
adaptation (Full DA / LoRA / ELoRA) requires roughly 5-9 GPU-hours per model;
Stage 2 task heads (ZS+MLP@n, LP-FT) train in under a second per head on
CPU. Inference is real-time (8-120 ms per window on CPU) for all five models.

## Citation

```bibtex
@inproceedings{amlight_tsfm_indis2026,
  title     = {A Multi-Task Benchmark for Time Series Foundation Models
               on R&E Backbone Telemetry},
  author    = {[AUTHOR NAMES — TODO]},
  booktitle = {Proceedings of the INDIS Workshop at SC26},
  year      = {2026}
}
```

If you use the AmLight-SNMP-TS dataset, please also cite it separately —
see its own repository for citation details.

## License

MIT — see [LICENSE](LICENSE).
