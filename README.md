# IMST-Mamba: Informative Missingness State-Space Model for Early Sepsis Prediction

> **MLHC 2026 Submission**

IMST-Mamba encodes a three-state missingness taxonomy (*never observed*, *recently observed*, *stale*) within a selective state-space backbone for early sepsis prediction in ICU patients. On the PhysioNet Sepsis Challenge 2019 dataset, it achieves AUROC 0.8514 on long-stay patients, outperforming Transformer (+3.9 points) and GRU-D in the clinically critical subgroup where temporal missingness patterns are most complex.

## Key Results

| Model | Overall AUROC | Long-stay AUROC | Se@Sp90 | Mean EWT |
|-------|:---:|:---:|:---:|:---:|
| LSTM | 0.7800 | 0.8419 | 0.469 | 48.0h |
| GRU-D | 0.7770 | 0.8468 | 0.462 | 46.1h |
| Transformer | **0.8517** | 0.8122 | **0.552** | 56.7h |
| **IMST-Mamba** | 0.7791 | **0.8514** | 0.484 | 44.7h |

Long stays are the only subgroup IMST-Mamba wins. Transformer is ahead
everywhere else, by a wide margin on short stays (Table 2 in
[`paper_draft.md`](paper_draft.md), §5.2):

| Subgroup | IMST-Mamba | Transformer | GRU-D | LSTM |
|---------|:---:|:---:|:---:|:---:|
| Overall | 0.7791 | **0.8517** | 0.7770 | 0.7800 |
| Low missingness | 0.7738 | **0.8253** | 0.7789 | 0.7645 |
| High missingness | 0.7772 | **0.8427** | 0.7710 | 0.7846 |
| High lab-missingness | 0.7459 | **0.8147** | 0.7714 | 0.7769 |
| Short stays | 0.6474 | **0.9392** | 0.6306 | 0.7093 |
| Long stays | **0.8514** | 0.8122 | 0.8468 | 0.8419 |

The short-stay gap (0.6474 vs 0.9392) is the honest counterpart to the
long-stay result: a short ICU stay has a brief, recent observation
history, so there is little staleness for this design to read, and
conditioning on it costs accuracy. The claim here is a subgroup where
missingness structure carries signal — not a better model overall.

**Ablation** — removing the staleness signal ($s=0$) drops AUROC by **−0.121** (vs −0.039 for the binary mask), confirming that time-since-last-observation is the single most predictive component.

## Architecture

```
Input: x (features) + m (mask) + s (staleness) + δt (inter-event gap)
         │
         ▼
┌─────────────────────────────┐
│  Three-State Missingness    │  state ∈ {never, recent, stale}
│  Encoder + Staleness Emb.  │  log(1 + s_tf) gated by state
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Temporal Decay Imputation  │  γ_f(s) = exp(−max(0,w_f)·s)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  L=4 IMST-Mamba Blocks      │  Selective SSM, Δt conditioned on time emb.
└─────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Sepsis    Mortality + SOFA   (multi-task regularization)
logit     (auxiliary heads)
```

## Project Structure

```
Triage/
├── src/
│   ├── models/
│   │   ├── imst_mamba.py          # Main model
│   │   ├── modules/
│   │   │   ├── missingness_encoder.py
│   │   │   ├── selective_ssm.py
│   │   │   ├── temporal_decay.py
│   │   │   └── time_embedding.py
│   │   ├── transformer_baseline.py
│   │   ├── grud.py
│   │   ├── lstm_baseline.py
│   │   └── retain.py
│   ├── data/
│   │   ├── load_challenge2019.py  # PhysioNet 2019 loader
│   │   ├── build_timeseries.py
│   │   ├── dataset.py
│   │   └── splits.py
│   ├── training/
│   │   ├── trainer.py             # Mixed precision, warmup, early stopping
│   │   └── losses.py              # Focal loss + multi-task loss
│   └── evaluation/
│       ├── metrics.py             # AUROC/AUPRC with bootstrap CI
│       ├── early_warning_time.py
│       └── calibration.py
├── scripts/
│   ├── run_pipeline.py            # End-to-end training script
│   ├── analyze_clinical_comparison.py   # B1: vs qSOFA/SOFA
│   ├── analyze_mc_dropout.py            # B2: Uncertainty quantification
│   ├── analyze_ewt.py                   # B3: Early Warning Time
│   ├── analyze_imputation_comparison.py # B4: Imputation strategies
│   ├── analyze_ablation.py              # Ablation study
│   ├── analyze_subgroups.py             # Subgroup analysis
│   └── generate_figures.py             # Paper figures
├── configs/
│   ├── base.yaml
│   ├── model_imst_mamba.yaml
│   ├── model_transformer.yaml
│   └── model_grud.yaml
├── tests/
├── paper_imst_mamba_standalone.tex  # Paper (compiles without jmlr2e)
├── paper_imst_mamba.tex             # Paper (MLHC/PMLR official template)
└── references.bib
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
# For Mamba SSM (requires CUDA):
# pip install mamba-ssm causal-conv1d
```

### Data

Download the PhysioNet Computing in Cardiology Challenge 2019 dataset:

```bash
python scripts/download_challenge2019.py   # requires PhysioNet credentials
# or manually: https://physionet.org/content/challenge-2019/1.0.0/
```

Place data in `data/raw/training_setA/` and `data/raw/training_setB/`.

### Training

```bash
# Full pipeline: data preprocessing → train all models → evaluation
python scripts/run_pipeline.py --stage data
python scripts/run_pipeline.py --stage train --model imst_mamba --seeds 42 123 456
python scripts/run_pipeline.py --stage eval
```

Or use the Kaggle notebooks (`notebook_1_data.ipynb` → `notebook_4_analysis.ipynb`).

### Analysis

```bash
python scripts/analyze_clinical_comparison.py   # qSOFA/SOFA comparison
python scripts/analyze_mc_dropout.py            # Uncertainty (MC Dropout)
python scripts/analyze_ewt.py                   # Early Warning Time
python scripts/analyze_imputation_comparison.py # Imputation strategies
python scripts/generate_figures.py              # Paper figures
```

## Citation

```bibtex
@misc{imstmamba2026,
  title  = {IMST-Mamba: Informative Missingness State-Space Model
            for Early Sepsis Prediction in ICU Patients},
  author = {Park, Yunhu},
  year   = {2026},
  note   = {Manuscript under review at the Machine Learning for
            Healthcare Conference (MLHC) 2026. Not yet peer reviewed;
            please do not cite as published.}
}
```

## License

MIT License. Note: The PhysioNet 2019 dataset is subject to its own
[Data Use Agreement](https://physionet.org/content/challenge-2019/1.0.0/).
