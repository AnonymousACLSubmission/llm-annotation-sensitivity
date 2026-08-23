# Annotation Sensitivity Analysis — Data & Code Pipeline

Reproducibility package for "Reliable but Sensitive: Evaluating LLM Annotation Beyond Performance"
August 2026

## Quick Start

```bash
cd data_work/
USE_CACHE=1 Rscript src/run_from_raw.R
```

This runs all 8 stages and produces every table and figure in the paper. Set `USE_CACHE=1` to skip raw-data processing when `processed/df_long.csv` already exists; omit it to rebuild from scratch.

## Requirements

- R ≥ 4.3
- Packages: `tidyverse`, `lme4`, `irr`, `ggplot2`, `patchwork`, `xtable`, `arrow`
- System: macOS or Linux; uses `grDevices::pdf()` for figures (no cairo dependency)

Install missing packages:

```r
install.packages(c("tidyverse", "lme4", "irr", "patchwork", "xtable", "arrow"))
```

## Directory Structure

```
data_work/
├── README.md                 ← this file
├── src/                      ← all analysis code
│   ├── run_from_raw.R        ← single entry point (raw → paper)
│   ├── run_all.R             ← orchestrator; checks all artifacts exist
│   ├── 00_utils.R            ← shared helpers (crossed_prevalence_components, etc.)
│   ├── process_raw_data.R    ← Stage 0: raw CSVs → processed data
│   ├── 01_agreement.R        ← Stage 1: Krippendorff's α, pairwise agreement
│   ├── 02_prevalence.R       ← Stage 2: prevalence estimates + Figure 1
│   ├── 03_deff.R             ← Stage 3: crossed-REML variance decomposition → deff
│   ├── 04_human_design.R     ← Stage 4: human benchmarks + framework table
│   ├── 05_confidence.R       ← Stage 5: two-panel confidence figure
│   ├── 06_figures.R          ← Stage 6: heatmaps, distributions, remaining plots
│   ├── 07_tables.R           ← Stage 7: LaTeX table fragments
│   └── 08_paper_tables.R     ← Stage 8: framework + cost tables
├── raw/                      ← raw LLM annotation outputs (input data)
│   ├── gpt4o/                ← 4 CSVs (conditions A, B, C + variants)
│   ├── gpt4o-mini/
│   ├── gpt4.1/
│   ├── gpt4.1-mini/
│   ├── gpt4.1-nano/
│   ├── claude-sonnet/
│   ├── claude-haiku/
│   ├── llama4-scout/
│   └── llama4-maverick/
│       └── *.csv             ← 36 CSVs total across 9 model directories
├── Tweets_CK/                ← Kern et al. human reference annotations
│   └── *.csv
├── processed/                ← generated intermediate data (gitignored or cached)
│   ├── df_long.csv           ← long-format item × model × condition × run
│   ├── df_agg.csv            ← aggregated prevalence by model × condition × run
│   ├── kern_full.csv         ← processed Kern reference data
│   └── prevalences.csv       ← final prevalence estimates with CIs
└── outputs/                  ← generated LaTeX table fragments
    ├── table_agreement.tex
    ├── table_prevalence.tex
    ├── table_deff.tex
    ├── table_framework.tex
    ├── table_cost.tex
    ├── table_confidence.tex
    ├── table_human.tex
    ├── table_variance.tex
    └── table_summary.tex

plots/                        ← generated figures (sibling to data_work/)
├── fig_prevalence.pdf
├── fig_prevalence.png
├── fig_confidence.pdf
├── fig_confidence.png
├── fig_heatmap_agreement.pdf
├── fig_heatmap_agreement.png
├── fig_distribution.pdf
└── fig_distribution.png
```

## Pipeline Stages

| # | Script | What it produces |
|---|--------|-----------------|
| 0 | `process_raw_data.R` | `processed/df_long.csv`, `df_agg.csv`, `kern_full.csv` |
| 1 | `01_agreement.R` | `outputs/table_agreement.tex` |
| 2 | `02_prevalence.R` | `outputs/table_prevalence.tex`, `plots/fig_prevalence.*` |
| 3 | `03_deff.R` | `outputs/table_deff.tex`, `outputs/table_variance.tex` |
| 4 | `04_human_design.R` | `outputs/table_framework.tex`, `outputs/table_human.tex` |
| 5 | `05_confidence.R` | `outputs/table_confidence.tex`, `plots/fig_confidence.*` |
| 6 | `06_figures.R` | `plots/fig_heatmap_agreement.*`, `plots/fig_distribution.*` |
| 7 | `07_tables.R` | `outputs/table_summary.tex` |
| 8 | `08_paper_tables.R` | `outputs/table_cost.tex` (finalizes framework table) |

## Key Methods

### Crossed-REML variance decomposition (Stage 3)

The design effect (deff) quantifies how much annotation sensitivity inflates variance relative to simple random sampling. We estimate variance components using a crossed random-effects model fit by REML:

```
prevalence ~ 1 + (1|model) + (1|condition) + (1|model:condition)
```

- **σ²_model**: variance attributable to model choice
- **σ²_condition**: variance attributable to prompt wording (task structure)
- **σ²_interaction**: residual model × condition interaction
- **σ²_residual**: within-cell (across-run) replication noise

The design effect is:

```
deff = (σ²_model + σ²_condition + σ²_interaction + σ²_residual) / σ²_residual
```

This replaces an earlier nested decomposition that double-counted shared variance. The crossed model correctly partitions total variance when models and conditions are fully crossed (every model is run under every condition).

## Reproducibility Notes

- All random seeds are fixed in the raw data (LLM temperature = 0 for deterministic runs where supported; 5 replicate runs per cell otherwise).
- The pipeline is idempotent: re-running produces identical outputs.
- Figure output uses `grDevices::pdf()` (not cairo) for macOS compatibility.
- REML is used for variance-component estimation (unbiased with few groups); AIC/BIC comparisons use ML refit via `lme4::refitML()`.
