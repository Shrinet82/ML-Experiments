# Chronos-2 vs Standard Baselines on M5 Retail Demand

An empirical benchmark evaluating **Amazon Chronos-2** ([arXiv:2510.15821](https://arxiv.org/abs/2510.15821), October 2025) against classical and machine learning baselines on the Walmart M5 retail demand dataset.

## Summary Verdict
> **Across 1000 M5 retail series over 3 rolling origins, Chronos-2 with covariates achieves a MASE of 1.254 and WQL of 12734.058, achieving a WIN verdict against Seasonal Naive (95% CI [0.379, 0.437]) and a WIN against AutoETS (95% CI [0.185, 0.226]). Compared against the feature-engineered LightGBM baseline, Chronos-2 (Covariates) records a WIN on MASE (mean paired difference +1.817, 95% CI [1.324, 2.433]) and a WIN on WQL (95% CI [17873.546, 57665.519]). Supplying in-context price, event, and SNAP covariates to Chronos-2 yields a LOSS relative to zero-shot univariate Chronos-2 with a paired MASE difference of -0.008 (95% CI [-0.013, -0.002]), with a quantile crossing rate of 1.17%.**

## Overall Performance Table
| Model | MASE (Mean ± Std) | WQL (0.1..0.9) | Newsvendor 3:1 ($) | Quantile Crossing Rate | Runtime / 1k Series |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AutoETS** | 1.459 ± 1.077 | 20245.017 ± 99293.355 | $69.59 | 0.00% | 190.2s |
| **Chronos-2 (Covariates)** | 1.254 ± 1.014 | 12734.058 ± 156496.920 | $69.45 | 1.17% | 55.5s |
| **Chronos-2 (Zero-Shot)** | 1.246 ± 1.012 | 5451.720 ± 43941.233 | $69.21 | 1.76% | 10.4s |
| **Chronos-Bolt** | 1.262 ± 1.024 | 5426.543 ± 39904.619 | $69.30 | 4.15% | 9.3s |
| **LightGBM** | 3.070 ± 9.059 | 49874.159 ± 308319.769 | $130.90 | 0.77% | 50.9s |
| **Seasonal Naive** | 1.661 ± 1.312 | 9227.526 ± 50034.802 | $75.33 | 0.00% | 0.4s |

## Paired Bootstrap Significance (10,000 Resamples, Seed 42)
| Baseline Model | Metric | Mean Paired Difference | 95% Bootstrap CI | Verdict | Multiple Correction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AutoETS** | MASE | +0.205 | [0.185, 0.226] | **WIN** | None (No correction applied) |
| **AutoETS** | WQL | +7510.959 | [-902.918, 14443.953] | **TIE** | None (No correction applied) |
| **AutoETS** | Newsvendor (3:1) | +0.140 | [-1.431, 1.979] | **TIE** | None (No correction applied) |
| **Chronos-2 (Zero-Shot)** | MASE | -0.008 | [-0.013, -0.002] | **LOSS** | None (No correction applied) |
| **Chronos-2 (Zero-Shot)** | WQL | -7282.338 | [-17211.446, -215.633] | **LOSS** | None (No correction applied) |
| **Chronos-2 (Zero-Shot)** | Newsvendor (3:1) | -0.240 | [-1.179, 0.709] | **TIE** | None (No correction applied) |
| **Chronos-Bolt** | MASE | +0.008 | [0.001, 0.015] | **WIN** | None (No correction applied) |
| **Chronos-Bolt** | WQL | -7307.515 | [-17893.013, 160.351] | **TIE** | None (No correction applied) |
| **Chronos-Bolt** | Newsvendor (3:1) | -0.148 | [-1.330, 1.214] | **TIE** | None (No correction applied) |
| **LightGBM** | MASE | +1.817 | [1.324, 2.433] | **WIN** | None (No correction applied) |
| **LightGBM** | WQL | +37140.101 | [17873.546, 57665.519] | **WIN** | None (No correction applied) |
| **LightGBM** | Newsvendor (3:1) | +61.457 | [51.989, 71.572] | **WIN** | None (No correction applied) |
| **Seasonal Naive** | MASE | +0.408 | [0.379, 0.437] | **WIN** | None (No correction applied) |
| **Seasonal Naive** | WQL | -3506.532 | [-12813.956, 3242.358] | **TIE** | None (No correction applied) |
| **Seasonal Naive** | Newsvendor (3:1) | +5.882 | [4.599, 7.210] | **WIN** | None (No correction applied) |

## Artifacts Generated
* [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md): Complete technical report with assertions and analysis.
* [`per_series_metrics.csv`](results/per_series_metrics.csv): Per-series, per-origin point and probabilistic metrics.
* [`summary.csv`](results/summary.csv): Aggregated metrics by model and demand granularity.
* [`win_tie_loss.csv`](results/win_tie_loss.csv): 10,000-resample paired bootstrap confidence intervals.
* [`newsvendor_sensitivity.csv`](results/newsvendor_sensitivity.csv): Newsvendor inventory cost evaluation across cost ratios.
* [`runtime.csv`](results/runtime.csv): Wall-clock timing per 1,000 series.
* Charts: `chart1_paired_mase_ci.png`, `chart2_covariates_gain.png`, `chart3_newsvendor_sensitivity.png`, `chart4_runtime_comparison.png`.
* [`Chronos2_Shipping_Label_v2.pdf`](Chronos2_Shipping_Label_v2.pdf): Executive summary shipping label / benchmark card.

## Reproducibility
* Kaggle Kernel: `madlunatic/chronos-2-m5-retail-demand-benchmark`
* Model: `amazon/chronos-2` (120M parameters, Apache-2.0)
* Execution: T4 GPU, Python 3.10+, PyTorch 2.5+, `chronos-forecasting>=2.0`
