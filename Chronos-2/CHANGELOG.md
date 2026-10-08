# Benchmark Changelog: Chronos-2 Replication on M5 Retail Demand

All notable events, model executions, crashes, fixes, and design decisions are logged here in chronological order.

---

## [2026-10-08] Full Benchmark Execution & Empirical Verification

### Kernel Version 1 & 2 Diagnostics and Fixes
- **CLI Flag Passing in Notebook (V1):** Jupyter kernel launcher appends `-f /root/.local/share/jupyter/...` internal connection flags. Guarded CLI parser with `parse_known_args()` and checked `ipykernel not in sys.modules` to prevent notebook execution halts.
- **Chronos-Bolt API in `chronos>=2.0` (V2):** `ChronosBoltPipeline.predict()` does not accept `context` keyword argument in `chronos-forecasting 2.3.2`. Replaced with official `.predict_quantiles(context=..., prediction_length=28, quantile_levels=...)` returning tensor shape `(batch_size, horizon, num_quantiles)` and added bidirectional dimension handling for `(B, horizon, Q)` vs `(B, Q, horizon)`.
- **AutoETS Index Lookup (V2):** Replaced `.loc[s_id]` on `StatsForecast` output DataFrame with `.reset_index()` and dictionary grouping on `unique_id.astype(str)` to prevent pandas indexing ambiguity.
- **LightGBM Validation Window Tuning (V3):** Built an explicit 15-trial random search over learning rate, num_leaves, min_child_samples, and colsample_bytree evaluated on an internal validation window ($d_{1830} \dots d_{1857}$) strictly non-overlapping with test origins before fitting final point and quantile models.

### Kernel Version 3: Full Run Execution Record
- **Environment:** Cloud Tesla T4 GPU (14.56 GB VRAM), Linux x86_64, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8.
- **Libraries:** `chronos` 2.3.2, `statsforecast` 2.1.1, `lightgbm` 4.6.0, `pandas` 2.3.3, `numpy` 2.1.3, `scipy` 1.16.3.
- **Dataset:** Walmart M5 Forecasting Accuracy (`aryayadav0513/m5-forecasting-accuracy`), 1,000 item-store series sampled deterministically (seed 42) from 28,387 series with $\ge 100$ non-zero days.
- **Rolling Origins (3 Origins, $H=28$ days each):**
  - Origin 1: Train $d_1 \dots d_{1857}$, Test $d_{1858} \dots d_{1885}$ (0 zero-scale series excluded).
  - Origin 2: Train $d_1 \dots d_{1885}$, Test $d_{1886} \dots d_{1913}$ (0 zero-scale series excluded).
  - Origin 3: Train $d_1 \dots d_{1913}$, Test $d_{1914} \dots d_{1941}$ (0 zero-scale series excluded).
- **Execution Wall-Clock Time:** 05:31:23 UTC to 05:37:46 UTC (**6 minutes and 23 seconds** total, well within the 3-hour budget).
- **Models Completed (100% Finished):**
  1. Seasonal Naive: Finished (0.4s / 1k series)
  2. AutoETS: Finished (190.2s / 1k series)
  3. LightGBM Global (Tuned): Finished (50.9s / 1k series)
  4. Chronos-Bolt (`amazon/chronos-bolt-base`): Finished (9.3s / 1k series)
  5. Chronos-2 Zero-Shot (`amazon/chronos-2`): Finished (10.4s / 1k series)
  6. Chronos-2 Covariates (`amazon/chronos-2`): Finished (55.5s / 1k series)
  7. TimesFM 2.5: **Not Tested** (skipped per protocol to preserve runtime budget and prevent JAX dependency conflicts)

### Statistical & Empirical Findings
- **Paired Bootstrap:** Aggregated across 3 origins per series, then evaluated with 10,000 bootstrap resamples (seed 42, 95% CI). Verdict = WIN/LOSS only if CI strictly excludes 0, else TIE. No multiple-comparison correction was applied.
- **Chronos-2 Covariates vs Baselines:**
  - vs Seasonal Naive: **WIN** on MASE (+0.408, CI [0.379, 0.437]), **TIE** on WQL, **WIN** on Newsvendor 3:1 (+5.882, CI [4.599, 7.210]).
  - vs AutoETS: **WIN** on MASE (+0.205, CI [0.185, 0.226]), **TIE** on WQL, **TIE** on Newsvendor 3:1.
  - vs LightGBM: **WIN** on MASE (+1.817, CI [1.324, 2.433]), **WIN** on WQL (+37140.10, CI [17873.55, 57665.52]), **WIN** on Newsvendor 3:1 (+61.457, CI [51.989, 71.572]).
  - vs Chronos-Bolt: **WIN** on MASE (+0.008, CI [0.001, 0.015]), **TIE** on WQL, **TIE** on Newsvendor 3:1.
  - vs Chronos-2 Zero-Shot (Impact of Covariates): **LOSS** on MASE (-0.008, CI [-0.013, -0.002]), **LOSS** on WQL (-7282.34, CI [-17211.45, -215.63]), **TIE** on Newsvendor 3:1 (-0.240, CI [-1.179, 0.709]).
- **Quantile Crossing Rate:** Evaluated monotonicity violations ($q_k > q_{k+1}$) across all series and time steps:
  - Chronos-2 Covariates: 1.17%
  - Chronos-2 Zero-Shot: 1.76%
  - Chronos-Bolt: 4.15%

### Publication Charts Generated
- `chart1_paired_mase_ci.png`: Paired MASE difference vs baselines with 95% bootstrap CIs (WIN/LOSS/TIE color-coded).
- `chart2_covariates_gain.png`: Impact of in-context price/event covariates on MASE, WQL, and Newsvendor cost (bars anchored at zero).
- `chart3_newsvendor_sensitivity.png`: Newsvendor inventory cost per series across cost ratios (1:1, 3:1, 5:1, 9:1) at $q=0.90$.
- `chart4_runtime_comparison.png`: Wall-clock runtime per 1,000 series (fit + predict).
- All charts conform to publication guidelines: thin marks, direct labels, bars start at zero, no dual axes, 1080px readable.

### Assertion Verification
- All 17 empirical claims asserted against saved CSVs in `generate_report_from_csvs.py` passed with 100% accuracy.
