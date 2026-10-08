# ML Experiments

A curated collection of empirical machine learning benchmarks, foundation model evaluations, tabular/time-series architectures, and production-grade ML experiments.

---

## Repository Structure & Modules

| Module | Description | Key Deliverables | Status |
| :--- | :--- | :--- | :---: |
| [**TabPFN**](./TabPFN) | **Tabular Foundation Models vs Tuned GBDTs** (Nature 2025 benchmark, in-context Bayesian inference vs CatBoost/XGBoost/LightGBM, 25-fold Stratified CV, epistemic uncertainty rejection curves) | Report, Benchmark Scripts, Notebook, Results & Presentation Deck | Completed |
| [**Chronos-2**](./Chronos-2) | **Chronos-2 Time Series Foundation Model vs Baselines on M5 Retail Demand** (arXiv:2510.15821 replication, 1,000 item-store series, 3 rolling origins, paired bootstrap CIs, Newsvendor cost) | Report, Benchmark Scripts, Notebook, Results & 4 Visual Charts | In Progress |

---

## Featured Experiments

### 1. [TabPFN: Tabular Foundation Models vs Tuned GBDTs](./TabPFN)
* **Focus:** Empirical investigation of TabPFN v2 (*Nature*, Jan 2025) compared to Bayesian/Optuna-tuned Gradient Boosted Decision Trees (CatBoost, XGBoost, LightGBM) across business-critical tabular datasets (`credit-g`, `bank-marketing`, `churn`, `adult`).
* **Key Findings:** TabPFN achieves competitive or superior ROC-AUC in a single forward pass without hyperparameter tuning, with calibrated epistemic uncertainty for selective classification and rejection inference.
* **Deliverables:**
  - **Empirical Report:** [`TabPFN/BENCHMARK_REPORT.md`](./TabPFN/BENCHMARK_REPORT.md)
  - **Presentation Deck:** [`TabPFN/TabPFN_Day1_Benchmark_Presentation.pdf`](./TabPFN/TabPFN_Day1_Benchmark_Presentation.pdf)
  - **Interactive Notebook:** [`TabPFN/tabpfn_benchmark.ipynb`](./TabPFN/tabpfn_benchmark.ipynb)
  - **Standalone Benchmark Script:** [`TabPFN/tabpfn_benchmark.py`](./TabPFN/tabpfn_benchmark.py)
  - **Plots & Metrics:** [`TabPFN/results/`](./TabPFN/results/)

### 2. [Chronos-2: Time Series Foundation Models on M5 Retail Demand](./Chronos-2)
* **Focus:** Rigorous replication and evaluation of Chronos-2 (*Amazon AI Labs*, arXiv:2510.15821, Oct 2025) against Seasonal Naive, AutoETS (`statsforecast`), global tuned LightGBM, and Chronos-Bolt across 1,000 Walmart series with 3 rolling origins.
* **Key Inquiries:** Univariate zero-shot accuracy, in-context value of future-known covariates (price, calendar, SNAP), asymmetric Newsvendor inventory costs, and 10,000-resample paired bootstrap hypothesis testing.
* **Deliverables:**
  - **Empirical Report:** [`Chronos-2/BENCHMARK_REPORT.md`](./Chronos-2/BENCHMARK_REPORT.md)
  - **Interactive Notebook:** [`Chronos-2/chronos2_benchmark.ipynb`](./Chronos-2/chronos2_benchmark.ipynb)
  - **Standalone Benchmark Script:** [`Chronos-2/chronos2_benchmark.py`](./Chronos-2/chronos2_benchmark.py)
  - **Plots & Metrics:** [`Chronos-2/results/`](./Chronos-2/results/)

---

## License & Citation
Code is provided under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0). All benchmark evaluations follow standard academic citation standards.
