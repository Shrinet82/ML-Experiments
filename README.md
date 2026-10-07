# 21 Days of Machine Learning & Foundation Models

A structured 21-day deep dive into modern machine learning, foundation models, tabular architectures, and production-grade empirical benchmarking.

---

## Roadmap & Directory Structure

| Day | Topic | Highlights & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| [**Day 1**](./Day%201) | **TabPFN: Tabular Foundation Models vs Tuned GBDTs** | In-context Bayesian inference vs CatBoost/XGBoost/LightGBM, 25-fold Stratified CV benchmark, calibration & uncertainty rejection curves, presentation deck | Completed |
| **Day 2** | *Coming Soon* | — | Planned |
| **Day 3** | *Coming Soon* | — | Planned |
| ... | ... | ... | Planned |
| **Day 21** | *Coming Soon* | — | Planned |

---

## Daily Modules

### [Day 1: TabPFN — Tabular Foundation Model & GBDT Benchmark](./Day%201)
* **Topic:** Empirical investigation of TabPFN v2 (Nature, Jan 2025) vs hyperparameter-tuned GBDTs (CatBoost, XGBoost, LightGBM).
* **Key Findings:** TabPFN achieves competitive or superior ROC-AUC in a single forward pass without hyperparameter tuning, with calibrated epistemic uncertainty for selective classification.
* **Deliverables:**
  - Full statistical report: [`BENCHMARK_REPORT.md`](./Day%201/BENCHMARK_REPORT.md)
  - Visual presentation deck: [`TabPFN_Day1_Benchmark_Presentation.pdf`](./Day%201/TabPFN_Day1_Benchmark_Presentation.pdf)
  - Interactive Jupyter notebook: [`tabpfn_benchmark.ipynb`](./Day%201/tabpfn_benchmark.ipynb)
  - Standalone execution script: [`tabpfn_benchmark.py`](./Day%201/tabpfn_benchmark.py)
  - Raw per-fold metrics, plots, and figures: [`results/`](./Day%201/results/)

---

## License & Citation
Code is provided under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0). All benchmark evaluations follow standard academic citation standards.
