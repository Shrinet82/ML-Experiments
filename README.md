# ML Experiments

A curated collection of empirical machine learning benchmarks, foundation model evaluations, tabular architectures, and production-grade ML experiments.

---

## Repository Structure & Modules

| Module | Description | Key Deliverables | Status |
| :--- | :--- | :--- | :---: |
| [**TabPFN**](./TabPFN) | **Tabular Foundation Models vs Tuned GBDTs** (Nature 2025 benchmark, in-context Bayesian inference vs CatBoost/XGBoost/LightGBM, 25-fold Stratified CV, epistemic uncertainty rejection curves) | Report, Benchmark Scripts, Notebook, Results & Presentation Deck | Completed |

---

## Featured Experiments

### [TabPFN: Tabular Foundation Models vs Tuned GBDTs](./TabPFN)
* **Focus:** Empirical investigation of TabPFN v2 (*Nature*, Jan 2025) compared to Bayesian/Optuna-tuned Gradient Boosted Decision Trees (CatBoost, XGBoost, LightGBM) across business-critical tabular datasets (`credit-g`, `bank-marketing`, `churn`, `adult`).
* **Key Findings:** TabPFN achieves competitive or superior ROC-AUC in a single forward pass without hyperparameter tuning, with calibrated epistemic uncertainty for selective classification and rejection inference.
* **Deliverables:**
  - **Empirical Report:** [`TabPFN/BENCHMARK_REPORT.md`](./TabPFN/BENCHMARK_REPORT.md)
  - **Presentation Deck:** [`TabPFN/TabPFN_Day1_Benchmark_Presentation.pdf`](./TabPFN/TabPFN_Day1_Benchmark_Presentation.pdf)
  - **Interactive Notebook:** [`TabPFN/tabpfn_benchmark.ipynb`](./TabPFN/tabpfn_benchmark.ipynb)
  - **Standalone Benchmark Script:** [`TabPFN/tabpfn_benchmark.py`](./TabPFN/tabpfn_benchmark.py)
  - **Plots & Metrics:** [`TabPFN/results/`](./TabPFN/results/)

---

## License & Citation
Code is provided under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0). All benchmark evaluations follow standard academic citation standards.
