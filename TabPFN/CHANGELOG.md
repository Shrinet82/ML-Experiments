# Benchmark Changelog & Methodological Corrections

This changelog documents all corrections, removals, and methodological improvements made to the TabPFN benchmark report, transitioning from preliminary exploratory runs to a publication-grade, reproducible evaluation.

---

### 1. Factual and Bibliographic Corrections

* **Reference 2 (arXiv:2505.20003):**
  * *Previous:* Incorrectly attributed to "Tian & Haris" and claimed the paper establishes in-context learning as an "implicit Bayesian model averaging over an ensemble of kernel predictors".
  * *Correction:* Correctly cited as **"TabPFN: One Model to Rule Them All?"** authored by **Qiong Zhang, Yan Shuo Tan, Qinglong Tian, and Pengfei Li** (arXiv:2505.20003, May 2025). The unsubstantiated kernel BMA claims were removed, and Section 2.3 was rewritten to reflect the actual paper's core contributions: interpreting TabPFN as approximate Bayesian inference under a synthetic prior, its capacity to adapt to both parametric and nonparametric structures, and empirical performance under covariate shift and heterogeneous treatment effect estimation.
* **Reference 3 (ICLR 2022):**
  * *Previous:* Titled *"Transformers Can Do Bayesian Inference in Seconds"*.
  * *Correction:* Corrected to the actual published title: **"Transformers Can Do Bayesian Inference"** (Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter, *ICLR 2022*).
* **Model Weight Licensing:**
  * *Previous:* Claimed TabPFN v2 weights are released under "plain Apache-2.0".
  * *Correction:* Clarified that while the TabPFN source code is Apache-2.0, the TabPFN v2 model weights are distributed under the **Prior Labs License** (Apache-2.0 with an additional mandatory attribution clause). Subsequent releases (v2.5, v3, v3.5) are governed by non-commercial research licenses requiring platform authentication tokens.

---

### 2. Metric and Statistical Integrity

* **Removal of "Normalized Score":**
  * *Previous:* Included a min-max normalized column $\frac{\text{AUC} - \min}{\max - \min}$ that artificially set TabPFN to $1.0000$ and XGBoost Default to $0.0000$.
  * *Correction:* Removed completely. Min-max normalization exaggerates marginal performance spreads (e.g., a 0.046 spread became a 1.000 gap). Replaced with raw ROC-AUC, 95% Confidence Intervals, and paired difference tests.
* **Cross-Validation Protocol:**
  * *Previous:* Evaluated on a single 5-fold cross-validation split on a single dataset (`credit-g`).
  * *Correction:* Upgraded to **Repeated Stratified Cross-Validation (5 Folds $\times$ 5 Repeats = 25 evaluations per model per dataset)** using identical fold splits for all models.
* **Hypothesis Testing:**
  * *Addition:* Added paired differences ($\Delta_i = \text{AUC}_{\text{TabPFN}, i} - \text{AUC}_{\text{Baseline}, i}$), 95% Student's $t$ confidence intervals, paired **Wilcoxon signed-rank tests**, and the **Nadeau & Bengio (2003) corrected resampled $t$-test** to account for non-independence across repeated folds.
* **Variance Reduction Claim:**
  * *Previous:* Claimed "3.6× to 6.4× lower variance" based on fold standard deviations from a single 5-fold run.
  * *Correction:* Removed the single-run claim. Recomputed empirical variance and standard deviation across 25 repeated evaluations per model.

---

### 3. Baseline Fairness & Leakage Prevention

* **Categorical Feature Handling:**
  * *Previous:* Used one-hot or default ordinal encodings across baselines.
  * *Correction:* Enabled CatBoost's **native categorical handling** (`cat_features`), LightGBM's native categorical dtypes, and XGBoost ordinal encodings.
* **Nested Cross-Validation for Hyperparameter Tuning:**
  * *Previous:* Tuned GBDT baselines on validation splits without formal nested controls and conjectured "overfitting" without empirical isolation.
  * *Correction:* Implemented nested cross-validation: for each fold, tuning is conducted on an internal 20% validation split (15 randomized trials across key hyperparameters) with zero test fold exposure.
* **Explicit Tuning Budget:**
  * *Addition:* Documented exact hyperparameter search spaces and recorded elapsed tuning wall-clock time per model.

---

### 4. Multi-Dataset Expansion

* *Previous:* Evaluated only on German Credit (`credit-g`, $N=1,000$).
* *Correction:* Expanded the benchmark across **4 business-relevant tabular datasets**:
  1. `credit-g` (OpenML 31, $N=1,000$)
  2. `telco-customer-churn` (OpenML 42178, $N=7,043$)
  3. `bank-marketing` (OpenML 1461, subsampled to $N=10,000$, seed 42)
  4. `adult` (OpenML 1590, subsampled to $N=10,000$, seed 42)
* *Addition:* Compiled a formal **Win / Tie / Loss table** comparing TabPFN against the strongest tuned baseline across all datasets.

---

### 5. Business Translation (Financial Metrics)

* *Addition:* Translated abstract classification metrics into concrete financial outcomes:
  * **German Credit (`credit-g`):** Implemented the standard cost matrix where approving bad credit (False Positive) incurs cost 5 and rejecting good credit (False Negative) incurs cost 1. Optimal probability threshold $\tau^*$ is selected on the internal validation split and scored on the test fold.
  * **Telco Churn:** Implemented an ROI payoff model assuming a $\$20$ retention incentive cost and $\$100$ preserved customer lifetime value, evaluating expected net profit per customer at the validation-optimized decision threshold.
  * Reported money-per-applicant and profit-per-customer differences with 95% confidence intervals.

---

### 6. Uncertainty, Rejection Curves & Calibration

* **Reframing "Anomaly Detection":**
  * *Previous:* Framed predictive Shannon entropy as out-of-distribution (OOD) anomaly detection and claimed "20 anomalies detected" (an artifact of taking a 10% threshold on 200 samples).
  * *Correction:* Correctly reframed predictive entropy as **decision boundary uncertainty** for **selective classification** ("abstain and route to human review"). Dropped the arbitrary "20 anomalies" count.
* **Selective Classification Validation:**
  * *Addition:* Plotted an empirical **Rejection Curve** measuring classification accuracy and ROC-AUC on retained vs. rejected cohorts as the abstention fraction increases from 0% to 50%.
* **Probability Calibration Analysis:**
  * *Addition:* Evaluated Brier score, log loss, Expected Calibration Error (ECE), and reliability diagrams comparing TabPFN against raw and Isotonic-calibrated CatBoost.

---

### 7. Empirical Scaling & Latency Profiling

* **Sample Size Scaling Curve:**
  * *Addition:* Evaluated ROC-AUC across training sample sizes $N_{\text{train}} \in [200, 500, 1000, 2000, 5000, 10000]$ on the Adult Census dataset across 5 seeds to empirically identify where TabPFN's advantage diminishes relative to tree models.
* **Per-Row Latency Benchmarking:**
  * *Addition:* Formally measured batch inference latency excluding GPU warm-up and computed the per-row latency ratio ($\text{TabPFN} / \text{CatBoost}$).
