"""
TabPFN Comprehensive Benchmark Suite: Statistical Rigor & Business Translation
Paper Reference: Hollmann et al., Nature 637, 319-326 (Jan 2025)

Protocol:
- Datasets: credit-g (31), telco-churn (42178), bank-marketing (1461, N=10k), adult (1590, N=10k)
- 5 Folds x 5 Repeats Stratified Cross-Validation (25 evaluations per model per dataset)
- Identical splits for all models
- Statistical testing: 95% CIs, paired differences, Wilcoxon signed-rank test, Nadeau & Bengio corrected t-test
- Fair baselines: CatBoost (native cat_features), LightGBM (categorical dtypes), XGBoost (ordinal encoded)
- Nested tuning: 15 randomized trials on 20% internal validation split per fold (zero test leakage)
- Business translation: credit-g cost matrix, telco-churn profit optimization
- Uncertainty quantification: predictive entropy selective classification (rejection curve)
- Calibration: Brier score, log loss, ECE, reliability diagrams
- Scaling analysis: AUC vs N_train in [200, 500, 1000, 2000, 5000, 10000]
- Hardware: Dedicated Nvidia T4 GPU (16GB VRAM, CUDA acceleration)
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.datasets import fetch_openml
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# Baselines imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

# Set visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# 1. Dataset Fetching & Standard Preprocessing
# ---------------------------------------------------------

def load_benchmark_dataset(dataset_name, max_samples=10000):
    """
    Loads one of 4 benchmark datasets from OpenML with explicit business targets.
    Subsamples to max_samples if dataset exceeds limit, preserving class ratio.
    """
    print(f"\n[Data] Loading '{dataset_name}' from OpenML...")
    if dataset_name == "credit-g":
        # German Credit Dataset (1,000 samples, 20 features)
        b = fetch_openml(data_id=31, as_frame=True, parser="auto")
        X = b.data.copy()
        y = (b.target == "good").astype(int)
    elif dataset_name == "telco-churn":
        # Telco Customer Churn (OpenML ID: 42178, 7,043 samples)
        b = fetch_openml(data_id=42178, as_frame=True, parser="auto")
        X = b.data.copy()
        # Coerce TotalCharges object to numeric and impute 0 for new customers
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
        y = (b.target == "Yes").astype(int)
    elif dataset_name == "bank-marketing":
        # Bank Marketing (OpenML ID: 1461, 45,211 samples -> subsampled to 10k)
        b = fetch_openml(data_id=1461, as_frame=True, parser="auto")
        X = b.data.copy()
        y = (b.target.astype(str) == "2").astype(int)
    elif dataset_name == "adult":
        # Adult Census Income (OpenML ID: 1590, 48,842 samples -> subsampled to 10k)
        b = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        X = b.data.copy()
        y = (b.target.astype(str).str.strip().str.startswith(">50K")).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample if exceeds max_samples
    if len(X) > max_samples:
        print(f"[Data] Subsampling {len(X)} down to {max_samples} with seed {RANDOM_SEED}.")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=RANDOM_SEED)
        idx, _ = next(sss.split(X, y))
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    print(f"[Data] Ready {dataset_name}: {X.shape[0]} rows, {X.shape[1]} features, positive class: {y.mean():.3%}")
    return X, y


# ---------------------------------------------------------
# 2. Baseline Model Builders & Preprocessors
# ---------------------------------------------------------

def prepare_data_representations(X, y):
    """
    Prepares format-specific representations for each model class:
    - CatBoost: native cat_features string encoding
    - LightGBM: category dtypes
    - XGBoost / RF: Ordinal encoded + imputed numpy
    - Linear: One-hot encoded + scaled
    - TabPFN: Clean pandas dataframe with category dtypes
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    # 1. CatBoost format
    X_catboost = X.copy()
    for c in cat_cols:
        X_catboost[c] = X_catboost[c].astype(str).fillna("missing")
    for c in num_cols:
        X_catboost[c] = pd.to_numeric(X_catboost[c], errors="coerce").fillna(X_catboost[c].median())
    cat_indices = [X_catboost.columns.get_loc(c) for c in cat_cols]

    # 2. LightGBM format
    X_lgbm = X.copy()
    for c in cat_cols:
        X_lgbm[c] = X_lgbm[c].astype("category")
    for c in num_cols:
        X_lgbm[c] = pd.to_numeric(X_lgbm[c], errors="coerce")

    # 3. Tree Ordinal format (XGBoost, Random Forest, TabPFN)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preprocessor_tree = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    X_tree = preprocessor_tree.fit_transform(X)

    # 4. Linear format (Logistic Regression)
    num_pipe_lin = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe_lin = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor_lin = ColumnTransformer([
        ("num", num_pipe_lin, num_cols),
        ("cat", cat_pipe_lin, cat_cols)
    ])
    X_linear = preprocessor_lin.fit_transform(X)

    # 5. TabPFN format (clean dataframe with category dtypes)
    X_tabpfn = X.copy()
    for c in cat_cols:
        X_tabpfn[c] = X_tabpfn[c].astype("category")
    for c in num_cols:
        X_tabpfn[c] = pd.to_numeric(X_tabpfn[c], errors="coerce").fillna(X_tabpfn[c].median())

    return {
        "catboost": (X_catboost, cat_indices),
        "lgbm": X_lgbm,
        "tree": X_tree,
        "linear": X_linear,
        "tabpfn": X_tabpfn
    }


def get_tabpfn_classifier(device="cuda"):
    """
    Initializes TabPFN v2 via official API.
    ModelVersion.V2 uses Prior Labs License open weights (Apache-2.0 + attribution).
    """
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)


def tune_catboost(X_tr, y_tr, X_val, y_val, cat_indices, n_trials=15, seed=42):
    """Random search tuning for CatBoost on internal validation split."""
    rng = np.random.RandomState(seed)
    best_score = -1.0
    best_params = None

    param_grid = [
        {"depth": rng.choice([4, 6, 8]),
         "l2_leaf_reg": float(rng.choice([1, 3, 5, 10])),
         "learning_rate": float(rng.choice([0.03, 0.08, 0.15])),
         "iterations": int(rng.choice([250, 500]))}
        for _ in range(n_trials)
    ]
    # Always include standard baseline
    param_grid[0] = {"depth": 6, "l2_leaf_reg": 3.0, "learning_rate": 0.08, "iterations": 500}

    for p in param_grid:
        cb = CatBoostClassifier(
            depth=p["depth"],
            l2_leaf_reg=p["l2_leaf_reg"],
            learning_rate=p["learning_rate"],
            iterations=p["iterations"],
            cat_features=cat_indices,
            verbose=0,
            thread_count=4,
            random_seed=seed
        )
        cb.fit(X_tr, y_tr)
        preds = cb.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_params = p

    return best_params


def tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials=15, seed=42):
    """Random search tuning for XGBoost on internal validation split."""
    rng = np.random.RandomState(seed)
    best_score = -1.0
    best_params = None

    param_grid = [
        {"max_depth": int(rng.choice([3, 5, 7])),
         "learning_rate": float(rng.choice([0.03, 0.08, 0.15])),
         "n_estimators": int(rng.choice([100, 200, 300])),
         "subsample": float(rng.choice([0.7, 0.9, 1.0])),
         "colsample_bytree": float(rng.choice([0.7, 0.9, 1.0]))}
        for _ in range(n_trials)
    ]
    param_grid[0] = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "subsample": 1.0, "colsample_bytree": 1.0}

    for p in param_grid:
        xgb = XGBClassifier(
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4
        )
        xgb.fit(X_tr, y_tr)
        preds = xgb.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_params = p

    return best_params


def tune_lightgbm(X_tr, y_tr, X_val, y_val, n_trials=15, seed=42):
    """Random search tuning for LightGBM on internal validation split."""
    rng = np.random.RandomState(seed)
    best_score = -1.0
    best_params = None

    param_grid = [
        {"num_leaves": int(rng.choice([15, 31, 63])),
         "max_depth": int(rng.choice([3, 5, 7, -1])),
         "learning_rate": float(rng.choice([0.03, 0.08, 0.15])),
         "min_child_samples": int(rng.choice([10, 20, 50])),
         "n_estimators": int(rng.choice([100, 200]))}
        for _ in range(n_trials)
    ]
    param_grid[0] = {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.1, "min_child_samples": 20, "n_estimators": 100}

    for p in param_grid:
        lgb = LGBMClassifier(
            num_leaves=p["num_leaves"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            min_child_samples=p["min_child_samples"],
            n_estimators=p["n_estimators"],
            verbose=-1,
            random_state=seed,
            n_jobs=4
        )
        lgb.fit(X_tr, y_tr)
        preds = lgb.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_params = p

    return best_params


# ---------------------------------------------------------
# 3. Business Metric Calculation Functions
# ---------------------------------------------------------

def evaluate_credit_cost(y_val, p_val, y_test, p_test):
    """
    credit-g cost matrix:
    - Predicting Good (1) for Bad (0) borrower: Cost = 5 (False Positive)
    - Predicting Bad (0) for Good (1) borrower: Cost = 1 (False Negative)
    - Correct predictions: Cost = 0
    Threshold tau selected on validation set to minimize cost.
    Evaluated on test set.
    """
    thresholds = np.linspace(0.02, 0.98, 97)
    best_cost_val = float("inf")
    best_tau = 0.5

    for tau in thresholds:
        pred_val = (p_val >= tau).astype(int)
        # FP: true 0, pred 1 (cost 5)
        fp = np.sum((y_val == 0) & (pred_val == 1))
        # FN: true 1, pred 0 (cost 1)
        fn = np.sum((y_val == 1) & (pred_val == 0))
        cost = 5.0 * fp + 1.0 * fn
        if cost < best_cost_val:
            best_cost_val = cost
            best_tau = tau

    # Evaluate on test fold
    pred_test = (p_test >= best_tau).astype(int)
    fp_test = np.sum((y_test == 0) & (pred_test == 1))
    fn_test = np.sum((y_test == 1) & (pred_test == 0))
    total_test_cost = 5.0 * fp_test + 1.0 * fn_test
    cost_per_applicant = total_test_cost / len(y_test)
    return cost_per_applicant, best_tau


def evaluate_churn_profit(y_val, p_val, y_test, p_test):
    """
    telco-churn profit optimization:
    - Retention offer / incentive cost C = $20
    - Preserved customer value V = $100
    - Targeting true churner (TP): net gain = $100 - $20 = +$80
    - Targeting non-churner (FP): wasted incentive = -$20
    - Non-targeted instances: $0 relative impact
    Threshold tau selected on validation set to maximize profit.
    Evaluated on test set.
    """
    thresholds = np.linspace(0.02, 0.98, 97)
    best_profit_val = -float("inf")
    best_tau = 0.5

    for tau in thresholds:
        pred_val = (p_val >= tau).astype(int)
        tp = np.sum((y_val == 1) & (pred_val == 1))
        fp = np.sum((y_val == 0) & (pred_val == 1))
        profit = 80.0 * tp - 20.0 * fp
        if profit > best_profit_val:
            best_profit_val = profit
            best_tau = tau

    # Evaluate on test fold
    pred_test = (p_test >= best_tau).astype(int)
    tp_test = np.sum((y_test == 1) & (pred_test == 1))
    fp_test = np.sum((y_test == 0) & (pred_test == 1))
    total_profit = 80.0 * tp_test - 20.0 * fp_test
    profit_per_customer = total_profit / len(y_test)
    return profit_per_customer, best_tau


# ---------------------------------------------------------
# 4. Statistical Testing Tools
# ---------------------------------------------------------

def compute_confidence_interval(data, confidence=0.95):
    """Computes mean and two-sided Student's t 95% confidence interval."""
    a = np.array(data, dtype=float)
    n = len(a)
    mean = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h, h


def nadeau_bengio_corrected_t_test(diffs, n_train, n_test, n_splits=5, n_repeats=5):
    """
    Nadeau and Bengio (2003) corrected resampled t-test for repeated cross-validation.
    Accounts for non-independence of folds:
    var_corrected = s^2 * (1 / (R * K) + n_test / n_train)
    """
    diffs = np.array(diffs, dtype=float)
    d_bar = np.mean(diffs)
    s2 = np.var(diffs, ddof=1)
    k_total = n_splits * n_repeats
    factor = (1.0 / k_total) + (n_test / n_train)
    corrected_variance = s2 * factor
    if corrected_variance <= 0:
        return 0.0, 1.0
    t_stat = d_bar / np.sqrt(corrected_variance)
    df = k_total - 1
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df))
    return float(t_stat), float(p_val)


# ---------------------------------------------------------
# 5. Core Repeated Cross-Validation Engine
# ---------------------------------------------------------

def run_repeated_cv_benchmark(dataset_name, n_splits=5, n_repeats=5, device="cuda"):
    """
    Executes repeated stratified cross-validation across all models with identical splits.
    """
    X, y = load_benchmark_dataset(dataset_name)
    reps = prepare_data_representations(X, y)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED)
    fold_indices = list(rskf.split(X, y))
    total_evals = len(fold_indices)

    models_to_evaluate = [
        "TabPFN (v2)",
        "CatBoost (Default)",
        "CatBoost (Tuned)",
        "XGBoost (Default)",
        "XGBoost (Tuned)",
        "LightGBM (Default)",
        "LightGBM (Tuned)",
        "Random Forest",
        "Logistic Regression",
    ]

    # Warmup GPU once for TabPFN
    try:
        dummy_clf = get_tabpfn_classifier(device=device)
        dummy_clf.fit(reps["tabpfn"].iloc[:20], y.iloc[:20])
        _ = dummy_clf.predict_proba(reps["tabpfn"].iloc[20:30])
        print("[TabPFN] GPU warm-up completed successfully.")
    except Exception as e:
        print(f"[TabPFN] GPU warm-up notice: {e}")

    fold_results = []
    prob_predictions = {m: [] for m in models_to_evaluate}
    true_labels = []

    print(f"\n[Benchmark] Commencing {total_evals} folds on '{dataset_name}'...")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
        n_train, n_test = len(train_idx), len(test_idx)

        # Internal validation split for nested tuning and threshold optimization (20% of train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED + fold_idx)
        inner_tr_rel, inner_val_rel = next(sss.split(train_idx, y_train))
        inner_tr_idx = train_idx[inner_tr_rel]
        inner_val_idx = train_idx[inner_val_rel]

        y_inner_tr = y.iloc[inner_tr_idx].values
        y_inner_val = y.iloc[inner_val_idx].values

        if fold_idx == 0:
            true_labels.extend(y_test)

        # -------------------------------------------------------------
        # Model 1: TabPFN (v2) - Foundation Model (0-Tuning)
        # -------------------------------------------------------------
        X_tab_tr = reps["tabpfn"].iloc[train_idx]
        X_tab_te = reps["tabpfn"].iloc[test_idx]
        X_tab_in_tr = reps["tabpfn"].iloc[inner_tr_idx]
        X_tab_in_val = reps["tabpfn"].iloc[inner_val_idx]

        t0 = time.perf_counter()
        tab_clf = get_tabpfn_classifier(device=device)
        tab_clf.fit(X_tab_tr, y_train)
        tab_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        tab_p_test = tab_clf.predict_proba(X_tab_te)[:, 1]
        tab_pred_time = time.perf_counter() - t0
        tab_auc = roc_auc_score(y_test, tab_p_test)

        # Internal val predictions for business threshold
        tab_val_clf = get_tabpfn_classifier(device=device)
        tab_val_clf.fit(X_tab_in_tr, y_inner_tr)
        tab_p_val = tab_val_clf.predict_proba(X_tab_in_val)[:, 1]

        # Business metrics
        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, tab_p_val, y_test, tab_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, tab_p_val, y_test, tab_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "TabPFN (v2)",
            "AUC": tab_auc, "Fit_Time_s": tab_fit_time, "Predict_Time_s": tab_pred_time,
            "Per_Row_Predict_ms": (tab_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["TabPFN (v2)"].extend(tab_p_test)

        # -------------------------------------------------------------
        # Model 2: CatBoost (Default) - Native cat_features
        # -------------------------------------------------------------
        X_cb, cb_cats = reps["catboost"]
        X_cb_tr, X_cb_te = X_cb.iloc[train_idx], X_cb.iloc[test_idx]
        X_cb_in_tr, X_cb_in_val = X_cb.iloc[inner_tr_idx], X_cb.iloc[inner_val_idx]

        t0 = time.perf_counter()
        cb_def = CatBoostClassifier(iterations=500, cat_features=cb_cats, verbose=0, thread_count=4, random_seed=RANDOM_SEED)
        cb_def.fit(X_cb_tr, y_train)
        cb_def_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        cb_def_p_test = cb_def.predict_proba(X_cb_te)[:, 1]
        cb_def_pred_time = time.perf_counter() - t0
        cb_def_auc = roc_auc_score(y_test, cb_def_p_test)

        cb_def_val = CatBoostClassifier(iterations=500, cat_features=cb_cats, verbose=0, thread_count=4, random_seed=RANDOM_SEED)
        cb_def_val.fit(X_cb_in_tr, y_inner_tr)
        cb_def_p_val = cb_def_val.predict_proba(X_cb_in_val)[:, 1]

        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, cb_def_p_val, y_test, cb_def_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, cb_def_p_val, y_test, cb_def_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "CatBoost (Default)",
            "AUC": cb_def_auc, "Fit_Time_s": cb_def_fit_time, "Predict_Time_s": cb_def_pred_time,
            "Per_Row_Predict_ms": (cb_def_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["CatBoost (Default)"].extend(cb_def_p_test)

        # -------------------------------------------------------------
        # Model 3: CatBoost (Tuned) - 15 trials nested on inner val
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        best_cb_params = tune_catboost(X_cb_in_tr, y_inner_tr, X_cb_in_val, y_inner_val, cb_cats, n_trials=15, seed=RANDOM_SEED + fold_idx)
        cb_tuned = CatBoostClassifier(
            depth=best_cb_params["depth"],
            l2_leaf_reg=best_cb_params["l2_leaf_reg"],
            learning_rate=best_cb_params["learning_rate"],
            iterations=best_cb_params["iterations"],
            cat_features=cb_cats,
            verbose=0,
            thread_count=4,
            random_seed=RANDOM_SEED
        )
        cb_tuned.fit(X_cb_tr, y_train)
        cb_tuned_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        cb_tuned_p_test = cb_tuned.predict_proba(X_cb_te)[:, 1]
        cb_tuned_pred_time = time.perf_counter() - t0
        cb_tuned_auc = roc_auc_score(y_test, cb_tuned_p_test)

        # Validation prediction under best params for thresholding
        cb_best_val_model = CatBoostClassifier(
            depth=best_cb_params["depth"],
            l2_leaf_reg=best_cb_params["l2_leaf_reg"],
            learning_rate=best_cb_params["learning_rate"],
            iterations=best_cb_params["iterations"],
            cat_features=cb_cats,
            verbose=0,
            thread_count=4,
            random_seed=RANDOM_SEED
        )
        cb_best_val_model.fit(X_cb_in_tr, y_inner_tr)
        cb_tuned_p_val = cb_best_val_model.predict_proba(X_cb_in_val)[:, 1]

        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, cb_tuned_p_val, y_test, cb_tuned_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, cb_tuned_p_val, y_test, cb_tuned_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "CatBoost (Tuned)",
            "AUC": cb_tuned_auc, "Fit_Time_s": cb_tuned_fit_time, "Predict_Time_s": cb_tuned_pred_time,
            "Per_Row_Predict_ms": (cb_tuned_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["CatBoost (Tuned)"].extend(cb_tuned_p_test)

        # -------------------------------------------------------------
        # Model 4 & 5: XGBoost (Default & Tuned)
        # -------------------------------------------------------------
        X_tr_np, X_te_np = reps["tree"][train_idx], reps["tree"][test_idx]
        X_in_tr_np, X_in_val_np = reps["tree"][inner_tr_idx], reps["tree"][inner_val_idx]

        # Default XGB
        t0 = time.perf_counter()
        xgb_def = XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=4)
        xgb_def.fit(X_tr_np, y_train)
        xgb_def_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        xgb_def_p_test = xgb_def.predict_proba(X_te_np)[:, 1]
        xgb_def_pred_time = time.perf_counter() - t0
        xgb_def_auc = roc_auc_score(y_test, xgb_def_p_test)

        xgb_def_val = XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=4)
        xgb_def_val.fit(X_in_tr_np, y_inner_tr)
        xgb_def_p_val = xgb_def_val.predict_proba(X_in_val_np)[:, 1]

        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, xgb_def_p_val, y_test, xgb_def_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, xgb_def_p_val, y_test, xgb_def_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "XGBoost (Default)",
            "AUC": xgb_def_auc, "Fit_Time_s": xgb_def_fit_time, "Predict_Time_s": xgb_def_pred_time,
            "Per_Row_Predict_ms": (xgb_def_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["XGBoost (Default)"].extend(xgb_def_p_test)

        # Tuned XGB
        t0 = time.perf_counter()
        best_xgb_params = tune_xgboost(X_in_tr_np, y_inner_tr, X_in_val_np, y_inner_val, n_trials=15, seed=RANDOM_SEED + fold_idx)
        xgb_tuned = XGBClassifier(
            max_depth=best_xgb_params["max_depth"],
            learning_rate=best_xgb_params["learning_rate"],
            n_estimators=best_xgb_params["n_estimators"],
            subsample=best_xgb_params["subsample"],
            colsample_bytree=best_xgb_params["colsample_bytree"],
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=4
        )
        xgb_tuned.fit(X_tr_np, y_train)
        xgb_tuned_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        xgb_tuned_p_test = xgb_tuned.predict_proba(X_te_np)[:, 1]
        xgb_tuned_pred_time = time.perf_counter() - t0
        xgb_tuned_auc = roc_auc_score(y_test, xgb_tuned_p_test)

        xgb_best_val_model = XGBClassifier(
            max_depth=best_xgb_params["max_depth"],
            learning_rate=best_xgb_params["learning_rate"],
            n_estimators=best_xgb_params["n_estimators"],
            subsample=best_xgb_params["subsample"],
            colsample_bytree=best_xgb_params["colsample_bytree"],
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=4
        )
        xgb_best_val_model.fit(X_in_tr_np, y_inner_tr)
        xgb_tuned_p_val = xgb_best_val_model.predict_proba(X_in_val_np)[:, 1]

        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, xgb_tuned_p_val, y_test, xgb_tuned_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, xgb_tuned_p_val, y_test, xgb_tuned_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "XGBoost (Tuned)",
            "AUC": xgb_tuned_auc, "Fit_Time_s": xgb_tuned_fit_time, "Predict_Time_s": xgb_tuned_pred_time,
            "Per_Row_Predict_ms": (xgb_tuned_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["XGBoost (Tuned)"].extend(xgb_tuned_p_test)

        # -------------------------------------------------------------
        # Model 6 & 7: LightGBM (Default & Tuned)
        # -------------------------------------------------------------
        X_lgb = reps["lgbm"]
        X_lgb_tr, X_lgb_te = X_lgb.iloc[train_idx], X_lgb.iloc[test_idx]
        X_lgb_in_tr, X_lgb_in_val = X_lgb.iloc[inner_tr_idx], X_lgb.iloc[inner_val_idx]

        # Default LGBM
        t0 = time.perf_counter()
        lgb_def = LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=4, verbose=-1)
        lgb_def.fit(X_lgb_tr, y_train)
        lgb_def_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        lgb_def_p_test = lgb_def.predict_proba(X_lgb_te)[:, 1]
        lgb_def_pred_time = time.perf_counter() - t0
        lgb_def_auc = roc_auc_score(y_test, lgb_def_p_test)

        lgb_def_val = LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=4, verbose=-1)
        lgb_def_val.fit(X_lgb_in_tr, y_inner_tr)
        lgb_def_p_val = lgb_def_val.predict_proba(X_lgb_in_val)[:, 1]

        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, lgb_def_p_val, y_test, lgb_def_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, lgb_def_p_val, y_test, lgb_def_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "LightGBM (Default)",
            "AUC": lgb_def_auc, "Fit_Time_s": lgb_def_fit_time, "Predict_Time_s": lgb_def_pred_time,
            "Per_Row_Predict_ms": (lgb_def_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["LightGBM (Default)"].extend(lgb_def_p_test)

        # Tuned LGBM
        t0 = time.perf_counter()
        best_lgb_params = tune_lightgbm(X_lgb_in_tr, y_inner_tr, X_lgb_in_val, y_inner_val, n_trials=15, seed=RANDOM_SEED + fold_idx)
        lgb_tuned = LGBMClassifier(
            num_leaves=best_lgb_params["num_leaves"],
            max_depth=best_lgb_params["max_depth"],
            learning_rate=best_lgb_params["learning_rate"],
            min_child_samples=best_lgb_params["min_child_samples"],
            n_estimators=best_lgb_params["n_estimators"],
            verbose=-1,
            random_state=RANDOM_SEED,
            n_jobs=4
        )
        lgb_tuned.fit(X_lgb_tr, y_train)
        lgb_tuned_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        lgb_tuned_p_test = lgb_tuned.predict_proba(X_lgb_te)[:, 1]
        lgb_tuned_pred_time = time.perf_counter() - t0
        lgb_tuned_auc = roc_auc_score(y_test, lgb_tuned_p_test)

        lgb_best_val_model = LGBMClassifier(
            num_leaves=best_lgb_params["num_leaves"],
            max_depth=best_lgb_params["max_depth"],
            learning_rate=best_lgb_params["learning_rate"],
            min_child_samples=best_lgb_params["min_child_samples"],
            n_estimators=best_lgb_params["n_estimators"],
            verbose=-1,
            random_state=RANDOM_SEED,
            n_jobs=4
        )
        lgb_best_val_model.fit(X_lgb_in_tr, y_inner_tr)
        lgb_tuned_p_val = lgb_best_val_model.predict_proba(X_lgb_in_val)[:, 1]

        credit_cost, churn_profit = np.nan, np.nan
        if dataset_name == "credit-g":
            credit_cost, _ = evaluate_credit_cost(y_inner_val, lgb_tuned_p_val, y_test, lgb_tuned_p_test)
        elif dataset_name == "telco-churn":
            churn_profit, _ = evaluate_churn_profit(y_inner_val, lgb_tuned_p_val, y_test, lgb_tuned_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "LightGBM (Tuned)",
            "AUC": lgb_tuned_auc, "Fit_Time_s": lgb_tuned_fit_time, "Predict_Time_s": lgb_tuned_pred_time,
            "Per_Row_Predict_ms": (lgb_tuned_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": credit_cost, "Churn_Profit_Per_Customer": churn_profit
        })
        prob_predictions["LightGBM (Tuned)"].extend(lgb_tuned_p_test)

        # -------------------------------------------------------------
        # Model 8: Random Forest
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=4)
        rf.fit(X_tr_np, y_train)
        rf_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        rf_p_test = rf.predict_proba(X_te_np)[:, 1]
        rf_pred_time = time.perf_counter() - t0
        rf_auc = roc_auc_score(y_test, rf_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "Random Forest",
            "AUC": rf_auc, "Fit_Time_s": rf_fit_time, "Predict_Time_s": rf_pred_time,
            "Per_Row_Predict_ms": (rf_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": np.nan, "Churn_Profit_Per_Customer": np.nan
        })
        prob_predictions["Random Forest"].extend(rf_p_test)

        # -------------------------------------------------------------
        # Model 9: Logistic Regression
        # -------------------------------------------------------------
        X_lin_tr, X_lin_te = reps["linear"][train_idx], reps["linear"][test_idx]
        t0 = time.perf_counter()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_lin_tr, y_train)
        lr_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        lr_p_test = lr.predict_proba(X_lin_te)[:, 1]
        lr_pred_time = time.perf_counter() - t0
        lr_auc = roc_auc_score(y_test, lr_p_test)

        fold_results.append({
            "Dataset": dataset_name, "Fold": fold_idx, "Model": "Logistic Regression",
            "AUC": lr_auc, "Fit_Time_s": lr_fit_time, "Predict_Time_s": lr_pred_time,
            "Per_Row_Predict_ms": (lr_pred_time / n_test) * 1000.0,
            "Credit_Cost_Per_Applicant": np.nan, "Churn_Profit_Per_Customer": np.nan
        })
        prob_predictions["Logistic Regression"].extend(lr_p_test)

        if (fold_idx + 1) % 5 == 0:
            print(f"[{dataset_name}] Completed fold {fold_idx + 1}/{total_evals} | TabPFN AUC: {tab_auc:.4f} | CatBoost Default AUC: {cb_def_auc:.4f}")

    df_folds = pd.DataFrame(fold_results)
    return df_folds, prob_predictions


# ---------------------------------------------------------
# 6. Scaling Curve Experiment (AUC vs Training Sample Size)
# ---------------------------------------------------------

def run_scaling_curve_experiment(dataset_name="adult", device="cuda"):
    """
    Evaluates how TabPFN, CatBoost, and XGBoost scale as N_train increases:
    N_train in [200, 500, 1000, 2000, 5000, 10000].
    Evaluated on a fixed holdout test set across 5 random seeds.
    """
    print(f"\n[Scaling] Executing training size curve on '{dataset_name}'...")
    X, y = load_benchmark_dataset(dataset_name, max_samples=12000)

    # Fixed holdout test set of 2,000 instances
    test_size = 2000
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    pool_idx, test_idx = next(sss.split(X, y))

    X_pool, y_pool = X.iloc[pool_idx].reset_index(drop=True), y.iloc[pool_idx].reset_index(drop=True)
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    reps_test = prepare_data_representations(X_test, y_test)
    X_cb_te, cb_cats = reps_test["catboost"]
    X_tree_te = reps_test["tree"]
    X_tab_te = reps_test["tabpfn"]

    sizes = [200, 500, 1000, 2000, 5000, 10000]
    seeds = [42, 43, 44, 45, 46]
    results = []

    for n_sub in sizes:
        print(f"[Scaling] Evaluating N_train = {n_sub} across {len(seeds)} seeds...")
        for s in seeds:
            sub_sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=s)
            sub_idx, _ = next(sub_sss.split(X_pool, y_pool))
            X_sub = X_pool.iloc[sub_idx].reset_index(drop=True)
            y_sub = y_pool.iloc[sub_idx].reset_index(drop=True)

            reps_sub = prepare_data_representations(X_sub, y_sub)

            # TabPFN
            t0 = time.perf_counter()
            tab = get_tabpfn_classifier(device=device)
            tab.fit(reps_sub["tabpfn"], y_sub)
            tab_fit_time = time.perf_counter() - t0
            p_tab = tab.predict_proba(X_tab_te)[:, 1]
            auc_tab = roc_auc_score(y_test, p_tab)
            results.append({"N_train": n_sub, "Seed": s, "Model": "TabPFN (v2)", "AUC": auc_tab, "Fit_Time_s": tab_fit_time})

            # CatBoost Default
            t0 = time.perf_counter()
            cb = CatBoostClassifier(iterations=500, cat_features=cb_cats, verbose=0, thread_count=4, random_seed=s)
            cb.fit(reps_sub["catboost"][0], y_sub)
            cb_fit_time = time.perf_counter() - t0
            p_cb = cb.predict_proba(X_cb_te)[:, 1]
            auc_cb = roc_auc_score(y_test, p_cb)
            results.append({"N_train": n_sub, "Seed": s, "Model": "CatBoost (Default)", "AUC": auc_cb, "Fit_Time_s": cb_fit_time})

            # XGBoost Default
            t0 = time.perf_counter()
            xgb = XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss", random_state=s, n_jobs=4)
            xgb.fit(reps_sub["tree"], y_sub)
            xgb_fit_time = time.perf_counter() - t0
            p_xgb = xgb.predict_proba(X_tree_te)[:, 1]
            auc_xgb = roc_auc_score(y_test, p_xgb)
            results.append({"N_train": n_sub, "Seed": s, "Model": "XGBoost (Default)", "AUC": auc_xgb, "Fit_Time_s": xgb_fit_time})

    df_scaling = pd.DataFrame(results)
    return df_scaling


# ---------------------------------------------------------
# 7. Uncertainty, Rejection Curve & Calibration Analysis
# ---------------------------------------------------------

def evaluate_uncertainty_and_calibration(dataset_name="credit-g", device="cuda"):
    """
    Rigorously tests:
    1. Predictive entropy selective prediction / rejection curves (abstention).
    2. Calibration: Brier score, log loss, and reliability diagrams for TabPFN vs CatBoost.
    """
    print(f"\n[Uncertainty] Evaluating calibration and selective prediction on '{dataset_name}'...")
    X, y = load_benchmark_dataset(dataset_name)
    reps = prepare_data_representations(X, y)

    # 80/20 train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X, y))

    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    X_tab_tr, X_tab_te = reps["tabpfn"].iloc[train_idx], reps["tabpfn"].iloc[test_idx]
    X_cb_tr, X_cb_te = reps["catboost"][0].iloc[train_idx], reps["catboost"][0].iloc[test_idx]
    cb_cats = reps["catboost"][1]

    # TabPFN fit & predict
    tab = get_tabpfn_classifier(device=device)
    tab.fit(X_tab_tr, y_train)
    p_tab = tab.predict_proba(X_tab_te)[:, 1]

    # CatBoost Uncalibrated
    cb = CatBoostClassifier(iterations=500, cat_features=cb_cats, verbose=0, thread_count=4, random_seed=RANDOM_SEED)
    cb.fit(X_cb_tr, y_train)
    p_cb_raw = cb.predict_proba(X_cb_te)[:, 1]

    # CatBoost Calibrated with Isotonic regression (5-fold internal CV)
    cb_cal = CalibratedClassifierCV(
        estimator=CatBoostClassifier(iterations=500, cat_features=cb_cats, verbose=0, thread_count=4, random_seed=RANDOM_SEED),
        method="isotonic",
        cv=5
    )
    cb_cal.fit(X_cb_tr, y_train)
    p_cb_cal = cb_cal.predict_proba(X_cb_te)[:, 1]

    # 1. Predictive Entropy & Rejection Curve for TabPFN
    # Shannon entropy: H(p) = -p ln p - (1-p) ln(1-p)
    eps = 1e-12
    p_tab_clipped = np.clip(p_tab, eps, 1.0 - eps)
    entropy_tab = - (p_tab_clipped * np.log(p_tab_clipped) + (1.0 - p_tab_clipped) * np.log(1.0 - p_tab_clipped))

    rejection_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rejection_data = []

    for k in rejection_rates:
        if k == 0.0:
            retained_mask = np.ones(len(y_test), dtype=bool)
            rejected_mask = np.zeros(len(y_test), dtype=bool)
        else:
            cutoff = np.quantile(entropy_tab, 1.0 - k)
            retained_mask = entropy_tab <= cutoff
            rejected_mask = entropy_tab > cutoff

        y_ret = y_test[retained_mask]
        p_ret = p_tab[retained_mask]
        pred_ret = (p_ret >= 0.5).astype(int)

        ret_acc = accuracy_score(y_ret, pred_ret)
        ret_auc = roc_auc_score(y_ret, p_ret) if len(np.unique(y_ret)) > 1 else np.nan

        if np.sum(rejected_mask) > 0:
            y_rej = y_test[rejected_mask]
            pred_rej = (p_tab[rejected_mask] >= 0.5).astype(int)
            rej_acc = accuracy_score(y_rej, pred_rej)
        else:
            rej_acc = np.nan

        rejection_data.append({
            "Rejection_Rate": k,
            "Retained_Fraction": 1.0 - k,
            "Retained_Accuracy": ret_acc,
            "Retained_AUC": ret_auc,
            "Rejected_Accuracy": rej_acc
        })

    df_rejection = pd.DataFrame(rejection_data)

    # 2. Calibration Metrics & Reliability
    def compute_ece(y_true, p_pred, n_bins=10):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_accs, bin_confs, bin_counts = [], [], []
        for i in range(n_bins):
            mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i+1] if i < n_bins - 1 else p_pred <= bin_edges[i+1])
            count = np.sum(mask)
            if count > 0:
                acc = np.mean(y_true[mask] == (p_pred[mask] >= 0.5))
                conf = np.mean(p_pred[mask])
                ece += (count / len(y_true)) * np.abs(acc - conf)
                bin_accs.append(acc)
                bin_confs.append(conf)
                bin_counts.append(count)
        return float(ece), bin_confs, bin_accs

    ece_tab, conf_tab, acc_tab = compute_ece(y_test, p_tab)
    ece_cb_raw, conf_cb_raw, acc_cb_raw = compute_ece(y_test, p_cb_raw)
    ece_cb_cal, conf_cb_cal, acc_cb_cal = compute_ece(y_test, p_cb_cal)

    cal_metrics = [
        {"Model": "TabPFN (v2)", "Brier_Score": brier_score_loss(y_test, p_tab), "Log_Loss": log_loss(y_test, p_tab), "ECE": ece_tab},
        {"Model": "CatBoost (Raw)", "Brier_Score": brier_score_loss(y_test, p_cb_raw), "Log_Loss": log_loss(y_test, p_cb_raw), "ECE": ece_cb_raw},
        {"Model": "CatBoost (Isotonic)", "Brier_Score": brier_score_loss(y_test, p_cb_cal), "Log_Loss": log_loss(y_test, p_cb_cal), "ECE": ece_cb_cal},
    ]
    df_cal = pd.DataFrame(cal_metrics)

    return df_rejection, df_cal, (y_test, p_tab, p_cb_raw, p_cb_cal)


# ---------------------------------------------------------
# 8. Plotting & Summary Artifact Generation
# ---------------------------------------------------------

def generate_publication_visualizations(df_all_folds, df_scaling, df_rejection, df_cal, cal_curves_data, output_dir="results"):
    """Generates all publication-quality comparison charts and saves to results/."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Boxplots of AUC across datasets
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("muted", n_colors=df_all_folds["Model"].nunique())
    g = sns.boxplot(
        data=df_all_folds,
        x="Dataset",
        y="AUC",
        hue="Model",
        palette=palette,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    plt.title("Repeated Stratified CV Performance (5 Folds x 5 Repeats = 25 Runs per Model)", fontsize=14, weight="bold")
    plt.xlabel("Benchmark Dataset", fontsize=12)
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auc_boxplots_by_dataset.png"), dpi=300)
    plt.close()

    # 2. Scaling Curve: AUC vs Sample Size
    plt.figure(figsize=(9, 6))
    summary_scaling = df_scaling.groupby(["N_train", "Model"])["AUC"].agg(["mean", "std"]).reset_index()
    for model_name, m_group in summary_scaling.groupby("Model"):
        plt.plot(m_group["N_train"], m_group["mean"], marker="o", linewidth=2.5, label=model_name)
        plt.fill_between(m_group["N_train"], m_group["mean"] - m_group["std"], m_group["mean"] + m_group["std"], alpha=0.15)
    plt.xscale("log")
    plt.title("Scaling Behavior: ROC-AUC vs. Training Set Size (Adult Census)", fontsize=13, weight="bold")
    plt.xlabel("Training Instances (N_train, Log Scale)", fontsize=11)
    plt.ylabel("ROC-AUC (Fixed 2,000 Instance Test Set)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scaling_curve_sample_size.png"), dpi=300)
    plt.close()

    # 3. Uncertainty Rejection Curve
    plt.figure(figsize=(9, 6))
    plt.plot(df_rejection["Rejection_Rate"] * 100, df_rejection["Retained_Accuracy"], marker="s", color="#1f77b4", linewidth=2.5, label="Retained Set Accuracy")
    plt.plot(df_rejection["Rejection_Rate"] * 100, df_rejection["Retained_AUC"], marker="o", color="#2ca02c", linewidth=2.5, label="Retained Set ROC-AUC")
    plt.plot(df_rejection["Rejection_Rate"] * 100, df_rejection["Rejected_Accuracy"], marker="^", color="#d62728", linestyle="--", label="Rejected Set Accuracy (Abstained)")
    plt.title("Selective Prediction Rejection Curve via TabPFN Predictive Entropy", fontsize=13, weight="bold")
    plt.xlabel("Abstention / Rejection Rate (% Most Uncertain Samples Routed to Human Review)", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rejection_curve_uncertainty.png"), dpi=300)
    plt.close()

    # 4. Calibration & Reliability Diagram
    y_test, p_tab, p_cb_raw, p_cb_cal = cal_curves_data
    from sklearn.calibration import calibration_curve
    prob_true_tab, prob_pred_tab = calibration_curve(y_test, p_tab, n_bins=10)
    prob_true_cb, prob_pred_cb = calibration_curve(y_test, p_cb_raw, n_bins=10)
    prob_true_cb_cal, prob_pred_cb_cal = calibration_curve(y_test, p_cb_cal, n_bins=10)

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred_tab, prob_true_tab, marker="o", linewidth=2.2, label=f"TabPFN v2 (ECE: {df_cal.loc[df_cal['Model']=='TabPFN (v2)', 'ECE'].values[0]:.3f})")
    plt.plot(prob_pred_cb, prob_true_cb, marker="s", linewidth=2.2, label=f"CatBoost Raw (ECE: {df_cal.loc[df_cal['Model']=='CatBoost (Raw)', 'ECE'].values[0]:.3f})")
    plt.plot(prob_pred_cb_cal, prob_true_cb_cal, marker="^", linewidth=2.2, label=f"CatBoost Isotonic (ECE: {df_cal.loc[df_cal['Model']=='CatBoost (Isotonic)', 'ECE'].values[0]:.3f})")
    plt.title("Reliability Diagrams & Probability Calibration (German Credit)", fontsize=13, weight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_reliability_curves.png"), dpi=300)
    plt.close()

    # 5. Business Financial Impact (Cost & Profit)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Credit-G Cost
    credit_df = df_all_folds[df_all_folds["Dataset"] == "credit-g"].dropna(subset=["Credit_Cost_Per_Applicant"])
    order = credit_df.groupby("Model")["Credit_Cost_Per_Applicant"].mean().sort_values().index
    sns.barplot(data=credit_df, x="Model", y="Credit_Cost_Per_Applicant", order=order, ax=axes[0], ci=95, palette="crest")
    axes[0].set_title("German Credit: Expected Cost per Applicant\n(Cost: 5 for approving bad debt, 1 for rejecting good)", fontsize=11, weight="bold")
    axes[0].set_ylabel("Expected Cost per Applicant (Lower is Better)", fontsize=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")

    # Telco Churn Profit
    churn_df = df_all_folds[df_all_folds["Dataset"] == "telco-churn"].dropna(subset=["Churn_Profit_Per_Customer"])
    order_profit = churn_df.groupby("Model")["Churn_Profit_Per_Customer"].mean().sort_values(ascending=False).index
    sns.barplot(data=churn_df, x="Model", y="Churn_Profit_Per_Customer", order=order_profit, ax=axes[1], ci=95, palette="viridis")
    axes[1].set_title("Telco Churn: Expected Profit per Customer\n(Value $100 retained vs $20 incentive cost)", fontsize=11, weight="bold")
    axes[1].set_ylabel("Expected Profit per Customer (Higher is Better)", fontsize=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "business_financial_impact.png"), dpi=300)
    plt.close()

    # 6. Fit Time & Per-Row Inference Latency
    plt.figure(figsize=(10, 6))
    time_summary = df_all_folds.groupby("Model")[["Fit_Time_s", "Per_Row_Predict_ms"]].mean().reset_index()
    sns.scatterplot(
        data=time_summary,
        x="Fit_Time_s",
        y="Per_Row_Predict_ms",
        hue="Model",
        s=250,
        palette="tab10"
    )
    for _, row in time_summary.iterrows():
        plt.text(row["Fit_Time_s"] * 1.05, row["Per_Row_Predict_ms"] * 1.02, row["Model"], fontsize=9)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Speed & Latency Profile: Fit Time vs. Per-Row Inference Latency", fontsize=13, weight="bold")
    plt.xlabel("Mean Fit Time (seconds, Log Scale)", fontsize=11)
    plt.ylabel("Inference Latency per Row (milliseconds, Log Scale)", fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_pareto_analysis.png"), dpi=300)
    plt.close()

    print("[Visuals] All publication-grade figures successfully generated.")


# ---------------------------------------------------------
# 9. Main Orchestration
# ---------------------------------------------------------

def main():
    device = "cuda"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("   TabPFN Multi-Dataset Statistical Benchmark (Nature 637)       ")
    print("=================================================================")

    datasets = ["credit-g", "telco-churn", "bank-marketing", "adult"]
    all_fold_dfs = []

    for d_name in datasets:
        df_d, _ = run_repeated_cv_benchmark(d_name, n_splits=5, n_repeats=5, device=device)
        all_fold_dfs.append(df_d)

    df_all_folds = pd.concat(all_fold_dfs, ignore_index=True)
    df_all_folds.to_csv(os.path.join(output_dir, "per_fold_metrics.csv"), index=False)
    print(f"\n[Saved] Raw per-fold results saved to {output_dir}/per_fold_metrics.csv")

    # Statistical Aggregation per Dataset & Model
    summary_rows = []
    tab_model = "TabPFN (v2)"

    for d_name in datasets:
        df_d = df_all_folds[df_all_folds["Dataset"] == d_name]
        tab_df = df_d[df_d["Model"] == tab_model].sort_values("Fold")
        tab_aucs = tab_df["AUC"].values

        for m in df_d["Model"].unique():
            m_df = df_d[df_d["Model"] == m].sort_values("Fold")
            m_aucs = m_df["AUC"].values

            mean_auc, ci_low, ci_high, h_err = compute_confidence_interval(m_aucs)
            std_auc = np.std(m_aucs, ddof=1)
            mean_fit = m_df["Fit_Time_s"].mean()
            mean_pred = m_df["Predict_Time_s"].mean()
            per_row_lat = m_df["Per_Row_Predict_ms"].mean()

            # Paired comparison with TabPFN
            diffs = tab_aucs - m_aucs
            mean_diff, diff_ci_low, diff_ci_high, _ = compute_confidence_interval(diffs)
            
            # Tests
            if m == tab_model:
                wilcox_p = 1.0
                nb_p = 1.0
            else:
                try:
                    wilcox_res = stats.wilcoxon(diffs)
                    wilcox_p = float(wilcox_res.pvalue)
                except Exception:
                    wilcox_p = 1.0
                n_train_est = int(len(df_d) / 25 * 0.8)
                n_test_est = int(len(df_d) / 25 * 0.2)
                _, nb_p = nadeau_bengio_corrected_t_test(diffs, n_train_est, n_test_est)

            summary_rows.append({
                "Dataset": d_name,
                "Model": m,
                "Mean_AUC": mean_auc,
                "Std_AUC": std_auc,
                "CI95_Lower": ci_low,
                "CI95_Upper": ci_high,
                "Margin_Error": h_err,
                "Mean_Fit_Time_s": mean_fit,
                "Mean_Predict_Time_s": mean_pred,
                "Per_Row_Predict_ms": per_row_lat,
                "Paired_Diff_vs_TabPFN": mean_diff,
                "Diff_CI95_Lower": diff_ci_low,
                "Diff_CI95_Upper": diff_ci_high,
                "Wilcoxon_p_value": wilcox_p,
                "Nadeau_Bengio_p_value": nb_p
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, "dataset_summary_metrics.csv"), index=False)
    print(f"[Saved] Aggregated statistical metrics saved to {output_dir}/dataset_summary_metrics.csv")

    # Win / Tie / Loss Table against best tuned baseline per dataset
    wtl_rows = []
    for d_name in datasets:
        d_sum = df_summary[(df_summary["Dataset"] == d_name) & (df_summary["Model"] != tab_model)]
        # Filter to tuned or competitive baselines
        tuned_baselines = d_sum[d_sum["Model"].str.contains("Tuned|CatBoost|XGBoost|LightGBM")]
        best_baseline_row = tuned_baselines.loc[tuned_baselines["Mean_AUC"].idxmax()]
        best_baseline_model = best_baseline_row["Model"]

        tab_row = df_summary[(df_summary["Dataset"] == d_name) & (df_summary["Model"] == tab_model)].iloc[0]
        tab_auc = tab_row["Mean_AUC"]
        base_auc = best_baseline_row["Mean_AUC"]
        diff = tab_auc - base_auc
        p_val = best_baseline_row["Wilcoxon_p_value"]

        if p_val < 0.05 and diff > 0:
            outcome = "Win (TabPFN)"
        elif p_val < 0.05 and diff < 0:
            outcome = "Loss (TabPFN)"
        else:
            outcome = "Tie / Inconclusive"

        wtl_rows.append({
            "Dataset": d_name,
            "Best_Baseline": best_baseline_model,
            "TabPFN_Mean_AUC": tab_auc,
            "Baseline_Mean_AUC": base_auc,
            "Paired_Diff": diff,
            "Wilcoxon_p": p_val,
            "Outcome": outcome
        })

    df_wtl = pd.DataFrame(wtl_rows)
    df_wtl.to_csv(os.path.join(output_dir, "win_tie_loss_table.csv"), index=False)
    print(f"[Saved] Win/Tie/Loss table saved to {output_dir}/win_tie_loss_table.csv")

    # Business Financial Summary Table
    bus_rows = []
    credit_df = df_all_folds[df_all_folds["Dataset"] == "credit-g"].dropna(subset=["Credit_Cost_Per_Applicant"])
    tab_costs = credit_df[credit_df["Model"] == tab_model]["Credit_Cost_Per_Applicant"].values
    for m in credit_df["Model"].unique():
        m_costs = credit_df[credit_df["Model"] == m]["Credit_Cost_Per_Applicant"].values
        mean_c, c_low, c_high, _ = compute_confidence_interval(m_costs)
        diff_c = m_costs - tab_costs  # baseline cost minus TabPFN cost (positive means TabPFN saves money)
        mean_d, d_low, d_high, _ = compute_confidence_interval(diff_c)
        bus_rows.append({
            "Dataset": "credit-g", "Scenario": "Cost per Applicant ($)", "Model": m,
            "Mean_Metric": mean_c, "CI95_Lower": c_low, "CI95_Upper": c_high,
            "Savings_vs_Baseline": mean_d, "Savings_CI95_Low": d_low, "Savings_CI95_High": d_high
        })

    churn_df = df_all_folds[df_all_folds["Dataset"] == "telco-churn"].dropna(subset=["Churn_Profit_Per_Customer"])
    tab_profits = churn_df[churn_df["Model"] == tab_model]["Churn_Profit_Per_Customer"].values
    for m in churn_df["Model"].unique():
        m_profits = churn_df[churn_df["Model"] == m]["Churn_Profit_Per_Customer"].values
        mean_p, p_low, p_high, _ = compute_confidence_interval(m_profits)
        diff_p = tab_profits - m_profits  # TabPFN profit minus baseline profit (positive means TabPFN earns more)
        mean_d, d_low, d_high, _ = compute_confidence_interval(diff_p)
        bus_rows.append({
            "Dataset": "telco-churn", "Scenario": "Profit per Customer ($)", "Model": m,
            "Mean_Metric": mean_p, "CI95_Lower": p_low, "CI95_Upper": p_high,
            "Incremental_Profit_vs_Baseline": mean_d, "Profit_CI95_Low": d_low, "Profit_CI95_High": d_high
        })

    df_bus = pd.DataFrame(bus_rows)
    df_bus.to_csv(os.path.join(output_dir, "business_financial_summary.csv"), index=False)
    print(f"[Saved] Business financial summary saved to {output_dir}/business_financial_summary.csv")

    # Scaling Experiment
    df_scaling = run_scaling_curve_experiment(dataset_name="adult", device=device)
    df_scaling.to_csv(os.path.join(output_dir, "scaling_curve_data.csv"), index=False)
    print(f"[Saved] Scaling curve raw data saved to {output_dir}/scaling_curve_data.csv")

    # Uncertainty, Rejection & Calibration Experiment
    df_rejection, df_cal, cal_curves_data = evaluate_uncertainty_and_calibration(dataset_name="credit-g", device=device)
    df_rejection.to_csv(os.path.join(output_dir, "rejection_curve_data.csv"), index=False)
    df_cal.to_csv(os.path.join(output_dir, "calibration_metrics.csv"), index=False)
    print(f"[Saved] Rejection curve and calibration metrics saved to {output_dir}/")

    # Generate Publication Figures
    generate_publication_visualizations(df_all_folds, df_scaling, df_rejection, df_cal, cal_curves_data, output_dir=output_dir)

    # Output artifacts saved in results directory

    print("\n=================================================================")
    print("   Benchmarking Complete. All CSVs & Figures Ready in /results   ")
    print("=================================================================")



if __name__ == "__main__":
    main()
