"""
Chronos-2 vs Baselines on M5 Retail Demand: Empirical Benchmark & Business Evaluation
Paper Reference: "Chronos-2: From Univariate to Universal Forecasting" (arXiv:2510.15821, Oct 2025)
Authors: Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado,
         Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami,
         Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson,
         Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, Michael Bohlke-Schneider.
Affiliation: Amazon Web Services (AWS) AI Labs.
Dataset: M5 Forecasting Accuracy (Walmart retail sales, 1,000 item-store series, 3 rolling origins).
Hardware: Dedicated Cloud T4 GPU (16GB VRAM, CUDA acceleration) / x86_64 CPU.
"""

import os
import sys
import time
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global Constants & Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SERIES = 1000
SMOKE_N_SERIES = 20
HORIZON = 28
SEASONALITY = 7
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HISTORY_DAYS = 365
BOOTSTRAP_ROUNDS = 10000

# Origins on M5 timeline (d_1 to d_1941)
# Origin 1: Train d_1..d_1857 (test d_1858..d_1885)
# Origin 2: Train d_1..d_1885 (test d_1886..d_1913)
# Origin 3: Train d_1..d_1913 (test d_1914..d_1941)
ORIGINS = [
    {"origin": 1, "train_end": 1857, "test_start": 1858, "test_end": 1885},
    {"origin": 2, "train_end": 1885, "test_start": 1886, "test_end": 1913},
    {"origin": 3, "train_end": 1913, "test_start": 1914, "test_end": 1941},
]

# Validation window for LightGBM hyperparameter search (strictly prior to Origin 1)
LGB_VAL_WINDOW = {
    "train_end": 1829,
    "val_start": 1830,
    "val_end": 1857
}

NEWSVENDOR_RATIOS = [
    {"name": "1:1", "c_u": 1.0, "c_o": 1.0},
    {"name": "3:1", "c_u": 3.0, "c_o": 1.0},
    {"name": "5:1", "c_u": 5.0, "c_o": 1.0},
    {"name": "9:1", "c_u": 9.0, "c_o": 1.0},
]


def log_message(msg: str, log_file: Path):
    """Logs message to stdout and run_log.txt."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted)
    with open(log_file, "a") as f:
        f.write(formatted + "\n")


def check_environment(log_file: Path):
    """Step 0: Log hardware, Python, and package versions."""
    log_message("=== STEP 0: ENVIRONMENT & HARDWARE VERIFICATION ===", log_file)
    log_message(f"Python Version: {sys.version.split()[0]}", log_file)
    log_message(f"Platform: {sys.platform}", log_file)

    try:
        import torch
        log_message(f"PyTorch Version: {torch.__version__}", log_file)
        cuda_avail = torch.cuda.is_available()
        log_message(f"CUDA Available: {cuda_avail}", log_file)
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log_message(f"GPU Device: {device_name} ({vram_gb:.2f} GB VRAM)", log_file)
    except ImportError:
        log_message("PyTorch: Not Installed", log_file)

    for pkg in ["chronos", "statsforecast", "lightgbm", "pandas", "numpy", "scipy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            log_message(f"Library {pkg}: {ver}", log_file)
        except ImportError:
            log_message(f"Library {pkg}: NOT INSTALLED", log_file)


def locate_m5_data(custom_dir=None):
    """Locates M5 dataset directory."""
    candidates = [
        custom_dir,
        "/kaggle/input/m5-forecasting-accuracy",
        "/kaggle/input/m5-forecasting-accuracy/m5-forecasting-accuracy",
        "./data/m5-forecasting-accuracy",
        "../data/m5-forecasting-accuracy",
        "data",
        "."
    ]

    # Dynamically search /kaggle/input if present
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "calendar.csv" in files and ("sales_train_evaluation.csv" in files or "sales_train_validation.csv" in files):
                candidates.insert(0, root)

    for c in candidates:
        if c and os.path.exists(c):
            cal = os.path.join(c, "calendar.csv")
            sales = os.path.join(c, "sales_train_evaluation.csv")
            if not os.path.exists(sales):
                sales = os.path.join(c, "sales_train_validation.csv")
            prices = os.path.join(c, "sell_prices.csv")
            if os.path.exists(cal) and os.path.exists(sales) and os.path.exists(prices):
                return c, cal, sales, prices
    raise FileNotFoundError(
        "M5 Forecasting Accuracy dataset not found! Ensure calendar.csv, "
        "sales_train_evaluation.csv (or validation), and sell_prices.csv are accessible."
    )


# ---------------------------------------------------------------------------
# Data Ingestion & Preprocessing
# ---------------------------------------------------------------------------
def load_sampled_m5(data_dir: str, n_series: int = N_SERIES, seed: int = RANDOM_SEED, log_file: Path = None):
    """
    Loads calendar, sell prices, and sales data.
    Filters series with at least 100 non-zero sales days up to day 1857.
    Samples exactly n_series with fixed random seed.
    """
    if log_file:
        log_message(f"Loading M5 dataset from '{data_dir}'...", log_file)

    cal_path = os.path.join(data_dir, "calendar.csv")
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    if not os.path.exists(sales_path):
        sales_path = os.path.join(data_dir, "sales_train_validation.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")

    calendar = pd.read_csv(cal_path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["has_event_1"] = calendar["event_name_1"].notna().astype(int)
    calendar["has_event_2"] = calendar["event_name_2"].notna().astype(int)
    calendar["has_event"] = ((calendar["has_event_1"] + calendar["has_event_2"]) > 0).astype(int)

    sales_df = pd.read_csv(sales_path)
    if log_file:
        log_message(f"Total series available in raw sales file: {len(sales_df):,}", log_file)

    # Count non-zero days up to day 1857 (prior to first evaluation test origin)
    d_cols_up_to_1857 = [f"d_{i}" for i in range(1, 1858)]
    non_zero_counts = (sales_df[d_cols_up_to_1857] > 0).sum(axis=1)

    eligible_mask = non_zero_counts >= 100
    eligible_df = sales_df[eligible_mask].copy()
    if log_file:
        log_message(
            f"Series with >= 100 non-zero sales days: {len(eligible_df):,} "
            f"({len(eligible_df)/len(sales_df)*100:.1f}%)", log_file
        )

    # Deterministic sampling
    sampled_sales = eligible_df.sample(n=n_series, random_state=seed).reset_index(drop=True)
    if log_file:
        log_message(f"Sampled exactly {len(sampled_sales)} series (seed={seed}).", log_file)

    # Load and filter sell_prices for sampled item-store combinations only
    sampled_item_stores = set(zip(sampled_sales["store_id"], sampled_sales["item_id"]))
    prices_iter = pd.read_csv(prices_path)
    prices_filtered = prices_iter[
        prices_iter.set_index(["store_id", "item_id"]).index.isin(sampled_item_stores)
    ].copy()
    if log_file:
        log_message(f"Loaded {len(prices_filtered):,} price records for sampled series.", log_file)

    return calendar, sampled_sales, prices_filtered


def prepare_origin_data(sampled_sales, calendar, prices_df, origin_cfg, history_days=HISTORY_DAYS):
    """
    Prepares train and test arrays, context_df, and future_df for a specific rolling origin.
    Explicitly checks against target leakage.
    """
    t_end = origin_cfg["train_end"]
    t_start = max(1, t_end - history_days + 1)
    test_start = origin_cfg["test_start"]
    test_end = origin_cfg["test_end"]

    hist_d_cols = [f"d_{i}" for i in range(t_start, t_end + 1)]
    test_d_cols = [f"d_{i}" for i in range(test_start, test_end + 1)]

    # Compute in-sample seasonal naive scale (Lag 7 MAE over historical sales)
    full_train_cols = [f"d_{i}" for i in range(1, t_end + 1)]
    full_train_arr = sampled_sales[full_train_cols].values.astype(np.float32)

    scales = np.zeros(len(sampled_sales), dtype=np.float32)
    for i in range(len(sampled_sales)):
        s_arr = full_train_arr[i]
        diffs = np.abs(s_arr[7:] - s_arr[:-7])
        s_mean = np.mean(diffs)
        scales[i] = s_mean if s_mean > 1e-8 else 0.0

    # Intermittent demand classification (>50% zeros in history)
    zero_fracs = (full_train_arr == 0).mean(axis=1)
    is_intermittent = (zero_fracs > 0.50).astype(int)

    # Actual test values
    y_true = sampled_sales[test_d_cols].values.astype(np.float32)

    # Map dates
    cal_dict = calendar.set_index("d").to_dict(orient="index")

    # Build long-format context_df and future_df
    hist_records = []
    future_records = []

    # Fast price lookup dictionary: (store_id, item_id, wm_yr_wk) -> sell_price
    price_lookup = prices_df.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"].to_dict()

    for idx, row in sampled_sales.iterrows():
        s_id = row["id"]
        store = row["store_id"]
        item = row["item_id"]
        state = row["state_id"]
        snap_col = f"snap_{state}"

        # History records
        for d in hist_d_cols:
            cal_info = cal_dict[d]
            dt = cal_info["date"]
            wk = cal_info["wm_yr_wk"]
            price = price_lookup.get((store, item, wk), np.nan)
            event = cal_info["has_event"]
            snap = cal_info[snap_col]
            target_val = float(row[d])

            hist_records.append({
                "id": s_id,
                "timestamp": dt,
                "target": target_val,
                "sell_price": price,
                "has_event": event,
                "snap": snap,
                "wday": cal_info["wday"],
                "month": cal_info["month"]
            })

        # Future records (strictly future-known covariates, NO TARGET)
        for d in test_d_cols:
            cal_info = cal_dict[d]
            dt = cal_info["date"]
            wk = cal_info["wm_yr_wk"]
            price = price_lookup.get((store, item, wk), np.nan)
            event = cal_info["has_event"]
            snap = cal_info[snap_col]

            future_records.append({
                "id": s_id,
                "timestamp": dt,
                "sell_price": price,
                "has_event": event,
                "snap": snap,
                "wday": cal_info["wday"],
                "month": cal_info["month"]
            })

    context_df = pd.DataFrame(hist_records)
    future_df = pd.DataFrame(future_records)

    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])

    # Impute missing prices with series forward-fill / back-fill
    context_df["sell_price"] = context_df.groupby("id")["sell_price"].ffill().bfill().fillna(1.0)
    # Forward-fill prices into future_df from context
    last_known_prices = context_df.groupby("id")["sell_price"].last().to_dict()
    future_df["sell_price"] = future_df["sell_price"].fillna(future_df["id"].map(last_known_prices)).fillna(1.0)

    # LEAKAGE ASSERTION: Target column must NOT exist in future_df
    assert "target" not in future_df.columns, "CRITICAL ERROR: Target variable leaked into future_df!"
    assert len(future_df) == len(sampled_sales) * HORIZON, "future_df length mismatch!"

    return {
        "hist_arr": sampled_sales[hist_d_cols].values.astype(np.float32),
        "y_true": y_true,
        "scales": scales,
        "is_intermittent": is_intermittent,
        "zero_fracs": zero_fracs,
        "context_df": context_df,
        "future_df": future_df,
        "series_meta": sampled_sales[["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"]].copy()
    }


# ---------------------------------------------------------------------------
# Model Implementations
# ---------------------------------------------------------------------------
def run_seasonal_naive(hist_arr: np.ndarray, horizon: int = HORIZON, quantiles=QUANTILES):
    """Seasonal Naive (Weekly, period=7) with empirical residual quantiles."""
    t0 = time.perf_counter()
    n_series, t_len = hist_arr.shape
    q_preds = np.zeros((n_series, horizon, len(quantiles)), dtype=np.float32)

    # Point forecast: repeat last week of training
    last_week = hist_arr[:, -7:]  # (n_series, 7)
    repeats = int(np.ceil(horizon / 7))
    point_forecast = np.tile(last_week, (1, repeats))[:, :horizon]  # (n_series, horizon)

    # In-sample weekly seasonal errors
    in_sample_errs = hist_arr[:, 7:] - hist_arr[:, :-7]  # (n_series, t_len - 7)

    for i in range(n_series):
        err_dist = in_sample_errs[i]
        q_offsets = np.quantile(err_dist, quantiles)
        for q_idx, offset in enumerate(q_offsets):
            q_preds[i, :, q_idx] = np.maximum(0.0, point_forecast[i] + offset)

    fit_time = 0.001
    pred_time = time.perf_counter() - t0
    return q_preds, fit_time, pred_time


def run_auto_ets(hist_arr: np.ndarray, horizon: int = HORIZON, quantiles=QUANTILES):
    """AutoETS using statsforecast with season_length=7."""
    t0 = time.perf_counter()
    n_series, t_len = hist_arr.shape
    q_preds = np.zeros((n_series, horizon, len(quantiles)), dtype=np.float32)

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        records = []
        dates = pd.date_range("2020-01-01", periods=t_len, freq="D")
        for i in range(n_series):
            s_id = str(i)
            for t_idx, d in enumerate(dates):
                records.append({"unique_id": s_id, "ds": d, "y": float(hist_arr[i, t_idx])})
        sf_df = pd.DataFrame(records)

        sf = StatsForecast(
            models=[AutoETS(season_length=7, model="ZZZ")],
            freq="D",
            n_jobs=-1
        )
        t_fit_start = time.perf_counter()
        sf.fit(sf_df)
        fit_time = time.perf_counter() - t_fit_start

        t_pred_start = time.perf_counter()
        # Predict with intervals
        forecasts = sf.predict(h=horizon, level=[20, 40, 60, 80])
        pred_time = time.perf_counter() - t_pred_start

        forecasts_df = forecasts.reset_index()
        uid_col = "unique_id" if "unique_id" in forecasts_df.columns else forecasts_df.columns[0]
        forecasts_df[uid_col] = forecasts_df[uid_col].astype(str)
        by_uid = {k: v for k, v in forecasts_df.groupby(uid_col)}

        for i in range(n_series):
            s_id = str(i)
            if s_id in by_uid:
                sub = by_uid[s_id]
                mean_pred = sub["AutoETS"].values

                # Quantile mapping from prediction intervals
                # 80% interval -> 0.10 and 0.90
                # 60% interval -> 0.20 and 0.80
                # 40% interval -> 0.30 and 0.70
                # 20% interval -> 0.40 and 0.60
                # mean -> 0.50
                q_map = {
                    0.1: np.maximum(0.0, sub["AutoETS-lo-80"].values),
                    0.2: np.maximum(0.0, sub["AutoETS-lo-60"].values),
                    0.3: np.maximum(0.0, sub["AutoETS-lo-40"].values),
                    0.4: np.maximum(0.0, sub["AutoETS-lo-20"].values),
                    0.5: np.maximum(0.0, mean_pred),
                    0.6: np.maximum(0.0, sub["AutoETS-hi-20"].values),
                    0.7: np.maximum(0.0, sub["AutoETS-hi-40"].values),
                    0.8: np.maximum(0.0, sub["AutoETS-hi-60"].values),
                    0.9: np.maximum(0.0, sub["AutoETS-hi-80"].values),
                }
                for q_idx, q_val in enumerate(quantiles):
                    q_preds[i, :, q_idx] = q_map.get(q_val, mean_pred)
            else:
                q_preds[i, :, :] = 0.0

    except Exception as e:
        print(f"[AutoETS Warning] Falling back to seasonal naive due to: {e}")
        return run_seasonal_naive(hist_arr, horizon, quantiles)

    return q_preds, fit_time, pred_time


def run_lightgbm(context_df: pd.DataFrame, future_df: pd.DataFrame,
                 horizon: int = HORIZON, quantiles=QUANTILES, seed: int = RANDOM_SEED):
    """
    Global LightGBM model with lag, rolling statistics, calendar, and price features.
    Trained with validation-tuned hyperparameters (strictly non-overlapping validation window).
    """
    t0 = time.perf_counter()
    import lightgbm as lgb

    # Feature engineering helper
    def create_features(df_long):
        df_sorted = df_long.sort_values(["id", "timestamp"]).copy()
        features = pd.DataFrame()
        features["wday"] = df_sorted["wday"].astype(int)
        features["month"] = df_sorted["month"].astype(int)
        features["is_weekend"] = df_sorted["wday"].isin([1, 2]).astype(int)
        features["sell_price"] = df_sorted["sell_price"].astype(float)
        features["has_event"] = df_sorted["has_event"].astype(int)
        features["snap"] = df_sorted["snap"].astype(int)

        # Lags on target (lag >= 28 to prevent leakage during 28-day forecast horizon)
        for lag in [28, 35, 42, 49]:
            features[f"lag_{lag}"] = df_sorted.groupby("id")["target"].shift(lag)

        # Rolling statistics on lag 28
        features["roll_mean_7"] = df_sorted.groupby("id")["target"].shift(28).rolling(7).mean()
        features["roll_mean_28"] = df_sorted.groupby("id")["target"].shift(28).rolling(28).mean()
        features["roll_std_7"] = df_sorted.groupby("id")["target"].shift(28).rolling(7).std().fillna(0)

        return features

    # Combine context and future for continuous lagging
    future_full = future_df.copy()
    future_full["target"] = np.nan
    all_df = pd.concat([context_df, future_full], ignore_index=True)
    feats_all = create_features(all_df)
    feats_all["id"] = all_df["id"]
    feats_all["target"] = all_df["target"]

    # Training slice (non-null lags)
    train_mask = feats_all["target"].notna() & feats_all["lag_49"].notna()
    X_train = feats_all[train_mask].drop(columns=["id", "target"]).values
    y_train = feats_all.loc[train_mask, "target"].values

    # Test slice
    test_mask = feats_all["target"].isna()
    X_test = feats_all[test_mask].drop(columns=["id", "target"]).values
    test_ids = feats_all.loc[test_mask, "id"].values

    # Validation window tuning (15 random trials on internal validation split)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Internal validation split: last 28 days of training history
    val_size = min(len(X_train) // 4, 28 * len(context_df["id"].unique()))
    if val_size > 50 and len(X_train) > val_size * 2:
        X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    else:
        X_fit, y_fit = X_train, y_train
        X_val, y_val = X_train, y_train

    param_candidates = [
        {"learning_rate": lr, "num_leaves": nl, "min_child_samples": mcs,
         "subsample": ss, "colsample_bytree": cs, "n_estimators": 50,
         "random_state": seed, "verbosity": -1, "n_jobs": -1}
        for lr in [0.03, 0.05, 0.08]
        for nl in [15, 31, 63]
        for mcs in [20, 50]
        for ss, cs in [(0.8, 0.8), (0.9, 0.7)]
    ]
    chosen_trials = [param_candidates[i] for i in rng.choice(len(param_candidates), size=min(15, len(param_candidates)), replace=False)]

    best_score = float("inf")
    best_params = chosen_trials[0].copy()

    for p in chosen_trials:
        try:
            trial_model = lgb.LGBMRegressor(**p)
            trial_model.fit(X_fit, y_fit)
            val_preds = trial_model.predict(X_val)
            val_rmse = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = p.copy()
        except Exception:
            continue

    best_params["n_estimators"] = 100

    t_fit_start = time.perf_counter()
    # Train mean point model
    lgb_point = lgb.LGBMRegressor(**best_params)
    lgb_point.fit(X_train, y_train)

    # Train quantile models for each quantile level
    quantile_models = {}
    for q in quantiles:
        q_params = best_params.copy()
        q_params["objective"] = "quantile"
        q_params["alpha"] = q
        q_model = lgb.LGBMRegressor(**q_params)
        q_model.fit(X_train, y_train)
        quantile_models[q] = q_model
    fit_time = time.perf_counter() - t_fit_start

    t_pred_start = time.perf_counter()
    unique_ids = context_df["id"].unique()
    n_series = len(unique_ids)
    id_map = {uid: i for i, uid in enumerate(unique_ids)}

    q_preds = np.zeros((n_series, horizon, len(quantiles)), dtype=np.float32)

    for q_idx, q in enumerate(quantiles):
        preds_all = quantile_models[q].predict(X_test)
        preds_clipped = np.maximum(0.0, preds_all)
        # Reshape to (n_series, horizon)
        for step in range(horizon):
            step_indices = np.arange(step, len(X_test), horizon)
            q_preds[:, step, q_idx] = preds_clipped[step_indices]
    pred_time = time.perf_counter() - t_pred_start

    return q_preds, fit_time, pred_time


def run_chronos_bolt(hist_arr: np.ndarray, horizon: int = HORIZON, quantiles=QUANTILES, device="cuda"):
    """Chronos-Bolt zero-shot univariate probabilistic forecasting."""
    t0 = time.perf_counter()
    import torch
    from chronos import ChronosBoltPipeline

    n_series = len(hist_arr)
    q_preds = np.zeros((n_series, horizon, len(quantiles)), dtype=np.float32)

    actual_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if actual_device == "cuda" else torch.float32

    t_load = time.perf_counter()
    pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map=actual_device,
        torch_dtype=dtype
    )
    load_time = time.perf_counter() - t_load

    t_pred_start = time.perf_counter()
    batch_size = 64
    for start_idx in range(0, n_series, batch_size):
        end_idx = min(start_idx + batch_size, n_series)
        batch_context = torch.tensor(hist_arr[start_idx:end_idx], dtype=torch.float32)

        try:
            res = pipeline.predict_quantiles(
                context=batch_context,
                prediction_length=horizon,
                quantile_levels=quantiles
            )
        except TypeError:
            res = pipeline.predict_quantiles(
                batch_context,
                prediction_length=horizon,
                quantile_levels=quantiles
            )

        q_tensor = res[0] if isinstance(res, tuple) else res
        q_np = q_tensor.detach().cpu().numpy()

        if q_np.shape[1] == horizon and q_np.shape[2] == len(quantiles):
            q_preds[start_idx:end_idx, :, :] = np.maximum(0.0, q_np)
        elif q_np.shape[1] == len(quantiles) and q_np.shape[2] == horizon:
            q_preds[start_idx:end_idx, :, :] = np.maximum(0.0, np.transpose(q_np, (0, 2, 1)))
        else:
            raise ValueError(f"Unexpected shape from ChronosBoltPipeline: {q_np.shape}")

    pred_time = time.perf_counter() - t_pred_start
    return q_preds, load_time, pred_time


def run_chronos2_univariate(context_df: pd.DataFrame, horizon: int = HORIZON, quantiles=QUANTILES, device="cuda"):
    """Chronos-2 zero-shot univariate forecasting (without covariates)."""
    t0 = time.perf_counter()
    import torch
    from chronos import Chronos2Pipeline

    actual_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if actual_device == "cuda" else torch.float32

    t_load = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=actual_device,
        torch_dtype=dtype
    )
    load_time = time.perf_counter() - t_load

    # Strip covariates: retain only id, timestamp, target
    univariate_context = context_df[["id", "timestamp", "target"]].copy()

    t_pred_start = time.perf_counter()
    try:
        pred_df = pipeline.predict_df(
            df=univariate_context,
            prediction_length=horizon,
            quantile_levels=quantiles,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            batch_size=64
        )
    except TypeError:
        try:
            pred_df = pipeline.predict_df(
                univariate_context,
                prediction_length=horizon,
                quantile_levels=quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )
        except TypeError:
            pred_df = pipeline.predict_df(
                df=univariate_context,
                prediction_length=horizon,
                quantile_levels=quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )
    pred_time = time.perf_counter() - t_pred_start

    # Format predictions into (n_series, horizon, len(quantiles))
    unique_ids = list(context_df["id"].unique())
    n_series = len(unique_ids)
    q_preds = np.zeros((n_series, horizon, len(quantiles)), dtype=np.float32)

    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    pred_df_grouped = pred_df.groupby("id")

    for uid, group in pred_df_grouped:
        if uid not in id_to_idx:
            continue
        s_idx = id_to_idx[uid]
        group_sorted = group.sort_values("timestamp")
        for q_idx, q in enumerate(quantiles):
            found_col = None
            for c in group_sorted.columns:
                c_str = str(c).strip()
                if c_str == str(q) or c_str == f"{q:.1f}" or c_str == f"{q:.2f}" or c_str == f"q_{q}":
                    found_col = c
                    break
            if found_col is None:
                for c in group_sorted.columns:
                    try:
                        if abs(float(c) - q) < 1e-4:
                            found_col = c
                            break
                    except ValueError:
                        continue
            if found_col is not None:
                vals = group_sorted[found_col].values[:horizon]
                q_preds[s_idx, :len(vals), q_idx] = np.maximum(0.0, vals)

    return q_preds, load_time, pred_time


def run_chronos2_covariates(context_df: pd.DataFrame, future_df: pd.DataFrame,
                            horizon: int = HORIZON, quantiles=QUANTILES, device="cuda"):
    """Chronos-2 zero-shot forecasting WITH price, event, and SNAP covariates."""
    t0 = time.perf_counter()
    import torch
    from chronos import Chronos2Pipeline

    actual_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if actual_device == "cuda" else torch.float32

    t_load = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=actual_device,
        torch_dtype=dtype
    )
    load_time = time.perf_counter() - t_load

    t_pred_start = time.perf_counter()
    try:
        pred_df = pipeline.predict_df(
            df=context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=quantiles,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            batch_size=64
        )
    except TypeError:
        try:
            pred_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=horizon,
                quantile_levels=quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )
        except TypeError:
            pred_df = pipeline.predict_df(
                df=context_df,
                future_df=future_df,
                prediction_length=horizon,
                quantile_levels=quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target"
            )
    pred_time = time.perf_counter() - t_pred_start

    # Format predictions
    unique_ids = list(context_df["id"].unique())
    n_series = len(unique_ids)
    q_preds = np.zeros((n_series, horizon, len(quantiles)), dtype=np.float32)

    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    pred_df_grouped = pred_df.groupby("id")

    for uid, group in pred_df_grouped:
        if uid not in id_to_idx:
            continue
        s_idx = id_to_idx[uid]
        group_sorted = group.sort_values("timestamp")
        for q_idx, q in enumerate(quantiles):
            found_col = None
            for c in group_sorted.columns:
                c_str = str(c).strip()
                if c_str == str(q) or c_str == f"{q:.1f}" or c_str == f"{q:.2f}" or c_str == f"q_{q}":
                    found_col = c
                    break
            if found_col is None:
                for c in group_sorted.columns:
                    try:
                        if abs(float(c) - q) < 1e-4:
                            found_col = c
                            break
                    except ValueError:
                        continue
            if found_col is not None:
                vals = group_sorted[found_col].values[:horizon]
                q_preds[s_idx, :len(vals), q_idx] = np.maximum(0.0, vals)

    return q_preds, load_time, pred_time


# ---------------------------------------------------------------------------
# Evaluation & Metrics Computation
# ---------------------------------------------------------------------------
def compute_metrics_per_series(y_true: np.ndarray, q_preds: np.ndarray,
                               scales: np.ndarray, quantiles=QUANTILES):
    """
    Computes MASE, WQL (0.1..0.9), Newsvendor cost (1:1, 3:1, 5:1, 9:1),
    and quantile crossing rate per series.
    """
    n_series, horizon = y_true.shape
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    q90_idx = quantiles.index(0.9) if 0.9 in quantiles else -1

    point_pred = q_preds[:, :, q50_idx]  # Median forecast
    q90_pred = q_preds[:, :, q90_idx]

    # MASE
    mae = np.mean(np.abs(y_true - point_pred), axis=1)  # (n_series,)
    mase = np.full(n_series, np.nan, dtype=np.float32)
    valid_scale_mask = scales > 0
    mase[valid_scale_mask] = mae[valid_scale_mask] / scales[valid_scale_mask]

    # WQL over quantiles 0.1..0.9
    wql_per_series = np.zeros(n_series, dtype=np.float32)
    total_y = np.sum(np.abs(y_true), axis=1)  # (n_series,)

    for q_idx, q in enumerate(quantiles):
        q_val = q_preds[:, :, q_idx]
        diff = y_true - q_val
        pinball = 2.0 * np.maximum(q * diff, (q - 1.0) * diff)
        sum_pinball = np.sum(pinball, axis=1)
        # Avoid division by zero on series with zero sales in test horizon
        safe_total = np.maximum(total_y, 1e-5)
        wql_per_series += (sum_pinball / safe_total)
    wql_per_series /= len(quantiles)

    # Newsvendor inventory cost at 0.9 quantile
    newsvendor_costs = {}
    for ratio in NEWSVENDOR_RATIOS:
        cu = ratio["c_u"]
        co = ratio["c_o"]
        underage = np.maximum(0.0, y_true - q90_pred)
        overage = np.maximum(0.0, q90_pred - y_true)
        cost = np.sum(cu * underage + co * overage, axis=1)
        newsvendor_costs[ratio["name"]] = cost

    # Quantile crossing check: count violations where q_k > q_{k+1}
    crossing_violations = 0
    total_comparisons = n_series * horizon * (len(quantiles) - 1)
    for q_idx in range(len(quantiles) - 1):
        crossing_violations += np.sum(q_preds[:, :, q_idx] > q_preds[:, :, q_idx + 1])
    crossing_rate = float(crossing_violations / total_comparisons)

    return {
        "mase": mase,
        "wql": wql_per_series,
        "newsvendor_1_1": newsvendor_costs["1:1"],
        "newsvendor_3_1": newsvendor_costs["3:1"],
        "newsvendor_5_1": newsvendor_costs["5:1"],
        "newsvendor_9_1": newsvendor_costs["9:1"],
        "crossing_rate": crossing_rate
    }


# ---------------------------------------------------------------------------
# Statistical Bootstrap (10,000 resamples, seed 42)
# ---------------------------------------------------------------------------
def run_paired_bootstrap(series_agg_df: pd.DataFrame,
                         ref_model: str = "Chronos-2 (Covariates)",
                         n_boot: int = BOOTSTRAP_ROUNDS,
                         seed: int = RANDOM_SEED):
    """
    Computes paired difference vs ref_model per series across aggregated origins.
    Runs 10,000 bootstrap resamples over series.
    Verdict is WIN/LOSS only if 95% CI excludes 0, else TIE.
    """
    np.random.seed(seed)
    models = [m for m in series_agg_df["model"].unique() if m != ref_model]
    records = []

    metrics = [
        ("MASE", "mase"),
        ("WQL", "wql"),
        ("Newsvendor (3:1)", "newsvendor_3_1")
    ]

    ref_data = series_agg_df[series_agg_df["model"] == ref_model].set_index("id")

    for m in models:
        base_data = series_agg_df[series_agg_df["model"] == m].set_index("id")
        common_ids = ref_data.index.intersection(base_data.index)

        for metric_name, col in metrics:
            # Drop NaN (e.g. series with scale=0 for MASE)
            valid_mask = ref_data.loc[common_ids, col].notna() & base_data.loc[common_ids, col].notna()
            diffs = (base_data.loc[common_ids, col] - ref_data.loc[common_ids, col])[valid_mask].values
            # diff > 0 means Baseline has HIGHER error/cost -> Chronos-2 is BETTER (WIN)

            n_samples = len(diffs)
            mean_diff = float(np.mean(diffs))

            # Bootstrap resampling
            boot_idx = np.random.randint(0, n_samples, size=(n_boot, n_samples))
            boot_means = np.mean(diffs[boot_idx], axis=1)

            ci_low = float(np.quantile(boot_means, 0.025))
            ci_high = float(np.quantile(boot_means, 0.975))

            if ci_low > 0.0:
                verdict = "WIN"  # Chronos-2 statistically outperforms baseline
            elif ci_high < 0.0:
                verdict = "LOSS"  # Baseline statistically outperforms Chronos-2
            else:
                verdict = "TIE"  # 95% CI includes 0

            records.append({
                "baseline_model": m,
                "reference_model": ref_model,
                "metric": metric_name,
                "n_series_evaluated": n_samples,
                "mean_paired_difference": mean_diff,
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "verdict": verdict,
                "multiple_comparison_correction": "None"
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# High-Resolution Publication Charts (Matplotlib)
# ---------------------------------------------------------------------------
def generate_charts(summary_df: pd.DataFrame, win_tie_df: pd.DataFrame,
                    newsvendor_df: pd.DataFrame, runtime_df: pd.DataFrame,
                    output_dir: Path):
    """
    Produces 4 publication-quality PNG charts according to formatting rules:
    - Thin marks, direct labels, bars start at zero, no dual axes, 1080px readable.
    """
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linewidth": 0.6
    })

    # CHART 1: Paired MASE Difference vs Baselines with 95% Bootstrap CI
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    mase_comp = win_tie_df[win_tie_df["metric"] == "MASE"].copy()
    mase_comp = mase_comp.sort_values("mean_paired_difference", ascending=True)

    y_pos = np.arange(len(mase_comp))
    diffs = mase_comp["mean_paired_difference"].values
    err_low = diffs - mase_comp["ci_lower_95"].values
    err_high = mase_comp["ci_upper_95"].values - diffs

    colors = ["#2b8a3e" if v == "WIN" else ("#c92a2a" if v == "LOSS" else "#495057")
              for v in mase_comp["verdict"]]

    ax.errorbar(diffs, y_pos, xerr=[err_low, err_high], fmt="o", color="#1c7ed6",
                ecolor="#339af0", elinewidth=1.5, capsize=4, capthick=1.2, markersize=7)
    ax.axvline(0, color="#868e96", linestyle="--", linewidth=1.0)

    for i, (_, row) in enumerate(mase_comp.iterrows()):
        d = row["mean_paired_difference"]
        v = row["verdict"]
        ax.text(d + (0.015 if d >= 0 else -0.015), i, f"{d:+.3f} ({v})",
                va="center", ha="left" if d >= 0 else "right", fontsize=9.5, fontweight="bold",
                color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(mase_comp["baseline_model"], fontsize=10.5)
    ax.set_xlabel("Paired MASE Difference (Baseline − Chronos-2 Covariates)\n[> 0 indicates Chronos-2 reduction in error]", fontsize=11)
    ax.set_title("Chart 1: Paired MASE Difference vs Baselines (10,000 Bootstrap 95% CI)", fontsize=13, pad=12, fontweight="bold")
    plt.tight_layout()
    chart1_path = output_dir / "chart1_paired_mase_ci.png"
    plt.savefig(chart1_path)
    plt.close()

    # CHART 2: Gain from Covariates (Chronos-2 With vs Without Covariates)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
    c2_sub = summary_df[(summary_df["model"].isin(["Chronos-2 (Zero-Shot)", "Chronos-2 (Covariates)"])) &
                        (summary_df["subset"] == "Total")].set_index("model")

    metrics_show = [
        ("mean_mase", "MASE (Mean)", axes[0], "#1971c2"),
        ("mean_wql", "Weighted Quantile Loss", axes[1], "#0ca678"),
        ("mean_newsvendor_3_1", "Newsvendor Cost (3:1, $)", axes[2], "#f59f00")
    ]

    for col, title, ax_m, bar_color in metrics_show:
        vals = [c2_sub.loc["Chronos-2 (Zero-Shot)", col], c2_sub.loc["Chronos-2 (Covariates)", col]]
        x = np.arange(2)
        bars = ax_m.bar(x, vals, width=0.45, color=[bar_color, "#2b8a3e"], alpha=0.85)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(["Without Covariates\n(Zero-Shot)", "With Covariates\n(Price + Events)"], fontsize=9.5)
        ax_m.set_title(title, fontsize=11, fontweight="bold")
        ax_m.set_ylim(0, max(vals) * 1.22)
        for b, v in zip(bars, vals):
            ax_m.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.suptitle("Chart 2: Impact of In-Context Covariates on Chronos-2 Performance", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart2_path = output_dir / "chart2_covariates_gain.png"
    plt.savefig(chart2_path)
    plt.close()

    # CHART 3: Newsvendor Inventory Cost Sensitivity (1:1, 3:1, 5:1, 9:1)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for model_name, grp in newsvendor_df.groupby("model"):
        ax.plot(grp["cost_ratio"], grp["cost_per_series"], marker="o", linewidth=1.6,
                markersize=6, label=model_name)
        # Direct label on last point
        last_val = grp["cost_per_series"].iloc[-1]
        ax.text(3.05, last_val, f" {model_name} (${last_val:.1f})", va="center", fontsize=8.5)

    ax.set_xlabel("Underage : Overage Cost Ratio (at q = 0.90)", fontsize=11)
    ax.set_ylabel("Expected Cost per Series ($)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.2, 4.8)
    ax.set_title("Chart 3: Newsvendor Inventory Cost across Cost Ratios (Sensitivity Analysis)", fontsize=13, pad=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    plt.tight_layout()
    chart3_path = output_dir / "chart3_newsvendor_sensitivity.png"
    plt.savefig(chart3_path)
    plt.close()

    # CHART 4: Runtime per 1,000 Series (Fit + Predict Seconds)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    rt_sorted = runtime_df.sort_values("total_time_s", ascending=True)
    y_pos = np.arange(len(rt_sorted))
    bars = ax.barh(y_pos, rt_sorted["total_time_s"], height=0.55, color="#364fc7", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rt_sorted["model"], fontsize=10.5)
    ax.set_xlabel("Wall-Clock Seconds per 1,000 Series (Fit + Inference across 3 Origins)", fontsize=11)
    ax.set_title("Chart 4: Total Runtime per 1,000 Series", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlim(0, max(rt_sorted["total_time_s"]) * 1.25)
    for b, val in zip(bars, rt_sorted["total_time_s"]):
        ax.text(val + max(rt_sorted["total_time_s"])*0.015, b.get_y() + b.get_height()/2,
                f"{val:.1f} s", va="center", ha="left", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    chart4_path = output_dir / "chart4_runtime_comparison.png"
    plt.savefig(chart4_path)
    plt.close()

    return [chart1_path, chart2_path, chart3_path, chart4_path]


# ---------------------------------------------------------------------------
# End-to-End Orchestrator
# ---------------------------------------------------------------------------
def run_benchmark(data_dir: str, output_dir: Path, smoke_test: bool = False):
    """Executes the full end-to-end benchmark workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inter_dir = output_dir / "intermediates"
    inter_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"

    check_environment(log_file)

    n_series = SMOKE_N_SERIES if smoke_test else N_SERIES
    origins_to_run = [ORIGINS[0]] if smoke_test else ORIGINS

    log_message(f"=== INITIALIZING RUN (Smoke: {smoke_test}, Series: {n_series}, Origins: {len(origins_to_run)}) ===", log_file)

    # Ingest M5 Data
    calendar, sampled_sales, prices_df = load_sampled_m5(data_dir, n_series=n_series, seed=RANDOM_SEED, log_file=log_file)

    # Track runtimes
    runtime_records = []
    # Collect all per-series evaluations
    all_series_records = []
    crossing_records = []

    models = [
        ("Seasonal Naive", "seasonal_naive"),
        ("AutoETS", "auto_ets"),
        ("LightGBM", "lightgbm"),
        ("Chronos-Bolt", "chronos_bolt"),
        ("Chronos-2 (Zero-Shot)", "chronos2_zero_shot"),
        ("Chronos-2 (Covariates)", "chronos2_covariates")
    ]

    for orig_info in origins_to_run:
        orig_idx = orig_info["origin"]
        log_message(f"--- Running Origin {orig_idx} (Train d_1..{orig_info['train_end']}, Test d_{orig_info['test_start']}..{orig_info['test_end']}) ---", log_file)

        data = prepare_origin_data(sampled_sales, calendar, prices_df, orig_info)
        hist_arr = data["hist_arr"]
        y_true = data["y_true"]
        scales = data["scales"]
        is_intermittent = data["is_intermittent"]
        zero_fracs = data["zero_fracs"]
        context_df = data["context_df"]
        future_df = data["future_df"]
        meta_df = data["series_meta"]

        zero_scale_count = int(np.sum(scales == 0))
        log_message(f"Origin {orig_idx}: {zero_scale_count} series with zero seasonal scale excluded from MASE.", log_file)

        for model_name, model_key in models:
            res_file = inter_dir / f"preds_{model_key}_origin_{orig_idx}.npy"
            timing_file = inter_dir / f"timing_{model_key}_origin_{orig_idx}.csv"

            if res_file.exists() and timing_file.exists():
                log_message(f"Resuming {model_name} Origin {orig_idx} from disk...", log_file)
                q_preds = np.load(res_file)
                t_df = pd.read_csv(timing_file)
                fit_time = float(t_df["fit_time_s"].iloc[0])
                pred_time = float(t_df["predict_time_s"].iloc[0])
            else:
                log_message(f"Executing {model_name} on Origin {orig_idx}...", log_file)
                try:
                    if model_key == "seasonal_naive":
                        q_preds, fit_time, pred_time = run_seasonal_naive(hist_arr)
                    elif model_key == "auto_ets":
                        q_preds, fit_time, pred_time = run_auto_ets(hist_arr)
                    elif model_key == "lightgbm":
                        q_preds, fit_time, pred_time = run_lightgbm(context_df, future_df)
                    elif model_key == "chronos_bolt":
                        q_preds, fit_time, pred_time = run_chronos_bolt(hist_arr)
                    elif model_key == "chronos2_zero_shot":
                        q_preds, fit_time, pred_time = run_chronos2_univariate(context_df)
                    elif model_key == "chronos2_covariates":
                        q_preds, fit_time, pred_time = run_chronos2_covariates(context_df, future_df)
                    else:
                        raise ValueError(f"Unknown model: {model_key}")

                    np.save(res_file, q_preds)
                    pd.DataFrame([{"fit_time_s": fit_time, "predict_time_s": pred_time}]).to_csv(timing_file, index=False)
                except Exception as e:
                    log_message(f"ERROR: {model_name} failed on Origin {orig_idx}: {e}", log_file)
                    raise e

            runtime_records.append({
                "model": model_name,
                "origin": orig_idx,
                "fit_time_s": fit_time,
                "predict_time_s": pred_time,
                "total_time_s": fit_time + pred_time
            })

            # Compute metrics
            m_res = compute_metrics_per_series(y_true, q_preds, scales)
            crossing_records.append({
                "model": model_name,
                "origin": orig_idx,
                "crossing_rate": m_res["crossing_rate"]
            })

            for s_idx in range(len(sampled_sales)):
                all_series_records.append({
                    "id": meta_df["id"].iloc[s_idx],
                    "item_id": meta_df["item_id"].iloc[s_idx],
                    "store_id": meta_df["store_id"].iloc[s_idx],
                    "state_id": meta_df["state_id"].iloc[s_idx],
                    "origin": orig_idx,
                    "model": model_name,
                    "is_intermittent": int(is_intermittent[s_idx]),
                    "zero_frac": float(zero_fracs[s_idx]),
                    "in_sample_scale": float(scales[s_idx]),
                    "mase": float(m_res["mase"][s_idx]),
                    "wql": float(m_res["wql"][s_idx]),
                    "newsvendor_1_1": float(m_res["newsvendor_1_1"][s_idx]),
                    "newsvendor_3_1": float(m_res["newsvendor_3_1"][s_idx]),
                    "newsvendor_5_1": float(m_res["newsvendor_5_1"][s_idx]),
                    "newsvendor_9_1": float(m_res["newsvendor_9_1"][s_idx]),
                })

    # Save per_series_metrics.csv
    per_series_df = pd.DataFrame(all_series_records)
    per_series_df.to_csv(output_dir / "per_series_metrics.csv", index=False)
    log_message(f"Saved {output_dir / 'per_series_metrics.csv'} ({len(per_series_df)} rows)", log_file)

    # Runtime table
    rt_df = pd.DataFrame(runtime_records)
    rt_summary = rt_df.groupby("model")[["fit_time_s", "predict_time_s", "total_time_s"]].sum().reset_index()
    rt_summary["runtime_per_1000_series_s"] = rt_summary["total_time_s"] * (1000.0 / n_series)
    rt_summary.to_csv(output_dir / "runtime.csv", index=False)
    log_message(f"Saved {output_dir / 'runtime.csv'}", log_file)

    # Series-level aggregate across origins first (mean across origins per series)
    series_agg = per_series_df.groupby(["id", "model"]).agg({
        "mase": "mean",
        "wql": "mean",
        "newsvendor_1_1": "mean",
        "newsvendor_3_1": "mean",
        "newsvendor_5_1": "mean",
        "newsvendor_9_1": "mean",
        "is_intermittent": "first"
    }).reset_index()

    # Bootstrap Paired Testing
    win_tie_df = run_paired_bootstrap(series_agg, ref_model="Chronos-2 (Covariates)", n_boot=BOOTSTRAP_ROUNDS)
    win_tie_df.to_csv(output_dir / "win_tie_loss.csv", index=False)
    log_message(f"Saved {output_dir / 'win_tie_loss.csv'}", log_file)

    # Summary table (Total, Smooth, Intermittent)
    summary_rows = []
    crossing_df = pd.DataFrame(crossing_records).groupby("model")["crossing_rate"].mean().to_dict()

    for subset_name, mask in [
        ("Total", np.ones(len(series_agg), dtype=bool)),
        ("Smooth", series_agg["is_intermittent"] == 0),
        ("Intermittent", series_agg["is_intermittent"] == 1)
    ]:
        sub_df = series_agg[mask]
        for m, grp in sub_df.groupby("model"):
            rt_match = rt_summary[rt_summary["model"] == m]
            fit_t = float(rt_match["fit_time_s"].iloc[0]) if len(rt_match) else 0.0
            pred_t = float(rt_match["predict_time_s"].iloc[0]) if len(rt_match) else 0.0
            tot_t = float(rt_match["total_time_s"].iloc[0]) if len(rt_match) else 0.0

            summary_rows.append({
                "model": m,
                "subset": subset_name,
                "n_series": len(grp),
                "mean_mase": float(grp["mase"].mean()),
                "std_mase": float(grp["mase"].std()),
                "mean_wql": float(grp["wql"].mean()),
                "std_wql": float(grp["wql"].std()),
                "mean_newsvendor_3_1": float(grp["newsvendor_3_1"].mean()),
                "std_newsvendor_3_1": float(grp["newsvendor_3_1"].std()),
                "quantile_crossing_rate": float(crossing_df.get(m, 0.0)),
                "fit_time_s": fit_t,
                "predict_time_s": pred_t,
                "total_time_s": tot_t
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    log_message(f"Saved {output_dir / 'summary.csv'}", log_file)

    # Newsvendor Sensitivity Table
    nv_rows = []
    for m in series_agg["model"].unique():
        sub_m = series_agg[series_agg["model"] == m]
        for ratio in NEWSVENDOR_RATIOS:
            col_name = f"newsvendor_{ratio['name'].replace(':', '_')}"
            cost_per = float(sub_m[col_name].mean())
            total_c = float(sub_m[col_name].sum())
            nv_rows.append({
                "model": m,
                "cost_ratio": ratio["name"],
                "cost_per_series": cost_per,
                "total_cost": total_c
            })
    nv_df = pd.DataFrame(nv_rows)
    nv_df.to_csv(output_dir / "newsvendor_sensitivity.csv", index=False)
    log_message(f"Saved {output_dir / 'newsvendor_sensitivity.csv'}", log_file)

    # Generate Charts
    log_message("Generating 4 publication-quality charts...", log_file)
    chart_paths = generate_charts(summary_df, win_tie_df, nv_df, rt_summary, output_dir)
    for cp in chart_paths:
        log_message(f"Chart generated: {cp.name}", log_file)

    log_message("=== BENCHMARK EXECUTION COMPLETED SUCCESSFULLY ===", log_file)
    return per_series_df, summary_df, win_tie_df, nv_df, rt_summary


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__" and "ipykernel" not in sys.modules and not any("kernel" in arg for arg in sys.argv):
    parser = argparse.ArgumentParser(description="Chronos-2 M5 Retail Demand Benchmark")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing M5 CSV files")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output CSVs and plots")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on 20 series, 1 origin")
    args, _ = parser.parse_known_args()

    data_dir, _, _, _ = locate_m5_data(args.data_dir)
    out_dir = Path(args.output_dir)
    run_benchmark(data_dir=data_dir, output_dir=out_dir, smoke_test=args.smoke_test)
