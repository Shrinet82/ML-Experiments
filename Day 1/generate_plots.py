"""
Generate publication-quality charts from the completed benchmark results.
Reads:
- results/per_fold_metrics.csv
- results/dataset_summary_metrics.csv
- results/business_financial_summary.csv
- results/win_tie_loss_table.csv
Outputs:
- results/auc_boxplots_by_dataset.png
- results/business_financial_impact.png
- results/latency_pareto_analysis.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

df_folds = pd.read_csv("results/per_fold_metrics.csv")
df_summary = pd.read_csv("results/dataset_summary_metrics.csv")
df_bus = pd.read_csv("results/business_financial_summary.csv")
df_wtl = pd.read_csv("results/win_tie_loss_table.csv")

# 1. Boxplots of AUC across datasets (5 folds x 5 repeats = 25 evaluations)
plt.figure(figsize=(15, 8))
palette = sns.color_palette("muted", n_colors=df_folds["Model"].nunique())
order_datasets = ["credit-g", "telco-churn", "bank-marketing", "adult"]
g = sns.boxplot(
    data=df_folds,
    x="Dataset",
    y="AUC",
    hue="Model",
    order=order_datasets,
    palette=palette,
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}
)
plt.title("Repeated Stratified CV Performance (5 Folds x 5 Repeats = 25 Runs per Model)", fontsize=14, weight="bold")
plt.xlabel("Benchmark Dataset", fontsize=12)
plt.ylabel("ROC-AUC", fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=10)
plt.tight_layout()
plt.savefig("results/auc_boxplots_by_dataset.png", dpi=300)
plt.close()
print("Saved results/auc_boxplots_by_dataset.png")

# 2. Business Financial Impact (German Credit Cost & Telco Churn Profit)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Credit-G Cost per Applicant
credit_df = df_folds[df_folds["Dataset"] == "credit-g"].dropna(subset=["Credit_Cost_Per_Applicant"])
order_cost = credit_df.groupby("Model")["Credit_Cost_Per_Applicant"].mean().sort_values().index
sns.barplot(
    data=credit_df,
    x="Model",
    y="Credit_Cost_Per_Applicant",
    order=order_cost,
    ax=axes[0],
    ci=95,
    palette="crest",
    capsize=0.1
)
axes[0].set_title("German Credit: Expected Cost per Applicant\n(Cost: 5 for approving bad debt, 1 for rejecting good)", fontsize=12, weight="bold")
axes[0].set_ylabel("Expected Cost per Applicant ($) — Lower is Better", fontsize=11)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right", fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.5)

# Telco Churn Profit per Customer
churn_df = df_folds[df_folds["Dataset"] == "telco-churn"].dropna(subset=["Churn_Profit_Per_Customer"])
order_profit = churn_df.groupby("Model")["Churn_Profit_Per_Customer"].mean().sort_values(ascending=False).index
sns.barplot(
    data=churn_df,
    x="Model",
    y="Churn_Profit_Per_Customer",
    order=order_profit,
    ax=axes[1],
    ci=95,
    palette="viridis",
    capsize=0.1
)
axes[1].set_title("Telco Churn: Expected Profit per Customer\n($100 preserved customer value vs. $20 incentive cost)", fontsize=12, weight="bold")
axes[1].set_ylabel("Expected Profit per Customer ($) — Higher is Better", fontsize=11)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right", fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/business_financial_impact.png", dpi=300)
plt.close()
print("Saved results/business_financial_impact.png")

# 3. Fit Time vs. Per-Row Inference Latency (Pareto Frontier)
plt.figure(figsize=(11, 6.5))
time_summary = df_folds.groupby("Model")[["Fit_Time_s", "Per_Row_Predict_ms"]].mean().reset_index()

sns.scatterplot(
    data=time_summary,
    x="Fit_Time_s",
    y="Per_Row_Predict_ms",
    hue="Model",
    s=280,
    palette="tab10"
)
for _, row in time_summary.iterrows():
    offset_x = 1.08
    offset_y = 1.05
    plt.text(row["Fit_Time_s"] * offset_x, row["Per_Row_Predict_ms"] * offset_y, row["Model"], fontsize=9, weight="semibold")

plt.xscale("log")
plt.yscale("log")
plt.title("Speed & Latency Profile: Fit Time vs. Per-Row Inference Latency (Log-Log Scale)", fontsize=13, weight="bold")
plt.xlabel("Mean Fit Time (seconds, Log Scale)", fontsize=11)
plt.ylabel("Inference Latency per Row (milliseconds, Log Scale)", fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/latency_pareto_analysis.png", dpi=300)
plt.close()
print("Saved results/latency_pareto_analysis.png")
