import os
import pandas as pd


def safe_read(path, dataset):
    """
    Safely read CSV and attach dataset name.
    """
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["dataset"] = dataset

    # Normalize fairness column
    df = normalize_dp(df)

    return df


def normalize_dp(df):
    """
    Ensure dataframe always contains a 'dp' column.
    Handles dp, dp_gender, dp_race, etc.
    """

    if "dp" in df.columns:
        return df

    fairness_cols = [c for c in df.columns if c.startswith("dp_")]

    if fairness_cols:
        df["dp"] = df[fairness_cols[0]]
    else:
        df["dp"] = pd.NA

    return df


def run_summary():

    tables = []

    # --------------------------------------------------
    # German Dataset
    # --------------------------------------------------
    tables.append(safe_read(
        "results/tables/german_baseline_metrics.csv", "German"))

    tables.append(safe_read(
        "results/tables/german_static_metrics.csv", "German"))

    tables.append(safe_read(
        "results/tables/german_fairlearn_metrics.csv", "German"))

    tables.append(safe_read(
        "results/tables/german_adaptive_metrics.csv", "German"))

    tables.append(safe_read(
        "results/tables/german_lightgbm_threshold_final.csv", "German"))

    # --------------------------------------------------
    # LendingClub Dataset
    # --------------------------------------------------
    tables.append(safe_read(
        "results/tables/lending_club_baseline_metrics.csv", "LendingClub"))

    tables.append(safe_read(
        "results/tables/lending_club_static_metrics.csv", "LendingClub"))

    tables.append(safe_read(
        "results/tables/lending_club_adaptive_metrics.csv", "LendingClub"))

    tables.append(safe_read(
        "results/tables/lending_club_fairlearn_metrics.csv", "LendingClub"))

    tables.append(safe_read(
        "results/tables/lending_club_lightgbm_threshold_final.csv", "LendingClub"))

    # --------------------------------------------------
    # Adult Dataset
    # --------------------------------------------------
    tables.append(safe_read(
        "results/tables/adult_baseline_metrics.csv", "Adult"))

    tables.append(safe_read(
        "results/tables/adult_static_metrics.csv", "Adult"))

    tables.append(safe_read(
        "results/tables/adult_adaptive_metrics.csv", "Adult"))

    tables.append(safe_read(
        "results/tables/adult_fairlearn_metrics.csv", "Adult"))

    tables.append(safe_read(
        "results/tables/adult_lightgbm_threshold_final.csv", "Adult"))

    # --------------------------------------------------
    # Combine all tables
    # --------------------------------------------------
    final_df = pd.concat(tables, ignore_index=True)

    final_df = final_df[
        ["dataset", "model", "accuracy", "roc_auc", "dp"]
    ]

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    os.makedirs("results/tables", exist_ok=True)

    final_df.to_csv(
        "results/tables/final_method_comparison.csv",
        index=False
    )

    print("\nClean Final Comparison Table:\n")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    run_summary()