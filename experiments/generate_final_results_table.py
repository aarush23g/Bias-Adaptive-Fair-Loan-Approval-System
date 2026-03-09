import os
import pandas as pd


# --------------------------------------------------
# Load dataset-specific results
# --------------------------------------------------
def load_dataset_results(dataset_key, dataset_label):

    base_path = "results/tables"

    baseline_path = os.path.join(
        base_path, f"{dataset_key}_baseline_metrics.csv"
    )
    static_path = os.path.join(
        base_path, f"{dataset_key}_static_metrics.csv"
    )
    adaptive_path = os.path.join(
        base_path, f"{dataset_key}_adaptive_metrics.csv"
    )

    baseline = pd.read_csv(baseline_path)
    static = pd.read_csv(static_path)
    adaptive = pd.read_csv(adaptive_path)

    combined = pd.concat([baseline, static, adaptive], ignore_index=True)

    # --------------------------------------------------
    # Fairness column selection (dataset-aware)
    # --------------------------------------------------
    fairness_cols = [col for col in combined.columns if col.startswith("dp_")]

    if len(fairness_cols) == 0:
        raise ValueError(f"No fairness column found for {dataset_label}")

    # Dataset-specific priority
    if dataset_key == "german" and "dp_age_group" in fairness_cols:
        fairness_col = "dp_age_group"

    elif dataset_key == "lending_club" and "dp_income_group" in fairness_cols:
        fairness_col = "dp_income_group"

    else:
        # fallback (first dp column)
        fairness_col = fairness_cols[0]

    combined = combined[["model", "accuracy", "roc_auc", fairness_col]]
    combined["dataset"] = dataset_label

    combined = combined.rename(columns={fairness_col: "dp"})

    return combined


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    final_results = []

    print("Loading German results...")
    german_results = load_dataset_results(
        dataset_key="german",
        dataset_label="German"
    )
    final_results.append(german_results)

    print("Loading LendingClub results...")
    lending_results = load_dataset_results(
        dataset_key="lending_club",
        dataset_label="LendingClub"
    )
    final_results.append(lending_results)

    final_df = pd.concat(final_results, ignore_index=True)

    final_df = final_df[
        ["dataset", "model", "accuracy", "roc_auc", "dp"]
    ]

    print("\nFinal Combined Results:\n")
    print(final_df)

    final_df.to_csv(
        "results/tables/final_combined_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()