import pandas as pd
from pathlib import Path


def load_results(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Missing results file: {file_path}")
    return pd.read_csv(file_path)


def compare_baselines():
    baseline_path = "results/tables/baseline_metrics.csv"
    static_path = "results/tables/static_fairness_metrics.csv"

    baseline_df = load_results(baseline_path)
    static_df = load_results(static_path)

    combined_df = pd.concat([baseline_df, static_df], ignore_index=True)

    # Optional: order models for readability
    model_order = [
        "LogisticRegression",
        "RandomForest",
        "FairLogisticRegression",
    ]

    combined_df["model"] = pd.Categorical(
        combined_df["model"], categories=model_order, ordered=True
    )
    combined_df = combined_df.sort_values("model")

    print("\nBaseline vs Static Fairness Comparison:\n")
    print(combined_df)

    combined_df.to_csv("results/tables/baseline_vs_static.csv", index=False)


if __name__ == "__main__":
    compare_baselines()