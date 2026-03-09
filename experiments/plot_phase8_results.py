import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_PATH = "results/tables"
FIG_PATH = "results/figures"

os.makedirs(FIG_PATH, exist_ok=True)


def load_dataset_results(dataset_key, fairness_col):

    baseline = pd.read_csv(
        f"{BASE_PATH}/{dataset_key}_baseline_metrics.csv"
    )
    static = pd.read_csv(
        f"{BASE_PATH}/{dataset_key}_static_metrics.csv"
    )
    adaptive = pd.read_csv(
        f"{BASE_PATH}/{dataset_key}_adaptive_metrics.csv"
    )

    df = pd.concat([baseline, static, adaptive], ignore_index=True)

    return df[["model", "accuracy", "roc_auc", fairness_col]].rename(
        columns={fairness_col: "dp"}
    )


# --------------------------------------------------
# 1️⃣ Fairness Bar Plot
# --------------------------------------------------
def plot_fairness_bar(dataset_key, dataset_label, fairness_col):

    df = load_dataset_results(dataset_key, fairness_col)

    color_map = {
        "LogisticRegression": "#1f77b4",
        "RandomForest": "#ff7f0e",
        "FairLogisticRegression": "#2ca02c",
        "AdaptiveControllerLR": "#d62728",
    }

    colors = [color_map.get(m, "gray") for m in df["model"]]

    plt.figure(figsize=(7, 4))
    plt.bar(df["model"], df["dp"], color=colors)

    plt.xticks(rotation=25)
    plt.ylabel("Demographic Parity Difference")
    plt.title(f"{dataset_label}: Fairness Comparison")
    plt.axhline(0, linestyle="--", color="black", alpha=0.7)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        f"{FIG_PATH}/{dataset_key}_fairness_bar.png",
        dpi=300
    )
    plt.close()


# --------------------------------------------------
# 2️⃣ Accuracy vs Fairness Tradeoff
# --------------------------------------------------
def plot_tradeoff(dataset_key, dataset_label, fairness_col):

    df = load_dataset_results(dataset_key, fairness_col)

    # Consistent color mapping
    color_map = {
        "LogisticRegression": "#1f77b4",
        "RandomForest": "#ff7f0e",
        "FairLogisticRegression": "#2ca02c",
        "AdaptiveControllerLR": "#d62728",
    }

    plt.figure(figsize=(6, 5))

    for _, row in df.iterrows():
        plt.scatter(
            abs(row["dp"]),
            row["accuracy"],
            s=120,
            color=color_map.get(row["model"], "gray"),
            label=row["model"]
        )

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8)

    plt.xlabel("|Demographic Parity Difference|")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_label}: Fairness–Accuracy Tradeoff")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        f"{FIG_PATH}/{dataset_key}_tradeoff.png",
        dpi=300
    )
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # German (age fairness)
    plot_fairness_bar(
        dataset_key="german",
        dataset_label="German",
        fairness_col="dp_age_group"
    )

    plot_tradeoff(
        dataset_key="german",
        dataset_label="German",
        fairness_col="dp_age_group"
    )

    # LendingClub (income fairness)
    plot_fairness_bar(
        dataset_key="lending_club",
        dataset_label="LendingClub",
        fairness_col="dp_income_group"
    )

    plot_tradeoff(
        dataset_key="lending_club",
        dataset_label="LendingClub",
        fairness_col="dp_income_group"
    )

    print("Phase 8 plots generated successfully.")


if __name__ == "__main__":
    main()