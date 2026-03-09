import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_TABLE = "results/tables/final_method_comparison.csv"
FIG_DIR = "results/figures"


COLOR_MAP = {
    "LogisticRegression": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "FairLogisticRegression": "#2ca02c",
    "AdaptiveController_LightGBM": "#d62728",
    "Fairlearn_ExponentiatedGradient": "#9467bd",
    "Fairlearn_GridSearch": "#8c564b",
    "LightGBM_ThresholdOptimized": "#e377c2",
}


def compute_pareto(df):
    """Return Pareto-efficient points"""
    pareto = []

    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (
                other["accuracy"] >= row["accuracy"]
                and abs(other["dp"]) <= abs(row["dp"])
                and (
                    other["accuracy"] > row["accuracy"]
                    or abs(other["dp"]) < abs(row["dp"])
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(row)

    return pd.DataFrame(pareto)


def plot_dataset(df, dataset):

    subset = df[df["dataset"] == dataset].copy()

    if subset.empty:
        return

    subset["abs_dp"] = subset["dp"].abs()

    plt.figure(figsize=(6,4))

    # Plot methods
    for _, row in subset.iterrows():

        color = COLOR_MAP.get(row["model"], "black")

        plt.scatter(
            row["abs_dp"],
            row["accuracy"],
            color=color,
            s=80,
            edgecolor="black"
        )

    # Plot Pareto frontier
    pareto = compute_pareto(subset)

    pareto = pareto.sort_values("abs_dp")

    plt.plot(
        pareto["abs_dp"],
        pareto["accuracy"],
        linestyle="--",
        color="black",
        label="Pareto Frontier"
    )

    plt.xlabel("|Demographic Parity Difference|")
    plt.ylabel("Accuracy")
    plt.title(f"Fairness–Accuracy Tradeoff ({dataset})")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend()

    os.makedirs(FIG_DIR, exist_ok=True)

    save_path = f"{FIG_DIR}/fairness_accuracy_tradeoff_{dataset.lower()}.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {save_path}")


def main():

    df = pd.read_csv(RESULT_TABLE)

    for dataset in df["dataset"].unique():

        print(f"Generating plot for {dataset}")

        plot_dataset(df, dataset)


if __name__ == "__main__":
    main()