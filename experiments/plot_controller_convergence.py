import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "results/tables/controller_training_log.csv"
FIG_DIR = "results/figures"


# --------------------------------------------------
# Safety checks
# --------------------------------------------------
def load_log():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"{LOG_PATH} not found. Run the adaptive controller first."
        )

    df = pd.read_csv(LOG_PATH)

    required_cols = {"epoch", "lambda", "dp", "accuracy", "roc_auc"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in log: {missing}")

    # ensure numeric
    for col in ["epoch", "lambda", "dp", "accuracy", "roc_auc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("epoch").reset_index(drop=True)

    # smooth DP for visualization
    df["dp_smooth"] = df["dp"].rolling(window=2, min_periods=1).mean()

    return df


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------
# Plotting functions
# --------------------------------------------------
def plot_lambda_convergence(df):
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["lambda"], marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Lambda (Fairness Weight)")
    plt.title("Lambda Convergence")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/lambda_convergence.png", dpi=300)
    plt.close()


def plot_fairness_convergence(df):
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["dp_smooth"], marker="o", linewidth=2, label="DP (smoothed)")
    plt.axhline(0, linestyle="--", linewidth=1, label="Fairness target")
    plt.xlabel("Epoch")
    plt.ylabel("Demographic Parity Difference")
    plt.title("Fairness Convergence")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # zoom into DP region for visibility
    dp_max = max(abs(df["dp"].max()), abs(df["dp"].min()))
    plt.ylim(-dp_max * 1.2, dp_max * 1.2)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fairness_convergence.png", dpi=300)
    plt.close()


def plot_accuracy_curve(df):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(df["epoch"], df["accuracy"], marker="o", linewidth=2, label="Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # secondary axis for ROC-AUC
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["roc_auc"], linestyle="--", linewidth=2, label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC")

    # combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title("Accuracy & ROC-AUC Stability During Controller Updates")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/controller_accuracy_curve.png", dpi=300)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    ensure_fig_dir()
    df = load_log()

    plot_lambda_convergence(df)
    plot_fairness_convergence(df)
    plot_accuracy_curve(df)

    print("Controller convergence plots saved to results/figures/.")


if __name__ == "__main__":
    main()