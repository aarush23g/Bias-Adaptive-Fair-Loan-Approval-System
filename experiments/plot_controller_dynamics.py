import os
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["dp", "eop", "eod"]


def plot_metric(metric):

    path = f"results/analysis/controller_dynamics_{metric}.csv"

    if not os.path.exists(path):
        print(f"Skipping {metric}: file not found")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(3, 1, figsize=(8,10))

    # Lambda convergence
    ax[0].plot(df["step"], df["lambda"])
    ax[0].set_title(f"{metric.upper()} - Lambda Convergence")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Lambda")

    # Threshold evolution
    ax[1].plot(df["step"], df["threshold"])
    ax[1].set_title(f"{metric.upper()} - Threshold Evolution")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Threshold")

    # Fairness violation
    ax[2].plot(df["step"], df["fairness_violation"])
    ax[2].set_title(f"{metric.upper()} - Fairness Violation")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Violation")

    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)

    save_path = f"results/figures/controller_dynamics_{metric}.png"

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved {save_path}")


def run_all():

    for m in METRICS:
        print(f"Generating plot for {m}")
        plot_metric(m)


if __name__ == "__main__":
    run_all()