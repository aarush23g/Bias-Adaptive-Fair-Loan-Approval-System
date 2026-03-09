import pandas as pd

def main():

    baseline = pd.read_csv("results/tables/baseline_vs_static.csv")
    adaptive = pd.read_csv("results/tables/adaptive_controller_metrics.csv")

    baseline_lr = baseline[baseline["model"] == "LogisticRegression"]
    static_lr = baseline[baseline["model"] == "FairLogisticRegression"]

    adaptive_lr = adaptive.copy()

    combined = pd.concat(
        [baseline_lr, static_lr, adaptive_lr],
        ignore_index=True,
        sort=False
    )

    print("\nMethod Comparison:\n", combined)

    combined.to_csv("results/tables/ablation_method_comparison.csv", index=False)


if __name__ == "__main__":
    main()