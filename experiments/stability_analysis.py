import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data

from src.evaluation.performance_metrics import compute_performance_metrics
from src.evaluation.fairness_evaluation import demographic_parity_difference

from src.fairness.static_fairness import FairLogisticRegression


SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------
# Adaptive symmetric fairness (same as your final)
# --------------------------------------------------
def symmetric_dp_correction(test_probs, s_test_attr):
    preds = (test_probs >= 0.5).astype(int)

    priv_mask = s_test_attr == 1
    unpriv_mask = s_test_attr == 0

    priv_rate = np.mean(preds[priv_mask])
    unpriv_rate = np.mean(preds[unpriv_mask])

    gap = priv_rate - unpriv_rate
    target_gap = gap * 0.5

    flips_applied = 0

    if gap > 0:
        candidates = np.where((unpriv_mask) & (preds == 0))[0]
        sorted_candidates = candidates[np.argsort(-test_probs[candidates])]

        needed_rate_increase = gap - target_gap
        needed_flips = int(needed_rate_increase * np.sum(unpriv_mask))

        flip_indices = sorted_candidates[:needed_flips]
        preds[flip_indices] = 1
        flips_applied = len(flip_indices)

    elif gap < 0:
        candidates = np.where((priv_mask) & (preds == 1))[0]
        sorted_candidates = candidates[np.argsort(test_probs[candidates])]

        needed_rate_decrease = (-gap) - (-target_gap)
        needed_flips = int(needed_rate_decrease * np.sum(priv_mask))

        flip_indices = sorted_candidates[:needed_flips]
        preds[flip_indices] = 0
        flips_applied = len(flip_indices)

    return preds, flips_applied


# --------------------------------------------------
# Run one seed
# --------------------------------------------------
def run_single_seed(seed):
    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s, random_state=seed)

    results = []

    # ---------------- Baseline ----------------
    base_model = LogisticRegression(max_iter=1000)
    base_model.fit(X_train, y_train)

    base_preds = base_model.predict(X_test)
    acc, roc_auc = compute_performance_metrics(base_model, X_test, y_test)
    dp = demographic_parity_difference(base_preds, s_test["age_group"].values)

    results.append({
        "model": "BaselineLR",
        "accuracy": acc,
        "roc_auc": roc_auc,
        "dp_age_group": dp
    })

    # ---------------- Static Fairness ----------------
    fair_model = FairLogisticRegression()
    fair_model.fit(X_train, y_train, s_train["age_group"].values)

    fair_preds = fair_model.predict(X_test)
    acc, roc_auc = compute_performance_metrics(fair_model, X_test, y_test)
    dp = demographic_parity_difference(fair_preds, s_test["age_group"].values)

    results.append({
        "model": "StaticFairLR",
        "accuracy": acc,
        "roc_auc": roc_auc,
        "dp_age_group": dp
    })

    # ---------------- Adaptive Fairness ----------------
    test_probs = base_model.predict_proba(X_test)[:, 1]
    adaptive_preds, _ = symmetric_dp_correction(
        test_probs,
        s_test["age_group"].values
    )

    acc = np.mean(adaptive_preds == y_test)
    roc_auc = compute_performance_metrics(base_model, X_test, y_test)[1]
    dp = demographic_parity_difference(adaptive_preds, s_test["age_group"].values)

    results.append({
        "model": "AdaptiveFairLR",
        "accuracy": acc,
        "roc_auc": roc_auc,
        "dp_age_group": dp
    })

    return results


# --------------------------------------------------
# Main stability experiment
# --------------------------------------------------
def run_stability_analysis():
    all_results = []

    for seed in SEEDS:
        seed_results = run_single_seed(seed)
        for r in seed_results:
            r["seed"] = seed
        all_results.extend(seed_results)

    df = pd.DataFrame(all_results)

    summary = df.groupby("model").agg({
        "accuracy": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "dp_age_group": ["mean", "std"]
    }).reset_index()

    print("\nStability Analysis Results:\n")
    print(summary)

    summary.to_csv("results/tables/stability_analysis.csv", index=False)
    df.to_csv("results/tables/stability_raw_runs.csv", index=False)


if __name__ == "__main__":
    run_stability_analysis()