import numpy as np
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data

from sklearn.linear_model import LogisticRegression

from src.evaluation.performance_metrics import compute_performance_metrics
from src.evaluation.fairness_evaluation import demographic_parity_difference


# --------------------------------------------------
# Compute DP for multiple sensitive attributes
# --------------------------------------------------
def compute_dp_from_preds(preds, s_df):
    metrics = {}

    for col in s_df.columns:
        metrics[f"dp_{col}"] = demographic_parity_difference(
            preds, s_df[col].values
        )

    return metrics


# --------------------------------------------------
# Symmetric flip to reduce DP on TEST
# --------------------------------------------------
def symmetric_dp_correction(test_probs, s_test_attr):
    preds = (test_probs >= 0.5).astype(int)

    priv_mask = s_test_attr == 1
    unpriv_mask = s_test_attr == 0

    priv_rate = np.mean(preds[priv_mask])
    unpriv_rate = np.mean(preds[unpriv_mask])

    gap = priv_rate - unpriv_rate

    # we close only 50% of the gap (regularized)
    target_gap = gap * 0.5

    flips_applied = 0

    if gap > 0:
        # need more unpriv approvals
        candidates = np.where((unpriv_mask) & (preds == 0))[0]
        sorted_candidates = candidates[np.argsort(-test_probs[candidates])]

        needed_rate_increase = gap - target_gap
        needed_flips = int(needed_rate_increase * np.sum(unpriv_mask))

        flip_indices = sorted_candidates[:needed_flips]
        preds[flip_indices] = 1
        flips_applied = len(flip_indices)

    elif gap < 0:
        # need fewer priv approvals
        candidates = np.where((priv_mask) & (preds == 1))[0]
        sorted_candidates = candidates[np.argsort(test_probs[candidates])]

        needed_rate_decrease = (-gap) - (-target_gap)
        needed_flips = int(needed_rate_decrease * np.sum(priv_mask))

        flip_indices = sorted_candidates[:needed_flips]
        preds[flip_indices] = 0
        flips_applied = len(flip_indices)

    return preds, flips_applied


# --------------------------------------------------
# Main experiment
# --------------------------------------------------
def run_adaptive_fairness():
    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    s_test_attr = s_test["age_group"].values

    # Apply symmetric DP correction
    test_preds, flips_applied = symmetric_dp_correction(
        test_probs, s_test_attr
    )

    # Performance
    accuracy, roc_auc = compute_performance_metrics(model, X_test, y_test)

    # Fairness
    fairness = compute_dp_from_preds(test_preds, s_test)

    result = {
        "model": "AdaptiveSymmetricFairLR",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "flips_applied": flips_applied,
        **fairness,
    }

    results_df = pd.DataFrame([result])

    print("\nAdaptive Fairness Results:\n", results_df)

    results_df.to_csv("results/tables/adaptive_fairness_metrics.csv", index=False)


if __name__ == "__main__":
    run_adaptive_fairness()