import os
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data

from src.fairness.fairness_metrics import (
    demographic_parity,
    equal_opportunity,
    equalized_odds
)

from src.fairness.adaptive_controller import (
    AdaptiveFairnessController,
    apply_controller
)


def compute_violation(metric, preds, y_true, sensitive):

    if metric == "dp":
        return demographic_parity(preds, sensitive)

    elif metric == "eop":
        return equal_opportunity(preds, y_true, sensitive)

    elif metric == "eod":
        return equalized_odds(preds, y_true, sensitive)

    else:
        raise ValueError(f"Unknown fairness metric: {metric}")


def run_experiment(metric_name):

    df = load_data()

    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    sensitive = s_test.values.flatten()
    y_test_np = y_test.values

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    controller = AdaptiveFairnessController(
        alpha=0.02,
        target=0.02
    )

    lambda_values = []
    thresholds = []
    violations = []

    # Controller iterations
    for step in range(30):

        preds, threshold = apply_controller(probs, controller.lambda_t)

        violation = compute_violation(
            metric_name,
            preds,
            y_test_np,
            sensitive
        )

        controller.update(violation)

        lambda_values.append(controller.lambda_t)
        thresholds.append(threshold)
        violations.append(violation)

    final_preds, final_threshold = apply_controller(
        probs,
        controller.lambda_t
    )

    acc = accuracy_score(y_test, final_preds)
    roc = roc_auc_score(y_test, probs)

    final_fairness = compute_violation(
        metric_name,
        final_preds,
        y_test_np,
        sensitive
    )

    # Save controller dynamics for R5
    dynamics = pd.DataFrame({
        "step": np.arange(len(lambda_values)),
        "lambda": lambda_values,
        "threshold": thresholds,
        "fairness_violation": violations,
        "metric": metric_name
    })

    os.makedirs("results/analysis", exist_ok=True)

    dynamics.to_csv(
        f"results/analysis/controller_dynamics_{metric_name}.csv",
        index=False
    )

    return {
        "metric": metric_name,
        "accuracy": acc,
        "roc_auc": roc,
        "fairness_violation": final_fairness,
        "final_lambda": controller.lambda_t,
        "final_threshold": final_threshold
    }


def run_all():

    os.makedirs("results/tables", exist_ok=True)

    metrics = ["dp", "eop", "eod"]

    results = []

    for m in metrics:

        print(f"\nRunning Adaptive Controller for metric: {m}")

        res = run_experiment(m)

        results.append(res)

    df = pd.DataFrame(results)

    df.to_csv(
        "results/tables/adaptive_multi_metric_results.csv",
        index=False
    )

    print("\nAdaptive Multi-Metric Results:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_all()