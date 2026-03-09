import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data
from src.evaluation.fairness_evaluation import demographic_parity_difference
from src.evaluation.performance_metrics import compute_performance_metrics


def compute_fairness_weights(y, s_attr, base_weights, lambda_):
    weights = base_weights.copy()

    priv_mask = s_attr == 1
    unpriv_mask = s_attr == 0

    priv_rate = np.mean(y[priv_mask])
    unpriv_rate = np.mean(y[unpriv_mask])

    gap = priv_rate - unpriv_rate

    if gap > 0:
        boost_mask = (unpriv_mask) & (y == 1)
        weights[boost_mask] *= (1 + lambda_)
    else:
        boost_mask = (priv_mask) & (y == 0)
        weights[boost_mask] *= (1 + lambda_)

    return weights


def run_controller(alpha):

    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    s_train_attr = s_train["age_group"].values

    model = LogisticRegression(max_iter=1000, warm_start=True)

    lambda_ = 0.0
    epochs = 15
    base_weights = np.ones(len(y_train))

    for _ in range(epochs):
        weights = compute_fairness_weights(
            y_train.values, s_train_attr, base_weights, lambda_
        )

        model.fit(X_train, y_train, sample_weight=weights)

        train_preds = model.predict(X_train)
        train_dp = demographic_parity_difference(train_preds, s_train_attr)

        lambda_ = max(0, lambda_ + alpha * train_dp)

    test_preds = model.predict(X_test)

    accuracy, roc_auc = compute_performance_metrics(model, X_test, y_test)

    test_dp = demographic_parity_difference(
        test_preds, s_test["age_group"].values
    )

    return accuracy, roc_auc, lambda_, test_dp


def run_ablation():

    alphas = [0.1, 0.3, 0.5]

    results = []

    for alpha in alphas:
        acc, auc, final_lambda, dp = run_controller(alpha)

        results.append({
            "alpha": alpha,
            "accuracy": acc,
            "roc_auc": auc,
            "final_lambda": final_lambda,
            "dp_age_group": dp,
        })

    df = pd.DataFrame(results)

    print("\nAblation Study (Alpha):\n", df)

    df.to_csv("results/tables/ablation_alpha.csv", index=False)


if __name__ == "__main__":
    run_ablation()