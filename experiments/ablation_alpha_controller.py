import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data
from src.evaluation.fairness_evaluation import demographic_parity_difference
from src.evaluation.performance_metrics import compute_performance_metrics


def run_controller_with_alpha(alpha):

    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    s_train_attr = s_train["age_group"].values
    s_val_attr = s_val["age_group"].values

    model = LogisticRegression(max_iter=1000, warm_start=True, solver="liblinear")

    lambda_ = 0.0
    epochs = 15
    base_weights = np.ones(len(y_train))

    for _ in range(epochs):

        weights = base_weights.copy()

        priv_mask = s_train_attr == 1
        unpriv_mask = s_train_attr == 0

        priv_rate = np.mean(y_train[priv_mask])
        unpriv_rate = np.mean(y_train[unpriv_mask])
        gap = priv_rate - unpriv_rate

        if gap > 0:
            weights[(unpriv_mask) & (y_train == 1)] *= (1 + lambda_)
        else:
            weights[(priv_mask) & (y_train == 0)] *= (1 + lambda_)

        model.fit(X_train, y_train, sample_weight=weights)

        val_preds = model.predict(X_val)
        val_dp = demographic_parity_difference(val_preds, s_val_attr)

        lambda_ = np.clip(lambda_ + alpha * val_dp, 0, 2)

    # test evaluation
    test_preds = model.predict(X_test)

    accuracy, roc_auc = compute_performance_metrics(model, X_test, y_test)
    test_dp = demographic_parity_difference(test_preds, s_test["age_group"].values)

    return {
        "alpha": alpha,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "final_lambda": lambda_,
        "dp_age_group": test_dp,
    }


def main():

    alphas = [0.1, 0.3, 0.5, 0.7]

    results = []

    for alpha in alphas:
        results.append(run_controller_with_alpha(alpha))

    df = pd.DataFrame(results)

    print("\nAblation Study — Alpha (Controller Strength):\n", df)

    df.to_csv("results/tables/ablation_alpha_controller.csv", index=False)


if __name__ == "__main__":
    main()