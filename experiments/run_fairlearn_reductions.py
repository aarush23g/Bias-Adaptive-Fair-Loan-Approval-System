import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import (
    ExponentiatedGradient,
    GridSearch,
    DemographicParity
)

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data
from src.evaluation.performance_metrics import compute_performance_metrics
from src.evaluation.fairness_evaluation import demographic_parity_difference


def run_fairlearn_experiments():

    os.makedirs("results/tables", exist_ok=True)

    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    # Select fairness attribute consistently
    if "age_group" in s.columns:
        sensitive_col = "age_group"
    elif "income_group" in s.columns:
        sensitive_col = "income_group"
    else:
        sensitive_col = s.columns[0]

    sensitive_train = s_train[sensitive_col].values
    sensitive_test = s_test[sensitive_col].values

    results = []

    base_estimator = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )

    constraint = DemographicParity()

    # -----------------------------------------
    # 1️⃣ Exponentiated Gradient
    # -----------------------------------------
    constraint_exp = DemographicParity()

    exp_grad = ExponentiatedGradient(
        base_estimator,
        constraints=constraint_exp
    )

    exp_grad.fit(X_train, y_train, sensitive_features=sensitive_train)

    exp_preds = exp_grad.predict(X_test)

    acc, roc_auc = compute_performance_metrics(exp_grad, X_test, y_test)
    dp = demographic_parity_difference(exp_preds, sensitive_test)

    results.append({
    "model": "Fairlearn_ExponentiatedGradient",
    "accuracy": acc,
    "roc_auc": roc_auc,
    "dp": dp
    })

    # -----------------------------------------
    # 2️⃣ Grid Search Reduction
    # -----------------------------------------
    unique_groups = np.unique(sensitive_train)

    if len(unique_groups) > 1:

        constraint_grid = DemographicParity()

        grid = GridSearch(
            base_estimator,
            constraints=constraint_grid,
            grid_size=10
        )

        grid.fit(X_train, y_train, sensitive_features=sensitive_train)

        grid_preds = grid.predict(X_test)

        acc, roc_auc = compute_performance_metrics(grid, X_test, y_test)
        dp = demographic_parity_difference(grid_preds, sensitive_test)

        results.append({
            "model": "Fairlearn_GridSearch",
            "accuracy": acc,
            "roc_auc": roc_auc,
            "dp": dp
        })

    else:

        print("Skipping GridSearch: only one sensitive group in training set.")

    results_df = pd.DataFrame(results)

    dataset_key = (
        "german"
        if "age_group" in s.columns
        else "lending_club"
        if "income_group" in s.columns
        else "adult"
    )

    results_df.to_csv(
        f"results/tables/{dataset_key}_fairlearn_metrics.csv",
        index=False
    )

    print("\nFairlearn Reduction Results:\n")
    print(results_df)


if __name__ == "__main__":
    run_fairlearn_experiments()