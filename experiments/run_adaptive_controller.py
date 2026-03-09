import os
import yaml
import numpy as np
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data
from src.evaluation.fairness_evaluation import demographic_parity_difference
from src.evaluation.performance_metrics import compute_performance_metrics
from src.models.model_factory import get_lightgbm_model


def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


def compute_fairness_weights(y, s_attr, lambda_):

    weights = np.ones(len(y))

    priv_mask = s_attr == 1
    unpriv_mask = s_attr == 0

    if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
        return weights

    priv_rate = np.mean(y[priv_mask])
    unpriv_rate = np.mean(y[unpriv_mask])

    gap = priv_rate - unpriv_rate

    if gap > 0:
        weights[unpriv_mask] *= (1 + lambda_)
    else:
        weights[priv_mask] *= (1 + lambda_)

    return weights


def clean_feature_names(df):
    """
    LightGBM does not allow special JSON characters in feature names.
    This function sanitizes column names.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
    )

    return df


def run_adaptive_controller():

    os.makedirs("results/tables", exist_ok=True)

    cfg = load_config()
    dataset_name = cfg["dataset"]

    df = load_data()
    X, y, s = preprocess(df)

    # --------------------------------------------------
    # FIX LIGHTGBM FEATURE NAME ISSUE
    # --------------------------------------------------
    X = clean_feature_names(X)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    sensitive_col = s.columns[0]

    s_train_attr = s_train[sensitive_col].values
    s_val_attr = s_val[sensitive_col].values
    s_test_attr = s_test[sensitive_col].values

    model = get_lightgbm_model()

    lambda_ = 0.0
    alpha = 0.3
    beta = 0.1
    epochs = 5
    lambda_max = 3.0

    for epoch in range(epochs):

        weights = compute_fairness_weights(
            y_train.values,
            s_train_attr,
            lambda_
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            eval_metric="auc"
        )

        val_preds = model.predict(X_val)

        val_dp = demographic_parity_difference(
            val_preds,
            s_val_attr
        )

        lambda_ = (1 - beta) * lambda_ + alpha * val_dp
        lambda_ = float(np.clip(lambda_, 0.0, lambda_max))

    test_preds = model.predict(X_test)

    accuracy, roc_auc = compute_performance_metrics(
        model,
        X_test,
        y_test
    )

    test_dp = demographic_parity_difference(
        test_preds,
        s_test_attr
    )

    result = {
        "model": "AdaptiveController_LightGBM",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "dp": test_dp
    }

    result_df = pd.DataFrame([result])

    print(f"\nAdaptive Controller Results ({dataset_name}):\n")
    print(result_df)

    result_df.to_csv(
        f"results/tables/{dataset_name}_adaptive_metrics.csv",
        index=False
    )


if __name__ == "__main__":
    run_adaptive_controller()