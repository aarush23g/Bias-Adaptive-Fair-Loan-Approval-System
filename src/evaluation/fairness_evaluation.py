import numpy as np


def demographic_parity_difference(y_pred, s_attr):
    privileged_mask = s_attr == 1
    unprivileged_mask = s_attr == 0

    if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
        return 0.0

    p_priv = y_pred[privileged_mask].mean()
    p_unpriv = y_pred[unprivileged_mask].mean()

    return float(p_priv - p_unpriv)


def compute_fairness_metrics(model, X_test, s_test):
    y_pred = model.predict(X_test)

    fairness_results = {}

    for col in s_test.columns:
        dp = demographic_parity_difference(y_pred, s_test[col].values)
        fairness_results[f"dp_{col}"] = dp

    return fairness_results