import numpy as np


def _rate_positive(y_pred, s, group):
    mask = s == group
    if mask.sum() == 0:
        return 0.0
    return np.mean(y_pred[mask] == 1)


def demographic_parity_difference(y_pred, s):
    p_unpriv = _rate_positive(y_pred, s, group=0)
    p_priv = _rate_positive(y_pred, s, group=1)
    return p_unpriv - p_priv


def true_positive_rate(y_true, y_pred, s, group):
    mask = (s == group) & (y_true == 1)
    if mask.sum() == 0:
        return 0.0
    return np.mean(y_pred[mask] == 1)


def equal_opportunity_difference(y_true, y_pred, s):
    tpr_unpriv = true_positive_rate(y_true, y_pred, s, group=0)
    tpr_priv = true_positive_rate(y_true, y_pred, s, group=1)
    return tpr_unpriv - tpr_priv


def disparate_impact_ratio(y_pred, s):
    p_unpriv = _rate_positive(y_pred, s, group=0)
    p_priv = _rate_positive(y_pred, s, group=1)
    if p_priv == 0:
        return 0.0
    return p_unpriv / p_priv