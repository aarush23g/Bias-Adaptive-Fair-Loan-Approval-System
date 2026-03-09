import numpy as np
from src.fairness.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
)


def test_demographic_parity():
    y_pred = np.array([1, 0, 1, 0])
    s = np.array([0, 0, 1, 1])

    dp = demographic_parity_difference(y_pred, s)
    assert isinstance(dp, float)


def test_equal_opportunity():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    s = np.array([0, 0, 1, 1])

    eo = equal_opportunity_difference(y_true, y_pred, s)
    assert isinstance(eo, float)


def test_disparate_impact():
    y_pred = np.array([1, 0, 1, 0])
    s = np.array([0, 0, 1, 1])

    di = disparate_impact_ratio(y_pred, s)
    assert isinstance(di, float)