import numpy as np
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# Reweighing weight computation
# --------------------------------------------------
def compute_reweighing_weights(y, s_attr):
    """
    Reweighing formula:
    weight = P(Y) * P(S) / P(Y, S)

    y      : binary target (0/1)
    s_attr : binary sensitive attribute (1=privileged, 0=unprivileged)
    """

    y = np.asarray(y)
    s_attr = np.asarray(s_attr)

    weights = np.ones(len(y), dtype=float)

    unique_y = np.unique(y)
    unique_s = np.unique(s_attr)

    for y_val in unique_y:
        for s_val in unique_s:
            mask = (y == y_val) & (s_attr == s_val)

            p_y = np.mean(y == y_val)
            p_s = np.mean(s_attr == s_val)
            p_ys = np.mean(mask)

            if p_ys > 0:
                weights[mask] = (p_y * p_s) / p_ys

    return weights


# --------------------------------------------------
# Static fairness logistic regression (reweighing)
# --------------------------------------------------
class FairLogisticRegression:
    """
    Logistic Regression with reweighing-based fairness.

    Uses sample weights computed from sensitive attribute
    to reduce demographic disparity during training.
    """

    def __init__(self, max_iter=1000):
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X, y, s_attr):
        """
        X      : features
        y      : target
        s_attr : binary sensitive attribute (1=privileged, 0=unprivileged)
        """

        weights = compute_reweighing_weights(y, s_attr)

        self.model.fit(X, y, sample_weight=weights)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)