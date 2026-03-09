import numpy as np
from sklearn.linear_model import LogisticRegression
from src.fairness.adaptive_controller import AdaptiveLambdaController
from src.evaluation.fairness_evaluation import demographic_parity_difference


def compute_group_weights(y, s_attr, lambda_):
    """
    Strong group rebalancing:
    Increase weight for unprivileged samples proportionally to lambda.
    """
    weights = np.ones(len(y))

    # privileged = 1, unprivileged = 0
    unprivileged_mask = s_attr == 0

    # strong scaling (not tiny)
    weights[unprivileged_mask] = 1 + 5 * lambda_

    return weights


def train_adaptive_fair_model(
    X_train, y_train, s_train, sensitive_col, X_val, s_val, iterations=10
):
    controller = AdaptiveLambdaController(
        lambda_init=0.2, alpha=0.1, beta=0.02, threshold=0.02
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear")

    for _ in range(iterations):
        s_attr = s_train[sensitive_col].values

        weights = compute_group_weights(y_train.values, s_attr, controller.lambda_)

        model.fit(X_train, y_train, sample_weight=weights)

        y_val_pred = model.predict(X_val)

        dp = demographic_parity_difference(
            y_val_pred, s_val[sensitive_col].values
        )

        controller.update(dp)

    return model, controller.lambda_