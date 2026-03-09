import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_performance_metrics(model, X_test, y_test):

    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    y_pred = model.predict(X_test)

    # -----------------------------------------
    # Probabilities (if available)
    # -----------------------------------------
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)

    else:
        # Fairlearn reductions may not expose probas
        # fallback to predicted labels (not ideal but valid)
        y_scores = y_pred

    # -----------------------------------------
    # Metrics
    # -----------------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = np.nan  # if only one class predicted

    return accuracy, roc_auc