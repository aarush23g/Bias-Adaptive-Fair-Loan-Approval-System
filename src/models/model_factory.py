from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier


# ---------------------------------------------
# Logistic Regression (Strong Version)
# ---------------------------------------------
def get_logistic_model(C=0.5):

    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            C=C,
            class_weight="balanced"
        ))
    ])


# ---------------------------------------------
# LightGBM Model (High Capacity)
# ---------------------------------------------
def get_lightgbm_model():

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=32,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )