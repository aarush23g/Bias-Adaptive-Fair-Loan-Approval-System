from sklearn.model_selection import train_test_split
import pandas as pd
import yaml


def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


def split_data(X, y, s, test_size=None, val_size=None, random_state=None):
    """
    Splits data into train, validation, and test sets.

    Improvements:
    - Stratifies by (target + sensitive attribute) when possible
    - Prevents fairness experiments from losing groups
    """

    cfg = load_config()

    # Use passed values or config defaults
    test_size = test_size if test_size is not None else cfg["test_size"]
    val_size = val_size if val_size is not None else cfg["val_size"]
    random_state = (
        random_state if random_state is not None else cfg["random_state"]
    )

    # --------------------------------------------------
    # Build stratification label
    # --------------------------------------------------
    if s.shape[1] > 0:
        stratify_label = y.astype(str) + "_" + s.iloc[:, 0].astype(str)
    else:
        stratify_label = y

    # --------------------------------------------------
    # First split: train vs temp
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X,
        y,
        s,
        test_size=test_size,
        stratify=stratify_label,
        random_state=random_state,
    )

    # --------------------------------------------------
    # Validation stratification
    # --------------------------------------------------
    stratify_temp = y_temp.astype(str) + "_" + s_temp.iloc[:, 0].astype(str)

    val_relative_size = val_size / (1 - test_size)

    # --------------------------------------------------
    # Second split: val vs test
    # --------------------------------------------------
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp,
        y_temp,
        s_temp,
        test_size=val_relative_size,
        stratify=stratify_temp,
        random_state=random_state,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        s_train,
        s_val,
        s_test,
    )