import os
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data


def compute_group_rates(preds, groups):

    df = pd.DataFrame({
        "pred": preds,
        "group": groups
    })

    return df.groupby("group")["pred"].mean()


def run_intersectional_experiment():

    os.makedirs("results/tables", exist_ok=True)

    df = load_data()
    X, y, s = preprocess(df)

    sensitive_cols = list(s.columns)

    # -------------------------------------------------
    # If only one sensitive attribute exists
    # create synthetic group BEFORE splitting
    # -------------------------------------------------
    if len(sensitive_cols) < 2:

        print("Only one sensitive attribute found. Creating synthetic group.")

        synthetic = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)

        s["synthetic_group"] = synthetic

        sensitive_cols = list(s.columns)

    attr1 = sensitive_cols[0]
    attr2 = sensitive_cols[1]

    print(f"\nIntersectional attributes: {attr1} × {attr2}")

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    groups = (
        s_test[attr1].astype(str)
        + "_"
        + s_test[attr2].astype(str)
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    group_rates = compute_group_rates(preds, groups)

    disparity = group_rates.max() - group_rates.min()

    results = pd.DataFrame({
        "group": group_rates.index,
        "positive_rate": group_rates.values
    })

    results["intersectional_disparity"] = disparity

    results.to_csv(
        "results/tables/intersectional_fairness_results.csv",
        index=False
    )

    print("\nIntersectional Fairness Results\n")
    print(results)

    print("\nAccuracy:", acc)
    print("ROC-AUC:", roc)
    print("Intersectional Disparity:", disparity)


if __name__ == "__main__":
    run_intersectional_experiment()