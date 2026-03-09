import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data


FIG_DIR = "results/figures"
TABLE_DIR = "results/tables"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# --------------------------------------------------
# Train baseline model
# --------------------------------------------------
def train_model():

    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    sensitive_col = s_test.columns[0]

    return model, X_train, X_test, s_test, sensitive_col


# --------------------------------------------------
# SHAP Global
# --------------------------------------------------
def shap_global(model, X_train, X_test, dataset):

    explainer = shap.LinearExplainer(model, X_train)

    shap_values = explainer(X_test)

    plt.figure()

    shap.plots.beeswarm(shap_values, show=False)

    plt.savefig(
        f"{FIG_DIR}/shap_global_{dataset}.png",
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# SHAP Group Comparison
# --------------------------------------------------
def shap_group_comparison(model, X_train, X_test, s_test, sensitive_col, dataset):

    explainer = shap.LinearExplainer(model, X_train)

    shap_values = explainer(X_test)

    shap_matrix = np.abs(shap_values.values)

    priv_mask = s_test[sensitive_col] == 1
    unpriv_mask = s_test[sensitive_col] == 0

    shap_priv = shap_matrix[priv_mask].mean(axis=0)
    shap_unpriv = shap_matrix[unpriv_mask].mean(axis=0)

    diff = shap_priv - shap_unpriv

    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "priv_mean_abs_shap": shap_priv,
        "unpriv_mean_abs_shap": shap_unpriv,
        "difference": diff,
    }).sort_values(by="difference", key=np.abs, ascending=False)

    shap_df.to_csv(
        f"{TABLE_DIR}/shap_group_difference_{dataset}.csv",
        index=False
    )

    top = shap_df.head(10)

    plt.figure()

    plt.barh(top["feature"], top["difference"])

    plt.xlabel("SHAP Difference (Priv - Unpriv)")

    plt.title(f"Feature Attribution Differences ({dataset})")

    plt.gca().invert_yaxis()

    plt.savefig(
        f"{FIG_DIR}/shap_group_difference_{dataset}.png",
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# Counterfactual Examples
# --------------------------------------------------
def generate_counterfactual_examples(model, X_test, s_test, sensitive_col, dataset):

    preds = model.predict(X_test)

    unpriv_mask = (s_test[sensitive_col] == 0) & (preds == 0)

    candidates = X_test[unpriv_mask].copy()

    cf_rows = []

    for idx, row in candidates.head(5).iterrows():

        for col in X_test.columns[:5]:

            modified = row.copy()

            modified[col] = modified[col] + 1

            new_pred = model.predict(
                modified.values.reshape(1, -1)
            )[0]

            if new_pred == 1:

                cf_rows.append({
                    "dataset": dataset,
                    "original_index": idx,
                    "changed_feature": col,
                    "original_prediction": 0,
                    "counterfactual_prediction": 1,
                })

                break

    cf_df = pd.DataFrame(cf_rows)

    cf_df.to_csv(
        f"{TABLE_DIR}/counterfactual_examples_{dataset}.csv",
        index=False
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def run_explainability():

    model, X_train, X_test, s_test, sensitive_col = train_model()

    # Detect dataset name from config
    import yaml
    with open("configs/data.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]

    shap_global(model, X_train, X_test, dataset)

    shap_group_comparison(
        model,
        X_train,
        X_test,
        s_test,
        sensitive_col,
        dataset
    )

    generate_counterfactual_examples(
        model,
        X_test,
        s_test,
        sensitive_col,
        dataset
    )

    print(f"\nExplainability artifacts saved for {dataset}.")


if __name__ == "__main__":
    run_explainability()