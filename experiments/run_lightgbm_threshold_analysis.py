import os
import yaml
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data
from src.evaluation.fairness_evaluation import demographic_parity_difference

from sklearn.metrics import accuracy_score, roc_auc_score


# -------------------------------------------------
# Load config
# -------------------------------------------------
def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Pareto Frontier Extraction
# -------------------------------------------------
def extract_pareto_frontier(df):

    pareto = []

    for i, row in df.iterrows():

        dominated = False

        for j, other in df.iterrows():

            if (
                other["accuracy"] >= row["accuracy"]
                and abs(other["dp"]) <= abs(row["dp"])
                and (
                    other["accuracy"] > row["accuracy"]
                    or abs(other["dp"]) < abs(row["dp"])
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(row)

    return pd.DataFrame(pareto)


# -------------------------------------------------
# Main Threshold Analysis
# -------------------------------------------------
def run_threshold_analysis():

    os.makedirs("results/tables", exist_ok=True)

    cfg = load_config()
    dataset_name = cfg["dataset"]

    df = load_data()
    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s)

    sensitive_col = s.columns[0]

    # -------------------------------------------------
    # Train LightGBM on TRAIN
    # -------------------------------------------------
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------
    # Get probabilities
    # -------------------------------------------------
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_roc = roc_auc_score(y_val, val_probs)
    test_roc = roc_auc_score(y_test, test_probs)

    print("\nValidation ROC-AUC:", val_roc)
    print("Test ROC-AUC:", test_roc)

    # -------------------------------------------------
    # Threshold sweep (VALIDATION)
    # -------------------------------------------------
    thresholds = np.linspace(0.05, 0.95, 60)

    val_results = []

    for t in thresholds:

        preds = (val_probs >= t).astype(int)

        acc = accuracy_score(y_val, preds)

        dp = demographic_parity_difference(
            preds,
            s_val[sensitive_col].values
        )

        val_results.append({
            "threshold": t,
            "accuracy": acc,
            "dp": dp
        })

    val_df = pd.DataFrame(val_results)

    val_df.to_csv(
        f"results/tables/{dataset_name}_lightgbm_validation_curve.csv",
        index=False
    )

    # -------------------------------------------------
    # Pareto frontier
    # -------------------------------------------------
    pareto_df = extract_pareto_frontier(val_df)

    pareto_df.to_csv(
        f"results/tables/{dataset_name}_lightgbm_pareto_validation.csv",
        index=False
    )

    print("\nPareto Frontier (Validation):")
    print(pareto_df.sort_values("accuracy", ascending=False))

    # -------------------------------------------------
    # Select best threshold
    # maximize accuracy subject to |dp| <= 0.02
    # -------------------------------------------------
    feasible = pareto_df[pareto_df["dp"].abs() <= 0.02]

    if len(feasible) == 0:
        best_row = pareto_df.sort_values(
            "accuracy", ascending=False
        ).iloc[0]
    else:
        best_row = feasible.sort_values(
            "accuracy", ascending=False
        ).iloc[0]

    best_threshold = best_row["threshold"]

    print("\nSelected Threshold:", best_threshold)

    # -------------------------------------------------
    # Final TEST evaluation
    # -------------------------------------------------
    final_preds = (test_probs >= best_threshold).astype(int)

    final_acc = accuracy_score(y_test, final_preds)

    final_dp = demographic_parity_difference(
        final_preds,
        s_test[sensitive_col].values
    )

    final_result = pd.DataFrame([{
        "model": "LightGBM_ThresholdOptimized",
        "accuracy": final_acc,
        "roc_auc": test_roc,
        "dp": final_dp,
        "threshold": best_threshold
    }])

    final_result.to_csv(
        f"results/tables/{dataset_name}_lightgbm_threshold_final.csv",
        index=False
    )

    print("\nFinal Test Result:")
    print(final_result)


if __name__ == "__main__":
    run_threshold_analysis()