import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data

from src.models.fairness_model import train_fair_logistic_regression

from src.evaluation.performance_metrics import compute_performance_metrics
from src.evaluation.fairness_evaluation import compute_fairness_metrics


def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


def extract_dp(fairness_dict):
    for k, v in fairness_dict.items():
        if k.startswith("dp_"):
            return v
    return None


def run_static_fairness():

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

    model = train_fair_logistic_regression(
        X_train, y_train, s_train
    )

    accuracy, roc_auc = compute_performance_metrics(model, X_test, y_test)

    fairness = compute_fairness_metrics(model, X_test, s_test)
    dp = extract_dp(fairness)

    result = {
        "model": "FairLogisticRegression",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "dp": dp
    }

    results_df = pd.DataFrame([result])

    print(f"\nStatic Fairness Results ({dataset_name}):\n")
    print(results_df)

    results_df.to_csv(
        f"results/tables/{dataset_name}_static_metrics.csv",
        index=False
    )


if __name__ == "__main__":
    run_static_fairness()