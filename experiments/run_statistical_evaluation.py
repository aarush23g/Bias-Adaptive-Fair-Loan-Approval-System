import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data

from src.models.baseline import (
    train_logistic_regression,
    train_random_forest
)

from src.models.fairness_model import train_fair_logistic_regression

from src.fairness.fairness_metrics import demographic_parity
from src.fairness.adaptive_controller import (
    AdaptiveFairnessController,
    apply_controller
)

from lightgbm import LGBMClassifier


SEEDS = [0,1,2,3,4]


def evaluate_model(model, X_test, y_test, sensitive):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    dp = demographic_parity(preds, sensitive)

    return acc, roc, dp


def run_baseline(seed):

    df = load_data()

    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s, random_state=seed)

    sensitive = s_test.values.flatten()

    results = []

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    for name, model in [("LogisticRegression", lr),
                        ("RandomForest", rf)]:

        acc, roc, dp = evaluate_model(
            model,
            X_test,
            y_test,
            sensitive
        )

        results.append({
            "model": name,
            "accuracy": acc,
            "roc_auc": roc,
            "dp": dp
        })

    return results


def run_adaptive(seed):

    df = load_data()

    X, y, s = preprocess(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        s_train, s_val, s_test
    ) = split_data(X, y, s, random_state=seed)

    sensitive = s_test.values.flatten()

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=seed,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]

    controller = AdaptiveFairnessController()

    for _ in range(30):

        preds,_ = apply_controller(
            probs,
            controller.lambda_t
        )

        violation = demographic_parity(
            preds,
            sensitive
        )

        controller.update(violation)

    final_preds,_ = apply_controller(
        probs,
        controller.lambda_t
    )

    acc = accuracy_score(y_test, final_preds)
    roc = roc_auc_score(y_test, probs)

    dp = demographic_parity(
        final_preds,
        sensitive
    )

    return [{
        "model":"AdaptiveController",
        "accuracy":acc,
        "roc_auc":roc,
        "dp":dp
    }]


def run_all():

    os.makedirs("results/statistics", exist_ok=True)

    all_results = []

    for seed in SEEDS:

        print(f"\nRunning Seed {seed}")

        baseline_results = run_baseline(seed)
        adaptive_results = run_adaptive(seed)

        all_results.extend(baseline_results)
        all_results.extend(adaptive_results)

    df = pd.DataFrame(all_results)

    summary = df.groupby("model").agg({
        "accuracy":["mean","std"],
        "roc_auc":["mean","std"],
        "dp":["mean","std"]
    }).reset_index()

    summary.columns = [
        "model",
        "accuracy_mean","accuracy_std",
        "roc_mean","roc_std",
        "dp_mean","dp_std"
    ]

    summary.to_csv(
        "results/statistics/statistical_summary.csv",
        index=False
    )

    print("\nStatistical Summary\n")
    print(summary)


if __name__ == "__main__":
    run_all()