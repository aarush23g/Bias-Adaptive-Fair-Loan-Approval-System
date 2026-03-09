import pandas as pd


def load_german_data(cfg):
    path = cfg["datasets"]["german"]["path"]

    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "sex", "other_debtors",
        "residence", "property", "age", "other_installment_plans", "housing",
        "existing_credits", "job", "num_dependents", "telephone",
        "foreign_worker", "credit_risk"
    ]

    # --- TRY WHITESPACE FORMAT (UCI) ---
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # If only one column → file is actually comma-separated
    if df.shape[1] == 1:
        df = pd.read_csv(path, header=None)

    # If still wrong, raise clear error
    if df.shape[1] < 21:
        raise ValueError(
            f"German dataset has {df.shape[1]} columns. "
            "Expected 21. Check file format."
        )

    # Trim extra columns if present
    df = df.iloc[:, :21]

    df.columns = columns

    # --- TARGET PROCESSING ---
    df["credit_risk"] = pd.to_numeric(df["credit_risk"], errors="coerce")
    df = df[df["credit_risk"].isin([1, 2])]
    df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})

    # --- AGE GROUP ---
    # ensure age is numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # create age group
    df["age_group"] = df["age"].apply(lambda x: "old" if x >= 25 else "young")

    df = df.reset_index(drop=True)

    return df