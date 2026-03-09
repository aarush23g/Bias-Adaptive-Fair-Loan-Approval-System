import pandas as pd


def load_lending_club_data(cfg):
    path = cfg["datasets"]["lending_club"]["path"]

    # --------------------------------------------------
    # LOAD (no dtype inference explosion)
    # --------------------------------------------------
    df = pd.read_csv(path, low_memory=False)

    # --------------------------------------------------
    # TEST MODE → USE SMALL SAMPLE FOR PYTEST
    # --------------------------------------------------
    if cfg.get("test_mode", False):
        df = df.sample(n=2000, random_state=cfg["random_state"])

    # --------------------------------------------------
    # EXPERIMENT MODE → MEMORY SAFE SAMPLE
    # --------------------------------------------------
    sample_size = cfg.get("sample_size_lending_club", None)
    if sample_size and not cfg.get("test_mode", False):
        df = df.sample(n=sample_size, random_state=cfg["random_state"])

    # --------------------------------------------------
    # TARGET PROCESSING
    # --------------------------------------------------
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

    df["loan_status"] = df["loan_status"].map(
        {"Fully Paid": 1, "Charged Off": 0}
    )

    # --------------------------------------------------
    # NUMERIC CLEANUPS
    # --------------------------------------------------
    if "int_rate" in df.columns:
        if df["int_rate"].dtype == "object":
            df["int_rate"] = (
                df["int_rate"].str.replace("%", "", regex=False)
            )
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    if "term" in df.columns:
        df["term"] = (
            df["term"].astype(str).str.extract(r"(\d+)")[0]
        )
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"].astype(str).str.extract(r"(\d+)")[0]
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    # --------------------------------------------------
    # INCOME → SENSITIVE ATTRIBUTE
    # --------------------------------------------------
    df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")

    median_income = df["annual_inc"].median()

    df["income_group"] = df["annual_inc"].apply(
        lambda x: "high" if x >= median_income else "low"
    )

    # --------------------------------------------------
    # DROP HIGH-CARDINALITY / USELESS COLUMNS
    # (prevents 1.6 TB dummy matrix)
    # --------------------------------------------------
    drop_cols = [
        "id",
        "member_id",
        "url",
        "desc",
        "title",
        "zip_code",
        "addr_state",
        "earliest_cr_line",
        "last_pymnt_d",
        "next_pymnt_d",
        "last_credit_pull_d",
        "issue_d",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --------------------------------------------------
    # DROP ROWS WITH CRITICAL NaNs
    # --------------------------------------------------
    df = df.dropna(subset=["loan_status", "annual_inc"])

    # --------------------------------------------------
    # OPTIONAL: KEEP ONLY TOP FEATURES FOR SPEED
    # (You can expand later for experiments)
    # --------------------------------------------------
    keep_cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "emp_length",
        "home_ownership",
        "verification_status",
        "purpose",
        "income_group",
        "loan_status",
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    if df.shape[0] == 0:
        raise ValueError("LendingClub dataset is empty after preprocessing.")

    return df