import pandas as pd
import yaml


def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


def encode_sensitive(df, cfg):
    dataset = cfg["dataset"]
    sens_cfg = cfg["datasets"][dataset]["sensitive_attributes"]

    s = pd.DataFrame(index=df.index)

    for attr, groups in sens_cfg.items():
        priv = groups["privileged"]
        s[attr] = df[attr].apply(lambda x: 1 if x == priv else 0)

    return s


def preprocess(df):
    cfg = load_config()
    dataset = cfg["dataset"]
    target_col = cfg["datasets"][dataset]["target"]

    # --------------------------------------------------
    # TARGET PROCESSING
    # --------------------------------------------------
    y = pd.to_numeric(df[target_col], errors="coerce")
    df = df.loc[y.notna()].copy()
    y = y.loc[y.notna()].astype(int)

    if y.nunique() != 2:
        raise ValueError("Target column must be binary after preprocessing.")

    # --------------------------------------------------
    # SENSITIVE ATTRIBUTES
    # --------------------------------------------------
    s = encode_sensitive(df, cfg)

    # --------------------------------------------------
    # FEATURES (keep sensitive columns in X for fairness-aware training)
    # --------------------------------------------------
    X = df.drop(columns=[target_col]).copy()

    # --------------------------------------------------
    # HANDLE NUMERIC FEATURES → MEDIAN IMPUTATION
    # --------------------------------------------------
    numeric_cols = X.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # --------------------------------------------------
    # HANDLE CATEGORICAL FEATURES
    # --------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns

    for col in categorical_cols:
        X[col] = X[col].fillna("missing")

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # --------------------------------------------------
    # FINAL SANITY CHECKS
    # --------------------------------------------------
    if X.isna().sum().sum() > 0:
        raise ValueError("NaN values found in features after preprocessing.")

    if len(X) == 0:
        raise ValueError("Feature matrix is empty after preprocessing.")

    if len(y) == 0:
        raise ValueError("Target vector is empty after preprocessing.")
    
    # --------------------------------------------------
    # CLEAN FEATURE NAMES FOR LIGHTGBM
    # --------------------------------------------------
    X.columns = (
        X.columns
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace("__+", "_", regex=True)
    )

    return X, y, s