import pandas as pd


def load_adult_data(cfg):

    dataset_cfg = cfg["datasets"]["adult"]
    path = dataset_cfg["path"]

    df = pd.read_csv(path)

    # ---------------------------------------
    # Clean column names
    # ---------------------------------------
    df.columns = df.columns.str.strip()

    # ---------------------------------------
    # Clean string values
    # ---------------------------------------
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ---------------------------------------
    # Convert income to binary
    # ---------------------------------------
    df["income"] = df["income"].apply(
        lambda x: 1 if ">50K" in str(x) else 0
    )

    return df