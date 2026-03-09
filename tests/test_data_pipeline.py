from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.data.split import split_data


def test_data_pipeline_runs():
    df = load_data()
    X, y, s = preprocess(df)

    splits = split_data(X, y, s)

    X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test = splits

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(X_train) == len(y_train)
    assert len(X_train) == len(s_train)