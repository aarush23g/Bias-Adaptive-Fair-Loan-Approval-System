import yaml
from sklearn.linear_model import LogisticRegression
from src.fairness.static_fairness import compute_reweighing_weights


def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


def get_sensitive_column(cfg):
    dataset = cfg["dataset"]
    sens_attrs = cfg["datasets"][dataset]["sensitive_attributes"]
    # take first sensitive attribute
    return list(sens_attrs.keys())[0]


def train_fair_logistic_regression(X_train, y_train, s_train):

    cfg = load_config()
    sensitive_col = get_sensitive_column(cfg)

    s_attr = s_train[sensitive_col].values

    # compute reweighing weights
    weights = compute_reweighing_weights(y_train.values, s_attr)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train, sample_weight=weights)

    return model