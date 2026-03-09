import yaml

from src.data.german_loader import load_german_data
from src.data.lending_loader import load_lending_club_data
from src.data.adult_loader import load_adult_data


def load_config():
    with open("configs/data.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data():

    cfg = load_config()
    dataset = cfg["dataset"]
    test_mode = cfg.get("test_mode", False)

    # --------------------------------------------------
    # FORCE SMALL DATA IN TEST MODE
    # --------------------------------------------------
    if test_mode:
        return load_german_data(cfg)

    if dataset == "german":
        return load_german_data(cfg)

    elif dataset == "lending_club":
        return load_lending_club_data(cfg)

    elif dataset == "adult":
        return load_adult_data(cfg)

    else:
        raise ValueError("Unsupported dataset")