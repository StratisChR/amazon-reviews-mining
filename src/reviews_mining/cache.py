"""Caching of slow results (clustering, recommendation and classification runs), so the
notebook can be re-run without recomputing them. Delete data/processed/<category>/*.pkl
to force a step to run again."""
import pickle

from . import config


def cached(key: str, name: str, compute):
    path = config.processed_dir(key) / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    result = compute()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result
