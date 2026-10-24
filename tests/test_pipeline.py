"""Tests on synthetic data (no download): run with `pytest`."""
import importlib
import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    os.environ["REVIEWS_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    from reviews_mining import config
    importlib.reload(config)                   # pick up the new data directory
    import make_synthetic
    importlib.reload(make_synthetic)
    make_synthetic.main()
    from reviews_mining import classify, data, recommend, sentiment, text
    return {"data": data, "text": text, "recommend": recommend,
            "sentiment": sentiment, "classify": classify}


def test_clean_texts_drops_stop_words_digits_and_keeps_empty_empty(pkg):
    out = pkg["text"].clean_texts(["The 3 guitars were LOUD!!! :) www.example.com", "",
                                   "123 !!!"])
    assert out[0] == "guitar loud"
    assert out[1] == "" and out[2] == ""


def test_review_sample_and_ratings_table(pkg):
    reviews = pkg["data"].load_reviews("allBeauty", sample_size=1000)
    ratings = pkg["data"].load_ratings("allBeauty")
    assert len(reviews) == 1000
    assert len(ratings) > len(reviews)
    assert set(reviews["parent_asin"]) <= set(ratings["parent_asin"])


def test_k_core_and_leave_last_out(pkg):
    R = pkg["recommend"]
    core = R.k_core(pkg["data"].load_ratings("musicalInstruments"), k=5)
    assert core.groupby("user_id").size().min() >= 5
    assert core.groupby("parent_asin").size().min() >= 5
    s = R.leave_last_out(core)
    for u in range(len(s.users)):
        assert s.train_val[u, s.test[u]] == 0          # test item never seen
        assert s.train[u, s.val[u]] == 0               # validation item not in train
        assert s.train_val[u, s.val[u]] == 1           # but used for the test run


def test_evaluation_constant_model_scores_like_random_and_oracle_scores_one(pkg):
    R = pkg["recommend"]
    s = R.leave_last_out(R.k_core(pkg["data"].load_ratings("musicalInstruments")))
    users = np.arange(len(s.users))
    n_items = len(s.items)

    class Constant:
        def scores(self, u):
            return np.zeros((len(u), n_items))

    class Oracle:
        def scores(self, u):
            z = np.zeros((len(u), n_items))
            z[np.arange(len(u)), s.test[u]] = 1
            return z

    unseen = n_items - np.asarray(s.train_val.sum(axis=1)).ravel()
    expected_hr = np.mean(np.minimum(10, unseen) / unseen)
    assert R.evaluate(Constant(), s.train_val, s.test, users)["HR@10"] == pytest.approx(expected_hr)
    assert R.evaluate(Oracle(), s.train_val, s.test, users) == {"HR@10": 1.0, "NDCG@10": 1.0}


def test_collaborative_filtering_beats_popularity(pkg):
    meta = pkg["text"].load_clean_metadata("musicalInstruments")
    result = pkg["recommend"].run(pkg["data"].load_ratings("musicalInstruments"), meta)
    t = result.table["test NDCG@10"]
    assert t["Item-based CF"] > t["Popularity"]


def test_classifier_cannot_see_the_label(pkg):
    """With the text shuffled across reviews, nothing left in the features is linked to
    the stars, so macro-F1 must fall to chance level. If the rating (or a score made
    from it) were a feature, it would stay near 1."""
    df = pkg["sentiment"].load_scored_reviews("allBeauty").copy()
    perm = np.random.default_rng(0).permutation(len(df))
    df["text_clean"] = df["text_clean"].to_numpy()[perm]
    df["vader"] = df["vader"].to_numpy()[perm]
    results, _ = pkg["classify"].evaluate(df, kv=None, folds=3, feature_sets=("TF-IDF",))
    assert results["test macro-F1"].max() < 0.45


def test_download_streams_to_disk_and_falls_back_to_the_second_name(pkg, tmp_path, monkeypatch):
    """The metadata file is tried under two names; the first one is missing here."""
    import http.server
    import threading
    from functools import partial
    from reviews_mining import config

    served = tmp_path / "served" / "raw" / "meta_categories"
    served.mkdir(parents=True)
    (served / "Gift_Cards.jsonl").write_bytes(b'{"parent_asin": "X"}\n' * 200_000)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "served"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(config, "DOWNLOAD_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw_dir")
        path = pkg["data"].meta_file("giftCards")
    finally:
        server.shutdown()
    assert path.name == "Gift_Cards.jsonl"
    assert path.read_bytes() == (served / "Gift_Cards.jsonl").read_bytes()
    assert not list(path.parent.glob("*.part"))
