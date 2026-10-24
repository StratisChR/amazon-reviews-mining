"""Sentiment classification: predict negative / neutral / positive from the review text.

Labels come from the stars: 1-2 negative, 3 neutral, 4-5 positive. The features may
therefore use only the text: the words (TF-IDF, or averaged GloVe embeddings) and the
VADER score, which is computed from the text alone. The rating, and anything derived
from it, is never a feature; with it, a model reads the answer instead of learning it.

Every learned transformation (TF-IDF vocabulary, scaling) is fitted inside the
cross-validation folds, on training data only. The test set (20%, stratified) is used
once, at the end. Classes are imbalanced (most reviews are positive), so the main
metric is macro-F1, and a majority-class baseline shows what accuracy alone is worth.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, make_scorer,
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

CLASSES = ["negative", "neutral", "positive"]
# zero_division=0: the majority-class baseline never predicts two of the classes.
SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}


def sentiment_label(rating: pd.Series) -> np.ndarray:
    r = rating.round()
    return np.select([r <= 2, r == 3], [0, 1], default=2)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

_glove = None


def load_glove():
    """Pre-trained GloVe vectors (100 dimensions, ~130 MB download the first time).
    REVIEWS_GLOVE_PATH can point to a local file in word2vec text format instead."""
    global _glove
    if _glove is None:
        path = os.environ.get("REVIEWS_GLOVE_PATH")
        if path:
            from gensim.models import KeyedVectors
            _glove = KeyedVectors.load_word2vec_format(path)
        else:
            import gensim.downloader
            _glove = gensim.downloader.load("glove-wiki-gigaword-100")
    return _glove


def average_embeddings(texts, kv) -> np.ndarray:
    """Mean GloVe vector of the words of each text (zeros if none is known)."""
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, text in enumerate(texts):
        vecs = [kv[w] for w in text.split() if w in kv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def _tfidf_features():
    return ColumnTransformer([
        ("words", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=3,
                                  sublinear_tf=True), "text_clean"),
        ("vader", "passthrough", ["vader"]),
    ])


def tfidf_models(seed=config.SEED):
    """name -> (model, n_jobs for cross-validation): -1 runs folds in parallel;
    models that already use every core themselves get 1."""
    return {
        "Majority class": (DummyClassifier(), -1),   # ignores the features
        "Naive Bayes": (make_pipeline(_tfidf_features(), MultinomialNB()), -1),
        "Logistic Regression": (make_pipeline(_tfidf_features(), LogisticRegression(
            max_iter=2000, class_weight="balanced")), -1),
        "KNN": (make_pipeline(_tfidf_features(), KNeighborsClassifier(
            n_neighbors=15, metric="cosine", algorithm="brute")), -1),
        "Random Forest": (make_pipeline(_tfidf_features(), RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)), 1),
    }


def glove_models(seed=config.SEED):
    return {
        "Majority class": (DummyClassifier(), -1),
        "Naive Bayes": (GaussianNB(), -1),
        "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced")), -1),
        "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)), -1),
        "Random Forest": (RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed), 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _test_scores(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p,
            "recall_macro": r, "f1_macro": f1}


def evaluate(df: pd.DataFrame, kv, folds: int = 5, seed: int = config.SEED,
             feature_sets=("TF-IDF", "GloVe")):
    """Cross-validation on the training part and a final score on the test part, for
    every model and feature set. Returns (results, confusion matrices on the test set)."""
    y = sentiment_label(df["rating"])
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2,
                                           stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    inputs = {}
    if "TF-IDF" in feature_sets:
        inputs["TF-IDF"] = (df[["text_clean", "vader"]], tfidf_models(seed))
    if "GloVe" in feature_sets:
        emb = average_embeddings(df["text_clean"].tolist(), kv)
        inputs["GloVe"] = (np.hstack([emb, df[["vader"]].to_numpy()]), glove_models(seed))

    rows, confusions = [], {}
    for feature_set, (X, models) in inputs.items():
        take = (lambda idx: X.iloc[idx]) if isinstance(X, pd.DataFrame) else (lambda idx: X[idx])
        X_train, X_test = take(idx_train), take(idx_test)
        for name, (model, jobs) in models.items():
            # Two folds at a time: each process holds its own copy of the data and the
            # fitted model (KNN's distance computations in particular), so running every
            # fold at once can use more memory than a 16 GB laptop has.
            cvr = cross_validate(model, X_train, y[idx_train], cv=cv, scoring=SCORING,
                                 n_jobs=2 if jobs == -1 else jobs)
            model.fit(X_train, y[idx_train])
            pred = model.predict(X_test)
            test = _test_scores(y[idx_test], pred)
            confusions[(feature_set, name)] = confusion_matrix(y[idx_test], pred,
                                                               labels=[0, 1, 2])
            rows.append({
                "features": feature_set, "model": name,
                "CV accuracy": cvr["test_accuracy"].mean(),
                "CV macro-F1": cvr["test_f1_macro"].mean(),
                "CV macro-F1 std": cvr["test_f1_macro"].std(),
                "test accuracy": test["accuracy"],
                "test precision": test["precision_macro"],
                "test recall": test["recall_macro"],
                "test macro-F1": test["f1_macro"],
            })
    return pd.DataFrame(rows).round(4), confusions


def class_shares(df: pd.DataFrame) -> pd.Series:
    y = sentiment_label(df["rating"])
    return pd.Series(np.bincount(y, minlength=3) / len(y), index=CLASSES).round(3)
