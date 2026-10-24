"""Clustering products by their text.

Text (title + description + feature bullets) -> TF-IDF -> truncated SVD (LSA, 100
dimensions) -> unit length -> k-means. k is chosen by the silhouette score.

Price and rating are not clustering inputs; they are used afterwards to describe the
clusters. Most products have no price, and a single dense rating column next to the
text dimensions would dominate the distances, so the clusters would just be rating
ranges. Describing text clusters by price and rating answers the more useful question:
do different kinds of products get different ratings?
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from . import config


@dataclass
class ClusterData:
    products: pd.DataFrame       # the products that were clustered
    tfidf: object                # sparse TF-IDF matrix
    vectors: np.ndarray          # LSA vectors, unit length
    terms: np.ndarray            # TF-IDF vocabulary


def build_features(meta: pd.DataFrame, max_features: int = 20_000,
                   n_components: int = 100, seed: int = config.SEED) -> ClusterData:
    products = meta[meta["product_text_clean"].str.len() > 0].reset_index(drop=True)
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.5,
                                 sublinear_tf=True)
    tfidf = vectorizer.fit_transform(products["product_text_clean"])
    n_components = min(n_components, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    vectors = Normalizer().fit_transform(svd.fit_transform(tfidf))
    print(f"{len(products):,} of {len(meta):,} products have text; "
          f"{tfidf.shape[1]:,} terms; LSA keeps {svd.explained_variance_ratio_.sum():.0%} "
          f"of the TF-IDF variance")
    return ClusterData(products, tfidf, vectors, vectorizer.get_feature_names_out())


def choose_k(vectors: np.ndarray, ks=range(2, 11), sample: int = 10_000,
             seed: int = config.SEED) -> pd.DataFrame:
    """Inertia and silhouette score (on a random sample of points) for each k."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(sample, len(vectors)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(vectors)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(vectors[idx], km.labels_[idx])})
    return pd.DataFrame(rows)


def plot_k_selection(scores: pd.DataFrame, title: str = "") -> int:
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(scores["k"], scores["inertia"], marker="o", color="tab:blue")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["silhouette"], marker="s", color="tab:red")
    ax2.set_ylabel("silhouette", color="tab:red")
    best = scores.loc[scores["silhouette"].idxmax(), "k"]
    ax1.axvline(best, ls="--", color="gray")
    ax1.set_title(f"{title}: choosing k (best silhouette at k={best})")
    fig.tight_layout()
    return int(best)


def cluster(vectors: np.ndarray, k: int, seed: int = config.SEED) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)


def profile(data: ClusterData, labels: np.ndarray, n_terms: int = 8) -> pd.DataFrame:
    """One row per cluster: size, top terms, price and rating statistics."""
    p = data.products.assign(cluster=labels)
    rows = []
    for c, part in p.groupby("cluster"):
        mask = (labels == c)
        mean_tfidf = np.asarray(data.tfidf[mask].mean(axis=0)).ravel()
        top = data.terms[np.argsort(mean_tfidf)[::-1][:n_terms]]
        rows.append({
            "cluster": c,
            "products": len(part),
            "top terms": ", ".join(top),
            "price known": f"{part['price'].notna().mean():.0%}",
            "median price": part["price"].median(),
            "mean rating": part["average_rating"].mean(),
            "% rated < 3.5": (part["average_rating"] < 3.5).mean() * 100,
            "median #ratings": part["rating_number"].median(),
        })
    return pd.DataFrame(rows).set_index("cluster").round(2)


def example_titles(data: ClusterData, labels: np.ndarray, n: int = 4,
                   seed: int = config.SEED) -> dict:
    p = data.products.assign(cluster=labels)
    return {c: part.sample(min(n, len(part)), random_state=seed)["title"].str[:90].tolist()
            for c, part in p.groupby("cluster")}


def project_2d(data: ClusterData, labels: np.ndarray, sample: int = 5000,
               seed: int = config.SEED):
    """2-D PCA coordinates of a random sample of products, with their cluster."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=min(sample, len(labels)), replace=False)
    return PCA(n_components=2, random_state=seed).fit_transform(data.vectors[idx]), labels[idx]


def plot_clusters_2d(xy: np.ndarray, labels: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7)
    ax.legend(*sc.legend_elements(), title="cluster", fontsize=8, loc="best")
    ax.set_title(f"{title}: clusters (PCA of the LSA vectors)")
    fig.tight_layout()
    return fig


@dataclass
class ClusterRun:
    k_scores: pd.DataFrame
    k: int
    profile: pd.DataFrame
    examples: dict
    xy: np.ndarray
    xy_labels: np.ndarray


def run(meta: pd.DataFrame, ks=range(2, 11), seed: int = config.SEED) -> ClusterRun:
    """The whole clustering step for one category; the result is small enough to cache."""
    data = build_features(meta, seed=seed)
    scores = choose_k(data.vectors, ks, seed=seed)
    k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    labels = cluster(data.vectors, k, seed)
    xy, xy_labels = project_2d(data, labels, seed=seed)
    return ClusterRun(scores, k, profile(data, labels), example_titles(data, labels, seed=seed),
                      xy, xy_labels)
