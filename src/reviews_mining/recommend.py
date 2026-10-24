"""Recommendation: predict which product a user reviews next, and measure it.

Data. Amazon reviews are extremely sparse: most users review one product, and with one
review there is nothing to learn from. So the data is reduced to its 5-core (every
user and every product has at least 5 reviews), the usual setup for this dataset.

Evaluation (leave-last-out). For each user, the most recent review is the test item
and the one before it the validation item; everything earlier is training data.
A model scores every product the user has not reviewed yet, and we look at where the
held-out product lands:
  HR@10   - share of users whose held-out product is in the top 10,
  NDCG@10 - the same, but a hit at position 1 counts more than at position 10.
When several products have the same score, the held-out one is counted as equally
likely to be at any of their positions. So a model that gives everything the same
score gets what a random guess would get on average, not a perfect score.

Models. Popularity (baseline), user-based and item-based collaborative filtering
(cosine similarity, k nearest neighbours), content-based (the product's text, as in
the clustering), and a hybrid of item-based CF and content, whose weight is chosen on
the validation items and then reported on the test items.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def k_core(ratings: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Keeps only users and products with at least k reviews, repeating until stable
    (removing a product can push a user below k, and the other way round)."""
    df = (ratings.sort_values("timestamp")
          .drop_duplicates(["user_id", "parent_asin"], keep="last"))
    while True:
        users = df.groupby("user_id", observed=True)["user_id"].transform("size")
        items = df.groupby("parent_asin", observed=True)["parent_asin"].transform("size")
        keep = (users >= k) & (items >= k)
        if keep.all():
            # Plain string ids from here on: the core is small, and categories would
            # keep listing every user and product of the full data.
            return df.astype({"user_id": str, "parent_asin": str}).reset_index(drop=True)
        df = df[keep]


@dataclass
class Split:
    users: pd.Index          # user_id of each row of the matrices
    items: pd.Index          # parent_asin of each column
    train: sp.csr_matrix     # reviews before the validation item
    train_val: sp.csr_matrix # train + validation item (used for the test)
    val: np.ndarray          # validation item of each user
    test: np.ndarray         # test item of each user


def leave_last_out(core: pd.DataFrame) -> Split:
    users = pd.Index(core["user_id"].unique())
    items = pd.Index(core["parent_asin"].unique())
    df = core.assign(u=users.get_indexer(core["user_id"]),
                     i=items.get_indexer(core["parent_asin"]))
    df = df.sort_values(["u", "timestamp"])
    pos_from_end = df.groupby("u").cumcount(ascending=False)
    test = df[pos_from_end == 0].set_index("u")["i"].reindex(range(len(users))).to_numpy()
    val = df[pos_from_end == 1].set_index("u")["i"].reindex(range(len(users))).to_numpy()
    shape = (len(users), len(items))

    def matrix(part):
        return sp.csr_matrix((np.ones(len(part), dtype=np.float32), (part["u"], part["i"])),
                             shape=shape)

    return Split(users, items, matrix(df[pos_from_end >= 2]), matrix(df[pos_from_end >= 1]),
                 val, test)


def item_text_vectors(meta: pd.DataFrame, items: pd.Index, n_components: int = 64,
                      seed: int = config.SEED) -> np.ndarray:
    """LSA vector of each product's text, aligned with `items` (zeros when unknown)."""
    text = (meta.drop_duplicates("parent_asin").set_index("parent_asin")
            ["product_text_clean"].reindex(items).fillna(""))
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(text)
    n_components = max(1, min(n_components, tfidf.shape[1] - 1))
    vectors = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(tfidf)
    return normalize(vectors)


# ---------------------------------------------------------------------------
# Models: fit(X) on a users x items 0/1 matrix, scores(users) -> dense array
# ---------------------------------------------------------------------------

class Popularity:
    name = "Popularity"

    def fit(self, X):
        self.counts = np.asarray(X.sum(axis=0)).ravel()
        return self

    def scores(self, users):
        return np.tile(self.counts, (len(users), 1))


def _keep_top_k_per_row(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
    S = S.tocsr()
    data, indices, indptr = [], [], [0]
    for r in range(S.shape[0]):
        lo, hi = S.indptr[r], S.indptr[r + 1]
        vals, cols = S.data[lo:hi], S.indices[lo:hi]
        if len(vals) > k:
            top = np.argpartition(-vals, k)[:k]
            vals, cols = vals[top], cols[top]
        data.append(vals)
        indices.append(cols)
        indptr.append(indptr[-1] + len(vals))
    return sp.csr_matrix((np.concatenate(data), np.concatenate(indices), indptr),
                         shape=S.shape)


class ItemKNN:
    """Score of product j for user u: sum of the similarities between j and the products
    u reviewed, keeping each product's k most similar products."""
    name = "Item-based CF"

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X):
        self.X = X.tocsr()
        Xn = normalize(X.tocsc(), axis=0)
        S = (Xn.T @ Xn).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        self.S = _keep_top_k_per_row(S, self.k)
        return self

    def scores(self, users):
        return (self.X[users] @ self.S).toarray()


class UserKNN:
    """Score of product j for user u: similarity-weighted votes of u's k most similar
    users who reviewed j."""
    name = "User-based CF"

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X):
        self.X = X.tocsr()
        self.Xn = normalize(self.X, axis=1)
        return self

    def scores(self, users):
        sims = (self.Xn[users] @ self.Xn.T).toarray()
        sims[np.arange(len(users)), users] = 0                   # not yourself
        k = min(self.k, sims.shape[1] - 1)
        cut = np.partition(sims, -k, axis=1)[:, -k][:, None]
        sims[sims < cut] = 0                                     # k nearest only
        return np.asarray((self.X.T @ sims.T).T)


class ContentBased:
    """Score of product j for user u: cosine similarity between j's text vector and the
    mean text vector of the products u reviewed."""
    name = "Content-based"

    def __init__(self, item_vectors: np.ndarray):
        self.V = item_vectors

    def fit(self, X):
        self.profiles = normalize(X @ self.V)
        return self

    def scores(self, users):
        return self.profiles[users] @ self.V.T


class Hybrid:
    """Weighted sum of two models' scores, each standardized per user first (the
    models score on different scales)."""

    def __init__(self, a, b, weight_a: float):
        self.a, self.b, self.w = a, b, weight_a
        self.name = f"Hybrid ({weight_a:.1f} {a.name} + {1 - weight_a:.1f} {b.name})"

    def fit(self, X):
        self.a.fit(X)
        self.b.fit(X)
        return self

    @staticmethod
    def _standardize(s):
        return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)

    def scores(self, users):
        return (self.w * self._standardize(self.a.scores(users))
                + (1 - self.w) * self._standardize(self.b.scores(users)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_seen: sp.csr_matrix, targets: np.ndarray, users: np.ndarray,
             k: int = 10, batch: int = 256) -> dict:
    """HR@k and NDCG@k of the held-out `targets` of `users`; products in X_seen (already
    reviewed) are excluded from the ranking."""
    cum_gain = np.concatenate([[0], np.cumsum(1 / np.log2(np.arange(k) + 2))])
    hits, gains = [], []
    for start in range(0, len(users), batch):
        u = users[start:start + batch]
        s = model.scores(u).astype(np.float64)
        seen = X_seen[u].nonzero()
        s[seen] = -np.inf
        t = s[np.arange(len(u)), targets[u]][:, None]
        # The target lands somewhere among the products with the same score, each
        # position equally likely: average HR and NDCG over those positions.
        better = (s > t).sum(axis=1)
        tied = (s == t).sum(axis=1)                       # includes the target itself
        lo, hi = np.minimum(better, k), np.minimum(better + tied, k)
        hits.append((hi - lo) / tied)
        gains.append((cum_gain[hi] - cum_gain[lo]) / tied)
    return {"HR@10": float(np.concatenate(hits).mean()),
            "NDCG@10": float(np.concatenate(gains).mean())}


@dataclass
class Result:
    stats: dict
    table: pd.DataFrame          # one row per model: validation and test metrics
    weights: pd.DataFrame        # hybrid weight search on the validation items
    split: Split
    models: dict                 # fitted on train + validation, for examples


def run(ratings: pd.DataFrame, meta: pd.DataFrame, k_core_min: int = 5,
        max_eval_users: int = 5000, seed: int = config.SEED) -> Result | None:
    core = k_core(ratings, k_core_min)
    if core.empty or core["user_id"].nunique() < 50:
        print(f"Only {core['user_id'].nunique()} users in the {k_core_min}-core: skipped.")
        return None
    split = leave_last_out(core)
    n_users, n_items = len(split.users), len(split.items)
    stats = {"reviews (all)": len(ratings), "reviews (5-core)": len(core),
             "users": n_users, "products": n_items,
             "density": len(core) / (n_users * n_items)}
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(n_users, size=min(max_eval_users, n_users), replace=False))
    V = item_text_vectors(meta, split.items)

    def make_models():
        return {"pop": Popularity(), "user": UserKNN(), "item": ItemKNN(),
                "content": ContentBased(V)}

    # Validation: fit on train, score the validation items.
    val_models = {n: m.fit(split.train) for n, m in make_models().items()}
    val = {n: evaluate(m, split.train, split.val, users) for n, m in val_models.items()}
    weights = []
    for w in np.round(np.linspace(0, 1, 11), 1):
        h = Hybrid(val_models["item"], val_models["content"], w)   # already fitted
        weights.append({"weight of item-based CF": w,
                        **evaluate(h, split.train, split.val, users)})
    weights = pd.DataFrame(weights)
    best_w = float(weights.loc[weights["NDCG@10"].idxmax(), "weight of item-based CF"])

    # Test: refit on train + validation, score the test items.
    test_models = make_models()
    test_models["hybrid"] = Hybrid(test_models["item"], test_models["content"], best_w)
    for name in ["pop", "user", "item", "content"]:
        test_models[name].fit(split.train_val)
    val["hybrid"] = weights.loc[weights["weight of item-based CF"] == best_w,
                                ["HR@10", "NDCG@10"]].iloc[0].to_dict()
    rows = []
    for name, m in test_models.items():
        t = evaluate(m, split.train_val, split.test, users)
        rows.append({"model": m.name,
                     "val HR@10": val[name]["HR@10"], "val NDCG@10": val[name]["NDCG@10"],
                     "test HR@10": t["HR@10"], "test NDCG@10": t["NDCG@10"]})
    table = pd.DataFrame(rows).set_index("model").round(4)
    stats["evaluated users"] = len(users)
    return Result(stats, table, weights, split, test_models)


def recommend(result: Result, meta: pd.DataFrame, user_index: int, model: str = "hybrid",
              n: int = 10) -> tuple[list[str], list[str]]:
    """Titles of a user's reviewed products and of the model's top-n recommendations."""
    titles = meta.drop_duplicates("parent_asin").set_index("parent_asin")["title"]
    X = result.split.train_val
    seen = X[user_index].indices
    s = result.models[model].scores(np.array([user_index]))[0].astype(float)
    s[seen] = -np.inf
    top = np.argsort(-s)[:n]
    name = lambda i: titles.get(result.split.items[i], result.split.items[i])  # noqa: E731
    return [name(i) for i in seen], [name(i) for i in top]
