"""Exploratory analysis: frequent words, rating distributions, ratings over time,
best and worst products, and the words that separate negative from positive reviews."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grid(n, ncols=3, size=(6, 3.6)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows),
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def token_counts(texts) -> Counter:
    counts = Counter()
    for t in texts:
        counts.update(t.split())
    return counts


def plot_top_tokens(reviews: dict[str, pd.DataFrame], n: int = 20):
    """The n most frequent words (after cleaning) of each category's reviews."""
    fig, axes = _grid(len(reviews), size=(6, 4.5))
    for ax, (key, df) in zip(axes, reviews.items()):
        words, counts = zip(*token_counts(df["text_clean"]).most_common(n))
        ax.barh(words[::-1], counts[::-1], color="tab:blue")
        ax.set_title(key)
        ax.set_xlabel("occurrences")
    fig.suptitle(f"Most frequent words in reviews (top {n})", fontsize=14)
    fig.tight_layout()
    return fig


def plot_rating_distributions(ratings: dict[str, pd.DataFrame]):
    """Share of 1-5 star ratings per category, over all reviews of the category."""
    shares = pd.DataFrame({key: df["rating"].round().value_counts(normalize=True)
                           for key, df in ratings.items()}).reindex([1, 2, 3, 4, 5]).fillna(0)
    ax = (shares * 100).plot.bar(figsize=(10, 4.5), width=0.8, rot=0)
    ax.set_xlabel("stars")
    ax.set_ylabel("% of reviews")
    ax.set_title("Rating distribution per category (all reviews)")
    for key, df in ratings.items():
        print(f"{key:20s} mean rating {df['rating'].mean():.2f} over {len(df):,} reviews")
    plt.tight_layout()
    return shares


def plot_monthly_ratings(ratings: dict[str, pd.DataFrame], years: int = 4,
                         min_reviews: int = 50):
    """Average rating per calendar month, one line per year, for the last `years` years
    of each category. Months with fewer than `min_reviews` reviews are left out, so a
    handful of reviews can't make a spike."""
    fig, axes = _grid(len(ratings))
    for ax, (key, df) in zip(axes, ratings.items()):
        d = df.assign(year=df["timestamp"].dt.year, month=df["timestamp"].dt.month)
        last = sorted(d["year"].unique())[-years:]
        g = (d[d["year"].isin(last)].groupby(["year", "month"])["rating"]
             .agg(["mean", "size"]).reset_index())
        g = g[g["size"] >= min_reviews]
        for year, part in g.groupby("year"):
            ax.plot(part["month"], part["mean"], marker="o", ms=3, label=str(year))
        ax.set_xticks(range(1, 13))
        ax.set_title(key)
        ax.set_xlabel("month")
        ax.set_ylabel("average rating")
        ax.legend(fontsize=8)
    fig.suptitle("Average rating per month", fontsize=14)
    fig.tight_layout()
    return fig


def best_products(meta: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Products with the most ratings (a proxy for the best sellers)."""
    cols = ["title", "average_rating", "rating_number"]
    return (meta.sort_values("rating_number", ascending=False)
            .drop_duplicates("title")[cols].head(n).reset_index(drop=True))


def worst_products(meta: pd.DataFrame, min_ratings: int = 1000, n: int = 5) -> pd.DataFrame:
    """Lowest average rating among products with at least `min_ratings` ratings."""
    cols = ["title", "average_rating", "rating_number"]
    popular = meta[meta["rating_number"] >= min_ratings]
    return (popular.sort_values(["average_rating", "rating_number"], ascending=[True, False])
            .drop_duplicates("title")[cols].head(n).reset_index(drop=True))


def distinctive_words(reviews: pd.DataFrame, n: int = 15, min_count: int = 20):
    """Words most over-represented in negative (1-2 stars) vs positive (4-5 stars)
    reviews, ranked by the log-odds ratio with add-one smoothing. Raw frequency would
    show the same words on both sides ("product", "use", ...); the ratio shows what
    people actually complain about or praise."""
    neg = token_counts(reviews.loc[reviews["rating"] <= 2, "text_clean"])
    pos = token_counts(reviews.loc[reviews["rating"] >= 4, "text_clean"])
    vocab = [w for w in set(neg) | set(pos) if neg[w] + pos[w] >= min_count]
    if not vocab:
        return pd.DataFrame(columns=["negative", "positive"])
    n_neg, n_pos = sum(neg.values()) + len(vocab), sum(pos.values()) + len(vocab)
    score = pd.Series({w: np.log((neg[w] + 1) / n_neg) - np.log((pos[w] + 1) / n_pos)
                       for w in vocab})
    return pd.DataFrame({
        "negative": score.nlargest(n).index,
        "positive": score.nsmallest(n).index,
    })
