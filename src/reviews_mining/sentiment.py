"""Sentiment scores: VADER on the review text, combined with the star rating.

VADER uses capitalization, punctuation, emoticons and negations ("not good"), so it is
run on the original text, not on the cleaned one (which has none of those).
"""
import pandas as pd

from . import config
from .text import load_clean_reviews

WEIGHTS = [0.7, 0.5, 0.3]   # weight of VADER; the rest goes to the rating


def rating_normalized(rating: pd.Series) -> pd.Series:
    return (rating - 1) / 4          # 1..5 stars -> 0..1


def load_scored_reviews(key: str) -> pd.DataFrame:
    """Cleaned reviews with `vader` (0..1) and the weighted scores (cached)."""
    out = config.processed_dir(key) / "reviews_scored.parquet"
    if out.exists():
        return pd.read_parquet(out)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df = load_clean_reviews(key)
    raw = (df["title"] + ". " + df["text"]).tolist()
    df["vader"] = [(analyzer.polarity_scores(t)["compound"] + 1) / 2 for t in raw]
    df["rating_normalized"] = rating_normalized(df["rating"])
    for w in WEIGHTS:
        df[f"score_{w:.1f}_{1 - w:.1f}"] = w * df["vader"] + (1 - w) * df["rating_normalized"]
    df.to_parquet(out)
    return df


def vader_agreement(df: pd.DataFrame) -> dict:
    """How well VADER's text-only score matches the stars."""
    stars = df["rating"].round()
    clear = df[stars != 3]
    vader_positive = clear["vader"] > 0.5
    star_positive = clear["rating"] > 3
    return {
        "correlation": df["vader"].corr(df["rating"]),
        "agreement (1-2 vs 4-5 stars)": (vader_positive == star_positive).mean(),
        "positive text, 1-2 stars": ((clear["vader"] > 0.75) & ~star_positive).mean(),
        "negative text, 4-5 stars": ((clear["vader"] < 0.25) & star_positive).mean(),
    }


def mean_vader_by_stars(scored: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({key: df.groupby(df["rating"].round())["vader"].mean()
                         for key, df in scored.items()}).rename_axis("stars")


def score_columns() -> list[str]:
    return [f"score_{w:.1f}_{1 - w:.1f}" for w in WEIGHTS]

