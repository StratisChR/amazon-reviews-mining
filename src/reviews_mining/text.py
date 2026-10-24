"""Text cleaning: lowercase, strip non-letters, drop stop words, lemmatize.

Lemmas come from spaCy's lookup table (spacy-lookups-data) on a blank English
pipeline: no part-of-speech tagger, so it runs at around half a million words per
second in a single process, with almost no memory. A tagger would give slightly better
lemmas ("amazing" stays "amazing" as an adjective instead of becoming "amaze"), but for
bag-of-words features the difference is negligible, while the tagger is about 25 times
slower and needs several processes, each with its own copy of the data, to be usable.
(The first version cleaned one row at a time with a full spaCy pipeline: 4.5 hours.)
"""
import re
import time

import pandas as pd

from . import config
from .data import load_metadata, load_reviews

_NON_LETTERS = re.compile(r"[^a-z\s]+")
_MENTIONS = re.compile(r"@\w+")
_URLS = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")
MAX_CHARS = 20_000   # longer texts are cut (a handful of descriptions are huge)

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.blank("en")
        _nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        _nlp.initialize()
    return _nlp


def _pre_clean(text: str) -> str:
    text = str(text)[:MAX_CHARS].lower()
    text = _URLS.sub(" ", text)
    text = _MENTIONS.sub(" ", text)
    text = _NON_LETTERS.sub(" ", text)      # also removes digits, punctuation and emoji
    return _SPACES.sub(" ", text).strip()


def clean_texts(texts, batch_size: int = 2000) -> list[str]:
    """Cleaned version of every text: lemmas of 3-15 letters, no stop words, joined by
    spaces. An empty result stays an empty string (not a placeholder word, which would
    end up in every word count and TF-IDF vector)."""
    nlp = get_nlp()
    out = []
    for doc in nlp.pipe((_pre_clean(t) for t in texts), batch_size=batch_size):
        out.append(" ".join(t.lemma_ for t in doc
                            if not (t.is_stop or t.is_space)
                            and 3 <= len(t.lemma_) <= 15))
    return out


def load_clean_reviews(key: str) -> pd.DataFrame:
    """The review sample with a `text_clean` column (cached)."""
    out = config.processed_dir(key) / "reviews_clean.parquet"
    if out.exists():
        return pd.read_parquet(out)
    df = load_reviews(key)
    t = time.time()
    df["text_clean"] = clean_texts((df["title"] + ". " + df["text"]).tolist())
    df.to_parquet(out)
    print(f"{key}: cleaned {len(df):,} reviews in {time.time() - t:.0f} s")
    return df


# What the analysis uses from the metadata after cleaning. The raw description, feature
# and category lists are the bulk of the memory (gigabytes for all five categories) and
# are not needed once the cleaned product text exists.
CLEAN_META_COLUMNS = ["parent_asin", "title", "average_rating", "rating_number", "price",
                      "product_text_clean"]


def load_clean_metadata(key: str) -> pd.DataFrame:
    """Product title, rating, price and cleaned product text (title, description and
    feature bullets together), one row per product (cached)."""
    out = config.processed_dir(key) / "metadata_clean.parquet"
    if not out.exists():
        df = load_metadata(key)
        t = time.time()
        # Each part is cleaned once; the product text is the cleaned parts joined.
        title = clean_texts(df["title"].tolist())
        description = clean_texts(df["description"].map(" ".join).tolist())
        features = clean_texts(df["features"].map(" ".join).tolist())
        df["product_text_clean"] = [" ".join(p for p in parts if p)
                                    for parts in zip(title, description, features)]
        df[CLEAN_META_COLUMNS].to_parquet(out)
        print(f"{key}: cleaned {len(df):,} product texts in {time.time() - t:.0f} s")
    return pd.read_parquet(out, columns=CLEAN_META_COLUMNS)
