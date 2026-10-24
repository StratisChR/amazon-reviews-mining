"""Loading the raw data and caching it as Parquet.

The raw files are one JSON object per line. Each category is read in a single pass:
  - a uniform random sample of reviews (reservoir sampling) keeps the full text,
  - every review contributes (user, product, rating, timestamp) to a ratings table,
    which the recommender and the rating-over-time analysis use.
Every result is cached under data/processed/<category>/, so later runs skip the reading.
"""
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

REVIEW_COLUMNS = ["rating", "title", "text", "asin", "parent_asin", "user_id",
                  "timestamp", "helpful_vote", "verified_purchase"]
META_COLUMNS = ["parent_asin", "title", "main_category", "store", "average_rating",
                "rating_number", "price", "features", "description", "categories"]


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------

def _download(rel: str, dest: Path) -> None:
    """Streams a file from the Hugging Face repository to disk in 1 MB pieces, so memory
    use stays flat however big the file is. Writes to a .part file first, so an
    interrupted download is never mistaken for a complete one."""
    url = f"{config.DOWNLOAD_URL}/{rel}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "reviews-mining"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done, next_report = 0, 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"  downloading {rel}: {done / 1e9:.1f} of {total / 1e9:.1f} GB",
                      flush=True)
                next_report += 500_000_000
    part.rename(dest)


def _raw_file(candidates: list[str]) -> Path:
    """Returns the first candidate that exists locally or can be downloaded."""
    for rel in candidates:
        local = config.RAW_DIR / rel
        if local.exists():
            return local
    errors = []
    for rel in candidates:
        try:
            _download(rel, config.RAW_DIR / rel)
            return config.RAW_DIR / rel
        except urllib.error.HTTPError as e:   # not in the repo under this name: next one
            errors.append(f"{rel}: HTTP {e.code}")
    raise FileNotFoundError("None of these files could be downloaded: " + "; ".join(errors))


def review_file(key: str) -> Path:
    cat = config.CATEGORIES[key]
    return _raw_file([f"raw/review_categories/{cat}.jsonl"])


def meta_file(key: str) -> Path:
    cat = config.CATEGORIES[key]
    return _raw_file([f"raw/meta_categories/meta_{cat}.jsonl",
                      f"raw/meta_categories/{cat}.jsonl"])


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

def _read_reviews(path: Path, sample_size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    users, items, ratings, times = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("rating") is None:
                continue
            n = len(users)                 # index of this review among the valid ones
            users.append(r["user_id"])
            items.append(r["parent_asin"])
            ratings.append(r["rating"])
            times.append(r["timestamp"])
            row = {c: r.get(c) for c in REVIEW_COLUMNS}
            # Reservoir sampling (Algorithm R): every review ends up in the sample
            # with the same probability, without holding the whole file in memory.
            if len(sample) < sample_size:
                sample.append(row)
            else:
                j = rng.randint(0, n)
                if j < sample_size:
                    sample[j] = row
    reviews = pd.DataFrame(sample, columns=REVIEW_COLUMNS)
    all_ratings = pd.DataFrame({
        "user_id": pd.Categorical(users),
        "parent_asin": pd.Categorical(items),
        "rating": np.asarray(ratings, dtype=np.float32),
        "timestamp": np.asarray(times, dtype=np.int64),
    })
    return reviews, all_ratings


def _clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)


def load_reviews(key: str, sample_size: int = config.REVIEW_SAMPLE_SIZE,
                 seed: int = config.SEED) -> pd.DataFrame:
    """A uniform random sample of the category's reviews (cached)."""
    out = config.processed_dir(key) / "reviews_sample.parquet"
    if not out.exists():
        _build_review_tables(key, sample_size, seed)
    return pd.read_parquet(out)


def load_ratings(key: str) -> pd.DataFrame:
    """(user_id, parent_asin, rating, timestamp) for every review of the category (cached)."""
    out = config.processed_dir(key) / "ratings.parquet"
    if not out.exists():
        _build_review_tables(key, config.REVIEW_SAMPLE_SIZE, config.SEED)
    # User and product ids as categories: each distinct id is stored once, which keeps
    # the millions of rows of the larger categories to a few hundred MB.
    df = pd.read_parquet(out)
    df["user_id"] = df["user_id"].astype("category")
    df["parent_asin"] = df["parent_asin"].astype("category")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _build_review_tables(key, sample_size, seed):
    t = time.time()
    path = review_file(key)
    reviews, ratings = _read_reviews(path, sample_size, seed)
    _clean_reviews(reviews).to_parquet(config.processed_dir(key) / "reviews_sample.parquet")
    ratings.to_parquet(config.processed_dir(key) / "ratings.parquet")
    print(f"{key}: {len(ratings):,} reviews read, {len(reviews):,} sampled "
          f"({time.time() - t:.0f} s)")


# ---------------------------------------------------------------------------
# Product metadata
# ---------------------------------------------------------------------------

def _to_price(x):
    """Price as a float, or NaN when it is missing or not a number. Missing prices are
    left missing: most products have none, and filling them with the mean would make
    price look the same everywhere."""
    if x is None:
        return np.nan
    try:
        return float(str(x).replace("$", "").replace(",", "").strip())
    except ValueError:
        return np.nan


def _as_list(x):
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [str(x)]


def load_metadata(key: str) -> pd.DataFrame:
    """Product metadata of the category, one row per product (cached)."""
    out = config.processed_dir(key) / "metadata.parquet"
    if out.exists():
        return pd.read_parquet(out)
    t = time.time()
    rows = []
    with open(meta_file(key), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({c: r.get(c) for c in META_COLUMNS})
    df = pd.DataFrame(rows, columns=META_COLUMNS)
    df["price"] = df["price"].map(_to_price)
    for col in ["features", "description", "categories"]:
        df[col] = df[col].map(_as_list)
    df["title"] = df["title"].fillna("").astype(str)
    df["store"] = df["store"].fillna("").astype(str)
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates("parent_asin").reset_index(drop=True)
    df.to_parquet(out)
    print(f"{key}: {len(df):,} products ({time.time() - t:.0f} s)")
    return df
