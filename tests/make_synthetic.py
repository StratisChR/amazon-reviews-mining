"""Writes small synthetic raw files in the dataset's format, for testing the pipeline
without downloading anything:

    REVIEWS_DATA_DIR=/tmp/synthetic python tests/make_synthetic.py

The data has structure the analysis should find: product topics (for clustering),
users who stick to one topic (for recommendation) and review words that depend on
the rating (for classification).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reviews_mining import config  # noqa: E402

TOPICS = [
    ["guitar", "string", "pickup", "bridge", "fret", "amplifier"],
    ["shampoo", "hair", "scalp", "conditioner", "curl", "fragrance"],
    ["game", "puzzle", "level", "player", "score", "adventure"],
    ["necklace", "silver", "pendant", "chain", "earring", "jewelry"],
]
POSITIVE = ["excellent", "love", "perfect", "great", "wonderful", "recommend", "happy"]
NEGATIVE = ["terrible", "broke", "awful", "refund", "disappointed", "waste", "useless"]
NEUTRAL = ["okay", "average", "decent", "fine", "expected", "mediocre"]
FILLER = ["product", "arrive", "package", "use", "week", "price", "quality", "item"]


def words(rng, pool, n):
    return " ".join(rng.choice(pool) for _ in range(n))


def make_category(cat, rng, n_items=600, n_users=900):
    items = []
    for i in range(n_items):
        topic = i % len(TOPICS)
        items.append({
            "parent_asin": f"{cat[:3].upper()}{i:06d}",
            "topic": topic,
            "quality": rng.uniform(1.5, 5.0),
        })
    meta = []
    for it in items:
        t = TOPICS[it["topic"]]
        has_desc = rng.random() > 0.25
        meta.append({
            "main_category": cat,
            "title": f"{words(rng, t, 3).title()} Model {rng.randint(1, 999)}",
            "average_rating": round(min(5, max(1, it["quality"] + rng.gauss(0, 0.3))), 1),
            "rating_number": rng.randint(1, 20000),
            "features": [words(rng, t + FILLER, 6) for _ in range(rng.randint(0, 3))],
            "description": [words(rng, t + FILLER, 25)] if has_desc else [],
            "price": f"{rng.uniform(3, 300):.2f}" if rng.random() > 0.6 else None,
            "images": [], "videos": [], "store": f"Store{rng.randint(1, 50)}",
            "categories": [], "details": "{}", "parent_asin": it["parent_asin"],
            "bought_together": None,
        })
    by_topic = {k: [it for it in items if it["topic"] == k] for k in range(len(TOPICS))}
    reviews = []
    ts = 1_500_000_000_000
    for u in range(n_users):
        user = f"U{cat[:3].upper()}{u:07d}"
        fav = rng.randrange(len(TOPICS))
        n = rng.choice([1, 1, 2, 3, 5, 6, 8, 10, 14])
        for _ in range(n):
            pool = by_topic[fav] if rng.random() < 0.85 else items
            # Popular items within a topic: a skewed choice, like real data.
            it = pool[min(len(pool) - 1, int(rng.paretovariate(1.2)) - 1)]
            rating = int(round(min(5, max(1, it["quality"] + rng.gauss(0, 0.8)))))
            sentiment = NEGATIVE if rating <= 2 else NEUTRAL if rating == 3 else POSITIVE
            text = (words(rng, sentiment, 3) + " " + words(rng, FILLER, 4) + " "
                    + words(rng, TOPICS[it["topic"]], 2))
            if rng.random() < 0.02:
                text = ""
            ts += rng.randint(1, 10_000_000)
            reviews.append({
                "rating": float(rating), "title": words(rng, sentiment, 2).title(),
                "text": text, "images": [], "asin": it["parent_asin"] + "X",
                "parent_asin": it["parent_asin"], "user_id": user, "timestamp": ts,
                "helpful_vote": rng.randint(0, 5), "verified_purchase": True,
            })
    rng.shuffle(reviews)
    return reviews, meta


def main():
    rng = random.Random(0)
    for cat in config.CATEGORIES.values():
        reviews, meta = make_category(cat, rng)
        rdir = config.RAW_DIR / "raw" / "review_categories"
        mdir = config.RAW_DIR / "raw" / "meta_categories"
        rdir.mkdir(parents=True, exist_ok=True)
        mdir.mkdir(parents=True, exist_ok=True)
        with open(rdir / f"{cat}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in reviews)
        with open(mdir / f"meta_{cat}.jsonl", "w") as f:
            f.writelines(json.dumps(m) + "\n" for m in meta)
        print(f"{cat}: {len(reviews)} reviews, {len(meta)} products")


if __name__ == "__main__":
    main()
