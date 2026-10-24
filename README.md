# Mining Amazon Reviews

![Tests](https://github.com/StratisChR/amazon-reviews-mining/actions/workflows/tests.yml/badge.svg)

Text mining, clustering, recommendation and sentiment classification on five categories
of the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset
(All Beauty, Software, Musical Instruments, Handmade Products, Gift Cards):
about 9.4 million reviews and 580,000 products.

| Step | Method | Evaluation |
|---|---|---|
| Preprocessing | streaming download, reservoir sampling of 100k reviews per category, spaCy lookup lemmatization | – |
| Exploratory analysis | rating distributions, frequent words, words that separate negative from positive reviews (log-odds), ratings over time, best and worst products | – |
| Sentiment scores | VADER on the review text, combined with the stars | agreement of text and stars |
| Product clustering | TF-IDF → LSA (truncated SVD) → k-means | silhouette score, cluster profiles |
| Recommendation | popularity, user- and item-based collaborative filtering, content-based, hybrid | HR@10 and NDCG@10 on each user's last review (leave-last-out) |
| Sentiment classification | TF-IDF or GloVe features; Naive Bayes, logistic regression, KNN, random forest | 5-fold CV + held-out test set, macro-F1 |

The analysis and its results are in
[`notebooks/amazon_reviews_mining.ipynb`](notebooks/amazon_reviews_mining.ipynb); the code
is in [`src/reviews_mining/`](src/reviews_mining).

## Results

- **Sentiment classification** (negative / neutral / positive from the review text):
  TF-IDF + logistic regression is the best model in all five categories, with test
  macro-F1 **0.61-0.68**, against 0.28-0.31 for the majority-class baseline. TF-IDF beats
  averaged GloVe embeddings by 0.05-0.07. Neutral (3-star) reviews are the hard class:
  about half are recognized.
- **Recommendation**: collaborative filtering beats the popularity baseline where there
  is enough data. In Software (146k users, 17.6k products) item-based CF reaches HR@10
  **0.15** against 0.07 for popularity. Content-based recommendation is the weakest model.
- **Clustering** finds coherent product types (wigs, nail art, skin care; pedals,
  microphones, cables, drums), and ratings differ by type: casino apps and human-hair
  bundles are the lowest-rated groups. Silhouette scores are low (0.03-0.14): product texts
  overlap rather than forming separate groups.
- **Ratings** are J-shaped (55-84% five stars, more one-star than two- or three-star
  reviews), and in three of the five categories they were lower in 2022-2023 than in 2020. VADER's text-only sentiment agrees
  with the stars on 87-94% of clearly positive or negative reviews.

## Design choices

- **The rating is never a classification feature.** Sentiment labels come from the
  stars, so the classifiers see only the review text and the VADER score computed from
  it. A test shuffles the text across reviews and checks that performance drops to
  chance level, which it would not if anything linked to the label were left in.
- **No leakage between training and evaluation.** TF-IDF and scaling are fitted inside
  each cross-validation fold; the recommender's hybrid weight is chosen on validation
  reviews and reported on separate test reviews.
- **Honest baselines and metrics.** Most reviews are positive, so classifiers are
  compared with a majority-class baseline using macro-F1, and recommenders with a
  popularity baseline. Ties in recommendation scores are counted at their average
  expected position, so a model that gives everything the same score gets the result
  of a random guess.
- **Missing values stay missing.** Most products have no price; instead of filling in
  the mean, price is reported with how often it is known.
- **Fast, memory-light preprocessing.** Raw files are streamed to disk and read line by
  line; lemmas come from spaCy's lookup tables (about half a million words per second
  in one process, against 4.5 hours for the original row-by-row version). Each step
  caches its result as Parquet, so the notebook re-runs in minutes.

## Running it

Python 3.10 or newer. The first run downloads the raw files from Hugging Face (several
GB) and processes them; later runs read the cache in `data/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"
jupyter notebook notebooks/amazon_reviews_mining.ipynb   # or open it in VS Code
```

Tests run on small synthetic data in the dataset's format, with no download:

```bash
pip install pytest
pytest -q tests
```

## Layout

```
src/reviews_mining/
  config.py       categories, paths, sample size
  data.py         reading the raw files, reservoir sampling, Parquet cache
  text.py         text cleaning with spaCy
  eda.py          exploratory analysis and plots
  sentiment.py    VADER scores
  clustering.py   TF-IDF + LSA + k-means, cluster profiles
  recommend.py    k-core filtering, leave-last-out split, recommenders, HR/NDCG
  classify.py     sentiment classification without label leakage
  cache.py        caching of slow results
notebooks/        the analysis
tests/            synthetic data generator and tests
```

Started as a course project for *Data Mining* at the Department of Informatics and
Telecommunications, University of Athens, then reworked.
