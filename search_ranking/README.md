# Product search ranking (XGBoost learning-to-rank)

Re-rank the candidate products returned for a shopping query so the most
relevant ones come first. This is the second stage of a typical e-commerce search
stack: a cheap retriever returns a few dozen candidates, and a learned ranker
orders them using many signals at once.

## Data

[Amazon Shopping Queries (ESCI)](https://github.com/amazon-science/esci-data).
Every (query, product) pair is graded **E**xact, **S**ubstitute, **C**omplement
or **I**rrelevant, used here as relevance grades 3 / 2 / 1 / 0.

- US locale, "small version" (the reduced set recommended for ranking)
- 12,000 sampled training queries (10,800 train / 1,200 validation, 241k pairs)
- 3,000 queries from the official test split (60,782 pairs), used once for the numbers below

Download `shopping_queries_dataset_examples.parquet` and
`shopping_queries_dataset_products.parquet` into `data/raw/esci/`.

## Features (`src/features.py`)

| Group | Features |
|---|---|
| Lexical | BM25 on title and on all product text (vectorised over sparse matrices), TF-IDF cosine, query-term coverage, Jaccard |
| Semantic | Cosine similarity of `all-MiniLM-L6-v2` sentence embeddings (query vs title) |
| Attribute | Brand named in query, colour named in query, numbers in the query found in the title (sizes, counts, dimensions), text lengths |
| Query-relative | Rank and gap-to-best of BM25, TF-IDF and embedding scores within the query's candidate set |

The query-relative features matter: a ranker only needs to order products
*within* one query, so "how does this product compare with the best candidate
for this query" is often more useful than the raw score.

## Model (`src/train.py`)

`XGBRanker` with `rank:pairwise` (and `rank:ndcg` for comparison), 1,500 trees
max, learning rate 0.05, depth 6, row/column subsampling 0.8, early stopping on
validation NDCG@10.

## Results (3,000 held-out test queries)

NDCG@10 uses graded gains (2^grade - 1). MRR@10 is the reciprocal rank of the
first Exact product.

| Ranker | NDCG@10 | NDCG@5 | MRR@10 (Exact) |
|---|---|---|---|
| Random order | 0.693 | 0.630 | 0.609 |
| TF-IDF cosine (title) | 0.769 | 0.724 | 0.741 |
| BM25 (title) | 0.775 | 0.731 | 0.752 |
| BM25 (all text) | 0.779 | 0.739 | 0.747 |
| MiniLM embedding cosine | 0.789 | 0.747 | 0.761 |
| XGBoost, lexical features only | 0.782 | 0.743 | 0.761 |
| XGBoost, lexical + attribute | 0.787 | 0.748 | 0.767 |
| XGBoost, semantic + attribute | 0.796 | 0.756 | 0.777 |
| **XGBoost pairwise, all features** | **0.804** | **0.770** | **0.794** |
| XGBoost `rank:ndcg`, all features | 0.806 | 0.771 | 0.799 |

What the numbers say:

- ESCI candidate lists are mostly relevant already, so even a random order scores
  0.693. The useful headroom is 0.693 to 1.0; the full ranker closes 36% of it vs
  27% for BM25.
- The embedding score is the strongest single signal, but combining it with
  lexical and attribute features still adds +1.5 NDCG@10 points and +3.3 MRR
  points on top of it.
- By gain, the top features are `number_match`, `emb_cos_rank`, `bm25_all_rank`,
  `emb_cos_gap` and `coverage` (`results/feature_importance_gain.csv`). Number
  matching helps most on queries that specify a size, count or dimension.
- Pairwise and listwise (`rank:ndcg`) objectives land within 0.002 of each other.

## Run

```bash
python -m search_ranking.run --train-queries 12000 --test-queries 3000
```

Features are cached under `data/cache/`, so re-runs only retrain. A full run
takes a few minutes on an Apple M-series laptop (embeddings run on MPS).

## Next steps

- Replace the bi-encoder feature with a fine-tuned cross-encoder score on ESCI.
- Embed product bullets/descriptions, not just titles.
- Add behavioural features (clicks, add-to-cart) if logs are available; ESCI has none.
