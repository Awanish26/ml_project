"""Load the Amazon Shopping Queries (ESCI) dataset as a flat query-product table.

ESCI grades every (query, product) pair as Exact, Substitute, Complement or
Irrelevant. We keep the US locale of the "small version" (the reduced set the
dataset authors recommend for ranking), sample queries for a laptop-sized run,
and join the product text onto every pair.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "esci"

# Graded relevance used both as the XGBoost label and as the NDCG gain.
LABEL_GRADE = {"E": 3, "S": 2, "C": 1, "I": 0}

PRODUCT_COLS = [
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


def _sample_queries(query_ids, n, rng):
    query_ids = np.asarray(query_ids)
    if n is None or n >= len(query_ids):
        return query_ids
    return rng.choice(query_ids, size=n, replace=False)


def load_pairs(
    locale: str = "us",
    n_train_queries: int = 12_000,
    n_test_queries: int = 3_000,
    valid_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one row per (query, product) pair with a `part` column of
    train / valid / test. Validation queries are carved out of the official
    train split so the official test split is only touched once."""
    examples = pd.read_parquet(RAW_DIR / "shopping_queries_dataset_examples.parquet")
    examples = examples[(examples.small_version == 1) & (examples.product_locale == locale)]

    rng = np.random.default_rng(seed)
    train_q = _sample_queries(examples.loc[examples.split == "train", "query_id"].unique(), n_train_queries, rng)
    test_q = _sample_queries(examples.loc[examples.split == "test", "query_id"].unique(), n_test_queries, rng)
    n_valid = int(len(train_q) * valid_frac)
    part_of = {q: "valid" for q in train_q[:n_valid]}
    part_of.update({q: "train" for q in train_q[n_valid:]})
    part_of.update({q: "test" for q in test_q})

    pairs = examples[examples.query_id.isin(part_of)].copy()
    pairs["part"] = pairs.query_id.map(part_of)
    pairs["label"] = pairs.esci_label.map(LABEL_GRADE).astype(int)

    products = pd.read_parquet(
        RAW_DIR / "shopping_queries_dataset_products.parquet",
        columns=PRODUCT_COLS + ["product_locale"],
        filters=[("product_locale", "=", locale)],
    )
    products = products[products.product_id.isin(pairs.product_id.unique())].drop(columns="product_locale")
    pairs = pairs.merge(products, on="product_id", how="left")

    text_cols = [c for c in PRODUCT_COLS if c != "product_id"]
    pairs[text_cols] = pairs[text_cols].fillna("")
    # XGBoost's ranking objectives need rows grouped by query.
    return pairs.sort_values(["part", "query_id", "product_id"]).reset_index(drop=True)
