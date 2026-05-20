"""Query-product features for learning-to-rank.

Three families, all computed per (query, product) pair:
  * lexical   - BM25 (title / all text), TF-IDF cosine, term coverage, Jaccard
  * semantic  - cosine similarity of sentence embeddings (MiniLM)
  * attribute - brand / colour / number matches, text-length signals
plus query-relative versions of the strongest scores (rank and gap to the best
candidate in the same query), which is what lets a ranker compare products
*within* a query rather than across queries.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
TOKEN_RE = re.compile(r"\w+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokens(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


def _pair_index(pairs: pd.DataFrame, col: str):
    """Unique values of `col` and, for every pair, the index of its value."""
    uniques, idx = np.unique(pairs[col].to_numpy(), return_inverse=True)
    return uniques, idx


def bm25(queries, docs, query_idx, doc_idx, k1=1.2, b=0.75) -> np.ndarray:
    """Okapi BM25 for each (query, doc) pair, computed with sparse matrices.

    IDF and average document length come from the unique documents, so every
    pair is scored against the same corpus statistics."""
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
    tf = vec.fit_transform(docs).tocsr().astype(np.float32)
    n_docs = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    norm = k1 * (1 - b + b * doc_len / doc_len.mean())

    # Saturated, IDF-weighted term frequency per (doc, term).
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    weighted = tf.copy()
    weighted.data = idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + norm[rows])

    q_terms = vec.transform(queries).tocsr()
    q_terms.data[:] = 1.0
    return np.asarray(q_terms[query_idx].multiply(weighted[doc_idx]).sum(axis=1)).ravel()


def tfidf_cosine(queries, docs, query_idx, doc_idx) -> np.ndarray:
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
    d = vec.fit_transform(docs)
    q = vec.transform(queries)
    return np.asarray(q[query_idx].multiply(d[doc_idx]).sum(axis=1)).ravel()


def embedding_cosine(queries, docs, query_idx, doc_idx, cache_key: str) -> np.ndarray:
    """Cosine similarity of L2-normalised MiniLM embeddings (cached to disk)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    q_path, d_path = CACHE_DIR / f"{cache_key}_q.npy", CACHE_DIR / f"{cache_key}_d.npy"
    if q_path.exists() and d_path.exists():
        q_emb, d_emb = np.load(q_path), np.load(d_path)
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(EMBED_MODEL, device=device)
        kw = dict(batch_size=256, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        q_emb = model.encode(list(queries), **kw).astype(np.float32)
        d_emb = model.encode(list(docs), **kw).astype(np.float32)
        np.save(q_path, q_emb)
        np.save(d_path, d_emb)
    return np.einsum("ij,ij->i", q_emb[query_idx], d_emb[doc_idx])


def lexical_overlap(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query, title, brand, color in zip(
        pairs["query"], pairs["product_title"], pairs["product_brand"], pairs["product_color"]
    ):
        q, t = tokens(query), tokens(title)
        b, c = tokens(brand), tokens(color)
        q_nums = set(NUMBER_RE.findall(query))
        rows.append(
            (
                len(q & t) / max(len(q), 1),                     # query term coverage
                len(q & t) / max(len(q | t), 1),                 # Jaccard
                float(bool(b) and b <= q),                       # brand named in query
                float(bool(c) and bool(c & q)),                  # colour named in query
                len(q_nums & set(NUMBER_RE.findall(title))) / len(q_nums) if q_nums else -1.0,
                len(q),
                len(t),
            )
        )
    return pd.DataFrame(
        rows,
        columns=["coverage", "jaccard", "brand_match", "color_match", "number_match", "query_len", "title_len"],
        index=pairs.index,
    )


def add_query_relative(df: pd.DataFrame, cols) -> pd.DataFrame:
    grp = df.groupby("query_id")
    for c in cols:
        df[f"{c}_rank"] = grp[c].rank(ascending=False, pct=True)
        df[f"{c}_gap"] = df[c] - grp[c].transform("max")
    return df


def build_features(pairs: pd.DataFrame, cache_key: str = "esci") -> pd.DataFrame:
    df = pairs.copy()
    queries, q_idx = _pair_index(df, "query")
    titles, t_idx = _pair_index(df, "product_title")
    df["all_text"] = (
        df["product_title"] + " " + df["product_brand"] + " " + df["product_bullet_point"] + " " + df["product_description"]
    )
    all_docs, a_idx = _pair_index(df, "all_text")

    df["bm25_title"] = bm25(queries, titles, q_idx, t_idx)
    df["bm25_all"] = bm25(queries, all_docs, q_idx, a_idx)
    df["tfidf_cos"] = tfidf_cosine(queries, titles, q_idx, t_idx)
    df["emb_cos"] = embedding_cosine(queries, titles, q_idx, t_idx, cache_key)
    df = pd.concat([df, lexical_overlap(df)], axis=1)
    df["has_bullets"] = (df["product_bullet_point"].str.len() > 0).astype(float)
    df["has_description"] = (df["product_description"].str.len() > 0).astype(float)
    df = add_query_relative(df, ["bm25_all", "tfidf_cos", "emb_cos"])
    return df.drop(columns="all_text")


LEXICAL = ["bm25_title", "bm25_all", "tfidf_cos", "coverage", "jaccard",
           "bm25_all_rank", "bm25_all_gap", "tfidf_cos_rank", "tfidf_cos_gap"]
SEMANTIC = ["emb_cos", "emb_cos_rank", "emb_cos_gap"]
ATTRIBUTE = ["brand_match", "color_match", "number_match", "query_len", "title_len",
             "has_bullets", "has_description"]
ALL_FEATURES = LEXICAL + SEMANTIC + ATTRIBUTE
