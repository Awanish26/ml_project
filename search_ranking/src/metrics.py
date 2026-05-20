"""Ranking metrics computed per query and averaged."""
import numpy as np
import pandas as pd


def _dcg(grades: np.ndarray, k: int) -> float:
    grades = grades[:k]
    return float(np.sum((2.0 ** grades - 1) / np.log2(np.arange(2, len(grades) + 2))))


def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = 10) -> float:
    """Mean NDCG@k with graded gains 2^grade - 1 (E=3, S=2, C=1, I=0).
    Queries with no relevant product (ideal DCG = 0) are skipped."""
    vals = []
    for _, g in df.groupby("query_id", sort=False):
        ideal = _dcg(np.sort(g["label"].to_numpy())[::-1], k)
        if ideal == 0:
            continue
        ranked = g.sort_values(score_col, ascending=False, kind="mergesort")["label"].to_numpy()
        vals.append(_dcg(ranked, k) / ideal)
    return float(np.mean(vals))


def mrr_at_k(df: pd.DataFrame, score_col: str, k: int = 10, grade: int = 3) -> float:
    """Mean reciprocal rank of the first Exact match within the top k."""
    vals = []
    for _, g in df.groupby("query_id", sort=False):
        if not (g["label"] == grade).any():
            continue
        ranked = g.sort_values(score_col, ascending=False, kind="mergesort")["label"].to_numpy()[:k]
        hits = np.flatnonzero(ranked == grade)
        vals.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)
    return float(np.mean(vals))
