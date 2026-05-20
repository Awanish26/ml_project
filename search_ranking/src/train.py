"""Train a pairwise XGBoost ranker and compare it with single-signal baselines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from .features import ALL_FEATURES, ATTRIBUTE, LEXICAL, SEMANTIC
from .metrics import mrr_at_k, ndcg_at_k

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PARAMS = dict(
    objective="rank:pairwise",
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="ndcg@10",
    early_stopping_rounds=100,
    random_state=42,
)


def fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, features, **overrides) -> XGBRanker:
    model = XGBRanker(**{**PARAMS, **overrides})
    model.fit(
        train[features], train["label"], qid=train["query_id"],
        eval_set=[(valid[features], valid["label"])], eval_qid=[valid["query_id"]],
        verbose=100,
    )
    return model


def evaluate(test: pd.DataFrame, score_col: str) -> dict:
    return {
        "ndcg@10": round(ndcg_at_k(test, score_col, 10), 4),
        "ndcg@5": round(ndcg_at_k(test, score_col, 5), 4),
        "mrr@10_exact": round(mrr_at_k(test, score_col, 10), 4),
    }


def run(df: pd.DataFrame) -> dict:
    train, valid, test = (df[df.part == p].copy() for p in ("train", "valid", "test"))
    results = {"data": {
        p: {"queries": int(d.query_id.nunique()), "pairs": int(len(d))}
        for p, d in (("train", train), ("valid", valid), ("test", test))
    }}

    # Single-signal baselines: rank each query's candidates by one score.
    test["random"] = np.random.default_rng(0).random(len(test))
    for col in ["random", "bm25_title", "bm25_all", "tfidf_cos", "emb_cos"]:
        results[f"baseline_{col}"] = evaluate(test, col)

    # Feature-group ablations, then the full model.
    variants = {
        "xgb_lexical": LEXICAL,
        "xgb_lexical+attribute": LEXICAL + ATTRIBUTE,
        "xgb_semantic+attribute": SEMANTIC + ATTRIBUTE,
        "xgb_pairwise_all": ALL_FEATURES,
    }
    for name, feats in variants.items():
        model = fit_ranker(train, valid, feats)
        test[name] = model.predict(test[feats])
        results[name] = {**evaluate(test, name), "best_iteration": int(model.best_iteration)}
        if name == "xgb_pairwise_all":
            imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
            imp.sort_values(ascending=False).round(2).to_csv(RESULTS_DIR / "feature_importance_gain.csv", header=["gain"])

    # Same features, listwise objective, for comparison.
    model = fit_ranker(train, valid, ALL_FEATURES, objective="rank:ndcg")
    test["xgb_ndcg_all"] = model.predict(test[ALL_FEATURES])
    results["xgb_ndcg_all"] = {**evaluate(test, "xgb_ndcg_all"), "best_iteration": int(model.best_iteration)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
