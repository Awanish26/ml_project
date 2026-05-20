"""End-to-end run: load data -> baselines -> train two-tower -> cold-start
analysis -> write results/metrics.json.

    python -m recommender.run --users 300000 --items 50000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from recommender.src.data import load
from recommender.src.evaluate import (covisitation_matrix, covisitation_scorer, evaluate_scorer,
                                      popularity_scorer, summarise)
from recommender.src.train import train_two_tower, two_tower_scorer

RESULTS = Path(__file__).resolve().parent / "results"
BUCKETS = {"1-3": (1, 3), "4-6": (4, 6), "7-10": (7, 10), "11+": (11, 10**9)}
SHORT_HISTORY = 3


def by_bucket(metrics, hist_len):
    return {name: summarise(metrics, (hist_len >= lo) & (hist_len <= hi)) for name, (lo, hi) in BUCKETS.items()}


def blended_scorer(base_score, item_counts, device, lam):
    """Cold-start fallback: for short histories, add a popularity prior
    (z-scored log counts) to the two-tower score."""
    lp = np.log1p(item_counts.astype(np.float64))
    prior = torch.as_tensor((lp - lp.mean()) / lp.std(), dtype=torch.float32, device=device)

    def score(hist):
        s = base_score(hist)
        short = ((hist > 0).sum(1) <= SHORT_HISTORY).float().unsqueeze(1)
        return s + lam * short * prior.unsqueeze(0)

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300_000)
    ap.add_argument("--items", type=int, default=50_000)
    ap.add_argument("--eval-users", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=12)
    # Defaults below were chosen on the validation split by recommender/sweep.py.
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--untied", action="store_true", help="separate history and candidate item tables")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    t0 = time.time()
    ds = load(n_users=args.users, n_items=args.items, n_eval_users=args.eval_users, seed=args.seed)
    info = {"users_sampled": args.users, "items": ds.n_items, "train_examples": int(len(ds.train_target)),
            "valid_users": int(len(ds.valid["target"])), "test_users": int(len(ds.test["target"])),
            "eval_users_short_history_le3": int((ds.test["full_hist_len"] <= 3).sum())}
    print(json.dumps(info), f"loaded in {time.time() - t0:.0f}s")

    results = {"data": info}
    pop = popularity_scorer(ds.item_counts, device)
    covis = covisitation_scorer(covisitation_matrix(ds.train_seqs, ds.n_items), ds.item_counts, device)
    for name, fn in (("popularity", pop), ("covisitation", covis)):
        m = evaluate_scorer(fn, ds.test, device, batch=512)
        results[name] = {**summarise(m), "by_history_length": by_bucket(m, ds.test["hist_len"])}
        print(name, results[name]["hit@10"], results[name]["ndcg@10"])

    model, history = train_two_tower(ds, device, dim=args.dim, epochs=args.epochs, seed=args.seed,
                                     temperature=args.temperature, weight_decay=args.weight_decay,
                                     dropout=args.dropout, tie_embeddings=not args.untied)
    results["two_tower_config"] = {"dim": args.dim, "temperature": args.temperature, "weight_decay": args.weight_decay,
                                   "dropout": args.dropout, "tie_embeddings": not args.untied}
    results["two_tower_training"] = history
    tt = two_tower_scorer(model, ds.n_items, device)
    m_tt = evaluate_scorer(tt, ds.test, device)
    results["two_tower"] = {**summarise(m_tt), "by_history_length": by_bucket(m_tt, ds.test["hist_len"])}

    # Tune the cold-start blend weight on validation users with short histories only.
    short_valid = ds.valid["hist_len"] <= SHORT_HISTORY
    grid = {}
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        mv = evaluate_scorer(blended_scorer(tt, ds.item_counts, device, lam), ds.valid, device)
        grid[lam] = summarise(mv, short_valid)["hit@10"]
    best_lam = max(grid, key=grid.get)
    m_bl = evaluate_scorer(blended_scorer(tt, ds.item_counts, device, best_lam), ds.test, device)
    results["two_tower_cold_start_fallback"] = {
        "lambda": best_lam, "valid_short_history_grid": grid,
        **summarise(m_bl), "by_history_length": by_bucket(m_bl, ds.test["hist_len"]),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("hit@10", "ndcg@10")} for k, v in results.items()
                      if isinstance(v, dict) and "hit@10" in v}, indent=2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
