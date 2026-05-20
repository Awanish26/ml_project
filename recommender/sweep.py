"""Hyper-parameter sweep scored on the VALIDATION split only (test is never
touched here). Results go to results/sweep.json.

    python -m recommender.sweep --users 300000
"""
import argparse
import json
from pathlib import Path

import torch

from recommender.src.data import load
from recommender.src.train import train_two_tower

CONFIGS = [
    dict(name="untied_t0.05", tie_embeddings=False, temperature=0.05),
    dict(name="tied_t0.05", tie_embeddings=True, temperature=0.05),
    dict(name="tied_t0.1", tie_embeddings=True, temperature=0.1),
    dict(name="tied_t0.1_wd1e-4_do0.2", tie_embeddings=True, temperature=0.1, weight_decay=1e-4, dropout=0.2),
    dict(name="untied_t0.1_wd1e-4_do0.2", tie_embeddings=False, temperature=0.1, weight_decay=1e-4, dropout=0.2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300_000)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = load(n_users=args.users)
    out = Path(__file__).resolve().parent / "results" / f"sweep_{args.users}.json"
    results = {}
    for cfg in CONFIGS:
        cfg = dict(cfg)
        name = cfg.pop("name")
        print(f"--- {name}")
        _, history = train_two_tower(ds, device, epochs=args.epochs, **cfg)
        best = max(history, key=lambda h: h["hit@10"])
        results[name] = {"config": cfg, "best_valid": best, "epochs_run": len(history)}
        out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v["best_valid"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
