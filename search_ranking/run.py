"""End-to-end run: load ESCI -> build features -> train rankers -> write results/.

    python -m search_ranking.run --train-queries 12000 --test-queries 3000
"""
import argparse
import json
import time

from search_ranking.src.data import load_pairs
from search_ranking.src.features import CACHE_DIR, build_features
from search_ranking.src.train import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-queries", type=int, default=12_000)
    ap.add_argument("--test-queries", type=int, default=3_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    feat_path = CACHE_DIR / f"esci_features_{args.train_queries}_{args.test_queries}_{args.seed}.parquet"
    if feat_path.exists():
        import pandas as pd
        df = pd.read_parquet(feat_path)
    else:
        pairs = load_pairs(n_train_queries=args.train_queries, n_test_queries=args.test_queries, seed=args.seed)
        print(f"loaded {len(pairs):,} pairs in {time.time() - t0:.0f}s")
        df = build_features(pairs, cache_key=f"esci_{args.train_queries}_{args.test_queries}_{args.seed}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feat_path)
    print(f"features ready in {time.time() - t0:.0f}s")

    results = run(df)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
