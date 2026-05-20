"""Amazon Reviews 2023, Home & Kitchen (5-core, leave-last-out split).

The benchmark ships one row per user in `valid` and `test`:
  valid: target = the user's second-to-last item, history = all earlier items
  test:  target = the user's last item,           history = earlier items + the valid item
so a user's training sequence is exactly the valid-row history.

For a laptop-sized run we keep the most-interacted items (counted over the full
dataset), sample users whose held-out items fall inside that catalogue, and turn
every training sequence into (history -> next item) examples.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "amazon_home"
PAD = 0


@dataclass
class Dataset:
    n_items: int                 # item ids are 1..n_items, 0 is padding
    item_ids: np.ndarray         # index -> parent_asin (index 0 = "<pad>")
    train_hist: np.ndarray       # [N, L] left-padded histories for training
    train_target: np.ndarray     # [N]
    item_counts: np.ndarray      # [n_items + 1] training interaction counts
    train_seqs: list             # per-user training sequence (for baselines)
    valid: dict                  # {"hist": [U, L], "target": [U], "hist_len": [U], "full_hist_len": [U]}
    test: dict


def _left_pad(seqs, max_len):
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s):
            out[i, -len(s):] = s
    return out


def load(n_users: int = 300_000, n_items: int = 50_000, max_len: int = 20,
         n_eval_users: int = 50_000, seed: int = 42) -> Dataset:
    """Build a dense laptop-sized subset: keep the `n_items` most-interacted
    items (counted over the full dataset), then sample users whose valid and
    test targets are both in that catalogue."""
    valid = pd.read_csv(RAW_DIR / "Home_and_Kitchen.valid.csv.gz", usecols=["user_id", "parent_asin", "history"])
    test = pd.read_csv(RAW_DIR / "Home_and_Kitchen.test.csv.gz", usecols=["user_id", "parent_asin", "history"])
    test = test.set_index("user_id").loc[valid.user_id].reset_index()

    raw_train = [h.split() if isinstance(h, str) else [] for h in valid.history]
    counts = pd.Series([i for s in raw_train for i in s]).value_counts()
    counts = counts.add(valid.parent_asin.value_counts(), fill_value=0).add(test.parent_asin.value_counts(), fill_value=0)
    vocab = counts.nlargest(n_items).index.to_numpy()
    index = {asin: i + 1 for i, asin in enumerate(vocab)}

    train_seqs_all = [[index[i] for i in s if i in index] for s in raw_train]
    eligible = np.flatnonzero(
        valid.parent_asin.map(index).notna().to_numpy()
        & test.parent_asin.map(index).notna().to_numpy()
        & np.array([len(s) > 0 for s in train_seqs_all])
    )
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))
    train_seqs = [train_seqs_all[i] for i in chosen]

    hists, targets = [], []
    for s in train_seqs:
        for t in range(1, len(s)):
            hists.append(s[max(0, t - max_len):t])
            targets.append(s[t])
    item_counts = np.zeros(n_items + 1, dtype=np.int64)
    for s in train_seqs:
        np.add.at(item_counts, s, 1)

    # Evaluate on a fixed random subset of the sampled users.
    eval_pos = np.sort(rng.choice(len(chosen), size=min(n_eval_users, len(chosen)), replace=False))
    eval_rows = chosen[eval_pos]

    def split(targets_asin, hist_seqs):
        return {
            "hist": _left_pad(hist_seqs, max_len),
            "target": np.array([index[a] for a in targets_asin], dtype=np.int64),
            "hist_len": np.array([min(len(h), max_len) for h in hist_seqs]),
            "full_hist_len": np.array([len(h) for h in hist_seqs]),
        }

    valid_hist = [train_seqs_all[i] for i in eval_rows]
    test_hist = [train_seqs_all[i] + [index[valid.parent_asin.iat[i]]] for i in eval_rows]
    return Dataset(
        n_items=n_items,
        item_ids=np.concatenate([["<pad>"], vocab]),
        train_hist=_left_pad(hists, max_len),
        train_target=np.array(targets, dtype=np.int64),
        item_counts=item_counts,
        train_seqs=train_seqs,
        valid=split(valid.parent_asin.to_numpy()[eval_rows], valid_hist),
        test=split(test.parent_asin.to_numpy()[eval_rows], test_hist),
    )
