"""Full-catalogue ranking evaluation (no negative sampling).

For each held-out user every item in the catalogue is scored, items already in
the user's history are excluded, and the rank of the single target item gives
Hit@K and NDCG@K."""
import numpy as np
import torch
from scipy import sparse


def rank_metrics(scores: torch.Tensor, hist: torch.Tensor, target: torch.Tensor, k: int = 10) -> dict:
    scores = scores.clone()
    scores[:, 0] = float("-inf")                                  # padding id
    scores.scatter_(1, hist, float("-inf"))                       # seen items
    target_score = scores.gather(1, target.unsqueeze(1))
    rank = (scores > target_score).sum(1)                         # 0-based
    hit = (rank < k).float()
    ndcg = hit / torch.log2(rank.float() + 2)
    return {"hit": hit.cpu().numpy(), "ndcg": ndcg.cpu().numpy(), "rank": rank.cpu().numpy()}


def evaluate_scorer(score_fn, split: dict, device, batch: int = 2048, k: int = 10) -> dict:
    out = {"hit": [], "ndcg": [], "rank": []}
    for s in range(0, len(split["target"]), batch):
        hist = torch.as_tensor(split["hist"][s:s + batch], device=device)
        target = torch.as_tensor(split["target"][s:s + batch], device=device)
        m = rank_metrics(score_fn(hist), hist, target, k)
        for key in out:
            out[key].append(m[key])
    return {key: np.concatenate(v) for key, v in out.items()}


def summarise(m: dict, mask=None) -> dict:
    mask = np.ones_like(m["hit"], dtype=bool) if mask is None else mask
    return {"users": int(mask.sum()), "hit@10": round(float(m["hit"][mask].mean()), 4),
            "ndcg@10": round(float(m["ndcg"][mask].mean()), 4)}


def popularity_scorer(item_counts: np.ndarray, device):
    pop = torch.as_tensor(np.log1p(item_counts), dtype=torch.float32, device=device)
    return lambda hist: pop.unsqueeze(0).expand(hist.size(0), -1)


def covisitation_matrix(train_seqs, n_items: int, window: int = 3) -> sparse.csr_matrix:
    """Item-to-item counts of 'bought j within `window` steps after i'."""
    rows, cols = [], []
    for s in train_seqs:
        for i, a in enumerate(s):
            for b in s[i + 1:i + 1 + window]:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    m = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_items + 1, n_items + 1))
    return m.tocsr()


def covisitation_scorer(covis: sparse.csr_matrix, item_counts: np.ndarray, device, last_n: int = 3):
    """Sum co-visitation rows of the user's last `last_n` items (recency-weighted),
    with a tiny popularity term to break ties."""
    pop = np.log1p(item_counts).astype(np.float32) * 1e-3

    def score(hist: torch.Tensor) -> torch.Tensor:
        h = hist[:, -last_n:].cpu().numpy()
        weights = np.array([0.5 ** (last_n - 1 - j) for j in range(last_n)], dtype=np.float32)
        out = np.tile(pop, (h.shape[0], 1))
        for j in range(last_n):
            rows = covis[h[:, j]].toarray()
            rows[h[:, j] == 0] = 0.0
            out += weights[j] * np.log1p(rows)
        return torch.as_tensor(out, device=device)

    return score
