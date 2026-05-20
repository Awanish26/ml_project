"""Two-tower retrieval model.

User tower: the user's recent items (id + position embeddings), masked mean
pooling, then an MLP. Because the user is represented by their history rather
than a user-id embedding, the same tower serves users it never saw in training.
Item tower: item-id embedding followed by an MLP. Both towers output
L2-normalised vectors, so the score is a cosine similarity."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, hidden: int = 256, max_len: int = 20,
                 dropout: float = 0.1, tie_embeddings: bool = False):
        super().__init__()
        self.hist_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.user_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        # Tying shares one item table between history and candidates: half the
        # parameters, and every training example updates the candidate table too.
        self.item_emb = self.hist_emb if tie_embeddings else nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.item_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        nn.init.normal_(self.hist_emb.weight, std=0.02)
        if not tie_embeddings:
            nn.init.normal_(self.item_emb.weight, std=0.02)

    def user_vec(self, hist: torch.Tensor) -> torch.Tensor:
        """hist: [B, L] item ids, left-padded with 0."""
        mask = (hist > 0).unsqueeze(-1).float()
        pos = torch.arange(hist.size(1), device=hist.device).unsqueeze(0)
        e = self.emb_dropout(self.hist_emb(hist) + self.pos_emb(pos)) * mask
        pooled = e.sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(pooled + self.user_mlp(pooled), dim=-1)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        e = self.item_emb(items)
        return F.normalize(e + self.item_mlp(e), dim=-1)


def in_batch_softmax_loss(user: torch.Tensor, item: torch.Tensor, target_ids: torch.Tensor,
                          log_q: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sampled softmax over the other positives in the batch.

    Popular items appear as in-batch negatives more often than rare ones, so we
    subtract log(sampling probability) from every logit (logQ correction).
    Duplicate items in a batch are masked so a user's true item is never
    treated as its own negative."""
    logits = user @ item.T / temperature - log_q[target_ids].unsqueeze(0)
    same = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)
    logits = logits.masked_fill(same & ~torch.eye(len(target_ids), dtype=torch.bool, device=user.device), float("-inf"))
    labels = torch.arange(len(target_ids), device=user.device)
    return F.cross_entropy(logits, labels)
