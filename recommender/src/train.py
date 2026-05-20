"""Train the two-tower model with in-batch sampled softmax, keeping the epoch
with the best validation Hit@10."""
import copy
import time

import numpy as np
import torch

from .evaluate import evaluate_scorer, summarise
from .model import TwoTower, in_batch_softmax_loss


def two_tower_scorer(model: TwoTower, n_items: int, device):
    model.eval()
    with torch.no_grad():
        item_vecs = model.item_vec(torch.arange(n_items + 1, device=device))

    @torch.no_grad()
    def score(hist):
        return model.user_vec(hist) @ item_vecs.T

    return score


def train_two_tower(ds, device, dim=64, hidden=256, epochs=10, batch_size=4096, lr=2e-3,
                    temperature=0.05, weight_decay=1e-5, dropout=0.1, tie_embeddings=False,
                    patience=3, seed=42, log=print):
    torch.manual_seed(seed)
    model = TwoTower(ds.n_items, dim=dim, hidden=hidden, max_len=ds.train_hist.shape[1],
                     dropout=dropout, tie_embeddings=tie_embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    freq = ds.item_counts.astype(np.float64)
    log_q = torch.as_tensor(np.log(freq / freq.sum() + 1e-12), dtype=torch.float32, device=device)
    hist_all = torch.as_tensor(ds.train_hist)
    tgt_all = torch.as_tensor(ds.train_target)
    rng = np.random.default_rng(seed)

    best, best_hit, best_epoch, history = None, -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(tgt_all))
        for s in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[s:s + batch_size])
            hist, tgt = hist_all[idx].to(device), tgt_all[idx].to(device)
            loss = in_batch_softmax_loss(model.user_vec(hist), model.item_vec(tgt), tgt, log_q, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        valid = summarise(evaluate_scorer(two_tower_scorer(model, ds.n_items, device), ds.valid, device))
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), **valid})
        log(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  valid hit@10 {valid['hit@10']:.4f}  "
            f"ndcg@10 {valid['ndcg@10']:.4f}  ({time.time() - t0:.0f}s)")
        if valid["hit@10"] > best_hit:
            best_hit, best, best_epoch = valid["hit@10"], copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            log(f"early stop: no validation gain for {patience} epochs (best epoch {best_epoch})")
            break

    model.load_state_dict(best)
    return model, history
