# Home-goods recommender (PyTorch two-tower)

Given a customer's purchase/review history, retrieve the product they are most
likely to interact with next, from a 50,000-item Home & Kitchen catalogue. This
is the candidate-retrieval stage of a recommender: embed users and items in the
same space so the top items for any user come from one nearest-neighbour lookup.

## Data

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), Home & Kitchen,
5-core benchmark with the leave-last-out split (`last_out_w_his`). The full set
is 2.92M users, 764K items and 28.2M interactions. For a laptop-sized run:

- catalogue = the 50,000 most-interacted items (57% of all interactions)
- 300,000 users sampled among those whose held-out items are in the catalogue
- 1,054,209 (history -> next item) training examples, history capped at 20 items
- 50,000 of the sampled users scored on validation (second-to-last item) and test (last item)

Download `Home_and_Kitchen.valid.csv.gz` and `Home_and_Kitchen.test.csv.gz` from
`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/`
into `data/raw/amazon_home/`. Each user's training sequence is the history
column of their validation row, so the large train file is not needed.

## Model (`src/model.py`, `src/train.py`)

- **User tower:** the user's last 20 items (item + position embeddings), masked
  mean pooling, residual MLP, L2-normalised. There is no user-id embedding, so the
  tower works for any user with at least one item, including users never seen in training.
- **Item tower:** item embedding (shared with the user tower's item table) and a
  residual MLP, L2-normalised.
- **Loss:** in-batch sampled softmax (batch 4,096) with **logQ correction**, which
  subtracts each item's log sampling probability so popular items are not
  over-penalised as negatives, plus masking of duplicate items within a batch.
- AdamW, cosine learning-rate schedule, early stopping on validation Hit@10.

Hyper-parameters were chosen on the **validation** split only (`sweep.py`,
results in `results/sweep_300000.json`):

| Config | Valid Hit@10 |
|---|---|
| separate item tables, temperature 0.05 | 0.0141 |
| tied item table, temperature 0.05 | 0.0143 |
| tied, temperature 0.1 | 0.0145 |
| **tied, temperature 0.1, weight decay 1e-4, dropout 0.2** | **0.0148** |
| separate tables, temperature 0.1, weight decay 1e-4, dropout 0.2 | 0.0137 |

Tying the tables and softening the temperature fixed the early overfitting seen
in the first run (validation peaked at epoch 2 and then fell).

## Results (50,000 held-out test users, full-catalogue ranking)

Every one of the 50,000 items is scored for every user, items already in the
user's history are excluded, and the rank of the true next item gives Hit@10 and
NDCG@10. No negative sampling in evaluation.

| Model | Hit@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0112 | 0.0060 |
| Item co-visitation (last 3 items, recency-weighted) | 0.0128 | 0.0077 |
| **Two-tower (PyTorch)** | **0.0150** | **0.0080** |

The two-tower model is **+34%** Hit@10 over popularity and **+17%** over
co-visitation. Absolute numbers are low because this is next-item prediction
over 50,000 items on sparse review data, where most users have only a handful of
interactions.

### By history length (Hit@10)

| History length | Users | Popularity | Co-visitation | Two-tower |
|---|---|---|---|---|
| 1-3 items | 16,649 | 0.0116 | 0.0138 | **0.0155** |
| 4-6 | 21,848 | 0.0114 | 0.0134 | **0.0165** |
| 7-10 | 7,330 | 0.0109 | 0.0105 | **0.0127** |
| 11+ | 4,173 | 0.0089 | **0.0101** | 0.0091 |

- **Short histories (cold-ish users, a third of the sample):** the history-pooled
  user tower beats popularity by 34% with as few as 1-3 items, because it needs no
  per-user parameters.
- **Long histories:** mean pooling over 20 items dilutes the most recent intent,
  and co-visitation on the last 3 items wins. A sequence model (self-attention
  over the history, as in SASRec) is the obvious next step.
- **Popularity blend for short histories:** adding a popularity prior to the
  two-tower score for users with 3 or fewer items did not help on validation
  (best weight = 0), so the final model does not use it. Users with no history at
  all fall back to popularity.

## Run

```bash
python -m recommender.run                  # tuned defaults, ~5 minutes on an M-series GPU (MPS)
python -m recommender.sweep --users 300000 # validation-only sweep
```

## Next steps

- Self-attention user tower (SASRec-style) to weight recent items.
- Item-side content features (title, category, price) so new items can be retrieved.
- Train on all 2.92M users; approximate nearest-neighbour index for serving.
