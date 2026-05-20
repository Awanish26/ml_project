# ml_project

Two applied-ML projects on public e-commerce data, built around the two problems
that sit at the centre of product discovery: **ranking products for a search
query** and **recommending products from a customer's history**.

| Project | Problem | Model | Data |
|---|---|---|---|
| [`search_ranking/`](search_ranking/) | Rank candidate products for a shopping query | XGBoost learning-to-rank (pairwise) over lexical, semantic and attribute features | Amazon Shopping Queries (ESCI) |
| [`recommender/`](recommender/) | Retrieve the next product a customer will buy | PyTorch two-tower model with in-batch sampled softmax | Amazon Reviews 2023, Home & Kitchen |

Each project folder has its own README with the method, results and how to run it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: XGBoost needs the OpenMP runtime
brew install libomp
```

Raw data goes under `data/raw/` (git-ignored). Download links are in each
project's README.
