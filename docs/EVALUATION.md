# Evaluation

## Offline Metrics

Evaluated on the held-out test split (chronological). A "hit" at rank K means the user's actual new contribution appears in the top-K recommendations.

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K recommendations that are actual contributions |
| Recall@K | Fraction of actual contributions captured in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain — rewards correct recommendations ranked higher |
| Catalogue coverage | Fraction of all repositories that appear in at least one recommendation |
| Diversity | Intra-list diversity (mean pairwise distance between recommended repos) |

K values: 5, 10, 20.

## Models Compared

| Model | Type |
|-------|------|
| Popularity baseline | Non-personalized |
| Item-based kNN | Collaborative filtering |
| Two-Tower NN | Neural (primary model) |

Every model must beat the Popularity baseline to be considered useful.

## Human Judgement

The live app collects two signals:
1. **Thumbs up / thumbs down** on each recommendation + explanation
2. **Relevance and explanation clarity ratings** (1–5 scale) collected from a group of developers

Human evaluation covers both recommendation quality and explanation quality separately.

## Results

*To be filled in after experiments.*

| Model | P@10 | R@10 | NDCG@10 | Coverage | Diversity |
|-------|------|------|---------|----------|-----------|
| Popularity | — | — | — | — | — |
| Item-based kNN | — | — | — | — | — |
| Two-Tower NN | — | — | — | — | — |
