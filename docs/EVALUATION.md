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

Evaluated on 3,274 users, 1,810 repositories, 8,855 interactions (avg 2.7 interactions/user).
Test split: 15% per user held out randomly. Metrics averaged across all test users.

| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | P@20 | R@20 | NDCG@20 | Coverage |
|-------|-----|-----|--------|------|------|---------|------|------|---------|----------|
| Popularity | 0.0026 | 0.0120 | 0.0074 | 0.0020 | 0.0186 | 0.0094 | 0.0020 | 0.0372 | 0.0142 | 0.0149 |
| Item-based kNN | **0.0089** | **0.0398** | **0.0283** | **0.0056** | **0.0480** | **0.0311** | **0.0035** | **0.0594** | **0.0340** | **0.3669** |
| Two-Tower NN | 0.0032 | 0.0148 | 0.0093 | 0.0029 | 0.0277 | 0.0133 | 0.0021 | 0.0398 | 0.0164 | 0.0232 |

**Observations:**
- Item-kNN beats both baselines across all metrics, outperforming Two-Tower NN 3–5× on NDCG@K.
- This is expected for a highly sparse dataset (avg 2.7 interactions/user, min 2). Neural models typically need denser interaction data to outperform neighbourhood methods.
- Item-kNN achieves significantly higher catalogue coverage (36.7% vs 2.3% for Two-Tower), indicating the neural model tends to recommend popular/central repositories.
- Popularity baseline scores lowest, confirming personalisation adds real value.
- Absolute metric values are low due to extreme sparsity; the relative ordering (kNN > Two-Tower > Popularity) is the meaningful signal.
