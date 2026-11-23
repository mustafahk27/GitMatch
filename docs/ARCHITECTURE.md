# Architecture

## Overview

GitMatch uses a two-stage recommendation pipeline. Stage 1 retrieves candidate repositories; Stage 2 ranks open issues within those candidates.

```
GitHub Username
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1 — Repository Retrieval     │
│                                     │
│  User Tower  ◄──── user history     │
│      │                              │
│      ▼  dot product                 │
│  Repo Tower  ◄──── repo embeddings  │
│                                     │
│  → Top-K repositories               │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 2 — Issue Ranking            │
│                                     │
│  User profile embedding             │
│      │  cosine similarity           │
│  Issue text embeddings              │
│                                     │
│  → Ranked open issues per repo      │
└─────────────────────────────────────┘
      │
      ▼
  Results + Explanations
```

---

## Stage 1 — Repository Recommendation

### Primary Model: Two-Tower Neural Network

The Two-Tower architecture encodes users and repositories into a shared latent space. The affinity score is the dot product of the two towers' output embeddings.

**User Tower inputs:**
- Interaction history (weighted implicit feedback scores per repo)
- Preferred programming languages (one-hot or embedding)
- Repository topics from past contributions (mean-pooled)

**Repo Tower inputs:**
- Dense text embedding of README + topics via `all-MiniLM-L6-v2`
- Primary language (one-hot or embedding)
- Activity signals (star count, contributor count — log-scaled)

**Training objective:** Binary cross-entropy on positive (contributed) vs. sampled negative (not contributed) user–repo pairs.

**Cold-start handling:** New users with no history fall back to repo tower similarity against a generic "beginner" profile. New repos are handled by their text embedding alone.

### Baseline: Item-based kNN

Cosine similarity on the user–repo interaction matrix. Used as the comparison baseline in evaluation. Explanation: *"Recommended because contributors to Repo A and Repo B — which you've contributed to — also contributed to this."*

### Baseline: Popularity

Most-contributed-to repositories. Every model must beat this.

---

## Stage 2 — Issue Ranking

### Model: Sentence Transformers

For each candidate repository from Stage 1, open issues are ranked by cosine similarity between:
- **Issue embedding** — `all-MiniLM-L6-v2` encoding of the issue title + body + labels
- **User profile embedding** — mean pool of `all-MiniLM-L6-v2` encodings of the user's merged PR titles and descriptions

This overcomes the vocabulary mismatch of TF-IDF: a user whose PRs mention `async/await` will score highly on an issue about `concurrency bugs` even without exact term overlap.

---

## Implicit Feedback Weights

There are no explicit ratings on GitHub, so we construct a weighted implicit score per user–repo pair:

| Event | Weight |
|-------|--------|
| PR merged | 5 |
| PR opened (not merged) | 2 |
| Issue opened | 1 |
| Issue comment | 0.5 |

The final score is the sum of weighted events, clipped and log-scaled before training.

---

## Explanation Layer

Every recommendation exposes its reasoning:

- **Stage 1 (NN):** Top overlapping topics and languages between the user's history and the repo's metadata. Example: *"Your past repos use Python + asyncio; this repo is tagged async, networking."*
- **Stage 1 (kNN fallback):** Nearest neighbour repos the user already contributed to.
- **Stage 2:** Top terms from the issue that overlap with or are semantically close to the user's PR history.

---

## Model Serialization

Trained models are saved as small files for fast serving:
- Two-Tower weights: `data/processed/two_tower.pt`
- Repo embeddings (pre-computed): `data/processed/repo_embeddings.npy`
- kNN similarity matrix: `data/processed/knn_similarity.npz`

Open issues are fetched live from the GitHub API at request time so the list is always current.
