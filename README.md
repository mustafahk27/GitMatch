# GitMatch

A deployed recommender system that takes a GitHub username and returns a ranked list of open-source repositories and issues the developer is likely to contribute to — with plain-language explanations for every recommendation.

## Problem

Finding a good first open-source contribution is hard. A developer must browse thousands of repositories and guess which issues match their skills. The "good first issue" label helps, but it has no knowledge of what the person has already built.

## Solution

GitMatch learns from a developer's GitHub history (merged PRs, issues opened, repos contributed to) and recommends:
1. **Repositories** most relevant to their skills and coding patterns
2. **Open issues** within those repositories that match their experience

Each recommendation includes a plain reason: *"You have merged Python PRs in repos tagged `async`, and this issue asks for async work."*

## Architecture

Two-stage pipeline:

| Stage | Task | Model |
|-------|------|-------|
| Stage 1 | Repository recommendation | Two-Tower Neural Network (primary) · Item-based kNN (baseline) |
| Stage 2 | Issue ranking within repos | Sentence Transformers (`all-MiniLM-L6-v2`) |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## Data

- **GH Archive** — hourly dumps of public GitHub events (PR + issue events, 1–3 month window)
- **GitHub REST API** — repository topics, primary language, README text, open issues

See [docs/DATA.md](docs/DATA.md) for the collection and preprocessing pipeline.

## Project Structure

```
GitMatch/
├── data/
│   ├── raw/              # Raw GH Archive + API dumps
│   └── processed/        # Cleaned interaction matrix, embeddings
├── src/
│   ├── data_collection/  # GH Archive downloader, GitHub API client
│   ├── models/           # Two-Tower NN, kNN baseline, popularity baseline
│   ├── recommendation/   # Stage 1 (repos) and Stage 2 (issues) pipelines
│   ├── evaluation/       # Precision@K, Recall@K, NDCG@K, coverage, diversity
│   └── app/              # FastAPI web app
├── notebooks/            # Exploratory analysis and model experiments
├── docs/                 # Architecture and data documentation
├── requirements.txt
└── .gitignore
```

## Evaluation

**Offline metrics:** Precision@K, Recall@K, NDCG@K on held-out contributions — compared across Popularity baseline, Item-based kNN, and Two-Tower NN.  
**Human judgement:** Thumbs up/down on the live app + developer relevance ratings (1–5 scale).

## Deployment

FastAPI app · Docker · Hugging Face Spaces or Render  
Issues are fetched live from the GitHub API so the list stays current.

## Setup

```bash
pip install -r requirements.txt
```

```bash
# Collect data
python src/data_collection/gh_archive.py
python src/data_collection/github_api.py

# Train models
python src/models/two_tower.py

# Run app
uvicorn src.app.main:app --reload
```

## Results

*To be filled in after evaluation.*

| Model | Precision@10 | Recall@10 | NDCG@10 |
|-------|-------------|----------|---------|
| Popularity baseline | — | — | — |
| Item-based kNN | — | — | — |
| Two-Tower NN | — | — | — |
