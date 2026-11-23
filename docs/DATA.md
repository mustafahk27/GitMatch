# Data Pipeline

## Sources

| Source | What we collect | How |
|--------|----------------|-----|
| GH Archive | Public GitHub events (PR open/merge, issue open/comment) | Download hourly `.json.gz` dumps |
| GitHub REST API | Repo topics, primary language, README text, open issues | `GET /repos/{owner}/{repo}`, `GET /repos/{owner}/{repo}/issues` |

---

## Collection Window

1–3 months of GH Archive data. Longer windows give richer interaction history but increase preprocessing time and storage.

---

## Filtering Rules

| Filter | Threshold | Reason |
|--------|-----------|--------|
| Minimum user contributions | ≥ 5 repos | Prevents hopelessly sparse rows |
| Minimum repo contributors | ≥ 10 users | Ensures enough signal for collaborative filtering |
| Remove bot accounts | username ends in `[bot]`, or in known bot list | Noise reduction |
| Remove forks | `fork: true` from API | Forks skew the interaction matrix |
| Remove private repos | Not in GH Archive by definition | — |

---

## Interaction Matrix

After filtering, we construct a user–repo matrix where each cell holds the weighted implicit feedback score:

```
score(u, r) = Σ weight(event_type) for all events by user u on repo r
```

Weights are defined in [ARCHITECTURE.md](ARCHITECTURE.md#implicit-feedback-weights).

Final scores are log-scaled: `log1p(score)`.

---

## Repository Features

For each repository in the matrix:

```python
{
    "repo_id": "owner/name",
    "topics": ["python", "async", "networking"],  # from API
    "language": "Python",
    "readme_embedding": [...],   # all-MiniLM-L6-v2 on README[:512 tokens]
    "topic_embedding": [...],    # all-MiniLM-L6-v2 on joined topics string
}
```

README text is truncated to 512 tokens before embedding. If no README exists, we fall back to the topic string alone.

---

## User Features

For each user:

```python
{
    "user_id": "github_username",
    "contributed_repos": ["owner/repo1", ...],
    "top_languages": {"Python": 12, "JavaScript": 3, ...},  # count of repos
    "top_topics": {"async": 5, "networking": 3, ...},
    "pr_embeddings": [...],   # mean pool of all-MiniLM-L6-v2 on PR titles+descriptions
}
```

---

## Train / Validation / Test Split

Chronological split to simulate real deployment:

| Split | Period | Used for |
|-------|--------|---------|
| Train | Months 1–2 | Model training |
| Validation | First half of month 3 | Hyperparameter tuning |
| Test | Second half of month 3 | Final evaluation |

A contribution in validation/test is a "positive" only if the first contribution to that repo occurs in that split (i.e., a user contributing to a new repo they hadn't touched in training).

---

## Open Issues (Stage 2)

Open issues are **not stored** — they are fetched live from the GitHub API at recommendation time:

```
GET /repos/{owner}/{repo}/issues?state=open&labels=good+first+issue&per_page=50
```

Each issue is embedded on the fly using `all-MiniLM-L6-v2` on `title + "\n" + body[:256 tokens]`.

---

## Reproducibility

All collection scripts, filtering parameters, and random seeds are committed to the repository. The raw data files are excluded from git (`.gitignore`) due to size, but the scripts can re-download them from GH Archive and the GitHub API.
