"""
FastAPI web app for GitMatch — lightweight Vercel deployment.

At serve time: pure numpy dot products for Stage 1 (no torch).
Stage 2 uses the HF Inference API for issue embeddings.
Model artifacts are downloaded from HF Hub into /tmp on cold start.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.recommendation.stage2_issues import rank_issues, build_user_profile_embedding
from src.data_collection.github_api import fetch_open_issues, fetch_user_pr_history

DATA_DIR = Path(__file__).parent / "data"

_state: dict = {}


def _load_artifacts():
    repo_index = json.loads((DATA_DIR / "repo_index.json").read_text())
    user_index = json.loads((DATA_DIR / "user_index.json").read_text())

    repo_embs = np.load(str(DATA_DIR / "repo_embeddings.npy")).astype(np.float32)
    norms = np.linalg.norm(repo_embs, axis=1, keepdims=True)

    _state["repo_index"] = repo_index
    _state["user_index"] = user_index
    _state["repo_embeddings"] = repo_embs / np.where(norms > 0, norms, 1)
    _state["ready"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="GitMatch",
    description="GitHub Issues Recommender System",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RecommendRequest(BaseModel):
    username: str
    top_k_repos: int = 5
    top_k_issues: int = 10


class IssueResult(BaseModel):
    number: int
    title: str
    url: str
    labels: list[str]
    relevance_score: float


class RepoResult(BaseModel):
    repo: str
    explanation: str
    issues: list[IssueResult]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _cold_start_features(username: str, repo_index: dict, repo_embs: np.ndarray) -> dict | None:
    """Build a minimal feature dict for a user not seen during training."""
    from src.data_collection.github_api import _get as gh_get
    # Fetch user's public repos
    data = gh_get(f"https://api.github.com/users/{username}/repos", {"per_page": 50, "sort": "pushed"})
    if not data:
        return None
    repo_names = [r["full_name"] for r in data if not r.get("fork")]
    if not repo_names:
        return None

    idx_to_find = [repo_index[r] for r in repo_names if r in repo_index]
    if idx_to_find:
        u_emb = repo_embs[idx_to_find].mean(axis=0)
        norm = np.linalg.norm(u_emb)
        u_emb = u_emb / norm if norm > 0 else u_emb
    else:
        # No overlap with known repos — use mean of all embeddings (popularity fallback)
        u_emb = repo_embs.mean(axis=0)

    langs = [r.get("language") for r in data if r.get("language")]
    from collections import Counter
    top_langs = [l for l, _ in Counter(langs).most_common(3)]

    pr_titles, pr_descriptions = fetch_user_pr_history(username)

    return {
        "contributed_repo_ids": idx_to_find,
        "top_languages": top_langs,
        "top_topics": [],
        "user_embedding": u_emb.tolist(),
        "pr_titles": pr_titles,
        "pr_descriptions": pr_descriptions,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _state.get("ready", False)}


@app.post("/recommend", response_model=list[RepoResult])
def recommend(request: RecommendRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Models not yet loaded. Check /health.")

    username = request.username
    repo_index = _state["repo_index"]
    repo_embs = _state["repo_embeddings"]

    feat = _cold_start_features(username, repo_index, repo_embs)
    if feat is None:
        raise HTTPException(status_code=404, detail=f"GitHub user '{username}' not found or has no public contributions.")

    # Stage 1 — numpy dot product against pre-normalised repo embeddings
    if "user_embedding" in feat:
        u_emb = np.array(feat["user_embedding"], dtype=np.float32)
    else:
        # Cold-start: use mean of contributed repo embeddings
        contrib_ids = feat.get("contributed_repo_ids", [])
        if contrib_ids:
            u_emb = repo_embs[contrib_ids].mean(axis=0)
        else:
            raise HTTPException(status_code=404, detail="No contribution history found.")

    scores = repo_embs @ u_emb
    exclude = set(feat.get("contributed_repo_ids", []))
    idx_to_repo = {v: k for k, v in repo_index.items()}

    ranked_idxs = np.argsort(scores)[::-1]
    top_repos = []
    for idx in ranked_idxs:
        if int(idx) in exclude:
            continue
        repo_name = idx_to_repo.get(int(idx), "")
        langs = feat.get("top_languages", [])
        topics = feat.get("top_topics", [])
        explanation_parts = []
        if langs:
            explanation_parts.append(f"languages: {', '.join(langs[:3])}")
        if topics:
            explanation_parts.append(f"topics: {', '.join(topics[:3])}")
        explanation = (
            f"Matches your profile ({'; '.join(explanation_parts)})."
            if explanation_parts else "Recommended based on your contribution history."
        )
        top_repos.append({"repo": repo_name, "explanation": explanation})
        if len(top_repos) >= request.top_k_repos:
            break

    # Stage 2 — rank open issues within each repo via HF Inference API
    user_emb = build_user_profile_embedding(
        feat.get("pr_titles", []),
        feat.get("pr_descriptions", []),
    )

    results = []
    for rec in top_repos:
        raw_issues = fetch_open_issues(rec["repo"])
        ranked = rank_issues(raw_issues, user_emb, top_k=request.top_k_issues)
        results.append(RepoResult(
            repo=rec["repo"],
            explanation=rec["explanation"],
            issues=[IssueResult(**i) for i in ranked],
        ))

    return results
