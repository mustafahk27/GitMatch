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
    lang_index = json.loads((DATA_DIR / "lang_index.json").read_text())

    repo_embs = np.load(str(DATA_DIR / "repo_embeddings.npy")).astype(np.float32)
    norms = np.linalg.norm(repo_embs, axis=1, keepdims=True)
    repo_lang_ids = np.load(str(DATA_DIR / "repo_lang_ids.npy"))

    eval_path = DATA_DIR / "eval_results.json"
    eval_results = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    _state["repo_index"] = repo_index
    _state["user_index"] = user_index
    _state["lang_index"] = lang_index
    _state["repo_lang_ids"] = repo_lang_ids
    _state["repo_embeddings"] = repo_embs / np.where(norms > 0, norms, 1)
    _state["eval_results"] = eval_results
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


def _cold_start_features(username: str, repo_index: dict, repo_embs: np.ndarray, lang_index: dict = None, repo_lang_ids=None) -> dict | None:
    """Build a minimal feature dict for a user not seen during training."""
    from src.data_collection.github_api import _get as gh_get
    from collections import Counter

    # Fetch user's public repos
    data = gh_get(f"https://api.github.com/users/{username}/repos", {"per_page": 50, "sort": "pushed"})
    if not data:
        return None
    repo_names = [r["full_name"] for r in data if not r.get("fork")]
    if not repo_names:
        return None

    owned_ids = [repo_index[r] for r in repo_names if r in repo_index]
    langs = [r.get("language") for r in data if r.get("language")]
    top_langs = [l for l, _ in Counter(langs).most_common(3)]

    pr_titles, pr_descriptions, pr_repos = fetch_user_pr_history(username)
    pr_repo_ids = [repo_index[r] for r in pr_repos if r in repo_index]

    # Also check starred repos — most likely to overlap with training data
    starred = gh_get(f"https://api.github.com/users/{username}/starred", {"per_page": 50}) or []
    star_names = [r["full_name"] for r in starred]
    star_ids = [repo_index[r] for r in star_names if r in repo_index]

    all_ids = list(set(owned_ids + pr_repo_ids + star_ids))

    if all_ids:
        u_emb = repo_embs[all_ids].mean(axis=0)
        norm = np.linalg.norm(u_emb)
        u_emb = u_emb / norm if norm > 0 else u_emb
    elif top_langs and lang_index is not None and repo_lang_ids is not None:
        # Language-based fallback: mean-pool embeddings of repos matching the user's top languages
        lang_ids_for_user = [lang_index[l] for l in top_langs if l in lang_index]
        if lang_ids_for_user:
            mask = np.isin(repo_lang_ids, lang_ids_for_user)
            lang_embs = repo_embs[mask]
            if len(lang_embs) > 0:
                u_emb = lang_embs.mean(axis=0)
                norm = np.linalg.norm(u_emb)
                u_emb = u_emb / norm if norm > 0 else u_emb
            else:
                u_emb = repo_embs.mean(axis=0)
        else:
            u_emb = repo_embs.mean(axis=0)
    else:
        u_emb = repo_embs.mean(axis=0)

    # Build signal description for explanations
    signal_parts = []
    if pr_repo_ids:
        signal_parts.append(f"contributed to {len(pr_repo_ids)} matched repos")
    if star_ids:
        signal_parts.append(f"starred {len(star_ids)} matched repos")
    if not signal_parts and top_langs:
        signal_parts.append(f"language profile ({', '.join(top_langs[:2])})")

    return {
        "contributed_repo_ids": all_ids,
        "top_languages": top_langs,
        "top_topics": [],
        "user_embedding": u_emb.tolist(),
        "pr_titles": pr_titles,
        "pr_descriptions": pr_descriptions,
        "signal_parts": signal_parts,
        "pr_repos": pr_repos[:5],
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _state.get("ready", False)}


@app.get("/metrics")
def metrics():
    return _state.get("eval_results", {})


@app.post("/recommend", response_model=list[RepoResult])
def recommend(request: RecommendRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Models not yet loaded. Check /health.")

    username = request.username
    repo_index = _state["repo_index"]
    repo_embs = _state["repo_embeddings"]

    feat = _cold_start_features(username, repo_index, repo_embs,
                                lang_index=_state.get("lang_index"),
                                repo_lang_ids=_state.get("repo_lang_ids"))
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
    langs = feat.get("top_languages", [])
    signal_parts = feat.get("signal_parts", [])
    pr_repos = feat.get("pr_repos", [])

    top_repos = []
    for idx in ranked_idxs:
        if int(idx) in exclude:
            continue
        repo_name = idx_to_repo.get(int(idx), "")
        # Build a richer explanation
        exp_parts = []
        if pr_repos:
            short = [r.split("/")[-1] for r in pr_repos[:2]]
            exp_parts.append(f"similar to repos you've contributed to ({', '.join(short)})")
        elif signal_parts:
            exp_parts.extend(signal_parts)
        if langs:
            exp_parts.append(f"matches your {'/'.join(langs[:2])} profile")
        explanation = (
            "Recommended because " + "; ".join(exp_parts) + "."
            if exp_parts else "Recommended based on your GitHub activity."
        )
        top_repos.append({"repo": repo_name, "explanation": explanation})
        if len(top_repos) >= request.top_k_repos:
            break

    # Stage 2 — rank open issues within each repo via HF Inference API
    try:
        user_emb = build_user_profile_embedding(
            feat.get("pr_titles", []),
            feat.get("pr_descriptions", []),
        )
        stage2_ok = True
    except Exception:
        user_emb = None
        stage2_ok = False

    results = []
    for rec in top_repos:
        if stage2_ok and user_emb is not None:
            try:
                raw_issues = fetch_open_issues(rec["repo"])
                ranked = rank_issues(raw_issues, user_emb, top_k=request.top_k_issues)
                issue_results = [IssueResult(**i) for i in ranked]
            except Exception:
                issue_results = []
        else:
            issue_results = []
        results.append(RepoResult(
            repo=rec["repo"],
            explanation=rec["explanation"],
            issues=issue_results,
        ))

    return results
