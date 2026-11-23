"""
FastAPI web app for GitMatch — lightweight Vercel deployment.

At serve time: pure numpy dot products for Stage 1 (no torch).
Stage 2 uses the HF Inference API for issue embeddings.
Model artifacts are downloaded from HF Hub into /tmp on cold start.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.recommendation.stage2_issues import rank_issues, build_user_profile_embedding, _tfidf_matrix
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

    desc_path = DATA_DIR / "repo_desc_index.json"
    repo_desc_index = json.loads(desc_path.read_text()) if desc_path.exists() else {}

    _state["repo_index"] = repo_index
    _state["user_index"] = user_index
    _state["lang_index"] = lang_index
    _state["repo_lang_ids"] = repo_lang_ids
    _state["repo_embeddings"] = repo_embs / np.where(norms > 0, norms, 1)
    _state["eval_results"] = eval_results
    _state["repo_desc_index"] = repo_desc_index
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


def _tfidf_match_repos(user_text: str, repo_desc_index: dict, repo_index: dict, top_n: int = 40) -> list[int]:
    """Return indices of training repos most similar to user_text via TF-IDF."""
    candidates = [(name, repo_index[name]) for name in repo_desc_index if name in repo_index]
    if not candidates or not user_text.strip():
        return []
    names, idxs = zip(*candidates)
    repo_texts = [repo_desc_index[n] for n in names]
    all_docs = list(repo_texts) + [user_text]
    matrix, _ = _tfidf_matrix(all_docs)
    scores = matrix[:-1] @ matrix[-1]
    top = np.argsort(scores)[::-1][:top_n]
    return [idxs[i] for i in top if scores[i] > 0]


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

    # Build a rich text profile from the user's PRs and their own repo descriptions
    own_repo_texts = [
        f"{r.get('description', '')} {r.get('language', '')}"
        for r in data if not r.get("fork")
    ]
    user_text = " ".join(pr_titles + pr_descriptions + own_repo_texts)

    if all_ids:
        u_emb = repo_embs[all_ids].mean(axis=0)
        norm = np.linalg.norm(u_emb)
        u_emb = u_emb / norm if norm > 0 else u_emb
        signal_source = "direct"
    else:
        # TF-IDF fallback: match user text against stored repo descriptions
        repo_desc_index = _state.get("repo_desc_index", {})
        tfidf_ids = _tfidf_match_repos(user_text, repo_desc_index, repo_index, top_n=40)
        if tfidf_ids:
            u_emb = repo_embs[tfidf_ids].mean(axis=0)
            norm = np.linalg.norm(u_emb)
            u_emb = u_emb / norm if norm > 0 else u_emb
            all_ids = tfidf_ids
            signal_source = "tfidf"
        elif top_langs and lang_index is not None and repo_lang_ids is not None:
            # Language-based fallback
            lang_ids_for_user = [lang_index[l] for l in top_langs if l in lang_index]
            if lang_ids_for_user:
                mask = np.isin(repo_lang_ids, lang_ids_for_user)
                lang_embs = repo_embs[mask]
                u_emb = lang_embs.mean(axis=0) if len(lang_embs) > 0 else repo_embs.mean(axis=0)
                norm = np.linalg.norm(u_emb)
                u_emb = u_emb / norm if norm > 0 else u_emb
            else:
                u_emb = repo_embs.mean(axis=0)
            signal_source = "language"
        else:
            u_emb = repo_embs.mean(axis=0)
            signal_source = "global"

    # Build signal description for explanations
    signal_parts = []
    if pr_repo_ids:
        signal_parts.append(f"contributed to {len(pr_repo_ids)} matched repos")
    if star_ids:
        signal_parts.append(f"starred {len(star_ids)} matched repos")
    if not signal_parts and signal_source == "tfidf":
        signal_parts.append(f"matches your project interests ({', '.join(top_langs[:2])})")
    elif not signal_parts and top_langs:
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


_BORING = {
    "that", "this", "with", "from", "have", "been", "will", "your", "their",
    "which", "also", "more", "into", "than", "other", "used", "using", "based",
    "open", "source", "code", "data", "work", "project", "tool", "library",
    "system", "platform", "application", "build", "support", "provide", "include",
    "make", "allow", "help", "easy", "simple", "new", "you", "not", "repository",
    "repo", "github", "package", "interface", "implement", "feature", "version",
    "user", "users", "file", "files", "list", "type", "types", "create", "object",
}

import re as _re


def _per_repo_explanation(repo_name: str, repo_desc_index: dict, top_langs: list[str], user_text: str) -> str:
    repo_text = repo_desc_index.get(repo_name, "")
    if not repo_text:
        return "Recommended based on your GitHub activity."

    repo_tokens = set(_re.findall(r'[a-z][a-z0-9]{3,}', repo_text.lower())) - _BORING
    user_tokens = set(_re.findall(r'[a-z][a-z0-9]{3,}', user_text.lower())) - _BORING

    lang_hits = [l for l in top_langs if l.lower() in repo_text.lower()]
    shared = sorted(repo_tokens & user_tokens, key=len, reverse=True)[:3]
    notable = sorted(repo_tokens - _BORING, key=len, reverse=True)[:2]

    if lang_hits and shared:
        return f"Matches your {'/'.join(lang_hits[:2])} work — {', '.join(shared[:2])}."
    if lang_hits:
        extra = f" — {', '.join(notable)}" if notable else ""
        return f"Aligns with your {'/'.join(lang_hits[:2])} projects{extra}."
    if shared:
        return f"Related to your interests in {', '.join(shared[:3])}."
    if notable:
        return f"Recommended based on your activity — {', '.join(notable)}."
    return "Recommended based on your GitHub activity."


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
    pr_repos = feat.get("pr_repos", [])
    repo_desc_index = _state.get("repo_desc_index", {})

    # Build user text for per-repo explanation matching
    user_text_for_exp = " ".join(
        feat.get("pr_titles", []) + feat.get("pr_descriptions", [])
    )

    top_repos = []
    for idx in ranked_idxs:
        if int(idx) in exclude:
            continue
        repo_name = idx_to_repo.get(int(idx), "")
        explanation = _per_repo_explanation(repo_name, repo_desc_index, langs, user_text_for_exp)
        top_repos.append({"repo": repo_name, "explanation": explanation})
        if len(top_repos) >= request.top_k_repos:
            break

    # Stage 2 — fetch open issues for all repos in parallel, then TF-IDF rank
    user_emb = build_user_profile_embedding(
        feat.get("pr_titles", []),
        feat.get("pr_descriptions", []),
    )

    def _fetch_and_rank(rec):
        try:
            raw_issues = fetch_open_issues(rec["repo"])
            ranked = rank_issues(raw_issues, user_emb, top_k=request.top_k_issues)
            return rec, [IssueResult(**i) for i in ranked]
        except Exception:
            return rec, []

    issue_map: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_and_rank, rec): rec["repo"] for rec in top_repos}
        for future in as_completed(futures, timeout=7):
            try:
                rec, issues = future.result()
                issue_map[rec["repo"]] = issues
            except Exception:
                issue_map[futures[future]] = []

    return [
        RepoResult(
            repo=rec["repo"],
            explanation=rec["explanation"],
            issues=issue_map.get(rec["repo"], []),
        )
        for rec in top_repos
    ]
