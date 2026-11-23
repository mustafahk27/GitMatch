"""
FastAPI web app for GitMatch.

Models are stored on Hugging Face Hub (mustafahk27/gitmatch-models) and
downloaded on first request into /tmp (the only writable path on Vercel).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.two_tower import TwoTowerModel
from src.models.knn_baseline import ItemKNN
from src.recommendation.stage1_repos import recommend_repos
from src.recommendation.stage2_issues import rank_issues, build_user_profile_embedding
from src.data_collection.github_api import fetch_open_issues

# Vercel is read-only except /tmp
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/gitmatch_models"))
HF_REPO_ID = os.getenv("HF_REPO_ID", "mustafahk27/gitmatch-models")

_state: dict = {}


def _load_models():
    """Download model artifacts from HF Hub if not cached, then load into memory."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return  # huggingface_hub not installed — models unavailable

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = ["two_tower.pt", "repo_embeddings.npy", "repo_lang_ids.npy",
             "repo_index.json", "user_index.json", "user_features.json", "knn_similarity.npz"]
    local = {}
    for fname in files:
        try:
            local[fname] = hf_hub_download(repo_id=HF_REPO_ID, filename=fname, cache_dir=str(MODEL_CACHE_DIR))
        except Exception:
            return  # artifact not yet uploaded — stay in stub mode

    import json, scipy.sparse as sp

    _state["repo_index"] = json.loads(Path(local["repo_index.json"]).read_text())
    _state["user_index"] = json.loads(Path(local["user_index.json"]).read_text())
    _state["user_features"] = json.loads(Path(local["user_features.json"]).read_text())
    _state["repo_embeddings"] = np.load(local["repo_embeddings.npy"])
    _state["repo_lang_ids"] = np.load(local["repo_lang_ids.npy"])

    cfg = _state["user_features"].get("__config__", {})
    model = TwoTowerModel(
        num_users=cfg.get("num_users", 1),
        num_languages=cfg.get("num_languages", 50),
        num_topics=cfg.get("num_topics", 200),
    )
    model.load_state_dict(torch.load(local["two_tower.pt"], map_location="cpu"))
    model.eval()
    _state["model"] = model

    knn = ItemKNN()
    data = np.load(local["knn_similarity.npz"])
    import scipy.sparse as sp
    knn.similarity_matrix = data["matrix"] if "matrix" in data else None
    knn.repo_ids = list(_state["repo_index"].keys())
    _state["knn"] = knn
    _state["ready"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="GitMatch", description="GitHub Issues Recommender System", lifespan=lifespan)


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


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _state.get("ready", False)}


@app.post("/recommend", response_model=list[RepoResult])
def recommend(request: RecommendRequest):
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="Models not yet loaded. Check /health.")

    repos = recommend_repos(
        username=request.username,
        user_index=_state["user_index"],
        repo_index=_state["repo_index"],
        model=_state["model"],
        knn=_state["knn"],
        repo_embeddings=_state["repo_embeddings"],
        repo_lang_ids=_state["repo_lang_ids"],
        user_features=_state["user_features"],
        top_k=request.top_k_repos,
    )

    user_feat = _state["user_features"].get(request.username, {})
    pr_titles = user_feat.get("pr_titles", [])
    pr_descriptions = user_feat.get("pr_descriptions", [])
    user_emb = build_user_profile_embedding(pr_titles, pr_descriptions)

    results = []
    for repo_rec in repos:
        raw_issues = fetch_open_issues(repo_rec["repo"])
        ranked = rank_issues(raw_issues, user_emb, top_k=request.top_k_issues)
        results.append(RepoResult(
            repo=repo_rec["repo"],
            explanation=repo_rec["explanation"],
            issues=[IssueResult(**i) for i in ranked],
        ))

    return results
