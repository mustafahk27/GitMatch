"""
Stage 2 — Rank open issues within a recommended repository.

At serve time (Vercel): calls the HF Inference API for embeddings — no local model.
At training/offline time: set USE_LOCAL_MODEL=1 to use sentence-transformers directly.
"""

import os
import time

import numpy as np
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")
USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "0") == "1"

_local_model = None


def _embed_hf_api(texts: list[str]) -> np.ndarray:
    """Call HF Inference API. Retries once on 503 (model loading)."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    for attempt in range(3):
        resp = requests.post(HF_API_URL, headers=headers, json={"inputs": texts}, timeout=30)
        if resp.status_code == 200:
            return np.array(resp.json(), dtype=np.float32)
        if resp.status_code == 503:
            time.sleep(10)  # model is loading on HF
        else:
            resp.raise_for_status()
    raise RuntimeError("HF Inference API unavailable after retries")


def _embed_local(texts: list[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def embed(texts: list[str]) -> np.ndarray:
    if USE_LOCAL:
        return _embed_local(texts)
    return _embed_hf_api(texts)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a @ b.T).flatten()


def build_user_profile_embedding(pr_titles: list[str], pr_descriptions: list[str]) -> list[float]:
    """Mean-pool embeddings of the user's merged PR titles + descriptions."""
    texts = [f"{t} {d}".strip() for t, d in zip(pr_titles, pr_descriptions) if t or d]
    if not texts:
        texts = ["open source contribution"]
    embs = embed(texts)
    profile = embs.mean(axis=0)
    norm = np.linalg.norm(profile)
    return (profile / norm).tolist() if norm > 0 else profile.tolist()


def rank_issues(
    issues: list[dict],
    user_profile_embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    if not issues:
        return []

    issue_texts = [f"{i['title']} {i.get('body', '')}".strip() for i in issues]
    issue_embs = embed(issue_texts)
    profile = np.array(user_profile_embedding, dtype=np.float32).reshape(1, -1)
    scores = _cosine_sim(profile, issue_embs)

    ranked = sorted(zip(issues, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [
        {**issue, "relevance_score": round(score, 4)}
        for issue, score in ranked[:top_k]
    ]
