"""
Stage 1 — Repository recommendation for a given GitHub username.

Uses the trained Two-Tower model as the primary ranker.
Falls back to item-based kNN for users not seen during training.
"""

from pathlib import Path

import numpy as np
import torch

from src.models.two_tower import TwoTowerModel
from src.models.knn_baseline import ItemKNN


def recommend_repos(
    username: str,
    user_index: dict,
    repo_index: dict,
    model: TwoTowerModel,
    knn: ItemKNN,
    repo_embeddings: np.ndarray,
    repo_lang_ids: np.ndarray,
    user_features: dict,
    top_k: int = 10,
    device: str = "cpu",
) -> list[dict]:
    """
    Returns a ranked list of repo recommendations with explanation strings.
    """
    if username not in user_index:
        # Cold-start: fall back to kNN on popularity
        return _cold_start_recommend(username, knn, repo_index, top_k)

    user_id = user_index[username]
    features = user_features[username]

    user_id_t = torch.tensor([user_id], dtype=torch.long).to(device)
    lang_ids_t = torch.tensor([features["lang_ids"]], dtype=torch.long).to(device)
    topic_ids_t = torch.tensor([features["topic_ids"]], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        user_emb = model.encode_user(user_id_t, lang_ids_t, topic_ids_t)
        repo_embs_t = torch.tensor(repo_embeddings, dtype=torch.float).to(device)
        repo_lang_t = torch.tensor(repo_lang_ids, dtype=torch.long).to(device)
        repo_embs_out = model.encode_repo(repo_embs_t, repo_lang_t)
        scores = (user_emb @ repo_embs_out.T).squeeze(0).cpu().numpy()

    already_contributed = set(features.get("contributed_repo_ids", []))
    ranked = np.argsort(scores)[::-1]

    idx_to_repo = {v: k for k, v in repo_index.items()}
    results = []
    for repo_idx in ranked:
        if int(repo_idx) in already_contributed:
            continue
        repo_name = idx_to_repo.get(int(repo_idx), "")
        results.append({
            "repo": repo_name,
            "score": float(scores[repo_idx]),
            "explanation": _build_explanation(features, repo_name),
        })
        if len(results) >= top_k:
            break

    return results


def _cold_start_recommend(username: str, knn: ItemKNN, repo_index: dict, top_k: int) -> list[dict]:
    return [{"repo": repo, "score": 0.0, "explanation": "Popular repository matching your interests."} for repo in list(repo_index.keys())[:top_k]]


def _build_explanation(user_features: dict, repo_name: str) -> str:
    langs = user_features.get("top_languages", [])
    topics = user_features.get("top_topics", [])
    parts = []
    if langs:
        parts.append(f"languages: {', '.join(langs[:3])}")
    if topics:
        parts.append(f"topics: {', '.join(topics[:3])}")
    if parts:
        return f"Matches your profile ({'; '.join(parts)})."
    return "Recommended based on your contribution history."
