"""
Stage 2 — Rank open issues within a recommended repository.

Uses Sentence Transformers (all-MiniLM-L6-v2) to embed issue text
and user profile, then scores by cosine similarity.
"""

from sentence_transformers import SentenceTransformer, util

_model: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_user_profile_embedding(pr_titles: list[str], pr_descriptions: list[str]) -> list[float]:
    """Mean-pool embeddings of the user's merged PR titles + descriptions."""
    model = _get_model()
    texts = [f"{t} {d}".strip() for t, d in zip(pr_titles, pr_descriptions) if t or d]
    if not texts:
        texts = ["open source contribution"]
    embeddings = model.encode(texts, convert_to_tensor=True)
    profile_emb = embeddings.mean(dim=0)
    return profile_emb.tolist()


def rank_issues(
    issues: list[dict],
    user_profile_embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    """
    issues: list of dicts with keys title, body, number, url, labels
    user_profile_embedding: from build_user_profile_embedding()
    Returns issues sorted by relevance, with a similarity score.
    """
    if not issues:
        return []

    model = _get_model()
    issue_texts = [f"{i['title']} {i.get('body', '')}".strip() for i in issues]
    issue_embs = model.encode(issue_texts, convert_to_tensor=True)

    import torch
    profile_t = torch.tensor(user_profile_embedding).unsqueeze(0)
    scores = util.cos_sim(profile_t, issue_embs).squeeze(0).tolist()

    ranked = sorted(zip(issues, scores), key=lambda x: x[1], reverse=True)
    return [
        {**issue, "relevance_score": round(score, 4)}
        for issue, score in ranked[:top_k]
    ]
