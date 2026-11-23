"""
Stage 2 — Rank open issues within a recommended repository.

Uses TF-IDF cosine similarity (pure stdlib + numpy) so there is no
external API call and no Vercel timeout risk.
"""

import math
import re
from collections import Counter

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_matrix(docs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (tfidf_matrix, vocab) for a list of documents."""
    tokenized = [_tokenize(d) for d in docs]
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    vocab = sorted(df.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in vocab}

    matrix = np.zeros((n, len(vocab)), dtype=np.float32)
    for row, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        for word, count in tf.items():
            if word in word2idx:
                matrix[row, word2idx[word]] = (count / total) * idf[word]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms > 0, norms, 1)
    return matrix, vocab


def build_user_profile_embedding(pr_titles: list[str], pr_descriptions: list[str]) -> list[float]:
    """Build a TF-IDF vector from the user's PR history (returned as a plain list for compat)."""
    texts = [f"{t} {d}".strip() for t, d in zip(pr_titles, pr_descriptions) if t or d]
    query = " ".join(texts) if texts else "open source contribution bug fix feature"
    return query  # pass the raw string; rank_issues handles vectorisation


def rank_issues(
    issues: list[dict],
    user_profile: list[float] | str,
    top_k: int = 10,
) -> list[dict]:
    if not issues:
        return []

    issue_texts = [f"{i['title']} {i.get('body', '')}".strip() for i in issues]

    # user_profile may be a raw query string (new path) or a list (legacy)
    if isinstance(user_profile, str):
        query = user_profile
    else:
        query = "open source contribution"

    all_docs = issue_texts + [query]
    matrix, _ = _tfidf_matrix(all_docs)
    issue_vecs = matrix[:-1]
    query_vec = matrix[-1]

    scores = issue_vecs @ query_vec

    ranked = sorted(zip(issues, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [
        {**issue, "relevance_score": round(float(score), 4)}
        for issue, score in ranked[:top_k]
    ]
