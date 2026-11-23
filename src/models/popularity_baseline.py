"""
Popularity baseline — recommends the most contributed-to repositories.
Every other model must beat this.
"""

import numpy as np
import scipy.sparse as sp


class PopularityBaseline:
    def __init__(self):
        self.repo_scores: np.ndarray | None = None
        self.repo_ids: list[str] = []

    def fit(self, interaction_matrix: sp.csr_matrix, repo_ids: list[str]):
        self.repo_ids = repo_ids
        # Sum of all implicit scores per repo
        self.repo_scores = np.asarray(interaction_matrix.sum(axis=0)).flatten()

    def recommend(self, user_history: list[int], top_k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        exclude = exclude or set(user_history)
        ranked = np.argsort(self.repo_scores)[::-1]
        results = []
        for idx in ranked:
            if int(idx) not in exclude:
                results.append((int(idx), float(self.repo_scores[idx])))
            if len(results) >= top_k:
                break
        return results
