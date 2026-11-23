"""
Item-based kNN collaborative filtering baseline.

Builds a cosine similarity matrix over repositories from the
user-repo interaction matrix, then recommends repos similar to
the ones a user has already contributed to.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN:
    def __init__(self, k: int = 50):
        self.k = k
        self.similarity_matrix: np.ndarray | None = None
        self.repo_ids: list[str] = []

    def fit(self, interaction_matrix: sp.csr_matrix, repo_ids: list[str]):
        """
        interaction_matrix: (num_users, num_repos) sparse matrix of implicit scores
        repo_ids: list of repo names in column order
        """
        self.repo_ids = repo_ids
        # Compute item-item cosine similarity on the transposed matrix
        repo_matrix = interaction_matrix.T  # (num_repos, num_users)
        self.similarity_matrix = cosine_similarity(repo_matrix, dense_output=True)
        # Zero out self-similarity
        np.fill_diagonal(self.similarity_matrix, 0.0)

    def recommend(self, user_history: list[int], top_k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float, list[int]]]:
        """
        user_history: list of repo indices the user has contributed to
        Returns list of (repo_idx, score, [neighbor_repo_idxs]) sorted by score descending
        """
        if not user_history or self.similarity_matrix is None:
            return []

        exclude = exclude or set(user_history)
        scores = np.zeros(len(self.repo_ids))
        neighbor_map: dict[int, list[int]] = {}

        for repo_idx in user_history:
            sims = self.similarity_matrix[repo_idx]
            scores += sims

        for candidate in np.argsort(scores)[::-1]:
            if int(candidate) in exclude:
                continue
            top_neighbors = sorted(
                user_history,
                key=lambda h: self.similarity_matrix[h, candidate],
                reverse=True,
            )[:3]
            neighbor_map[int(candidate)] = top_neighbors

        results = []
        for candidate in np.argsort(scores)[::-1]:
            if int(candidate) in exclude:
                continue
            results.append((int(candidate), float(scores[candidate]), neighbor_map.get(int(candidate), [])))
            if len(results) >= top_k:
                break

        return results

    def explain(self, candidate_idx: int, neighbor_idxs: list[int]) -> str:
        neighbors = [self.repo_ids[i] for i in neighbor_idxs]
        candidate = self.repo_ids[candidate_idx]
        return f"Recommended because contributors to {', '.join(neighbors)} also contributed to {candidate}."
