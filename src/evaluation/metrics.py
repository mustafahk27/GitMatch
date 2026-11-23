"""
Offline evaluation metrics: Precision@K, Recall@K, NDCG@K, coverage, diversity.
"""

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, r in enumerate(top_k)
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    recommendations: dict[str, list],
    ground_truth: dict[str, set],
    k_values: list[int] = (5, 10, 20),
) -> dict:
    """
    recommendations: {username: [repo1, repo2, ...]} (ranked)
    ground_truth: {username: {repo_a, repo_b, ...}} (held-out contributions)
    """
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

    for user, recs in recommendations.items():
        relevant = ground_truth.get(user, set())
        if not relevant:
            continue
        for k in k_values:
            results[k]["precision"].append(precision_at_k(recs, relevant, k))
            results[k]["recall"].append(recall_at_k(recs, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(recs, relevant, k))

    summary = {}
    for k in k_values:
        summary[f"P@{k}"] = np.mean(results[k]["precision"])
        summary[f"R@{k}"] = np.mean(results[k]["recall"])
        summary[f"NDCG@{k}"] = np.mean(results[k]["ndcg"])

    return summary


def catalogue_coverage(recommendations: dict[str, list], all_repos: set) -> float:
    recommended_repos = {r for recs in recommendations.values() for r in recs}
    return len(recommended_repos) / len(all_repos) if all_repos else 0.0
