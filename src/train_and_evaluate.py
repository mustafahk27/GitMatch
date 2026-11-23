"""
Train Two-Tower model, train kNN baseline, and evaluate both.

Reads:  data/processed/interaction_matrix.npz
        data/processed/user_features.json
        data/processed/repo_index.json
        data/processed/repo_embeddings.npy
        data/processed/repo_lang_ids.npy
Writes:
  data/processed/two_tower.pt        — trained Two-Tower weights
  data/processed/knn_similarity.npz  — kNN item-item similarity matrix
  results.json                       — evaluation metrics table

Usage:
    python src/train_and_evaluate.py
"""

import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from src.models.two_tower import TwoTowerModel, InteractionDataset, train as train_two_tower
from src.models.knn_baseline import ItemKNN
from src.models.popularity_baseline import PopularityBaseline
from src.evaluation.metrics import evaluate_model, catalogue_coverage

PROCESSED = Path("data/processed")
RESULTS_PATH = Path("results.json")

TEST_RATIO = 0.15
VAL_RATIO = 0.10
SEED = 42


def chronological_split(matrix: sp.csr_matrix):
    """
    Simple random split (interactions_filtered.tsv has no timestamps after aggregation).
    Per-user: last TEST_RATIO contributions → test, next VAL_RATIO → val, rest → train.
    """
    rng = random.Random(SEED)
    train_rows, train_cols, train_vals = [], [], []
    val_gt: dict[int, set] = {}
    test_gt: dict[int, set] = {}

    cx = matrix.tocoo()
    user_items: dict[int, list[tuple[int, float]]] = {}
    for u, r, v in zip(cx.row, cx.col, cx.data):
        user_items.setdefault(int(u), []).append((int(r), float(v)))

    for u, items in user_items.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        test_items = items[:n_test]
        val_items = items[n_test:n_test + n_val]
        train_items = items[n_test + n_val:]

        test_gt[u] = {r for r, _ in test_items}
        val_gt[u] = {r for r, _ in val_items}
        for r, v in train_items:
            train_rows.append(u)
            train_cols.append(r)
            train_vals.append(v)

    train_matrix = sp.csr_matrix(
        (train_vals, (train_rows, train_cols)), shape=matrix.shape, dtype=np.float32
    )
    return train_matrix, val_gt, test_gt


def build_positives_list(matrix: sp.csr_matrix, user_features: dict, user_index: dict, repo_index: dict) -> list[dict]:
    idx_to_user = {v: k for k, v in user_index.items() if k != "__config__"}
    idx_to_repo = {v: k for k, v in repo_index.items()}
    cx = matrix.tocoo()
    positives = []
    for u, r in zip(cx.row, cx.col):
        username = idx_to_user.get(int(u))
        if username not in user_features:
            continue
        feat = user_features[username]
        positives.append({
            "user_id": int(u),
            "user_lang_ids": feat["lang_ids"],
            "user_topic_ids": feat["topic_ids"],
            "repo": {"text_emb": None, "lang_id": r},  # filled below
            "repo_idx": int(r),
        })
    return positives


def evaluate_all(models_dict: dict, test_gt: dict[int, set], repo_ids: list[str], n_repos: int) -> dict:
    results = {}
    for name, get_recs in models_dict.items():
        recs: dict[str, list] = {}
        idx_to_repo = {i: r for i, r in enumerate(repo_ids)}
        for user_id, relevant in test_gt.items():
            if not relevant:
                continue
            rec_idxs = get_recs(user_id, list(relevant))
            recs[str(user_id)] = [idx_to_repo[i] for i in rec_idxs]
        gt_named = {str(u): {idx_to_repo[r] for r in rel} for u, rel in test_gt.items()}
        summary = evaluate_model(recs, gt_named)
        coverage = catalogue_coverage(recs, set(repo_ids))
        summary["coverage"] = round(coverage, 4)
        results[name] = {k: round(float(v), 4) for k, v in summary.items()}
        print(f"\n{name}:")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
    return results


def main():
    print("Loading data ...")
    matrix = sp.load_npz(str(PROCESSED / "interaction_matrix.npz"))
    repo_index: dict[str, int] = json.loads((PROCESSED / "repo_index.json").read_text())
    user_features: dict = json.loads((PROCESSED / "user_features.json").read_text())
    user_index: dict[str, int] = {k: v for k, v in json.loads(
        (PROCESSED / "user_index.json").read_text()).items()}
    repo_embeddings = np.load(str(PROCESSED / "repo_embeddings.npy"))
    repo_lang_ids = np.load(str(PROCESSED / "repo_lang_ids.npy"))

    repo_ids = sorted(repo_index.keys(), key=lambda r: repo_index[r])
    cfg = user_features.get("__config__", {})
    num_users = matrix.shape[0]
    num_languages = cfg.get("num_languages", 50)
    num_topics = cfg.get("num_topics", 200)

    print(f"Matrix: {matrix.shape}, {matrix.nnz:,} interactions")

    # --- Split ---
    print("\nSplitting train/val/test ...")
    train_matrix, val_gt, test_gt = chronological_split(matrix)
    print(f"Train: {train_matrix.nnz:,} | Val users: {len(val_gt)} | Test users: {len(test_gt)}")

    # --- Popularity baseline ---
    print("\nFitting popularity baseline ...")
    pop = PopularityBaseline()
    pop.fit(train_matrix, repo_ids)

    # --- kNN baseline ---
    print("Fitting item-based kNN ...")
    knn = ItemKNN(k=50)
    knn.fit(train_matrix, repo_ids)
    np.savez_compressed(
        str(PROCESSED / "knn_similarity.npz"),
        matrix=knn.similarity_matrix,
    )
    print("  Saved knn_similarity.npz")

    # --- Two-Tower ---
    print("\nBuilding Two-Tower training dataset ...")
    idx_to_user = {v: k for k, v in user_index.items()}
    cx = train_matrix.tocoo()
    positives = []
    for u_idx, r_idx in zip(cx.row, cx.col):
        username = idx_to_user.get(int(u_idx))
        if username not in user_features:
            continue
        feat = user_features[username]
        positives.append({
            "user_id": int(u_idx),
            "user_lang_ids": feat["lang_ids"],
            "user_topic_ids": feat["topic_ids"],
            "repo": {
                "text_emb": repo_embeddings[r_idx].tolist(),
                "lang_id": int(repo_lang_ids[r_idx]),
            },
        })

    all_repos = [
        {"text_emb": repo_embeddings[i].tolist(), "lang_id": int(repo_lang_ids[i])}
        for i in range(len(repo_ids))
    ]

    model = TwoTowerModel(num_users=num_users, num_languages=num_languages, num_topics=num_topics)
    dataset = InteractionDataset(positives, all_repos, num_negatives=4)
    print(f"Dataset size: {len(dataset):,} samples  ({len(positives):,} positives)")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device} ...")
    model = train_two_tower(model, dataset, epochs=10, lr=1e-3, device=device)
    torch.save(model.state_dict(), str(PROCESSED / "two_tower.pt"))
    print("  Saved two_tower.pt")

    # --- Evaluate ---
    print("\nEvaluating ...")
    repo_embs_t = torch.tensor(repo_embeddings, dtype=torch.float)
    repo_lang_t = torch.tensor(repo_lang_ids, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        all_repo_embs = model.encode_repo(repo_embs_t.to(device), repo_lang_t.to(device)).cpu().numpy()

    def nn_recs(user_id: int, exclude: list[int], k: int = 20) -> list[int]:
        username = idx_to_user.get(user_id)
        if username not in user_features:
            return []
        feat = user_features[username]
        u_t = torch.tensor([user_id], dtype=torch.long)
        l_t = torch.tensor([feat["lang_ids"]], dtype=torch.long)
        tp_t = torch.tensor([feat["topic_ids"]], dtype=torch.long)
        with torch.no_grad():
            u_emb = model.encode_user(u_t.to(device), l_t.to(device), tp_t.to(device)).cpu().numpy()
        scores = (u_emb @ all_repo_embs.T).flatten()
        excl = set(exclude)
        ranked = [i for i in np.argsort(scores)[::-1] if i not in excl]
        return ranked[:k]

    def knn_recs(user_id: int, exclude: list[int], k: int = 20) -> list[int]:
        history = [r for r in range(matrix.shape[1]) if train_matrix[user_id, r] > 0]
        recs = knn.recommend(history, top_k=k, exclude=set(exclude))
        return [idx for idx, _, _ in recs]

    def pop_recs(user_id: int, exclude: list[int], k: int = 20) -> list[int]:
        recs = pop.recommend([], top_k=k + len(exclude), exclude=set(exclude))
        return [idx for idx, _ in recs[:k]]

    all_results = evaluate_all(
        {
            "Popularity": pop_recs,
            "Item-kNN": knn_recs,
            "Two-Tower NN": nn_recs,
        },
        test_gt,
        repo_ids,
        n_repos=len(repo_ids),
    )

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
