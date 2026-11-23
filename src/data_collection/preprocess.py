"""
Filter raw interactions and build user/repo indexes.

Reads:  data/processed/interactions_raw.tsv
Writes:
  data/processed/interactions_filtered.tsv  — filtered user-repo pairs
  data/processed/user_index.json            — {username: int_id}
  data/processed/repo_index.json            — {owner/repo: int_id}
  data/processed/repo_list.txt              — one repo per line (for github_api.py)
  data/processed/interaction_matrix.npz    — scipy sparse CSR (users x repos)

Usage:
    python src/data_collection/preprocess.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROCESSED = Path("data/processed")
RAW_TSV = PROCESSED / "interactions_raw.tsv"

MIN_USER_REPOS = 2    # drop users with fewer distinct repos
MIN_REPO_USERS = 3    # drop repos with fewer distinct users


def load_raw(path: Path) -> dict[tuple[str, str], float]:
    interactions: dict[tuple[str, str], float] = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            user, repo, score = parts
            interactions[(user, repo)] = float(score)
    return interactions


def filter_interactions(
    interactions: dict[tuple[str, str], float],
    min_user_repos: int = MIN_USER_REPOS,
    min_repo_users: int = MIN_REPO_USERS,
) -> dict[tuple[str, str], float]:
    # Iteratively filter until stable
    while True:
        user_counts: dict[str, int] = defaultdict(int)
        repo_counts: dict[str, int] = defaultdict(int)
        for user, repo in interactions:
            user_counts[user] += 1
            repo_counts[repo] += 1

        keep = {
            (u, r): s
            for (u, r), s in interactions.items()
            if user_counts[u] >= min_user_repos and repo_counts[r] >= min_repo_users
        }
        if len(keep) == len(interactions):
            break
        interactions = keep

    return interactions


def build_indexes(interactions: dict[tuple[str, str], float]):
    users = sorted({u for u, _ in interactions})
    repos = sorted({r for _, r in interactions})
    user_index = {u: i for i, u in enumerate(users)}
    repo_index = {r: i for i, r in enumerate(repos)}
    return user_index, repo_index


def build_sparse_matrix(
    interactions: dict[tuple[str, str], float],
    user_index: dict[str, int],
    repo_index: dict[str, int],
) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for (user, repo), score in interactions.items():
        rows.append(user_index[user])
        cols.append(repo_index[repo])
        vals.append(math.log1p(score))

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_index), len(repo_index)),
        dtype=np.float32,
    )


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_TSV} ...")
    raw = load_raw(RAW_TSV)
    print(f"  {len(raw):,} raw user-repo pairs")

    print("Filtering ...")
    filtered = filter_interactions(raw)
    print(f"  {len(filtered):,} pairs after filtering  "
          f"({len({u for u,_ in filtered}):,} users, {len({r for _,r in filtered}):,} repos)")

    user_index, repo_index = build_indexes(filtered)

    # Write filtered TSV
    out_tsv = PROCESSED / "interactions_filtered.tsv"
    with open(out_tsv, "w") as f:
        f.write("user\trepo\tscore\n")
        for (user, repo), score in filtered.items():
            f.write(f"{user}\t{repo}\t{score:.4f}\n")

    # Write indexes
    (PROCESSED / "user_index.json").write_text(json.dumps(user_index, indent=2))
    (PROCESSED / "repo_index.json").write_text(json.dumps(repo_index, indent=2))

    # Write repo list for github_api.py
    (PROCESSED / "repo_list.txt").write_text("\n".join(repo_index.keys()))

    # Write sparse matrix
    matrix = build_sparse_matrix(filtered, user_index, repo_index)
    sp.save_npz(str(PROCESSED / "interaction_matrix.npz"), matrix)

    print(f"Wrote indexes ({len(user_index):,} users, {len(repo_index):,} repos)")
    print(f"Matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")


if __name__ == "__main__":
    main()
