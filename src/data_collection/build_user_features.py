"""
Build user feature dicts from filtered interactions + repo metadata.

Also pre-computes a user_embedding (mean of contributed repo embeddings,
L2-normalised) so the Vercel server can do pure-numpy dot-product retrieval
without loading torch at serve time.

Reads:  data/processed/interactions_filtered.tsv
        data/processed/repo_features.json
        data/processed/repo_index.json
        data/processed/lang_index.json
        data/processed/repo_embeddings.npy   (optional — skipped if absent)
Writes: data/processed/user_features.json

Usage:
    python src/data_collection/build_user_features.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROCESSED = Path("data/processed")
MAX_LANG_IDS = 10
MAX_TOPIC_IDS = 20


def main():
    repo_index: dict[str, int] = json.loads((PROCESSED / "repo_index.json").read_text())
    repo_features: dict[str, dict] = json.loads((PROCESSED / "repo_features.json").read_text())
    lang_index: dict[str, int] = json.loads((PROCESSED / "lang_index.json").read_text())

    # Build topic index from all repo topics
    all_topics = sorted({t for f in repo_features.values() for t in f.get("topics", [])})
    topic_index = {t: i + 1 for i, t in enumerate(all_topics)}  # 0 = padding

    user_repos: dict[str, list[str]] = defaultdict(list)
    with open(PROCESSED / "interactions_filtered.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                user, repo, _ = parts
                user_repos[user].append(repo)

    user_features: dict[str, dict] = {}
    for user, repos in user_repos.items():
        lang_counts: Counter = Counter()
        topic_counts: Counter = Counter()
        for repo in repos:
            feat = repo_features.get(repo, {})
            lang = feat.get("language", "")
            if lang:
                lang_counts[lang] += 1
            for t in feat.get("topics", []):
                topic_counts[t] += 1

        top_langs = [l for l, _ in lang_counts.most_common(MAX_LANG_IDS)]
        top_topics = [t for t, _ in topic_counts.most_common(MAX_TOPIC_IDS)]

        # Pad to fixed length
        lang_ids = [lang_index.get(l, 0) for l in top_langs]
        lang_ids += [0] * (MAX_LANG_IDS - len(lang_ids))

        topic_ids = [topic_index.get(t, 0) for t in top_topics]
        topic_ids += [0] * (MAX_TOPIC_IDS - len(topic_ids))

        user_features[user] = {
            "contributed_repo_ids": [repo_index[r] for r in repos if r in repo_index],
            "top_languages": top_langs,
            "top_topics": top_topics,
            "lang_ids": lang_ids,
            "topic_ids": topic_ids,
            "pr_titles": [],       # populated later from GitHub API if available
            "pr_descriptions": [],
        }

    # Pre-compute user embeddings (mean of contributed repo embeddings) for numpy serving
    emb_path = PROCESSED / "repo_embeddings.npy"
    if emb_path.exists():
        repo_embs = np.load(str(emb_path)).astype(np.float32)
        repo_index_map: dict[str, int] = json.loads((PROCESSED / "repo_index.json").read_text())
        for username, feat in user_features.items():
            if username == "__config__":
                continue
            contrib_ids = feat.get("contributed_repo_ids", [])
            valid_ids = [i for i in contrib_ids if i < len(repo_embs)]
            if valid_ids:
                u_emb = repo_embs[valid_ids].mean(axis=0)
                norm = np.linalg.norm(u_emb)
                u_emb = u_emb / norm if norm > 0 else u_emb
                feat["user_embedding"] = u_emb.tolist()
        print("  Pre-computed user embeddings (mean of contributed repos)")
    else:
        print("  repo_embeddings.npy not found — run embed_repos.py first to add user_embedding")

    # Store config so the app can re-instantiate the model
    user_features["__config__"] = {
        "num_users": len(user_repos),
        "num_languages": len(lang_index),
        "num_topics": len(topic_index),
    }

    (PROCESSED / "user_features.json").write_text(json.dumps(user_features, indent=2))
    (PROCESSED / "topic_index.json").write_text(json.dumps(topic_index, indent=2))
    print(f"Wrote user_features.json for {len(user_repos):,} users")


if __name__ == "__main__":
    main()
