"""
Build user feature dicts from filtered interactions + repo metadata.

Reads:  data/processed/interactions_filtered.tsv
        data/processed/repo_features.json
        data/processed/repo_index.json
        data/processed/lang_index.json
Writes: data/processed/user_features.json

Usage:
    python src/data_collection/build_user_features.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

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
