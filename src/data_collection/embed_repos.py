"""
Generate Sentence Transformer embeddings for all repos in repo_metadata.jsonl.

Reads:  data/processed/repo_metadata.jsonl
        data/processed/repo_index.json
Writes:
  data/processed/repo_embeddings.npy   — float32 array (num_repos, 384)
  data/processed/repo_lang_ids.npy     — int32 array (num_repos,)
  data/processed/repo_features.json    — {repo: {topics, language, ...}} for explanations
  data/processed/lang_index.json       — {language: int_id}

Usage:
    python src/data_collection/embed_repos.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED = Path("data/processed")
MODEL_NAME = "all-MiniLM-L6-v2"


def build_repo_text(meta: dict) -> str:
    parts = []
    if meta.get("description"):
        parts.append(meta["description"])
    if meta.get("topics"):
        parts.append(" ".join(meta["topics"]))
    readme = meta.get("readme", "")
    # First 300 chars of README is usually the most informative
    if readme:
        parts.append(readme[:300])
    return " ".join(parts) or meta.get("repo", "")


def main():
    repo_index: dict[str, int] = json.loads((PROCESSED / "repo_index.json").read_text())
    metadata_path = PROCESSED / "repo_metadata.jsonl"

    meta_by_repo: dict[str, dict] = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            for line in f:
                m = json.loads(line)
                meta_by_repo[m["repo"]] = m

    print(f"Repos in index: {len(repo_index):,}  |  Repos with metadata: {len(meta_by_repo):,}")

    # Build language index
    all_languages = sorted({m.get("language", "") or "" for m in meta_by_repo.values()} | {""})
    lang_index = {lang: i for i, lang in enumerate(all_languages)}
    (PROCESSED / "lang_index.json").write_text(json.dumps(lang_index, indent=2))

    # Order repos by their integer index
    repos_ordered = sorted(repo_index.keys(), key=lambda r: repo_index[r])
    num_repos = len(repos_ordered)

    print(f"Loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = []
    lang_ids = np.zeros(num_repos, dtype=np.int32)
    features_out: dict[str, dict] = {}

    for i, repo in enumerate(repos_ordered):
        meta = meta_by_repo.get(repo, {"repo": repo})
        texts.append(build_repo_text(meta))
        lang = meta.get("language", "") or ""
        lang_ids[i] = lang_index.get(lang, 0)
        features_out[repo] = {
            "topics": meta.get("topics", []),
            "language": lang,
            "description": meta.get("description", ""),
        }

    print(f"Encoding {num_repos:,} repos ...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(str(PROCESSED / "repo_embeddings.npy"), embeddings)
    np.save(str(PROCESSED / "repo_lang_ids.npy"), lang_ids)
    (PROCESSED / "repo_features.json").write_text(json.dumps(features_out, indent=2))

    print(f"Saved embeddings {embeddings.shape}  lang_ids {lang_ids.shape}")


if __name__ == "__main__":
    main()
