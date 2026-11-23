"""
Upload trained GitMatch model artifacts to Hugging Face Hub.

Usage:
    export HF_TOKEN=hf_...
    python scripts/upload_to_hf.py

Uploads to: mustafahk27/gitmatch-models  (created as a dataset repo if it doesn't exist)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

PROCESSED = Path("data/processed")
HF_REPO_ID = os.getenv("HF_REPO_ID", "mustafak27/gitmatch-models")

ARTIFACTS = [
    PROCESSED / "repo_embeddings.npy",
    PROCESSED / "repo_lang_ids.npy",
    PROCESSED / "repo_index.json",
    PROCESSED / "user_index.json",
    PROCESSED / "user_features.json",
    PROCESSED / "two_tower.pt",
    PROCESSED / "knn_similarity.npz",
    PROCESSED / "lang_index.json",
    PROCESSED / "topic_index.json",
    PROCESSED / "repo_features.json",
]


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Get one at https://huggingface.co/settings/tokens and run:")
        print("  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    login(token=token)
    api = HfApi()

    # Create the repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"WARNING: Could not create repo (may already exist): {e}")

    missing = [f for f in ARTIFACTS if not f.exists()]
    if missing:
        print(f"WARNING: {len(missing)} artifact(s) missing, skipping:")
        for f in missing:
            print(f"  {f}")

    for artifact in ARTIFACTS:
        if not artifact.exists():
            continue
        size_mb = artifact.stat().st_size / 1_048_576
        print(f"Uploading {artifact.name} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(artifact),
            path_in_repo=artifact.name,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"  ✓ {artifact.name}")

    print(f"\nAll artifacts uploaded to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
