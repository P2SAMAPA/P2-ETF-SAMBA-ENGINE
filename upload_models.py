# upload_models.py — Push models and signals to HuggingFace

import glob
import os
from huggingface_hub import HfApi
import config as cfg


def upload_models() -> None:
    token   = os.environ.get("HF_TOKEN", "") or cfg.HF_TOKEN
    repo_id = os.environ.get("HF_MODELS_REPO", "") or cfg.HF_MODELS_REPO

    if not token:
        raise ValueError("HF_TOKEN is not set.")
    if not repo_id:
        raise ValueError("HF_MODELS_REPO is not set.")

    print(f"Uploading to: {repo_id}")
    api = HfApi(token=token)

    patterns = ["*.pt", "*.pkl", "*.json"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(cfg.MODELS_DIR, pat)))

    if not files:
        print("No model files found.")
        return

    for f in files:
        repo_path = f"models/{os.path.basename(f)}"
        print(f"  Uploading {os.path.basename(f)} → {repo_path}")
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="[auto] Update SAMBA models and signals",
        )

    print(f"Pushed {len(files)} file(s) to {repo_id}")


if __name__ == "__main__":
    upload_models()
