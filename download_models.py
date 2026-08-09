# download_models.py — Pull all models/*.json, *.pt, *.pkl from HF_MODELS_REPO
# into the local models/ dir before running predict.py. Shared by
# train_and_predict.yml's predict_and_upload job and predict_only.yml, so the
# download logic (and its debug logging) lives in exactly one place.
#
# Usage:
#   python download_models.py

import os
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

import config as cfg

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries


def download_models() -> None:
    repo_id = os.environ.get("HF_MODELS_REPO", "") or cfg.HF_MODELS_REPO
    token   = os.environ.get("HF_TOKEN", "") or cfg.HF_TOKEN

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    print(f"[debug] repo_id  = {repo_id!r}")
    print(f"[debug] HF_TOKEN set = {bool(token)}")

    if not repo_id:
        raise ValueError("HF_MODELS_REPO is not set.")

    api = HfApi(token=token or None)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token or None)
    except Exception as e:
        print(f"[debug] list_repo_files FAILED: {type(e).__name__}: {e}")
        raise

    print(f"[debug] total files in repo: {len(files)}")
    if len(files) <= 30:
        for f in sorted(files):
            print(f"[debug]   {f}")
    else:
        print("[debug] (too many to list individually, showing first 10)")
        for f in sorted(files)[:10]:
            print(f"[debug]   {f}")

    model_files = [f for f in files if f.startswith("models/")]
    print(f"[debug] files matching 'models/' prefix: {len(model_files)}")

    if not model_files:
        print(
            "[debug] ⚠️  No files matched the 'models/' prefix filter. Either "
            "the repo is genuinely empty (no successful upload has happened "
            "yet), HF_MODELS_REPO points at the wrong repo, or upload_models.py "
            "is writing to a different path than this filter expects. Nothing "
            "to download — predict.py WILL report 'No model found' for every "
            "option."
        )

    failed = []
    for f in model_files:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f,
                    repo_type="dataset",
                    token=token or None,
                    local_dir=".",
                    local_dir_use_symlinks=False,
                    force_download=True,
                )
                print(f"Downloaded {f}")
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"  ⚠️  Attempt {attempt} failed for {f}: {e}")
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ All {MAX_RETRIES} attempts failed for {f}: {e}")
                    failed.append(f)

    local_models = sorted(os.listdir(cfg.MODELS_DIR)) if os.path.isdir(cfg.MODELS_DIR) else []
    print(f"[debug] local {cfg.MODELS_DIR}/ dir now contains {len(local_models)} file(s):")
    for f in local_models:
        print(f"[debug]   {f}")

    if failed:
        print(f"\n❌ {len(failed)} file(s) failed to download after {MAX_RETRIES} attempts each:")
        for f in failed:
            print(f"   {f}")
        sys.exit(1)


if __name__ == "__main__":
    download_models()
