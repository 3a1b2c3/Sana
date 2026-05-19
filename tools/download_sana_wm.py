import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_DEST = Path("output/pretrained_models/SANA-WM_bidirectional")


def main() -> int:
    p = argparse.ArgumentParser(description=f"Download {REPO} (~94 GB) from Hugging Face.")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--revision", default="main")
    p.add_argument("--include", nargs="+", help="glob patterns to include (e.g. 'dit/*' 'vae/*')")
    p.add_argument("--exclude", nargs="+", help="glob patterns to exclude (e.g. 'refiner/text_encoder/*')")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    kwargs = {"repo_id": REPO, "local_dir": str(args.dest)}
    if args.revision != "main":
        kwargs["revision"] = args.revision
    if args.include:
        kwargs["allow_patterns"] = args.include
    if args.exclude:
        kwargs["ignore_patterns"] = args.exclude
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token

    path = snapshot_download(**kwargs)
    print(f"Downloaded to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
