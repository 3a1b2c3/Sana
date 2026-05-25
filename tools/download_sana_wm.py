"""Download SANA-WM_bidirectional from HuggingFace.

Defaults max_workers=1 to dodge the Windows + Python 3.10/3.12 thread-pool
shutdown race in ``snapshot_download`` (``concurrent.futures.thread.shutdown
-> t.join -> RuntimeError: cannot join thread before it is started`` on ^C,
or hang-then-crash mid-download). Pass --workers N to opt back into parallel
fetch on Linux / when you know your stack tolerates it.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Efficient-Large-Model/SANA-WM_bidirectional"


def main() -> int:
    p = argparse.ArgumentParser(description=f"Download {REPO} (~94 GB) from Hugging Face.")
    p.add_argument("--dest", type=Path, default=None,
                   help="optional local copy dir. Default: download to HF hub cache "
                        "(~/.cache/huggingface/hub/) so other tools that resolve via "
                        "snapshot_download() can re-use the same files.")
    p.add_argument("--revision", default="main")
    p.add_argument("--include", nargs="+", help="glob patterns to include (e.g. 'dit/*' 'vae/*')")
    p.add_argument("--exclude", nargs="+", help="glob patterns to exclude (e.g. 'refiner/text_encoder/*')")
    p.add_argument("--workers", type=int, default=1,
                   help="snapshot_download max_workers (default 1 = serial; "
                        "raise for parallel fetch when not on Windows/py3.12).")
    args = p.parse_args()

    kwargs = {
        "repo_id": REPO,
        "max_workers": args.workers,
    }
    if args.dest is not None:
        args.dest.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(args.dest)
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
