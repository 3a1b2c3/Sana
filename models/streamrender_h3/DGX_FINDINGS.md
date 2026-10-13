# StreamRender-H3 on DGX (pmgb300ws-0304) — Findings

Single-GPU (GB300 Blackwell, 256GB VRAM) bring-up of upstream `scripts/setup_env.sh`,
`download_models.sh`, `launch_local.sh`. Project defaults to 4/8 GPU; patched for 1-GPU testing.

## Build fixes (magi-attention from source)

- **C++20, two separate build paths, both needed patching**:
  - `csrc/extensions/CMakeLists.txt`: `CXX_STANDARD`/`CUDA_STANDARD` hardcoded to 17 → 20.
  - `setup.py` `cxx_flags`: had no `-std=` flag at all → added `-std=c++20`.
  - `setup.py` `nvcc_flags`: only had `-Xcompiler -std=c++20` (host-forwarded). Still failed with
    identical `char8_t`/`requires requires` errors until a standalone top-level `-std=c++20` was
    added for nvcc's own frontend (separate from the `-Xcompiler` variant).
- **Missing `psutil`** at build time (setup.py imports it, not declared as a build dep).

## Build fixes (flash-attn from source, pinned commit `0251105a2fb1`)

- **Same C++20 issue as magi-attention, independently**: the FA2 extension's `setup.py` pins both
  nvcc and host cxx flags to `-std=c++17` (lines ~314/333), which fails identically
  (`char8_t`/`requires requires`) against torch 2.14's headers. `uv pip install git+...` builds in
  an ephemeral temp dir that can't be patched — fixed by cloning to `.build/flash-attention`,
  sed-patching `setup.py`'s `c++17` → `c++20`, then `uv pip install` from that local path instead.
  Baked into `setup_env.sh`.
- **Silent no-op install**: an ad-hoc `nohup bash -c 'uv pip install ...'` exited 0 and looked
  successful, but `uv` wasn't actually on `PATH` in that non-interactive shell — the whole command
  silently did nothing. Not caught until `ModuleNotFoundError: No module named 'flash_attn'`
  ~2000 log lines into a model run (after backbone/text_encoder/video_decoder/audio_vae had all
  already loaded). Fixed two ways: always resolve `uv`'s absolute path before backgrounding it, and
  added an explicit `python -c "import flash_attn"` check immediately after the build step in
  `setup_env.sh` so a no-op install fails loudly right there instead of downstream.
- **Missing `ninja`**: `magi_attn_comm` uses `dlink=True` device-code linking, which
  `torch.cpp_extension` hard-requires ninja for. Baked into `setup_env.sh` as an explicit install
  step (was previously only a one-off manual fix, lost on venv recreation).

## Build fixes (flash-attn-4 / CuTe DSL backend)

- `from flash_attn.cute.interface import flash_attn_varlen_func` (used by the video decoder's
  native fast path, `causal_vit_stream.py`) is **not** part of the `flash-attn` package built
  above — upstream's own `setup.py` explicitly excludes `flash_attn.cute`/`flash_attn.cute.*` from
  that build (`packages=find_packages(exclude=(..., "flash_attn.cute", "flash_attn.cute.*"))`).
  It ships as a separate PyPI package, `flash-attn-4` (per `flash_attn/cute/README.md`): pure
  Python, JIT-compiled at runtime via `nvidia-cutlass-dsl`, so a normal wheel install
  (`uv pip install "flash-attn-4[cu13]"`), not a source build. Installing it bumped
  `nvidia-cutlass-dsl` 4.4.2 → 4.8.0 as a side effect (expected — see the versions-get-bumped note
  below).
- `magi_attention.utils.debug` unconditionally imports `debugpy` (an attach-a-debugger helper, not
  used at runtime here) but never declares it as a dependency — `import magi_attention` fails
  without it. `uv pip install debugpy` fixes it. Both added to `setup_env.sh` with an import-check
  guard afterward.

## Runtime fix (FSDP2 + `torch.inference_mode()` incompatible)

- `render/pipeline.py`'s `start()`/`render()` were decorated `@torch.inference_mode()`. Once
  text_encoder's FSDP wrap actually ran forward (previously never reached — it crashed earlier on
  the shard_size assertion), this failed: `RuntimeError: Inference tensors do not track version
  counter`, inside FSDP2's `wait_for_unshard` → `torch.autograd._unsafe_preserve_version_counter`.
  This is a known FSDP2 (`torch.distributed.fsdp.fully_shard`) limitation — `inference_mode()`
  tensors don't support the version-counter tracking FSDP2's unshard synchronization needs.
  Independent of `shard_size`/`world_size`; would hit any FSDP-wrapped module run this way. Fixed
  by changing both decorators to `@torch.no_grad()` (same inference-only effect, version counters
  stay tracked). Patched directly on the DGX (`render/pipeline.py`, backed up as `.bak`).

## Environment/tooling gotchas

- `uv venv` is destructive and unconditional — reran once and wiped the whole venv (ninja,
  magi-attention, everything already built). `setup_env.sh` made idempotent (skip if venv exists).
- `uv`-managed venv's `python` symlinks to a shared base interpreter outside the venv dir —
  `.resolve()`'ing that path before use breaks `pyvenv.cfg` venv detection.
- No sudo/apt on this host — Node 22 installed as a local tarball under `.tools/node`, no root.
- Git `core.autocrlf` on Windows silently converts LF→CRLF on commit, breaking bash scripts on
  Linux (`set -euo pipefail\r` → "invalid option name"). Fixed with `.gitattributes`
  (`*.sh text eol=lf`).
- `scp` with multiple sources to one destination flattens by basename, ignoring subdirectory
  structure — caused one stray misplaced file.
- torch pin moved from `2.9.1+cu128` (environment-tested.json) to `2.14.1+cu130` mid-session
  (manual upgrade); `cu132` does not exist as a wheel index, only `cu130`.

## Single-GPU patches (upstream hardcodes 4/8 GPU)

- `runtime/worker.py:30`: `if world not in (4, 8):` → `if world not in (1, 4, 8):`.
- `scripts/launch_local.sh`: `--nproc-per-node="${GPUS:-8}"` → `"${GPUS:-1}"`.
- `configs/runtime.json`: `inference.models.text_encoder.placement.fsdp.shard_size` hardcoded to
  `4`, asserted via `assert world_size % shard_size == 0`
  (`render/vendor/.../placement.py:329`). **Not a checkpoint-format constraint** — PyTorch DCP
  checkpoints support resharding to a different world size on load — just a plain config value.
  Patched to `1`.
- Backbone (33.1B params) placement had no shard_size issue at 1 GPU — only text_encoder's FSDP
  wrap hit the assertion. **Confirmed fixed**: with shard_size=1, backbone, text_encoder,
  video_decoder, and audio_vae all load and place on GPU successfully (~68 GiB VRAM used).

## Runtime gotchas

- Port 5191/8765 conflicts recurred repeatedly: when the worker crashes before `frontend_pid` is
  set, `cleanup()`'s trap doesn't kill vite's child, leaving the port bound for the next run.
  Fixed with an auto-cleanup block in `launch_local.sh` (`fuser -k -TERM` on both ports before
  start).
- `download_assets.py` silently "succeeded" once without producing the merged backbone
  safetensors file (only the 9 raw shards existed). Rerun fixed it; verified via SHA-256.
- A local (uncommitted) patch to `worker.py` was lost after a `git pull` whose own diff only
  touched `launch_local.sh` — `git status` showed worker.py clean. Reapplied and committed locally
  (not pushed) to prevent recurrence; same done for `launch_local.sh` and `.gitattributes`.
- `examples/media/` (30 videos + 60 pictures, ~1.7GB, deliberately **gitignored** per
  `download_examples.sh`) was wiped during a cleanup pass that removed what looked like a redundant
  duplicate tar — the tar removal was fine (`download_examples.sh` already `rm -f`s it after
  extraction), but the media directory itself got deleted too and, being gitignored, was NOT
  recoverable via `git restore` (unlike 4 tracked files — `README.md`, `media_inventory.json`,
  `requests_A.jsonl`, `requests_B.jsonl` — that were also accidentally deleted alongside it and
  *were* trivially restored). Surfaced later as `FileNotFoundError` on
  `examples/media/pictures/*.png` mid-run, well past model loading. Fixed by rerunning
  `scripts/download_examples.sh`. Lesson: gitignored directories have no safety net — `git status`
  showing "clean" after a cleanup does not mean gitignored content wasn't lost.

## Accessing the frontend (SSH tunnel required)

`launch_local.sh` binds both the worker's `/health` endpoint and the vite frontend to
**`127.0.0.1` only** on the DGX (confirmed via `ss -ltnp`: `127.0.0.1:8765` and `127.0.0.1:5191`,
not `0.0.0.0`). Browsing directly to `http://10.74.11.118:5191` from another machine gets
`ERR_CONNECTION_REFUSED` — not a firewall issue, the services simply aren't listening on the
external interface. Tunnel both ports over SSH instead:

```
ssh -N -L 5191:127.0.0.1:5191 -L 8765:127.0.0.1:8765 kschmid@10.74.11.118
```

Then browse to `http://127.0.0.1:5191` (and `http://127.0.0.1:8765/health` to verify readiness)
on the *local* machine — not the DGX's IP — while the tunnel stays open.

## Status: working end-to-end on 1 GPU

With all of the above fixes applied, a full run (`launch_local.sh` → `STREAMRENDER_READY` →
`scripts/replay_example.py --example B10_e-track-3_bluehour`) completed successfully: steady
`STREAMRENDER_CHUNK` events (~1.15s publication interval, ~927ms render_ms/round) through 150+
rounds, output frames saved to `outputs/replay/<session>_NNNNNN.jpg` (~166MB, 1000+ JPEGs).
Visually confirmed — frames hold the reference image's car livery, track, and lighting correctly
through the render.

## Customizing the scene (prompt + reference image)

- The reference image **is** used — `_validation_picture()` encodes it into a real image latent
  (`encode_reference_image()` via `video_vae`), which anchors the opening frame's composition,
  camera angle, lighting, and color grading. It is not ignored. What's separately true: when the
  prompt *text* and the image's *pixels* disagree about one subject's identity (a red race car
  drawn vs. "yellow taxi" described), text conditioning tends to win for that subject's
  color/livery, while the image still drives everything else (framing, lighting, scene geometry).
  Confirmed by swapping only the reference-car description from a red race sedan to a yellow taxi
  (checkered livery, roof sign, door number) via text edits to three repeated passages
  (`subject_definitions`, `retention_analysis`, the `[Shot 1]` paragraph in `detailed_description`)
  — no image editing involved, since no image-gen/edit tool is available in this environment. A new
  reference PNG was still uploaded separately (see below) to keep the image and text in agreement
  rather than relying on text to override a conflicting picture.
- Did a swap-and-restore for this instead of overwriting the only example config in place: backed
  up `configs/example.local.json` before editing, then once the taxi edits were working, split them
  into a separate `configs/taxi.local.json` and restored `example.local.json` to its untouched
  original (`cp .bak` back over it). Keeps both the stock red-sedan example and the yellow-taxi
  variant runnable independently via `--assets configs/<name>.local.json`.
- The reference PNG still needs *a* new image uploaded (even though the car's described appearance
  overrides what's drawn in it) — `prepare_reference_image()` auto-resizes via LANCZOS to whatever
  `output.height`/`output.width` says (1344×768 here), so any image matching that aspect ratio
  (1.75:1) works without pre-resizing; verified with `PIL.Image.verify()` before wiring it in.
- Added optional 3D traffic-cone objects to the TORCS live-driving scene itself (not just prompt
  text) in `coding/threejs-racing/src/world.js`: a new `cone` semantic class in `public/palette.json`
  (required — `SemanticView.render()` throws `Unknown semantic` for any object whose label isn't in
  the palette), geometry placed just ahead of the spawn point along the track curve's samples. Gated
  behind `?cones=1` in the URL (off by default) so it doesn't change the existing scene/prompt
  pairing unless explicitly requested.

## Runtime gotcha: one session per worker process

- `runtime/server.py`'s `Bridge.create()` (`server.py:69`) refuses a second `/session` while any
  prior session is still open: `{"error": "One active GPU session is supported; close it before
  restarting"}` (HTTP 400). There's no "list sessions" or "force-close" endpoint — only
  `/close/<id>`, which needs the session's own id. A `replay_example.py` run that gets interrupted
  (Ctrl-C, crashed client, connection reset) leaves its session open with no way to recover the id
  from outside that process. In practice, the only reliable recovery is restarting the whole worker.
- Relatedly: `launch_local.sh`'s trap-based cleanup (`cleanup() { kill "$worker_pid"
  "${frontend_pid:-$worker_pid}" ... }`) can still leave vite orphaned if the worker side exits via
  a path that doesn't go through the trap cleanly — recurred again here as a single orphaned `node
  vite` process holding port 5191 after a worker exit, separate from the originally-fixed crash-
  before-frontend-starts case. Killed it by exact PID after explicit confirmation (never blanket
  `pkill -f` on this host — that was attempted once this session and corrected immediately per
  standing policy on this DGX).

## Open items

- `configs/runtime.json` shard_size=1 patch validated (`json.load` confirms `1`) and committed
  locally on the DGX (not pushed).
- Two copies of `runtime.json` exist on the DGX (`~/Sana/models/streamrender_h3/configs/` and
  `~/Sana/models/StreamRender-H3/configs/`, the latter still `shard_size: 4`) — confirmed via
  `runtime/config.py`'s `ROOT = Path(__file__).resolve().parents[1]` that `worker.py` only ever
  reads the first (`streamrender_h3/configs/runtime.json`, the patched one). The second copy is
  unused dead weight from the asset download, not a risk.
- `render/pipeline.py`'s `inference_mode`→`no_grad` patch and the DGX-side `runtime.json` edit are
  local-only (not scp'd back / committed into the repo's tracked diff yet beyond the local commit
  already made for `runtime.json`) — worth reconciling into a clean diff if this is to be upstreamed
  or reproduced on another host.
