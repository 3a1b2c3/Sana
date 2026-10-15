# StreamRender-H3: MXFP8 option, live metrics, and running on one GB300

Companion to [DGX_FINDINGS.md](DGX_FINDINGS.md). Covers what was added after that bring-up, how to run it, and
what is and is not measured. Status: the MXFP8 path and the metrics changes are written but have not been run.

## What was added

| File | Change |
| --- | --- |
| `render/mxfp8.py` | New. Block-scaled MXFP8 quantize kernel, `MXFP8Linear`, `MXActivation`, and three fused producers (RMSNorm+modulate, residual+gate+RMSNorm+modulate, SiLU-and-mul), ported from `models/minimax_h3/Sol-H3/h3_runtime/mxfp8.py`. The SiLU-and-mul producer is adapted: this model computes `silu(first half) * second half`, the reverse of Sol-H3's layout. |
| `render/compute_quant.py` | New. `install()` swaps the attention and FFN linears of selected DiT blocks for MXFP8 GEMMs. `install_fused_producers()` replaces those blocks' forward with one that feeds the GEMMs from the fused producers. |
| `runtime/worker.py` | Calls `install()` when `STREAMRENDER_COMPUTE_QUANT=mxfp8` and `install_fused_producers()` when `STREAMRENDER_MXFP8_FUSED=1`. Records every chunk's metrics per session and logs `STREAMRENDER_SUMMARY` (with the mode) when the session ends. |
| `tests/test_mxfp8_producers.py` | New. GPU checks of each producer against an FP32 reference and of `MXFP8Linear` on a prequantized input. Skipped without an SM100-family GPU. |
| `coding/threejs-racing/src/runtime-client.js` | Status panel shows fps, stage timings, rolling median and session totals. |

## MXFP8 compute profile

Opt-in, default off. Environment variables:

| Variable | Values | Meaning |
| --- | --- | --- |
| `STREAMRENDER_COMPUTE_QUANT` | `none` (default), `mxfp8` | Enable the profile. |
| `STREAMRENDER_QUANT_BLOCKS` | `FIRST-LAST`, default `2-46` | Inclusive block range to quantize. The backbone has 50 blocks. |
| `STREAMRENDER_MXFP8_FUSED` | `0` (default), `1` | Also fuse each quantized block's elementwise chains into the activation quantize (see below). Requires `STREAMRENDER_COMPUTE_QUANT=mxfp8`; setting it without that raises an error. |

- Replaced per block: `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`.
- Kept in BF16: blocks outside the range, AdaLN projections, token refiners, VAE, text and audio encoders. This matches
  the validated Sol-H3 profile.
- Installation is one-way: the BF16 weights are released after quantization, so VRAM drops along with compute.
- Requires an SM100-family GPU (GB200/GB300, compute capability major 10), `scaled_mm` with `ScalingType`/`SwizzleType`,
  and `torch.float8_e8m0fnu`. `install()` raises a clear error otherwise.
- Requires plain tensor weights. If the backbone is FSDP-wrapped and the weights are sharded tensors, `install()` raises
  a `TypeError` instead of quantizing incorrectly.
- Startup prints `STREAMRENDER_MXFP8 blocks=2-46 linears=... X GiB -> Y GiB`.

### Fused producers (`STREAMRENDER_MXFP8_FUSED=1`)

Three modes can be compared: BF16, MXFP8 (separate activation quantize before each GEMM), and MXFP8 with fused
producers. In the third, for each block in the range:

1. `norm1` + modulation is computed and quantized in one kernel, feeding `qkv_proj`.
2. The attention gated residual, `norm2` and modulation are one kernel, producing the BF16 residual and the MXFP8
   input of `fc1`.
3. SiLU-and-mul is computed and quantized in one kernel, feeding `fc2`. The final gated residual stays eager.

Blocks outside the range keep the stock forward. If a block in the range is not MXFP8-quantized, or has an AdaLN
condition head, installation raises instead of patching it. The startup line reports `fused_producers=True`, and the
summary carries `compute_quant` and `mxfp8_fused_producers`.

Numerics: the producers keep the chain in FP32 and round once, where the eager chain rounds to BF16 after every
operation, so results differ from eager by BF16 rounding on top of the MXFP8 quantization. This is the same
approximate class as the MXFP8 GEMMs themselves, not a bitwise-equal change.

Test on the DGX from `models/streamrender_h3` (it skips without an SM100 GPU):
`python -m unittest tests.test_mxfp8_producers -v`. It checks each producer against an FP32 reference, that the
SiLU-and-mul uses the first half as the gate (the layout trap), and that `MXFP8Linear` gives identical results for a
BF16 input and the same input prequantized.

Expected effect, not measured: roughly 1.3 to 1.6 times faster DiT compute is a planning estimate. Quality has not been
checked. Compare against the BF16 run on the same example before relying on it.

## Metrics

### Per chunk (`STREAMRENDER_CHUNK` log line, `timings.jsonl`, and the UI status)

| Field | Meaning |
| --- | --- |
| `round` | Chunk index within the session. Round 0 is the bootstrap chunk and includes first-use compile cost. |
| `output_frames` | RGB frames published for this chunk. |
| `encode_ms`, `render_ms`, `decode_ms` | GPU-side stage times (reference encode, two-NFE diffusion step, causal decode). |
| `rgb_ready_wall_ms` | Encode plus render plus decode, wall clock. Excludes JPEG delivery, network, and browser paint. |
| `publication_interval_ms` | Time since the previous chunk was ready. Null for round 0. |
| `output_fps` | `output_frames * 1000 / publication_interval_ms`, the sustained frame rate of this chunk. Null for round 0. Logged by the worker and also derived in the browser panel. |
| `oldest_input_to_rgb_ready_ms` | From receipt of the oldest input frame in the chunk to RGB ready. An upper bound for that chunk, not the lag of one keypress. |

Sustained fps is `output_frames * 1000 / publication_interval_ms`, logged per chunk as `output_fps`.

### Browser status panel

Three lines: the current chunk, the median of the last 10 chunks, and session totals (chunks, frames, average fps over
the sum of publication intervals). Totals reset each time the stream is started. The first chunk's frames are left out
of the average because that chunk has no interval.

### Session summary (`STREAMRENDER_SUMMARY`, `outputs/<session>/summary.json`)

Written when a session ends by closing or by going idle for `session_idle_seconds` (120 in `configs/runtime.json`).
Contains `chunks`, `output_frames`, `seconds`, `average_fps`, `median_fps` and `median_*` of each timing field. Medians
and the average exclude chunk 0. Stopping the worker with Ctrl+C while a session is open writes no summary, so end the
session first.

## Running on a single GB300

Run from `~/Sana/models/streamrender_h3` on the DGX.

1. `source .venv/bin/activate && nvidia-smi` (check the GPU is free first).
2. Baseline: `PYTHON_BIN=$(which python) bash scripts/launch_local.sh --assets configs/example.local.json`
3. Wait for `STREAMRENDER_READY`.
4. From the PC: `ssh -N -L 5191:127.0.0.1:5191 -L 8765:127.0.0.1:8765 kschmid@10.74.11.118`
5. Open `http://127.0.0.1:5191`, click "Start semantic stream", and drive.
6. End the session, then save the `STREAMRENDER_SUMMARY` line.
7. MXFP8: stop the worker, then launch with `STREAMRENDER_COMPUTE_QUANT=mxfp8` in front of the same command. Repeat
   steps 3 to 6 and compare the two summaries.

## Pitfalls hit during bring-up

- `scripts/launch_local.sh` runs `${PYTHON_BIN:-python3}`. Activating the venv is not enough; pass `PYTHON_BIN`.
  Without it the system `python3` has no torch.
- `configs/assets.local.json` can be an unfilled copy of `assets.example.json` (placeholder paths such as
  `/path/to/backbone`), which fails with `FileNotFoundError`. Pass a real manifest with `--assets`, for example
  `configs/example.local.json`.
- `runtime/worker.py` needs `world not in (1, 4, 8)` for single-GPU runs. A copy taken from a checkout without that
  patch fails the world-size check.
- A `{"error": "Unknown session"}` reply from port 8765 is normal for any path other than `/health` before a session
  exists. Use the UI on port 5191 and click "Start semantic stream".
- The long `cutlass` `aux_data` warning at startup is the FlashAttention-4 kernel compiling on first use.
- Both services listen on the DGX loopback only; use the SSH tunnel.
- The port-cleanup block in `launch_local.sh` runs `fuser -k` on ports 5191 and 8765 before starting. It terminates
  whatever is bound there, and it prints the PIDs.

## What we learned

- **The runtime already logs an input-latency metric.** `oldest_input_to_rgb_ready_ms` is added in `runtime/worker.py`
  per chunk. It is an upper bound for the chunk (oldest input frame to RGB ready), not single-keypress lag.
- **No public end-to-end input-latency number exists** for StreamRender-H3. The project page reports DiT and VAE stage
  times only (218 ms per DiT step on 8x GB200, 23.8 ms VAE decode on one GB200). The AlayaCrafter project has no
  published fps or latency either, and its repository has no code or paper yet.
- **StreamRender-H3 is not vanilla H3.** It is a distilled streaming variant (`MiniMaxH3VideoRefStreamingPrefixDMD`,
  two sampling steps, causal model with prefix cache and sink tokens, streaming decoder). Its speed numbers do not
  apply to the base model.
- **Chunk 0 is a bootstrap chunk.** It covers 30 reference frames and 9 native latents, produced 26 output frames, and
  includes the FlashAttention-4 compile. Do not use it as steady state. An earlier guess of about 7 frames per chunk
  came from dividing the saved JPEG count by the round count and should be replaced by the measured `output_frames`.
- **Single-GPU limits.** Context parallelism (Ulysses) does nothing on one GPU, so the 8-GPU speedup does not transfer.
  The DiT is the bottleneck; of the optimizations the page does not list, low-precision compute (this MXFP8 path),
  lower resolution plus upscaling, motion-vector reprojection of in-between frames, and DiT/VAE overlap are the open
  ones. Their size is an estimate until measured.
- **The vendored model runs every elementwise chain eagerly.** `_SGL_KERNELS_AVAILABLE = False` is hardcoded in
  `model.py`, which disables all of the upstream fused fast paths (indexed modulate, gated residual, SiLU-and-mul,
  fused qk-norm+RoPE) and leaves the eager PyTorch fallbacks. That makes fusion a larger opportunity than it looks, and
  BF16 fused kernels (without MXFP8) would be a lossless-leaning option to try. The fused qk-norm+RoPE is not yet
  covered by the new producers.
- **Layout trap.** Sol-H3's SwiGLU kernel takes the first half as the value and the second as the gate. This model
  does the reverse (`silu(first) * second`), so the kernel was adapted and is covered by a test.
- **Connection refused through the tunnel** (`channel N: open failed: connect failed: Connection refused`) means the
  tunnel works but nothing listens on the DGX loopback. The worker or frontend is not running or has not finished
  loading. Check with `ss -ltnp | grep -E "5191|8765"`.
- **`launch_local.sh` stops by Ctrl+C.** Its trap stops both the worker and the frontend. A blanket kill on the DGX
  is not needed and could hit other jobs.

## Known numbers and what is missing

Measured on the DGX before these changes (see DGX_FINDINGS.md): about 927 ms `render_ms` per round and about 1.15 s
publication interval on one GB300. The first chunk of a later run: `render_ms` 2890, `decode_ms` 144,
`rgb_ready_wall_ms` 3091, 26 output frames, which is the bootstrap chunk and not steady state.

Not yet measured: steady-state fps and `oldest_input_to_rgb_ready_ms` per chunk, the MXFP8 speedup and its quality
effect, and true glass-to-glass input latency. A marker-injection test in `benchmarks/browser_loop.mjs` or a
high-speed-camera measurement would cover the last one.
