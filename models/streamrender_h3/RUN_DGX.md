# Running StreamRender-H3 on the DGX (single GPU)

Run guide for `pmgb300ws-0304` (GB300, 1 GPU used). The build fixes, patches and debugging history are in
[`DGX_FINDINGS.md`](DGX_FINDINGS.md); the upstream description of the runtime is in [`README.md`](README.md).

**Status.** A full run (launch, `STREAMRENDER_READY`, then `scripts/replay_example.py --example B10_e-track-3_bluehour`) completed on this
host with steady `STREAMRENDER_CHUNK` events (about 1.15 s publication interval, about 927 ms render time per round) and correct frames
(see `DGX_FINDINGS.md`, "Status"). The commands below are the ones used there.

## 1. Start the worker and the frontend

```bash
cd ~/Sana/models/streamrender_h3
PYTHON_BIN=.venv/bin/python bash scripts/launch_local.sh --assets configs/example.local.json
```

- `GPUS` defaults to 1 here (the upstream default of 8 was patched in `launch_local.sh`).
- The worker loads the models, warms up and then prints `STREAMRENDER_READY`. The frontend (Vite) is on port 5191 and the worker's
  `/health` endpoint on port 8765. Both listen on `127.0.0.1` only.
- Before starting, the script runs `fuser -k -TERM` on ports 5191 and 8765. That frees leftovers from a previous run, and it also
  stops any other process of yours holding those ports.
- Other scene configs in `configs/`: `taxi.local.json` (yellow taxi variant of the example prompt), `taxi_10s.local.json`,
  `cones.local.json` and `taxi_cones.local.json` (names as in the repo; see `DGX_FINDINGS.md`, "Customizing the scene", for the taxi and cone
  work). Use one with `--assets configs/<name>.local.json`.

## 2. Open it from your Windows machine

The ports are not reachable from other machines directly (browsing to `http://10.74.11.118:5191` gives `ERR_CONNECTION_REFUSED`).
Tunnel them from a second PowerShell window and keep it open:

```
ssh -N -L 5191:127.0.0.1:5191 -L 8765:127.0.0.1:8765 kschmid@10.74.11.118
```

Then browse to `http://127.0.0.1:5191`; `http://127.0.0.1:8765/health` shows whether the worker is ready. In the page enable autopilot or drive with
WASD, then click the live streaming-render button. Add `?cones=1` to the URL for the optional cones, or `?carColor=yellow` for the yellow car.

## 3. Or replay an example without a browser

With the worker ready, in a second SSH session:

```bash
cd ~/Sana/models/streamrender_h3
.venv/bin/python scripts/replay_example.py --example B10_e-track-3_bluehour
```

Output frames go to `outputs/replay/<session>_NNNNNN.jpg` (about 166 MB and 1000+ JPEGs in the recorded run).

## 4. Take results back to Windows

Exports saved by the page are in `coding/threejs-racing/exports/<timestamp>-<id>/` (`controls.jsonl`, `manifest.json`, `palette.json`,
`ref.png` and a `semantic/` folder of frames). Copy one with the DGX's IP, not its hostname:

```
scp -r kschmid@10.74.11.118:/localhome/kschmid/Sana/models/streamrender_h3/coding/threejs-racing/exports/<folder> C:\workspace\world\Sana\models\streamrender_h3\coding\threejs-racing\exports\
```

## Known problems

| Symptom | Cause and fix |
|---|---|
| `set: pipefail: invalid option name` (the line number varies) | The script has Windows line endings. The tracked copy of `launch_local.sh` had CRLF; strip them with `sed -i 's/\r$//' scripts/launch_local.sh`. Windows copies of `launch_slurm.sh`, `run_worker_node.sh` and `setup.sh` also have CRLF. Always copy `.sh` files as LF. |
| Port 5191 or 8765 already in use | A previous worker or Vite is still running. The launcher clears the ports itself, but an orphaned `node vite` can survive a worker exit. Find it with `ss -ltnp`, then stop it by its exact PID. Never `pkill -f` on this host. |
| A second session is refused ("restarting") | One session per worker process. An interrupted `replay_example.py` leaves its session open, and there is no endpoint to list or force-close sessions, so restart the whole worker. |
| `ERR_CONNECTION_REFUSED` in the browser | The tunnel is not open, or the page was opened with the DGX's address instead of `127.0.0.1`. |
| Out of memory or a very slow start | The DGX GPU is shared; check `nvidia-smi` for other jobs first. |

## Not verified here

- The Windows `setup_env.sh` and `download_models.sh` have not been run by me; setup on the DGX is as recorded in `DGX_FINDINGS.md`.
- I have not run the worker myself in this session, and I do not know which of the extra configs were last run.
