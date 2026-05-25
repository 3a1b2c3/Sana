"""Gradio app for SANA-WM_bidirectional.

Wraps :class:`inference_sana_wm.SanaWMPipeline` with a browser UI. Loads the
pipeline once at startup (~5 min) and reuses it across generations, so each
subsequent click only pays the diffusion + refiner cost.

Launch:
    python app/app_sana_wm.py
        [--config <local config.yaml or hf://...>]
        [--model_path <local .safetensors or hf://...>]
        [--no_refiner | --offload_vae | --offload_refiner]
        [--server_port 7860]
        [--share]

Or run via test_sana_wm.bat (which activates .venv-wm + sets the right CLI args).
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import imageio.v3 as iio
import numpy as np
import pyrallis
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference_video_scripts"))

from inference_sana_wm import (
    GenerationParams,
    HF_DEFAULTS,
    InferenceConfig,
    RefinerSettings,
    SanaWMPipeline,
    action_string_to_c2w,
    apply_overlay,
    load_intrinsics,
    resize_and_center_crop,
    resolve_hf_path,
    transform_intrinsics_for_crop,
)

ASSET_DIR = REPO_ROOT / "asset" / "sana_wm"
OUTPUT_DIR = REPO_ROOT / "results" / "sana_wm_gradio"

DEMO_PRESETS: dict[str, dict] = {}
for stem in ("demo_0", "demo_1", "demo_2"):
    img = ASSET_DIR / f"{stem}.png"
    if not img.exists():
        continue
    txt = ASSET_DIR / f"{stem}.txt"
    intr = ASSET_DIR / f"{stem}_intrinsics.npy"
    DEMO_PRESETS[stem] = {
        "image": img,
        "prompt": txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else "",
        "intrinsics": intr if intr.exists() else None,
    }

PIPELINE: SanaWMPipeline | None = None


def _build_pipeline(args: argparse.Namespace) -> SanaWMPipeline:
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig,
        config_path=resolve_hf_path(args.config),
        args=[],
    )
    refiner = None if args.no_refiner else RefinerSettings(
        checkpoint=args.refiner_checkpoint,
        gemma_root=args.refiner_gemma_root,
        sink_size=args.sink_size,
        seed=args.refiner_seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sana-wm-gradio] device={device}, refiner={'on' if refiner else 'off'}, "
          f"offload_vae={args.offload_vae}, offload_refiner={args.offload_refiner}", flush=True)
    return SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(args.model_path),
        device=device,
        refiner=refiner,
        offload_vae=args.offload_vae,
        offload_refiner=args.offload_refiner,
    )


def _load_demo(demo_name: str):
    if demo_name not in DEMO_PRESETS:
        return None, "", None
    p = DEMO_PRESETS[demo_name]
    img = Image.open(p["image"]).convert("RGB")
    intr_path = str(p["intrinsics"]) if p["intrinsics"] else None
    return img, p["prompt"], intr_path


def _generate(
    image,
    prompt: str,
    action: str,
    intrinsics_file,
    num_frames: float,
    fps: float,
    step: float,
    cfg_scale: float,
    flow_shift: float,
    seed: float,
    translation_speed: float,
    rotation_speed_deg: float,
    skip_refiner: bool,
    skip_overlay: bool,
    sampling_algo: str,
    negative_prompt: str,
    progress=gr.Progress(),
):
    if PIPELINE is None:
        raise gr.Error("Pipeline isn't built yet — wait for startup to finish.")
    if image is None:
        raise gr.Error("Upload or pick a first-frame image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")
    if not action or not action.strip():
        raise gr.Error("Action DSL cannot be empty (e.g. `w-80,jw-40`).")
    if intrinsics_file is None:
        raise gr.Error("Provide an intrinsics .npy (Pi3X intrinsics estimation crashes on Windows SDPA — skip it).")

    progress(0.0, desc="Preparing inputs")
    image = image if isinstance(image, Image.Image) else Image.open(image)
    image = image.convert("RGB")
    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)

    c2w_full = action_string_to_c2w(
        action,
        translation_speed=float(translation_speed),
        rotation_speed_deg=float(rotation_speed_deg),
    )
    nf = int(min(num_frames, c2w_full.shape[0]))
    c2w = c2w_full[:nf]

    intr_path = Path(intrinsics_file.name if hasattr(intrinsics_file, "name") else intrinsics_file)
    intr_src = load_intrinsics(intr_path, nf)
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    params = GenerationParams(
        num_frames=nf,
        fps=int(fps),
        step=int(step),
        cfg_scale=float(cfg_scale),
        flow_shift=float(flow_shift) if flow_shift else None,
        seed=int(seed),
        negative_prompt=(negative_prompt or "").strip(),
        sampling_algo=sampling_algo,
    )

    progress(0.05, desc=f"Generating {nf} frames @ {fps} fps, {step} DiT steps")
    if skip_refiner and PIPELINE.refiner_settings is not None:
        saved = PIPELINE.refiner_settings
        PIPELINE.refiner_settings = None
        try:
            out = PIPELINE.generate(cropped, prompt, c2w, intrinsics_vec4, params)
        finally:
            PIPELINE.refiner_settings = saved
    else:
        out = PIPELINE.generate(cropped, prompt, c2w, intrinsics_vec4, params)

    video_hwc = out["video"]
    if not skip_overlay:
        progress(0.95, desc="Applying WASD/joystick overlay")
        video_hwc = apply_overlay(video_hwc, out["c2w"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"gradio_{int(np.random.randint(0, 1_000_000)):06d}"
    out_path = OUTPUT_DIR / f"{stem}.mp4"
    iio.imwrite(str(out_path), video_hwc, fps=int(fps))
    print(f"[sana-wm-gradio] Saved {out_path}", flush=True)
    return str(out_path)


def _build_ui() -> gr.Blocks:
    presets = list(DEMO_PRESETS.keys())
    with gr.Blocks(title="SANA-WM bidirectional", analytics_enabled=False) as demo:
        gr.Markdown("# SANA-WM bidirectional — local Gradio")
        gr.Markdown(
            "Generate camera-controlled 720p videos. Pick a preset for a one-click smoke test, "
            "or upload your own image + intrinsics. Action DSL is `<keys>-<frames>` joined by commas "
            "(`w`/`a`/`s`/`d` translate, `i`/`j`/`k`/`l` rotate, `none` holds)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                preset_dd = gr.Dropdown(label="Preset (loads image + prompt + intrinsics)", choices=presets, value=None)
                image_in = gr.Image(label="First frame", type="pil", height=300)
                intrinsics_in = gr.File(label="Intrinsics .npy", file_types=[".npy"])
                prompt_in = gr.Textbox(label="Prompt", lines=3)
                action_in = gr.Textbox(label="Action DSL", value="w-80,jw-40,w-40,lw-60,w-100")
                with gr.Row():
                    trans_in = gr.Number(label="translation_speed", value=0.055)
                    rot_in = gr.Number(label="rotation_speed_deg", value=1.2)
            with gr.Column(scale=1):
                with gr.Row():
                    nf_in = gr.Slider(label="num_frames", value=321, minimum=21, maximum=601, step=4)
                    fps_in = gr.Slider(label="fps", value=16, minimum=8, maximum=30, step=1)
                with gr.Row():
                    step_in = gr.Slider(label="DiT steps", value=60, minimum=10, maximum=100, step=2)
                    cfg_in = gr.Slider(label="cfg_scale", value=5.0, minimum=1.0, maximum=10.0, step=0.5)
                with gr.Row():
                    fs_in = gr.Number(label="flow_shift (0 = config default)", value=8.0)
                    seed_in = gr.Number(label="seed", value=42, precision=0)
                with gr.Row():
                    no_ref_in = gr.Checkbox(label="--no_refiner (faster, lower quality)", value=False)
                    no_ov_in = gr.Checkbox(label="--no_action_overlay", value=False)
                algo_in = gr.Dropdown(
                    label="sampling_algo",
                    choices=["flow_euler_ltx", "flow_euler", "flow_dpm-solver"],
                    value="flow_euler_ltx",
                )
                neg_in = gr.Textbox(label="negative_prompt (optional)", lines=2)
                go_btn = gr.Button("Generate", variant="primary")
                video_out = gr.Video(label="Output mp4", interactive=False, height=400)

        preset_dd.change(_load_demo, inputs=[preset_dd], outputs=[image_in, prompt_in, intrinsics_in])
        go_btn.click(
            _generate,
            inputs=[
                image_in, prompt_in, action_in, intrinsics_in,
                nf_in, fps_in, step_in, cfg_in, fs_in, seed_in,
                trans_in, rot_in,
                no_ref_in, no_ov_in, algo_in, neg_in,
            ],
            outputs=[video_out],
        )
    return demo


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=HF_DEFAULTS["config"])
    p.add_argument("--model_path", default=HF_DEFAULTS["model_path"])
    p.add_argument("--no_refiner", action="store_true")
    p.add_argument("--refiner_checkpoint", default=HF_DEFAULTS["refiner_checkpoint"])
    p.add_argument("--refiner_gemma_root", default=HF_DEFAULTS["refiner_gemma_root"])
    p.add_argument("--refiner_seed", type=int, default=42)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument("--offload_vae", action="store_true")
    p.add_argument("--offload_refiner", action="store_true")
    p.add_argument("--server_name", default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main() -> None:
    global PIPELINE
    args = _parse_args()
    print("[sana-wm-gradio] Building pipeline — this takes a few minutes on first launch.", flush=True)
    PIPELINE = _build_pipeline(args)
    print("[sana-wm-gradio] Pipeline ready. Launching Gradio UI.", flush=True)
    demo = _build_ui()
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
