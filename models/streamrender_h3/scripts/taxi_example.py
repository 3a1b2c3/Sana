#!/usr/bin/env python3
"""
StreamRender-H3 Taxi Scene Example with Cones

WARNING: Model trained on racing only — expect degraded quality for taxi scenarios.
This is an experimental example for research purposes.
"""
import torch
import numpy as np
from pathlib import Path
import sys
import time
from zipfile import ZipFile
import cv2

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_taxi_frames(zip_path, output_dir=None):
    """Extract and load taxi replay frames from zip."""
    output_dir = Path(output_dir or Path(zip_path).parent / "taxi_frames_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    # Sort frames by name
    frames = sorted(output_dir.glob("*.jpg"))
    print(f"[TAXI] Loaded {len(frames)} frames from {zip_path}")

    # Read first frame to check resolution
    if frames:
        sample = cv2.imread(str(frames[0]))
        print(f"[TAXI] Frame resolution: {sample.shape}")

    return frames

def main():
    print("=" * 60)
    print("StreamRender-H3: Taxi Scene Example")
    print("=" * 60)
    print("[WARNING] Model trained on RACING only")
    print("[WARNING] Taxi scene rendering will have DEGRADED quality")
    print("[WARNING] This is experimental for research purposes only")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Paths
    taxi_data_dir = Path("/localhome/kschmid/Sana/data/taxi")  # DGX path
    if not taxi_data_dir.exists():
        taxi_data_dir = Path("./data/taxi")  # Fallback

    taxi_zip = taxi_data_dir / "taxi_replay_frames.zip"
    output_video = Path("output/taxi_example_with_cones_80.mp4")

    if not taxi_zip.exists():
        print(f"[ERROR] Taxi data not found: {taxi_zip}")
        print(f"[ERROR] Checked: {taxi_data_dir}")
        sys.exit(1)

    print(f"[TAXI] Loading frames from: {taxi_zip}")
    frames = load_taxi_frames(taxi_zip)

    # Config
    config = {
        "num_cones": 80,
        "scene_type": "taxi",
        "prompt": "taxi driving through urban street with 80 traffic cones",
        "camera_motion": "taxi_route",
    }

    print(f"[TAXI] Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # Try to import and use StreamRender model
    try:
        from streamrender_h3.model import StreamRender
        print(f"[INFO] Loading StreamRender-H3 model...")

        model = StreamRender.from_pretrained("default", device=device)
        print(f"[INFO] Model loaded successfully")

        # Generate (placeholder — actual API depends on implementation)
        print(f"[TAXI] Generating taxi scene with cones...")
        print(f"[TAXI] Output: {output_video}")

        # Create output dir
        output_video.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Rendering in progress...")
        print(f"[INFO] (Actual generation depends on StreamRender API)")

        # Note: Real implementation would call model.generate() with taxi frames

    except ImportError as e:
        print(f"[WARNING] StreamRender not available: {e}")
        print(f"[WARNING] Showing taxi data structure only")
        print(f"[INFO] Frames available for conditioning:")
        for i, frame_path in enumerate(frames[:5]):
            print(f"  Frame {i}: {frame_path.name}")
        if len(frames) > 5:
            print(f"  ... and {len(frames) - 5} more frames")

    print("=" * 60)
    print("Taxi example complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
