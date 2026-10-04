"""Upsample the 256x192 LiDAR depth to RGB resolution with PromptDA (geometry config B).

    python -m foodscan.depth_upsample SCAN_DIR [--stride 1] [--out-sub depth_hr]

Writes SCAN_DIR/depth_hr/NNNNNN.f32 as float32 metres at 1440x1920 (row-major) and
SCAN_DIR/depth_hr/meta.json. Requires the PromptDA repo on PYTHONPATH and a GPU.

Contract (from promptda/promptda.py): image float32 (1,3,H,W) in [0,1] with H, W
multiples of 14; prompt depth float32 (1,1,h,w) in metres at any resolution; output
(1,1,H,W) metres. Holes (0 / low confidence) are not handled by the model, so they
are KNN-filled first (the author's own workaround, issue #3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .loader import Scan

PAD_SIZE = (1904, 1428)  # (w, h): nearest multiples of 14 below 1920x1440


def fill_holes(d: np.ndarray, conf: np.ndarray | None, min_conf: int = 1, k: int = 4) -> np.ndarray:
    from scipy.spatial import cKDTree
    bad = (d <= 0) | ~np.isfinite(d)
    if conf is not None:
        bad |= conf < min_conf
    if bad.any() and (~bad).any():
        yy, xx = np.nonzero(~bad)
        by, bx = np.nonzero(bad)
        _, idx = cKDTree(np.c_[yy, xx]).query(np.c_[by, bx], k=k)
        d = d.copy()
        d[by, bx] = d[yy[idx], xx[idx]].mean(1)
    return d


class Upsampler:
    def __init__(self, repo_id: str = "depth-anything/prompt-depth-anything-vitl", device: str = "cuda"):
        import torch
        from promptda.promptda import PromptDA
        self.torch = torch
        self.device = device
        self.model = PromptDA.from_pretrained(repo_id).to(device).eval()

    def __call__(self, rgb_u8: np.ndarray, depth_m: np.ndarray, conf: np.ndarray | None) -> np.ndarray:
        import cv2
        torch = self.torch
        H0, W0 = rgb_u8.shape[:2]
        img = cv2.resize(rgb_u8.astype(np.float32) / 255.0, PAD_SIZE, interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        pd = torch.from_numpy(fill_holes(depth_m, conf).astype(np.float32))[None, None].to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.predict(img, pd)
        out = torch.nn.functional.interpolate(out.float(), (H0, W0), mode="bilinear", align_corners=False)
        return out[0, 0].cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out-sub", default="depth_hr")
    ap.add_argument("--min-conf", type=int, default=1, help="pixels below this are hole-filled before prompting")
    args = ap.parse_args()

    scan = Scan(args.scan_dir)
    out = Path(args.scan_dir) / args.out_sub
    out.mkdir(exist_ok=True)
    up = Upsampler()
    frames = scan.good_frames(args.stride)
    for f in frames:
        d = up(f.rgb(), f.depth(), f.confidence())
        d.astype("<f4").tofile(out / f"{f.index:06d}.f32")
    (out / "meta.json").write_text(json.dumps({
        "model": "depth-anything/prompt-depth-anything-vitl", "shape_hw": list(frames[0].rgb_size[::-1]),
        "units": "meters float32 LE row-major", "frames": len(frames), "hole_fill_min_conf": args.min_conf,
    }, indent=1))
    print(f"wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
