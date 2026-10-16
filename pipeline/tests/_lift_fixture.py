"""Shared fixture: hard_v10 synthetic scans with oracle per-frame masks standing in for SAM output."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.labels import estimate_normals  # noqa: E402
from foodscan.loader import Scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "data/synth/hard_v10/scans"


def scene_dirs(n: int = 3) -> list[Path]:
    return sorted(p for p in ROOT.glob("s*") if (p / "gt/points.npz").exists())[:n]


def load(scene: Path):
    """(scan, xyz float64, gt labels, oracle masks {frame: uint8}, normals).

    Oracle masks: food ids from gt/masks, containers as 254, no clutter masks (a real
    segmenter would give 253; here clutter points just vote background).
    """
    scan = Scan(scene)
    z = np.load(scene / "gt/points.npz")
    xyz, gt = z["xyz"].astype(np.float64), z["label"]
    masks = {}
    for f in scan.frames:
        m = np.asarray(Image.open(scene / f"gt/masks/{f.index:06d}.png")).copy()
        c = np.asarray(Image.open(scene / f"gt/container/{f.index:06d}.png"))
        m[(c > 0) & (m == 0)] = 254
        masks[f.index] = m
    return scan, xyz, gt, masks, estimate_normals(xyz)
