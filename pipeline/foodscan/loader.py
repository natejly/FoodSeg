"""Read a FoodScan capture directory produced by the iPad app.

Layout (see capture-app/Sources/ScanRecorder.swift):
    meta.json, frames.jsonl, rgb/NNNNNN.jpg, depth/NNNNNN.f32,
    smooth_depth/NNNNNN.f32, conf/NNNNNN.u8

Poses are converted from ARKit's camera convention (+Y up, -Z forward) to
OpenCV's (+Y down, +Z forward), which is what Open3D expects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

# Right-multiply an ARKit camera-to-world by this to get an OpenCV one.
_ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

CONF_LOW, CONF_MEDIUM, CONF_HIGH = 0, 1, 2


@dataclass
class Frame:
    index: int
    timestamp: float
    tracking_state: str
    K_rgb: np.ndarray          # 3x3 intrinsics for the full-res color image
    K_depth: np.ndarray        # 3x3 intrinsics scaled to the depth map
    T_cw: np.ndarray           # 4x4 camera-to-world, OpenCV convention
    rgb_size: tuple[int, int]  # (w, h)
    depth_size: tuple[int, int]
    _dir: Path

    @property
    def T_wc(self) -> np.ndarray:
        """World-to-camera (Open3D's 'extrinsic')."""
        return np.linalg.inv(self.T_cw)

    @property
    def _name(self) -> str:
        return f"{self.index:06d}"

    depth_source: str = "depth"      # "depth" (256x192 LiDAR) or "depth_hr" (PromptDA at RGB res)
    lidar_size: tuple[int, int] = (256, 192)

    def rgb(self) -> np.ndarray:
        return np.asarray(Image.open(self._dir / "rgb" / f"{self._name}.jpg").convert("RGB"))

    def depth(self, smoothed: bool = False) -> np.ndarray:
        if self.depth_source == "depth_hr":
            sub = "depth_hr"
        else:
            sub = "smooth_depth" if smoothed else "depth"
        w, h = self.depth_size
        raw = np.fromfile(self._dir / sub / f"{self._name}.f32", dtype="<f4")
        return raw.reshape(h, w)

    def confidence(self) -> np.ndarray | None:
        p = self._dir / "conf" / f"{self._name}.u8"
        if not p.exists():
            return None
        lw, lh = self.lidar_size
        c = np.fromfile(p, dtype=np.uint8).reshape(lh, lw)
        if self.depth_source == "depth_hr":
            # nearest-neighbour upsample of the LiDAR confidence to the HR grid
            w, h = self.depth_size
            ys = (np.arange(h) * lh) // h
            xs = (np.arange(w) * lw) // w
            c = c[ys][:, xs]
        return c

    def masked_depth(self, min_conf: int = CONF_HIGH, smoothed: bool = False,
                     max_depth: float | None = None) -> np.ndarray:
        """Depth with low-confidence (and optionally far) pixels zeroed."""
        d = self.depth(smoothed).copy()
        c = self.confidence()
        if c is not None:
            d[c < min_conf] = 0.0
        if max_depth is not None:
            d[d > max_depth] = 0.0
        d[~np.isfinite(d)] = 0.0
        return d


class Scan:
    """A capture directory.

    poses: None (ARKit) or "refined" (SCAN_DIR/poses_refined.json from refine_poses.py;
           frames without a refined pose are dropped).
    depth: "depth" (LiDAR 256x192) or "depth_hr" (SCAN_DIR/depth_hr from depth_upsample.py;
           frames without an HR file are dropped).
    """

    def __init__(self, path: str | Path, poses: str | None = None, depth: str = "depth"):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.depth_source = depth
        refined = None
        if poses == "refined":
            refined = {int(k): np.array(v) for k, v in
                       json.loads((self.dir / "poses_refined.json").read_text()).items()}
        self.frames: list[Frame] = []
        with open(self.dir / "frames.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if refined is not None:
                    if r["index"] not in refined:
                        continue
                    r["transform"] = refined[r["index"]].tolist()
                if depth == "depth_hr" and not (self.dir / "depth_hr" / f"{r['index']:06d}.f32").exists():
                    continue
                self.frames.append(self._parse(r))

    def _parse(self, r: dict) -> Frame:
        K = np.array(r["intrinsics"], dtype=np.float64)
        T_arkit = np.array(r["transform"], dtype=np.float64)
        rgb_w, rgb_h = r["imageWidth"], r["imageHeight"]
        d_w, d_h = r["depthWidth"], r["depthHeight"]
        if self.depth_source == "depth_hr":
            K_depth, depth_size = K.copy(), (rgb_w, rgb_h)
        else:
            K_depth, depth_size = np.diag([d_w / rgb_w, d_h / rgb_h, 1.0]) @ K, (d_w, d_h)
        # ARKit (and COLMAP) put the centre of pixel (0,0) at (0.5, 0.5); Open3D's
        # unprojection treats the integer index as the pixel centre. Shift the
        # principal point by half a pixel for everything that goes through Open3D
        # (K_depth). K_rgb keeps the ARKit convention for COLMAP / SAM 3 / PromptDA.
        K_depth[0, 2] -= 0.5
        K_depth[1, 2] -= 0.5
        return Frame(
            index=r["index"],
            timestamp=r["timestamp"],
            tracking_state=r["trackingState"],
            K_rgb=K,
            K_depth=K_depth,
            T_cw=T_arkit @ _ARKIT_TO_CV,
            rgb_size=(rgb_w, rgb_h),
            depth_size=depth_size,
            _dir=self.dir,
            depth_source=self.depth_source,
            lidar_size=(d_w, d_h),
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def good_frames(self, stride: int = 1) -> list[Frame]:
        """Frames with normal tracking, every `stride`-th one."""
        return [f for f in self.frames if f.tracking_state == "normal"][::stride]
