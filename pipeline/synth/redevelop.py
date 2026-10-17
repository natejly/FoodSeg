"""Refresh the RGB of an existing batch with the current phone-camera pipeline, without re-rendering.

    python -m synth.redevelop BATCH_ROOT [--out OUT_ROOT] [--seed 0] [--jobs 8] [--scans s000000 ...]

BATCH_ROOT has raw/<sid>/frame_*.npz (needs `--keep-raw` renders with `rgb_linear`) and scans/<sid>/.
Without --out the JPEGs and gt.json["phone_camera"] of BATCH_ROOT/scans are rewritten in place;
with --out, OUT_ROOT/scans/<sid>/rgb/*.jpg + gt.json are written and OUT_ROOT/raw is symlinked to the
batch's raw dir, so OUT_ROOT can be fed straight to `python -m synth.realism --synth OUT_ROOT`.
Camera parameters are re-sampled exactly like synth.finalize does (rng [seed, scene number]).
Depth, seg, LiDAR files are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .camera import PhoneCamera, scene_key, save_jpeg


def redevelop_scan(raw_dir: Path, scan_dir: Path, out_dir: Path, seed: int = 0) -> int:
    frames = sorted(raw_dir.glob("frame_*.npz"))
    if not frames or "rgb_linear" not in np.load(frames[0]).files:
        return 0
    sid = raw_dir.name
    rng = np.random.default_rng([seed, int(sid.lstrip("s") or 0)])
    phone = PhoneCamera.sample(rng)
    key = scene_key([np.load(p)["rgb_linear"] for p in frames[::4]])
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(frames):
        img = phone.process(np.load(p)["rgb_linear"], key, rng, i)
        save_jpeg(img, out_dir / "rgb" / f"{i:06d}.jpg", phone.jpeg_quality)
    gtp = scan_dir / "gt.json"
    if gtp.exists():
        gt = json.loads(gtp.read_text())
        gt["phone_camera"] = phone.to_json()
        (out_dir / "gt.json").write_text(json.dumps(gt, indent=1))
    return len(frames)


def _job(a):
    return redevelop_scan(*a)


def redevelop(root: Path, out: Path | None = None, seed: int = 0, jobs: int = 8, scans=None) -> int:
    raws = sorted(d for d in (root / "raw").iterdir() if d.is_dir() and (scans is None or d.name in scans))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        link = out / "raw"
        if not link.exists():
            os.symlink((root / "raw").resolve(), link)
    todo = [(d, root / "scans" / d.name, (out or root) / "scans" / d.name, seed) for d in raws]
    with ProcessPoolExecutor(jobs) as ex:
        return sum(ex.map(_job, todo))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--scans", nargs="*", default=None)
    a = ap.parse_args()
    n = redevelop(Path(a.root), Path(a.out) if a.out else None, a.seed, a.jobs, a.scans)
    print(f"re-developed {n} frames ->", a.out or a.root)


if __name__ == "__main__":
    main()
