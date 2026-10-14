"""Convex decomposition (CoACD) of food scans, for compound rigid bodies in render.py.

    python -m synth.collision --assets ../data/synth/assets_v3.jsonl --cache ../data/synth/collision [--ids ...]

One convex hull per scan made items rest on an invisible bubble: they could not nest,
lean into each other or sit in a hollow, which made scenes look stiff. A decomposition
into a few convex parts follows the real shape. Parts are cached per asset in
CACHE/<asset id>.npz (V: all part vertices, n: vertices per part) in the coordinates the
mesh has *in Blender* after import (glTF Y-up converted to Z-up), before scaling.
`synth.batch` computes missing ones before each render (synth env: coacd, trimesh,
fast-simplification).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAX_FACES = 20000          # decimate scans (up to 1.6 M faces) before decomposing
MAX_PARTS = 12


def to_blender(V: np.ndarray, fmt: str) -> np.ndarray:
    """Blender's glTF importer maps glTF (Y up) to Z up: (x, y, z) -> (x, -z, y). OBJ and npz
    are imported 1:1 (render.py loads OBJ with up_axis Z / forward Y for every Z-up source)."""
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1) if fmt == "gltf" else V


def decompose(path: str, fmt: str, threshold: float = 0.06) -> list[np.ndarray]:
    import coacd
    import trimesh
    coacd.set_log_level("error")
    m = trimesh.load(path, force="mesh")
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if len(m.faces) > MAX_FACES:
        m = m.simplify_quadric_decimation(face_count=MAX_FACES)
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)), threshold=threshold,
                            max_convex_hull=MAX_PARTS, preprocess_mode="auto", seed=0)
    return [to_blender(np.asarray(v, np.float64), fmt) for v, _ in parts]


def ensure(asset: dict, cache: Path) -> Path:
    """Path of the cached decomposition of `asset`, computing it if missing."""
    p = cache / f"{asset['id']}.npz"
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        parts = decompose(asset["path"], asset["fmt"])
        np.savez_compressed(p, V=np.concatenate(parts).astype(np.float32), n=np.array([len(v) for v in parts]))
    return p


def load(p: str | Path) -> list[np.ndarray]:
    z = np.load(p)
    return np.split(z["V"], np.cumsum(z["n"])[:-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    for line in open(args.assets):
        a = json.loads(line)
        if a["procedural"] or a.get("role", "food") != "food" or (args.ids and a["id"] not in args.ids):
            continue
        p = ensure(a, Path(args.cache))
        print(a["id"], len(np.load(p)["n"]), "parts")


if __name__ == "__main__":
    main()
