"""Pre-render QC of a settled spec: off-dish, volume mismatch and a coarse visibility estimate."""
from __future__ import annotations

import numpy as np

from .geometry import mesh_volume
from .mpm import load_local

LOW_W = 160            # width of the visibility z-buffer (px)
MIN_PX = 4             # an item counts as seen in a frame when >= MIN_PX low-res pixels are its own
MIN_FRAMES = 6         # fewer estimated frames than this -> resample (render threshold is 3; margin for the coarse test)
SAMPLE_MM = 2.0        # surface sample spacing


def item_world_mesh(it: dict, scene_dir) -> tuple[np.ndarray, np.ndarray] | None:
    """Settled world-space mesh, or None for items whose pose is only known after the render (cloth leaves)."""
    if it.get("cloth"):
        return None
    V, F = load_local(str(scene_dir / it["mesh"]) if not str(it["mesh"]).startswith("/") else it["mesh"],
                      it["fmt"], it.get("up_axis", "Z"))
    if "matrix_world" in it:
        M = np.asarray(it["matrix_world"], float)
        V = V @ M[:3, :3].T + M[:3, 3]
    return V, F


def _samples(V: np.ndarray, F: np.ndarray, rng, spacing: float) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.round(area / spacing ** 2)).astype(int)
    tri = np.repeat(np.arange(len(F)), n)
    u, v = rng.random((2, len(tri)))
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return a[tri] + u[:, None] * (b - a)[tri] + v[:, None] * (c - a)[tri]


def settled_problems(spec: dict, scene_dir) -> list[str]:
    """Flags (manifest names) that the settled spec already shows: off_container, volume_mismatch,
    barely_visible (estimated from a low-res depth test over the camera poses)."""
    from .manifest import VOLUME_TOL
    cj = spec["container"]
    reach = cj["R"] + cj.get("rim_w", 0.0)
    rng = np.random.default_rng(0)
    bad: set[str] = set()
    pts, lab = [], []
    for it in spec["items"]:
        m = item_world_mesh(it, scene_dir)
        if m is None:
            continue
        V, F = m
        c = 0.5 * (V.min(0) + V.max(0))
        if np.hypot(c[0], c[1]) > reach:
            bad.add("off_container")
        vol = mesh_volume(V, F) * 1e6
        if it.get("volume_ml") and abs(vol - it["volume_ml"]) > VOLUME_TOL * it["volume_ml"]:
            bad.add("volume_mismatch")
        p = _samples(V, F, rng, SAMPLE_MM * 1e-3)
        pts.append(p), lab.append(np.full(len(p), int(it["id"])))
    if not pts:
        return sorted(bad)
    P, L = np.concatenate(pts), np.concatenate(lab)
    cam = spec["camera"]
    K = np.asarray(cam["K"], float)
    s = LOW_W / cam["width"]
    w, h = LOW_W, int(round(cam["height"] * s))
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, (K[0, 2] - .5) * s, (K[1, 2] - .5) * s
    ids = sorted(set(L.tolist()))
    frames = dict.fromkeys(ids, 0)
    for T in cam["c2w"]:
        T = np.asarray(T, float)
        q = (P - T[:3, 3]) @ T[:3, :3]                       # camera coords, OpenGL: looks along -z
        z = -q[:, 2]
        ok = z > 0.05
        u = np.round(fx * q[:, 0] / np.where(ok, z, 1) + cx).astype(int)
        v = np.round(fy * -q[:, 1] / np.where(ok, z, 1) + cy).astype(int)
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, zz, ll = u[ok], v[ok], z[ok], L[ok]
        zb = np.full(w * h, np.inf)
        pix = v * w + u
        np.minimum.at(zb, pix, zz)
        own = zz <= zb[pix] + 0.004                          # within 4 mm of the nearest surface in the pixel
        cnt = {}
        for i_ in ids:
            cnt[i_] = len(np.unique(pix[own & (ll == i_)]))
        for i_, c_ in cnt.items():
            frames[i_] += c_ >= MIN_PX
    if any(n < MIN_FRAMES for n in frames.values()):
        bad.add("barely_visible")
    settled_problems.last_frames = frames
    return sorted(bad)
