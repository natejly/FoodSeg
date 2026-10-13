"""Write a fake FoodScan capture: a 6x4x3 cm box on a table seen from an orbit.

Lets the loader/fuse code be tested before the iPad app exists. The box's
true volume (72 cm^3) is the number the volume code should recover later.

    python tests/make_synthetic_scan.py OUT_DIR [--frames 24]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

RGB_W, RGB_H = 1920, 1440
D_W, D_H = 256, 192
BOX = np.array([0.06, 0.04, 0.03])  # x, y extents and height (m)


def look_at_arkit(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """ARKit camera-to-world: +X right, +Y up, -Z forward."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])  # world Z up (table normal)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = cam_up
    T[:3, 2] = -fwd
    T[:3, 3] = eye
    return T


BOWL_R, BOWL_D = 0.06, 0.02     # rim radius, depth at centre (m); rim flush with the table (z = 0)


def bowl_food_volume_cm3(fill: float) -> float:
    """Volume of a flat fill of height `fill` (m) in the paraboloid bowl z = -D + D r^2/R^2."""
    return float(np.pi * fill**2 * BOWL_R**2 / (2 * BOWL_D) * 1e6)


def _rays(T_cw_arkit: np.ndarray, K_depth: np.ndarray):
    us, vs = np.meshgrid(np.arange(D_W) + 0.5, np.arange(D_H) + 0.5)
    fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    # Rays in OpenCV camera coords, then to ARKit coords (flip y, z).
    d_cv = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], -1)
    d_ar = d_cv * np.array([1, -1, -1])
    R, o = T_cw_arkit[:3, :3], T_cw_arkit[:3, 3]
    d_w = d_ar @ R.T
    d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
    return d_cv, d_w, o


def _t_plane(o, d_w, z0: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z0 - o[2]) / d_w[..., 2]
    t[(t <= 0) | ~np.isfinite(t)] = np.inf
    return t


def _finish(t, d_cv):
    # OpenCV depth is the z component in camera coords = t * d_cv_z / |d_cv|
    depth = t / np.linalg.norm(d_cv, axis=-1)
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def render_depth(T_cw_arkit: np.ndarray, K_depth: np.ndarray) -> np.ndarray:
    """Ray-cast the table plane z=0 and the box centered at the origin."""
    d_cv, d_w, o = _rays(T_cw_arkit, K_depth)
    t_plane = _t_plane(o, d_w, 0.0)

    # Axis-aligned box [-BOX/2, BOX/2] in x,y and [0, h] in z (slab method)
    lo = np.array([-BOX[0] / 2, -BOX[1] / 2, 0.0])
    hi = np.array([BOX[0] / 2, BOX[1] / 2, BOX[2]])
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - o) / d_w
        t1 = (hi - o) / d_w
    tmin = np.max(np.minimum(t0, t1), axis=-1)
    tmax = np.min(np.maximum(t0, t1), axis=-1)
    hit = (tmax >= tmin) & (tmax > 0)
    t_box = np.where(hit, np.maximum(tmin, 0), np.inf)
    return _finish(np.minimum(t_plane, t_box), d_cv)


def render_depth_bowl(T_cw_arkit: np.ndarray, K_depth: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Ray-cast a table with a sunken paraboloid bowl, optionally filled flat to height `fill`.

    Bowl interior: z = -D + D r^2 / R^2 for r <= R (rim flush with the table at z = 0).
    Food: flat surface at z = -D + fill, present for r <= R sqrt(fill / D).
    """
    d_cv, d_w, o = _rays(T_cw_arkit, K_depth)
    R2, D = BOWL_R**2, BOWL_D

    # table outside the rim
    t_tab = _t_plane(o, d_w, 0.0)
    p = o + t_tab[..., None] * d_w
    t_tab[(p[..., 0]**2 + p[..., 1]**2) <= R2] = np.inf

    # paraboloid: D((ox+t dx)^2 + (oy+t dy)^2)/R2 - D - (oz + t dz) = 0
    a = D * (d_w[..., 0]**2 + d_w[..., 1]**2) / R2
    b = 2 * D * (o[0] * d_w[..., 0] + o[1] * d_w[..., 1]) / R2 - d_w[..., 2]
    c = D * (o[0]**2 + o[1]**2) / R2 - D - o[2]
    disc = b**2 - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
    t_bowl = np.full_like(t_tab, np.inf)
    for tc in (t1, t2):
        pc = o + tc[..., None] * d_w
        ok = np.isfinite(tc) & (tc > 0) & ((pc[..., 0]**2 + pc[..., 1]**2) <= R2) & (pc[..., 2] <= 0)
        t_bowl = np.where(ok, np.minimum(t_bowl, tc), t_bowl)

    t = np.minimum(t_tab, t_bowl)
    if fill > 0:
        t_food = _t_plane(o, d_w, -D + fill)
        pf = o + t_food[..., None] * d_w
        rf2 = R2 * fill / D
        t_food[(pf[..., 0]**2 + pf[..., 1]**2) > rf2] = np.inf
        t = np.minimum(t, t_food)
    return _finish(t, d_cv)


def smear_edges(depth: np.ndarray, px: int) -> np.ndarray:
    """Mimic ARKit sceneDepth at silhouettes: within `px` pixels of a depth jump, replace depth
    with a local average, i.e. a ramp of "flying pixels" between foreground and background."""
    from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
    k = 2 * px + 1
    jump = (maximum_filter(depth, k) - minimum_filter(depth, k)) > 0.01
    out = depth.copy()
    out[jump] = uniform_filter(depth, k)[jump]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--noise", type=float, default=0.002, help="depth noise sigma (m)")
    ap.add_argument("--scene", choices=["box", "bowl"], default="box")
    ap.add_argument("--fill", type=float, default=0.0, help="bowl scene: flat food fill height (m)")
    ap.add_argument("--edge-smear", type=int, default=0,
                    help="smear depth across silhouettes over this many px, like LiDAR (0 = off)")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ["rgb", "depth", "smooth_depth", "conf"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Plausible iPad intrinsics for 1920x1440.
    K_rgb = np.array([[1450.0, 0, 960.0], [0, 1450.0, 720.0], [0, 0, 1]])
    S = np.diag([D_W / RGB_W, D_H / RGB_H, 1.0])
    K_depth = S @ K_rgb

    rng = np.random.default_rng(0)
    with open(out / "frames.jsonl", "w") as f:
        for i in range(args.frames):
            ang = 2 * np.pi * i / args.frames
            eye = np.array([0.30 * np.cos(ang), 0.30 * np.sin(ang), 0.30])
            T = look_at_arkit(eye, np.zeros(3))
            if args.scene == "bowl":
                depth = render_depth_bowl(T, K_depth, fill=args.fill)
            else:
                depth = render_depth(T, K_depth)
            if args.edge_smear:
                depth = smear_edges(depth, args.edge_smear)
            noisy = depth + rng.normal(0, args.noise, depth.shape).astype(np.float32) * (depth > 0)
            noisy.astype("<f4").tofile(out / "depth" / f"{i:06d}.f32")
            depth.astype("<f4").tofile(out / "smooth_depth" / f"{i:06d}.f32")
            np.full((D_H, D_W), 2, np.uint8).tofile(out / "conf" / f"{i:06d}.u8")
            Image.fromarray(np.full((RGB_H, RGB_W, 3), 180, np.uint8)).save(out / "rgb" / f"{i:06d}.jpg")
            rec = {
                "index": i, "timestamp": i / 15.0, "trackingState": "normal",
                "intrinsics": K_rgb.tolist(), "transform": T.tolist(),
                "imageWidth": RGB_W, "imageHeight": RGB_H,
                "depthWidth": D_W, "depthHeight": D_H,
                "exposureDuration": 0.01, "exposureOffset": 0.0,
            }
            f.write(json.dumps(rec) + "\n")
    meta = {"formatVersion": 1, "app": "synthetic", "device": "synthetic",
            "savedFrames": args.frames, "scene": args.scene}
    if args.scene == "bowl":
        meta.update({"bowlRadiusM": BOWL_R, "bowlDepthM": BOWL_D, "fillM": args.fill,
                     "foodVolumeCm3": bowl_food_volume_cm3(args.fill),
                     "bowlCapacityCm3": bowl_food_volume_cm3(BOWL_D)})
        truth = meta["foodVolumeCm3"]
    else:
        meta["boxVolumeCm3"] = float(np.prod(BOX) * 1e6)
        truth = meta["boxVolumeCm3"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.frames} frames to {out}; {args.scene} true volume {truth:.1f} cm^3")


if __name__ == "__main__":
    main()
