"""Sample synthetic meal scenes as JSON specs that render.py turns into images.

    python -m synth.spec --assets ../data/synth/assets.jsonl --out ../data/synth/specs --n 50 [--seed 0]

Each scene gets OUT/<scene_id>/spec.json plus the procedural meshes it needs
(container.npz, fill.npz). Frames follow docs/collection_protocol.md: a hand-held
orbit at ~45 deg elevation and 25-35 cm, then a near top-down pass.

World frame here is Blender's (Z up, table z = 0, container axis at the origin).
`world_to_arkit` maps it to an ARKit-like world (Y up, gravity -Y) with a random
yaw and offset, the way an ARKit session origin is wherever tracking started.
Camera poses use the Blender/ARKit camera convention (+X right, +Y up, -Z forward).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assets import load_index
from .geometry import fill_mesh, mesh_volume, sample_container

# Placeholder iPad intrinsics at 1920x1440 (ARKit convention, pixel centres at +0.5).
# TODO: replace with K from a real FoodScan frames.jsonl (it varies a little per frame with autofocus).
RGB_W, RGB_H = 1920, 1440
FX = 1440.0

# Fill foods: name, base colour (linear RGB), roughness. Appearance only.
FILLS = [
    ("white rice", (0.80, 0.78, 0.70), 0.7), ("mashed potato", (0.85, 0.78, 0.55), 0.6),
    ("oatmeal", (0.62, 0.48, 0.30), 0.5), ("lentil stew", (0.35, 0.20, 0.08), 0.3),
    ("mac and cheese", (0.85, 0.60, 0.15), 0.4), ("fried rice", (0.65, 0.50, 0.25), 0.6),
    ("yogurt", (0.90, 0.90, 0.86), 0.2), ("chili", (0.35, 0.08, 0.04), 0.35),
]
DISH_COLOURS = [(0.80, 0.80, 0.78), (0.78, 0.76, 0.68), (0.72, 0.68, 0.56), (0.5, 0.5, 0.5), (0.03, 0.03, 0.035),
                (0.1, 0.06, 0.04), (0.45, 0.40, 0.32)]    # mostly white / ivory stoneware, a few dark glazes (v0 scenes)


def look_at(eye: np.ndarray, target: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Camera-to-world, +X right, +Y up, -Z forward, world Z up."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ up) > 0.999:          # straight down: pick any horizontal 'up'
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    c, s = np.cos(roll), np.sin(roll)
    right, cam_up = c * right + s * cam_up, -s * right + c * cam_up
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, cam_up, -fwd, eye
    return T


def orbit_poses(rng: np.random.Generator, n: int, centre: np.ndarray,
                dish_r: float | None = None) -> tuple[list[np.ndarray], list[float]]:
    """~75 % of frames on an elevation-45 ring, the rest a near top-down pass, with hand-held jitter.
    With `dish_r` (outline radius, m) the orbit distance follows the dish size, so the dish spans 55-80 % of the
    frame height (a phone food scan), not a bowl lost in empty table; top-down frames sit a little closer."""
    n_ring = max(int(round(0.75 * n)), 1)
    n_top = n - n_ring
    a0 = rng.uniform(0, 2 * np.pi)
    sweep = rng.uniform(1.6, 2.0) * np.pi * rng.choice([-1, 1])
    dist0, el0 = rng.uniform(0.25, 0.35), np.deg2rad(rng.uniform(38, 55))
    if dish_r is not None:
        dist0 = float(np.clip(2 * dish_r / rng.uniform(0.55, 0.8), 0.2, 0.45))
    poses, times = [], []
    duration = rng.uniform(20, 40)
    for i in range(n):
        if i < n_ring:
            u = i / max(n_ring - 1, 1)
            az = a0 + sweep * u
            el = el0 + np.deg2rad(rng.normal(0, 3))
        else:
            u = (i - n_ring) / max(n_top - 1, 1)
            az = a0 + sweep + np.pi * u
            el = np.deg2rad(rng.uniform(72, 86))
        d = dist0 * (1 + rng.normal(0, 0.05)) * (0.95 if (dish_r is not None and i >= n_ring) else 1.0)
        eye = centre + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        tgt = centre + rng.normal(0, 0.012, 3) * [1, 1, 0.3]
        poses.append(look_at(eye, tgt, roll=np.deg2rad(rng.normal(0, 4))))
        times.append(duration * i / max(n - 1, 1))
    return poses, times


def world_to_arkit(rng: np.random.Generator) -> np.ndarray:
    """Blender world (Z up) -> ARKit world (Y up): axis swap, random yaw about up, offset."""
    S = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    yaw = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(yaw), np.sin(yaw)
    Y = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    Y[:3, 3] = [rng.uniform(-0.3, 0.3), rng.uniform(-1.0, -0.6), rng.uniform(-0.6, -0.2)]
    return Y @ S


def intrinsics(rng: np.random.Generator, scale: float) -> tuple[np.ndarray, int, int]:
    fx = FX * (1 + rng.normal(0, 0.02))
    K = np.array([[fx, 0, RGB_W / 2 + rng.normal(0, 4)], [0, fx, RGB_H / 2 + rng.normal(0, 4)], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    return S @ K, int(round(RGB_W * scale)), int(round(RGB_H * scale))


def sample_scene(rng: np.random.Generator, assets: list[dict], out: Path, scene_id: str,
                 n_views: int = 24, res_scale: float = 1.0, max_items: int = 4,
                 p_fill: dict | None = None, allow_open: bool = False, hdris: list[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    p_fill = p_fill or {"bowl": 0.7, "plate": 0.35}
    c = sample_container(rng)
    V, F = c.mesh()
    np.savez_compressed(out / "container.npz", V=V.astype(np.float32), F=F.astype(np.int32))
    spec: dict = {"scene_id": scene_id, "container": {
        **c.to_json(), "mesh": "container.npz", "capacity_ml": c.capacity_ml(),
        "color": DISH_COLOURS[rng.integers(len(DISH_COLOURS))], "roughness": float(rng.uniform(0.05, 0.6))}}

    items: list[dict] = []
    support_top = c.floor
    if rng.random() < p_fill[c.kind]:
        name, col, rough = FILLS[rng.integers(len(FILLS))]
        h = c.floor + rng.uniform(0.3, 0.85) * (c.H - c.floor)
        mound = rng.uniform(0.0, 0.025 if c.kind == "bowl" else 0.02)
        wobble = rng.uniform(0, 0.004)
        Vf, Ff = fill_mesh(c, h, mound, wobble=wobble, seed=int(rng.integers(1 << 31)))
        np.savez_compressed(out / "fill.npz", V=Vf.astype(np.float32), F=Ff.astype(np.int32))
        items.append({"name": name, "category": name, "source": "procedural_fill", "asset_id": None,
                      "mesh": "fill.npz", "fmt": "npz", "to_m": 1.0, "scale": 1.0,
                      "volume_ml": mesh_volume(Vf, Ff) * 1e6, "mass_g": None, "dynamic": False,
                      "fill": {"h": h, "mound": mound, "wobble": wobble},
                      "color": col, "roughness": rough, "procedural": True, "generated": False})
        support_top = h + mound

    # TODO: place container_shaped assets (MetaFood3D "X(bowl)") in a bowl fitted to their underside
    # only items whose footprint fits the dish's flat area (else they slide off, e.g. a fried egg in a bowl)
    fit_r = c.well_r if c.kind == "plate" else c.r_at_height(c.floor + 0.5 * (c.H - c.floor))
    pool = [a for a in assets if (a.get("volume_ok", a["watertight"]) or allow_open)
            and not a.get("container_shaped") and max(a["extent_m"][:2]) <= 2 * fit_r]
    real = [a for a in pool if not a["procedural"]]
    n_items = int(rng.integers(0 if items else 1, max_items - len(items) + 1))
    cursor = support_top + 0.005          # items spawn stacked by bounding radius, never overlapping
    for _ in range(n_items):
        src = real if real and rng.random() < 0.7 else [a for a in pool if a["procedural"]]
        if not src:
            break
        a = src[rng.integers(len(src))]
        ext = np.asarray(a["extent_m"])
        rad = float(np.linalg.norm(ext)) / 2      # any rotation fits in this sphere
        r_max = max((c.well_r if c.kind == "plate" else c.R) * 0.7 - ext[:2].max() / 2, 0.0)
        rr, th = r_max * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
        drop_z = cursor + rad
        cursor += 2 * rad + 0.005
        items.append({"name": a["name"], "category": a["category"], "source": a["source"], "asset_id": a["id"],
                      "mesh": a["path"], "fmt": a["fmt"], "to_m": a["to_m"], "scale": 1.0,
                      "volume_ml": a["volume_ml"], "mass_g": a.get("mass_g"), "dynamic": True,
                      "init_location": [rr * np.cos(th), rr * np.sin(th), drop_z],
                      # scans rest on their scanned underside: yaw + small tilt; primitives any way up
                      "init_euler": (rng.uniform(0, 2 * np.pi, 3) if a["procedural"] else
                                     np.r_[rng.normal(0, 0.1, 2), rng.uniform(0, 2 * np.pi)]).tolist(),
                      "up_axis": a.get("up_axis", "Y"),
                      "color": a.get("color"), "roughness": float(rng.uniform(0.3, 0.8)),
                      "procedural": a["procedural"], "generated": a.get("generated", False)})
    for i, it in enumerate(items):
        it["id"] = i + 1                      # mask label; 0 = background

    K, w, h = intrinsics(rng, res_scale)
    poses, times = orbit_poses(rng, n_views, centre=np.array([0, 0, c.H * 0.5]))
    spec.update({
        "items": items,
        "table": {"color": rng.uniform(0.05, 0.8, 3).tolist(), "roughness": float(rng.uniform(0.2, 0.9)),
                  "size_m": 4.0},
        "lighting": {"hdri": (hdris[rng.integers(len(hdris))] if hdris else None),
                     "hdri_strength": float(rng.uniform(0.5, 1.5)),
                     "ambient": float(rng.uniform(0.2, 0.8)),
                     "key": {"location": [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
                                          float(rng.uniform(0.8, 1.6))],
                             "energy_w": float(rng.uniform(60, 300)), "size_m": float(rng.uniform(0.2, 1.0))}},
        "camera": {"K": K.tolist(), "width": w, "height": h, "res_scale": res_scale,
                   "c2w": [p.tolist() for p in poses], "timestamps": times},
        "world_to_arkit": world_to_arkit(rng).tolist(),
    })
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first scene index (for sharding)")
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--res-scale", type=float, default=1.0, help="0.5 -> 960x720 renders")
    ap.add_argument("--max-items", type=int, default=4)
    ap.add_argument("--allow-open", action="store_true", help="use non-watertight meshes (volume = hull)")
    ap.add_argument("--hdri-dir", default=None)
    ap.add_argument("--preset", choices=["v0", "hard"], default="v0", help="hard: see synth/spec_hard.py")
    args = ap.parse_args()
    assets = load_index(args.assets)
    hdris = sorted(str(p.resolve()) for p in Path(args.hdri_dir).rglob("*.hdr")) if args.hdri_dir else None
    for i in range(args.start, args.start + args.n):
        rng = np.random.default_rng([args.seed, i])
        sid = f"s{i:06d}"
        if args.preset == "hard":
            from .spec_hard import sample_scene_hard
            s = sample_scene_hard(rng, assets, Path(args.out) / sid, sid, n_views=args.views,
                                  res_scale=args.res_scale, hdris=hdris)
        else:
            s = sample_scene(rng, assets, Path(args.out) / sid, sid, n_views=args.views,
                             res_scale=args.res_scale, max_items=args.max_items,
                             allow_open=args.allow_open, hdris=hdris)
        print(sid, s["container"]["kind"], [(it["name"], round(it["volume_ml"], 1)) for it in s["items"]])


if __name__ == "__main__":
    main()
