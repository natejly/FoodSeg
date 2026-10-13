"""Harder scene specs: the segmentation stress test (`--preset hard` in synth.spec / synth.batch).

What v0 scenes lack and real meals have:
  - amorphous foods touching along a shared boundary: one heap split into 2-3 wedges
    with wavy edges, each a different food (rice | curry | beans), sometimes of
    similar colour;
  - toppings lying on another food (sauce, cheese), fully inside it;
  - piles of 2-5 scans of one category (nuggets, cookies) dropped together;
  - non-food clutter on the table (cutlery, a mug, a napkin; label 253).
Procedural pieces are closed meshes that never interpenetrate, so every volume stays
exact. Same world / camera conventions as synth.spec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import Mound, box_mesh, lathe_solid, mesh_volume, sample_container, split_angles, topping_mesh
from .spec import DISH_COLOURS, FILLS, intrinsics, orbit_poses, world_to_arkit

# Foods that lie on another food: name, base colour (linear RGB), roughness.
TOPPINGS = [
    ("tomato sauce", (0.45, 0.06, 0.03), 0.25), ("melted cheese", (0.90, 0.65, 0.15), 0.35),
    ("gravy", (0.30, 0.17, 0.07), 0.2), ("pesto", (0.20, 0.35, 0.08), 0.4),
    ("sour cream", (0.92, 0.92, 0.88), 0.3), ("salsa", (0.55, 0.15, 0.05), 0.5),
    ("guacamole", (0.40, 0.50, 0.15), 0.5), ("curry sauce", (0.75, 0.45, 0.08), 0.3),
]
DISTRACTOR_ID = 253


def _procedural(name: str, mesh: str, V: np.ndarray, F: np.ndarray, out: Path, col, rough: float,
                source: str, shape: dict) -> dict:
    np.savez_compressed(out / mesh, V=V.astype(np.float32), F=F.astype(np.int32))
    return {"name": name, "category": name, "source": source, "asset_id": None, "mesh": mesh, "fmt": "npz",
            "to_m": 1.0, "scale": 1.0, "volume_ml": mesh_volume(V, F) * 1e6, "mass_g": None, "dynamic": False,
            "fill": None, "color": list(col), "roughness": rough, "procedural": True, "generated": False,
            "texture": True, "shape": shape}


def distractors(rng: np.random.Generator, out: Path, r_clear: float) -> list[dict]:
    """1-3 of cutlery / mug / napkin on the table outside radius r_clear of the dish axis."""
    kinds = rng.choice(["knife", "fork", "spoon", "mug", "napkin"], size=int(rng.integers(1, 4)), replace=False)
    res, ang = [], rng.uniform(0, 2 * np.pi)
    for i, k in enumerate(kinds):
        if k == "mug":
            r, hgt = rng.uniform(0.035, 0.045), rng.uniform(0.08, 0.10)
            V, F = lathe_solid(np.array([[0, 0], [r, 0], [r, hgt], [r - 0.004, hgt], [r - 0.004, 0.006], [0, 0.006]]))
            dist, col, metal, rough = r_clear + r + rng.uniform(0.02, 0.08), rng.uniform(0.1, 0.9, 3), 0.0, 0.3
        elif k == "napkin":
            V, F = box_mesh(rng.uniform(0.12, 0.18), rng.uniform(0.12, 0.18), 0.003)
            dist, col, metal, rough = r_clear + rng.uniform(0.1, 0.14), rng.uniform(0.5, 0.95, 3), 0.0, 0.9
        else:
            V, F = box_mesh({"knife": 0.21, "fork": 0.19, "spoon": 0.17}[k], rng.uniform(0.014, 0.024), 0.004)
            dist, col, metal, rough = r_clear + rng.uniform(0.03, 0.06), (0.8, 0.8, 0.82), 1.0, 0.25
        a = ang + i * rng.uniform(0.8, 1.4)
        mesh = f"distractor_{i}.npz"
        np.savez_compressed(out / mesh, V=V.astype(np.float32), F=F.astype(np.int32))
        yaw = a + np.pi / 2 if k in ("knife", "fork", "spoon") else rng.uniform(0, 2 * np.pi)  # cutlery tangent
        res.append({"name": str(k), "mesh": mesh, "location": [float(dist * np.cos(a)), float(dist * np.sin(a)), 0.0],
                    "euler": [0.0, 0.0, float(yaw)], "color": [float(v) for v in col], "metallic": metal,
                    "roughness": rough})
    return res


def _similar(rng: np.random.Generator, n: int) -> np.ndarray:
    """n fill indices; with p = 0.3 the closest colours to a random first one (the hard case)."""
    idx = rng.choice(len(FILLS), n, replace=False)
    if n > 1 and rng.random() < 0.3:
        cols = np.array([f[1] for f in FILLS])
        idx = np.argsort(np.linalg.norm(cols - cols[idx[0]], axis=1))[:n]
    return idx


def _topping_site(rng: np.random.Generator, rf: float, centre, fa, fb):
    """A topping centre and radius whose (1.25x) footprint lies inside wedge [fa, fb] of the heap."""
    ph = np.linspace(0, 2 * np.pi, 64)
    for _ in range(30):
        rad = float(rng.uniform(0.012, 0.03))
        r0 = float(rng.uniform(0.1, 0.75) * rf)
        lo, hi = fa(np.array([r0]))[0], fb(np.array([r0]))[0]
        t0 = rng.uniform(lo, hi)
        bx, by = r0 * np.cos(t0) + 1.25 * rad * np.cos(ph), r0 * np.sin(t0) + 1.25 * rad * np.sin(ph)
        br, bt = np.hypot(bx, by), np.arctan2(by, bx)
        start = fa(br)
        if ((br < 0.92 * rf) & (np.mod(bt - start, 2 * np.pi) < fb(br) - start)).all():
            return (centre[0] + r0 * np.cos(t0), centre[1] + r0 * np.sin(t0)), rad
    return None


def sample_scene_hard(rng: np.random.Generator, assets: list[dict], out: Path, scene_id: str,
                      n_views: int = 24, res_scale: float = 1.0, max_items: int = 8,
                      hdris: list[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c = sample_container(rng, "plate" if rng.random() < 0.6 else "bowl")
    V, F = c.mesh()
    np.savez_compressed(out / "container.npz", V=V.astype(np.float32), F=F.astype(np.int32))
    spec: dict = {"scene_id": scene_id, "preset": "hard", "container": {
        **c.to_json(), "mesh": "container.npz", "capacity_ml": c.capacity_ml(),
        "color": DISH_COLOURS[rng.integers(len(DISH_COLOURS))], "roughness": float(rng.uniform(0.05, 0.6))}}
    items: list[dict] = []

    heap: list = []                          # (item id, Mound, wedge start, wedge end)
    # 1. a heap split into touching wedges, each a different food
    if rng.random() < 0.8:
        if c.kind == "bowl":
            lev = c.floor + rng.uniform(0.35, 0.8) * (c.H - c.floor)
            rf, centre = c.r_at_height(lev), (0.0, 0.0)
        else:
            lev, rf = None, c.well_r * rng.uniform(0.55, 0.9)
            t = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(0, c.well_r - rf)
            centre = (float(off * np.cos(t)), float(off * np.sin(t)))
        n_pieces = int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
        fns = split_angles(rng, n_pieces, rf)
        heap = []
        for k, fi in enumerate(_similar(rng, n_pieces)):
            mnd = Mound(rf=rf, level=lev, centre=centre, seed=int(rng.integers(1 << 31)),
                        mound=float(rng.uniform(0.004, 0.025) if c.kind == "bowl" else rng.uniform(0.01, 0.035)),
                        p=float(rng.uniform(2, 4)), wobble=float(rng.uniform(0, 0.004)))
            name, col, rough = FILLS[fi]
            items.append(_procedural(name, f"piece_{k}.npz", *mnd.piece(c, fns[k], fns[k + 1]), out, col, rough,
                                     "procedural_piece", {"kind": "piece", "n_pieces": n_pieces, "rf": rf,
                                                          "level": lev, "mound": mnd.mound, "centre": centre}))
            heap.append((len(items), mnd, fns[k], fns[k + 1]))

        # 2. toppings lying on a piece
        for item_id, mnd, fa, fb in heap:
            site = _topping_site(rng, rf, centre, fa, fb) if rng.random() < 0.35 else None
            if site is None:
                continue
            name, col, rough = TOPPINGS[rng.integers(len(TOPPINGS))]
            Vt, Ft = topping_mesh(c, mnd, site[0], site[1], float(rng.uniform(0.002, 0.008)),
                                  seed=int(rng.integers(1 << 31)))
            items.append(_procedural(name, f"topping_{item_id}.npz", Vt, Ft, out, col, rough, "procedural_topping",
                                     {"kind": "topping", "on_item": item_id, "radius": site[1]}))

    # 3. discrete items dropped onto whatever is there: a same-category pile and/or singles
    fit_r = c.well_r if c.kind == "plate" else c.r_at_height(c.floor + 0.5 * (c.H - c.floor))
    pool = [a for a in assets if a.get("volume_ok", a["watertight"]) and not a.get("container_shaped")
            and max(a["extent_m"][:2]) <= 2 * fit_r]
    real = [a for a in pool if not a["procedural"]]
    by_cat: dict[str, list[dict]] = {}
    for a in real:
        by_cat.setdefault(a["category"], []).append(a)
    small = sorted(k for k, v in by_cat.items() if np.median([max(a["extent_m"]) for a in v]) < 0.6 * fit_r)
    # Drop each single / pile just above the surface at its own spot (a high drop bounces off
    # the dish): spots don't overlap each other, and avoid the heap unless deliberately on it.
    mounds = [m for _, m, _, _ in heap]
    placed: list[tuple[np.ndarray, float]] = []

    def support_z(xy: np.ndarray) -> float:
        z = float(c.z_in(np.hypot(*xy)))
        if mounds:
            z = max([z] + [float(m.top(c, xy[0], xy[1])) for m in mounds])
            if np.hypot(xy[0] - mounds[0].centre[0], xy[1] - mounds[0].centre[1]) < mounds[0].rf:
                z += 0.009                   # clear any topping
        return z

    def spot(r_xy: float, on_heap: bool) -> np.ndarray | None:
        for _ in range(60):
            rr = max(0.85 * fit_r - r_xy, 0) * np.sqrt(rng.random())
            th = rng.uniform(0, 2 * np.pi)
            xy = np.array([rr * np.cos(th), rr * np.sin(th)])
            if any(np.linalg.norm(xy - p) < r + r_xy + 0.002 for p, r in placed):
                continue
            if mounds and not on_heap:
                if np.linalg.norm(xy - mounds[0].centre) < mounds[0].rf + 0.3 * r_xy:
                    continue
            placed.append((xy, r_xy))
            return xy
        return None if on_heap or not mounds else spot(r_xy, on_heap=True)   # no free dish: land on food

    def add(a: dict, xy: np.ndarray, z: float, kind: str) -> None:
        items.append({"name": a["name"], "category": a["category"], "source": a["source"], "asset_id": a["id"],
                      "mesh": a["path"], "fmt": a["fmt"], "to_m": a["to_m"], "scale": 1.0,
                      "volume_ml": a["volume_ml"], "mass_g": a.get("mass_g"), "dynamic": True, "fill": None,
                      "init_location": [float(xy[0]), float(xy[1]), float(z)],
                      "init_euler": (rng.uniform(0, 2 * np.pi, 3) if a["procedural"] else
                                     np.r_[rng.normal(0, 0.1, 2), rng.uniform(0, 2 * np.pi)]).tolist(),
                      "up_axis": a.get("up_axis", "Y"), "color": a.get("color"),
                      "roughness": float(rng.uniform(0.3, 0.8)), "procedural": a["procedural"],
                      "generated": a.get("generated", False), "texture": bool(a["procedural"]),
                      "damping": 0.8, "shape": {"kind": kind}})

    rad_of = lambda a: float(np.linalg.norm(a["extent_m"])) / 2       # any rotation fits
    if small and max_items - len(items) >= 2 and rng.random() < 0.5:
        grp = by_cat[small[rng.integers(len(small))]]
        k = int(min(rng.integers(2, 6), max_items - len(items)))
        sel = [grp[j] for j in rng.choice(len(grp), k, replace=len(grp) < k)]
        spread = 0.6 * max(max(a["extent_m"][:2]) for a in sel)
        ctr = spot(spread + 0.5 * max(max(a["extent_m"][:2]) for a in sel), on_heap=rng.random() < 0.3)
        if ctr is not None:
            z = support_z(ctr) + 0.002        # a pile: stacked by bounding radius, settles into contact
            for a in sel:
                z += rad_of(a)
                add(a, ctr + rng.normal(0, spread / 3, 2), z, "pile")
                z += rad_of(a) + 0.002
    room = max_items - len(items)
    n_single = int(rng.integers(0 if items else 1, min(3, room) + 1)) if room > 0 else 0
    for _ in range(n_single):
        src = real if real and rng.random() < 0.8 else [a for a in pool if a["procedural"]]
        a = src[rng.integers(len(src))]
        xy = spot(0.5 * max(a["extent_m"][:2]), on_heap=rng.random() < 0.25)
        if xy is not None:
            add(a, xy, support_z(xy) + rad_of(a) + 0.002, "single")
    for i, it in enumerate(items):
        it["id"] = i + 1                      # mask label; 0 = background

    K, w, h = intrinsics(rng, res_scale)
    poses, times = orbit_poses(rng, n_views, centre=np.array([0, 0, c.H * 0.5]))
    spec.update({
        "items": items,
        "distractors": distractors(rng, out, c.R + c.rim_w) if rng.random() < 0.7 else [],
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
    (out / "spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))
    return spec
