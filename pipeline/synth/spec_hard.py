"""Harder scene specs: the segmentation stress test (`--preset hard` in synth.spec / synth.batch).

What v0 scenes lack and real meals have:
  - amorphous foods touching along a shared boundary: one heap split into 2-3 wedges
    with wavy edges, each a different food (rice | curry | beans), sometimes of
    similar colour;
  - toppings lying on another food (sauce, cheese), fully inside it;
  - piles of 2-5 scans of one category (nuggets, cookies) dropped together;
  - non-food clutter on the table (cutlery, a mug, a napkin; label 253).
Procedural pieces are closed meshes that never interpenetrate; synth.mpm then settles them with
the dropped items (soft food flows round them) and volumes are those of the settled meshes.
Same world / camera conventions as synth.spec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import (Container, Mound, box_mesh, lathe_solid, mesh_volume, salad_sheets, sample_container,
                       split_angles, topping_mesh)
from .looks import PIECES, kind_of, look, physics, shape
from .meals import BOWL_ONLY, FOODS, sample_meal
from .spec import DISH_COLOURS, FILLS as V0_FILLS, intrinsics, orbit_poses, world_to_arkit

# v0's fills plus Nutrition5k staples that render as amorphous heaps (synth/composition.py)
FILLS = V0_FILLS + [
    ("scrambled eggs", (0.95, 0.78, 0.25), 0.6), ("brown rice", (0.55, 0.40, 0.22), 0.7),
    ("quinoa", (0.75, 0.65, 0.45), 0.6), ("egg whites", (0.92, 0.90, 0.80), 0.5),
] + [(n, col, rough) for n, (col, rough) in FOODS.items()]       # synth.meals: noodles, curries, soups ...
FILL_BY_NAME = {f[0]: f for f in FILLS}

# Foods that lie on another food: name, base colour (linear RGB), roughness.
GREENS = [  # leafy greens (Nutrition5k: mixed greens, arugula, spinach, lettuce, kale ...)
    ("lettuce", (0.45, 0.65, 0.18), 0.5), ("spinach (raw)", (0.10, 0.28, 0.06), 0.4),
    ("arugula", (0.18, 0.38, 0.08), 0.45), ("mixed greens", (0.25, 0.45, 0.10), 0.5),
    ("kale", (0.12, 0.25, 0.10), 0.6), ("radicchio", (0.35, 0.05, 0.12), 0.4),
]
TOPPINGS = [
    ("tomato sauce", (0.45, 0.06, 0.03), 0.25), ("melted cheese", (0.90, 0.65, 0.15), 0.35),
    ("gravy", (0.30, 0.17, 0.07), 0.2), ("pesto", (0.20, 0.35, 0.08), 0.4),
    ("sour cream", (0.92, 0.92, 0.88), 0.3), ("salsa", (0.55, 0.15, 0.05), 0.5),
    ("guacamole", (0.40, 0.50, 0.15), 0.5), ("curry sauce", (0.75, 0.45, 0.08), 0.3),
]
DISTRACTOR_ID = 253


def _procedural(name: str, mesh: str, V: np.ndarray, F: np.ndarray, out: Path, col, rough: float,
                source: str, shape: dict, rng: np.random.Generator) -> dict:
    np.savez_compressed(out / mesh, V=V.astype(np.float32), F=F.astype(np.int32))
    return {"name": name, "category": name, "source": source, "asset_id": None, "mesh": mesh, "fmt": "npz",
            "to_m": 1.0, "scale": 1.0, "volume_ml": mesh_volume(V, F) * 1e6, "mass_g": None, "dynamic": False,
            "fill": None, "color": list(col), "roughness": rough, "procedural": True, "generated": False,
            "texture": True, "shape": shape,
            # appearance and MPM material by consistency (synth.looks, synth.mpm)
            "look": look(name, rng), "physics": physics(name), "pieces": PIECES.get(name, [])}


REAL_PROPS = {"knife": ("032_knife",), "fork": ("030_fork",), "spoon": ("031_spoon",), "mug": ("mug", "cup")}


def distractors(rng: np.random.Generator, out: Path, r_clear: float, assets: list[dict] | None = None) -> list[dict]:
    """1-3 of cutlery / mug / napkin on the table outside radius r_clear of the dish axis; 80 %
    of them real scans (YCB cutlery, GSO / YCB mugs) when the asset index has them."""
    kinds = rng.choice(["knife", "fork", "spoon", "mug", "napkin"], size=int(rng.integers(1, 4)), replace=False)
    props = [a for a in (assets or []) if a.get("role") == "clutter"]
    res, ang = [], rng.uniform(0, 2 * np.pi)
    for i, k in enumerate(kinds):
        real = [a for a in props if any(t in a["id"].lower() for t in REAL_PROPS.get(str(k), ()))]
        if real and rng.random() < 0.8:
            a = real[rng.integers(len(real))]
            half = 0.5 * max(a["extent_m"][:2])
            dist = r_clear + (0.5 * min(a["extent_m"][:2]) if k != "mug" else half) + rng.uniform(0.02, 0.06)
            ang_i = ang + i * rng.uniform(0.8, 1.4)
            yaw = ang_i + np.pi / 2 if k != "mug" else rng.uniform(0, 2 * np.pi)   # cutlery tangent
            res.append({"name": str(k), "mesh": a["path"], "fmt": a["fmt"], "up_axis": a.get("up_axis", "Z"),
                        "asset_id": a["id"], "location": [float(dist * np.cos(ang_i)), float(dist * np.sin(ang_i)), 0.0],
                        "euler": [0.0, 0.0, float(yaw)]})
            continue
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


# Table surfaces people eat on: (base colour range low, high, roughness range, texture scale range).
TABLES = [
    ((0.25, 0.13, 0.06), (0.55, 0.33, 0.16), (0.3, 0.6), (8, 30)),      # wood
    ((0.55, 0.55, 0.52), (0.85, 0.85, 0.82), (0.2, 0.5), (2, 10)),      # light laminate / stone
    ((0.03, 0.03, 0.03), (0.12, 0.12, 0.12), (0.3, 0.7), (5, 20)),      # dark laminate
    ((0.1, 0.1, 0.1), (0.8, 0.8, 0.8), (0.7, 0.95), (150, 400)),        # tablecloth / placemat weave
]


def table_look(rng: np.random.Generator, surfaces: list[dict] | None = None) -> dict:
    """A photographed PBR tabletop (synth.fetch_assets surface sets: wood, stone, laminate,
    cloth) at its real-world tile size, rotated and slightly tinted; else the procedural one."""
    if surfaces and rng.random() < 0.9:
        kinds = sorted({t["kind"] for t in surfaces})
        kind = kinds[rng.integers(len(kinds))] if rng.random() < 0.5 else None   # don't let 17 woods dominate
        pool = [t for t in surfaces if kind is None or t["kind"] == kind]
        t = pool[rng.integers(len(pool))]
        return {"pbr": {"dir": t["dir"], "kind": t["kind"], "tile_m": float(t["tile_m"]),
                        "rotation": float(rng.uniform(0, np.pi)), "offset": rng.uniform(0, 1, 2).tolist(),
                        "tint": rng.normal(1.0, 0.06, 3).clip(0.8, 1.2).tolist(),
                        "brightness": float(rng.uniform(0.8, 1.15))},
                "color": [0.5, 0.5, 0.5], "roughness": 0.5, "size_m": 4.0}
    lo, hi, rough, scale = TABLES[rng.integers(len(TABLES))]
    col = rng.uniform(lo, hi) if rng.random() < 0.85 else rng.uniform(0.1, 0.8, 3)   # some colourful cloths
    return {"color": [float(v) for v in col], "roughness": float(rng.uniform(*rough)),
            "texture_scale": float(rng.uniform(*scale)), "size_m": 4.0}


def glaze(rng: np.random.Generator) -> dict:
    """Procedural dish finish (render.ceramic_material): glossy glaze over a body colour,
    stoneware speckle, sometimes a coloured rim band; porcelain is slightly translucent."""
    rim = rng.random() < 0.3
    return {"coat": float(rng.uniform(0.6, 1.0)), "coat_rough": float(rng.uniform(0.02, 0.12)),
            "speckle": float(rng.uniform(0, 0.6) if rng.random() < 0.4 else 0.0),
            "rim_color": rng.uniform(0.05, 0.8, 3).tolist() if rim else None, "rim_frac": float(rng.uniform(0.85, 0.95)),
            "sss": float(rng.uniform(0.0, 0.15))}


def lighting(rng: np.random.Generator, hdris: list[str] | None) -> dict:
    """Indoor HDRI (window light, room bounce, real reflections) at a random rotation and
    strength; a soft window / lamp key on top in some scans. Without HDRIs: the old flat ambient."""
    L = {"hdri": (hdris[rng.integers(len(hdris))] if hdris else None),
         "hdri_strength": float(rng.uniform(0.6, 1.6)), "hdri_rotation": float(rng.uniform(0, 2 * np.pi)),
         "ambient": float(rng.uniform(0.15, 0.5)), "ambient_tint": light_tint(rng, 0.08),
         "key": {"location": [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(0.8, 1.6))],
                 "energy_w": float(rng.uniform(40, 200)), "size_m": float(rng.uniform(0.2, 1.0)),
                 "color": light_tint(rng, 0.12)}}
    if L["hdri"]:
        if rng.random() < 0.5:
            L["key"] = None
        else:
            L["key"].update(energy_w=float(rng.uniform(8, 50)), size_m=float(rng.uniform(0.5, 1.5)))
    return L


def light_tint(rng: np.random.Generator, spread: float) -> list[float]:
    """Light colour along the warm (tungsten) - cool (daylight / LED) axis."""
    t = rng.uniform(-1, 1) * spread
    return [float(1 + t), 1.0, float(1 - t)]


def _similar(rng: np.random.Generator, n: int, bowl: bool) -> list[tuple]:
    """n random fills (no soups on a plate); with p = 0.3 the closest colours to a random first
    one (the hard case)."""
    ok = [f for f in FILLS if bowl or f[0] not in BOWL_ONLY]
    idx = rng.choice(len(ok), n, replace=False)
    if n > 1 and rng.random() < 0.3:
        cols = np.array([f[1] for f in ok])
        idx = np.argsort(np.linalg.norm(cols - cols[idx[0]], axis=1))[:n]
    return [ok[i] for i in idx]


def _dish_kind(a: dict) -> str:
    ext = a["extent_m"]
    return "bowl" if ext[2] > 0.25 * float(min(ext[:2])) else "plate"


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
                      hdris: list[str] | None = None, surfaces: list[dict] | None = None,
                      n5k: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    # what is served (synth.meals): a cuisine template, a real Nutrition5k dish, or random
    meal = sample_meal(rng, n5k)
    want = meal["dish"]
    if want == "any":
        want = "bowl" if any(f in BOWL_ONLY for f in meal["heap"] or []) else ("plate" if rng.random() < 0.6 else "bowl")
    c = sample_container(rng, want)
    V, F = c.mesh()
    np.savez_compressed(out / "container.npz", V=V.astype(np.float32), F=F.astype(np.int32))
    spec: dict = {"scene_id": scene_id, "preset": "hard", "container": {
        **c.to_json(), "mesh": "container.npz", "capacity_ml": c.capacity_ml(),
        "color": DISH_COLOURS[rng.integers(len(DISH_COLOURS))], "roughness": float(rng.uniform(0.05, 0.6)),
        "glaze": glaze(rng)}}
    items: list[dict] = []

    # 0. sometimes a real scanned dish (GSO / YCB). Heaps are built to fit the procedural
    # dish's analytic surface, so real-dish scenes hold only physics-dropped items; `c`
    # becomes an approximate stand-in used for placement radii and the off-dish check.
    dishes = [a for a in assets if a.get("role") == "container" and _dish_kind(a) == c.kind]
    # a meal with a heap needs the procedural dish (heaps follow its analytic surface)
    p_real = 0.2 if meal["source"] == "random" else (0.0 if meal["heap"] else 0.5)
    real_dish = dishes[rng.integers(len(dishes))] if dishes and rng.random() < p_real else None
    if real_dish:
        ext = real_dish["extent_m"]
        w = float(min(ext[:2]))
        kind = "bowl" if ext[2] > 0.25 * w else "plate"
        c = Container(kind, R=0.45 * w, H=float(ext[2]), floor=0.006, q=2.5, well_r=0.3 * w, rim_w=0.05 * w)
        spec["container"] = {**c.to_json(), "mesh": None, "capacity_ml": None,
                             "asset": {"mesh": real_dish["path"], "fmt": real_dish["fmt"], "name": real_dish["id"],
                                       "up_axis": real_dish.get("up_axis", "Z"),
                                       "yaw": float(rng.uniform(0, 2 * np.pi))},
                             "source": real_dish["source"], "licence": real_dish.get("licence")}

    heap: list = []                          # (item id, Mound, wedge start, wedge end)
    # 1. a heap split into touching wedges, each a different food
    if meal["source"] != "random":
        foods = [FILL_BY_NAME[n] for n in meal["heap"] if n in FILL_BY_NAME
                 and (c.kind == "bowl" or n not in BOWL_ONLY)][:3]
    if real_dish is None and (rng.random() < 0.95 if meal["source"] == "random" else bool(foods)):
        if c.kind == "bowl":
            lev = c.floor + rng.uniform(0.35, 0.8) * (c.H - c.floor)
            rf, centre = c.r_at_height(lev), (0.0, 0.0)
        else:
            lev, rf = None, c.well_r * rng.uniform(0.7, 0.95)        # a served portion fills the well
            t = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(0, c.well_r - rf)
            centre = (float(off * np.cos(t)), float(off * np.sin(t)))
        if meal["source"] == "random":
            n_pieces = int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
            foods = _similar(rng, n_pieces, c.kind == "bowl")
        n_pieces = len(foods)
        fns = split_angles(rng, n_pieces, rf)
        # the served shape (synth.mpm then lets it flow / slump / heap by consistency): liquids
        # level out, purees mound, grains heap; near a seam both sides ease to the heap's mean
        # profile, so a dome of rice meets a pool of curry, not a cliff
        shapes = [shape(name, c.kind, rf, rng) for name, _, _ in foods]
        shared = (float(np.mean([s["mound"] for s in shapes])), float(np.mean([s["p"] for s in shapes])))
        for k, ((name, col, rough), sh) in enumerate(zip(foods, shapes)):
            mnd = Mound(rf=rf, level=lev, centre=centre, seed=int(rng.integers(1 << 31)), mound=sh["mound"],
                        p=sh["p"], wobble=sh["wobble"], relief=sh["relief"][0], relief_scale=sh["relief"][1],
                        shared=shared if n_pieces > 1 else None, edges=(fns[k], fns[k + 1]),
                        blend=float(rng.uniform(0.01, 0.02)))
            items.append(_procedural(name, f"piece_{k}.npz", *mnd.piece(c, fns[k], fns[k + 1]), out, col, rough,
                                     "procedural_piece", {"kind": "piece", "n_pieces": n_pieces, "rf": rf,
                                                          "level": lev, "mound": mnd.mound, "centre": centre}, rng))
            heap.append((len(items), mnd, fns[k], fns[k + 1]))
        if n_pieces > 1:
            # how the foods share the heap: synth.mpm re-partitions the heap's particles before it
            # settles them (the wedge meshes stay for --physics bullet). Real plates: foods side by
            # side, a sauce / curry poured over part of a base, irregular patches; wedges rarely
            layout = str(rng.choice(["sides", "layers", "patches", "wedges"], p=[0.35, 0.25, 0.2, 0.2]))
            if meal.get("layout"):                    # the template knows how it's served
                layout = str(meal["layout"][rng.integers(len(meal["layout"]))])
            ids = [h[0] for h in heap]                # piece item ids (ids are list positions + 1)
            if layout == "layers":                    # the template's top food, else the runniest, goes on top (last)
                rank = {"liquid": 0, "porridge": 1, "puree": 2, "curd": 3, "grain": 4}
                top = meal.get("top")
                ids = sorted(ids, key=lambda i: (items[i - 1]["name"] == top, -rank[kind_of(items[i - 1]["name"])]))
            spec["heap"] = {"layout": layout, "items": ids, "rf": rf, "centre": list(centre),
                            "seed": int(rng.integers(1 << 31)),
                            "fractions": rng.dirichlet(np.ones(n_pieces) * 4).tolist(),
                            "top_thick": float(rng.uniform(0.005, 0.012)), "top_r": float(rng.uniform(0.45, 0.85) * rf)}

        # 2. toppings lying on a piece
        for item_id, mnd, fa, fb in heap:
            menu = TOPPINGS if meal["source"] == "random" else [t for t in TOPPINGS if t[0] in meal["toppings"]]
            p_top = 0.35 if meal["source"] == "random" else 0.5
            site = _topping_site(rng, rf, centre, fa, fb) if menu and rng.random() < p_top else None
            if site is None:
                continue
            name, col, rough = menu[rng.integers(len(menu))]
            Vt, Ft = topping_mesh(c, mnd, site[0], site[1], float(rng.uniform(0.004, 0.01)),   # >= 3 particles deep
                                  seed=int(rng.integers(1 << 31)))
            items.append(_procedural(name, f"topping_{item_id}.npz", Vt, Ft, out, col, rough, "procedural_topping",
                                     {"kind": "topping", "on_item": item_id, "radius": site[1]}, rng))

    fit_r = c.well_r if c.kind == "plate" else c.r_at_height(c.floor + 0.5 * (c.H - c.floor))

    # 2b. leafy greens: thin leaves dropped as cloth (render.py), one item per portion
    salad_spot = None
    greens = (GREENS[rng.integers(len(GREENS))] if rng.random() < 0.35 else None) if meal["source"] == "random" \
        else (next((g for g in GREENS if g[0] == meal["greens"]), GREENS[3]) if meal["greens"] else None)   # [3] mixed greens
    if len(items) < max_items and greens:
        name, col, rough = greens
        t = rng.uniform(0, 2 * np.pi)
        salad_spot = rng.uniform(0, 0.25 * fit_r) * np.array([np.cos(t), np.sin(t)])
        top = max([float(np.load(out / it["mesh"])["V"][:, 2].max()) for it in items], default=c.H)
        n_leaves = int(rng.integers(6, 17))
        Vl, Fl = salad_sheets(rng, n_leaves, salad_spot, max(top, c.H) + 0.01, 0.01,   # tight column: stays on
                              "arugula" if name == "arugula" else "mixed")
        np.savez_compressed(out / "salad.npz", V=Vl.astype(np.float32), F=Fl.astype(np.int32))
        thick = 0.0006
        a, b, cc = Vl[Fl[:, 0]], Vl[Fl[:, 1]], Vl[Fl[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1).sum()
        items.append({"name": name, "category": name, "source": "procedural_leaves", "asset_id": None,
                      "mesh": "salad.npz", "fmt": "npz", "to_m": 1.0, "scale": 1.0,
                      "volume_ml": float(area * thick * 1e6),     # replaced by the simulated solid's volume
                      "mass_g": None, "dynamic": False, "fill": None, "color": list(col), "roughness": rough,
                      "procedural": True, "generated": False, "texture": True,
                      "cloth": {"thickness_m": thick, "bending": float(rng.uniform(0.1, 1.0)), "grid": [12, 5]},
                      "shape": {"kind": "leaves", "n_leaves": n_leaves}})

    # 3. discrete items dropped onto whatever is there: a same-category pile and/or singles
    pool = [a for a in assets if a.get("role", "food") == "food" and a.get("volume_ok", a["watertight"])
            and not a.get("container_shaped") and max(a["extent_m"][:2]) <= 2 * fit_r]
    real = [a for a in pool if not a["procedural"]]
    by_cat: dict[str, list[dict]] = {}
    for a in real:
        by_cat.setdefault(a["category"], []).append(a)
    small = sorted(k for k, v in by_cat.items() if np.median([max(a["extent_m"]) for a in v]) < 0.6 * fit_r)
    # Drop each single / pile just above the surface at its own spot (a high drop bounces off
    # the dish): spots don't overlap each other, and avoid the heap unless deliberately on it.
    mounds = [m for _, m, _, _ in heap]
    placed: list[tuple[np.ndarray, float]] = [(salad_spot, 0.04)] if salad_spot is not None else []

    def support_z(xy: np.ndarray) -> float:
        if real_dish:                         # surface unknown analytically: drop from above the rim
            return c.H + 0.003
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
                      # pile pieces are poured: any orientation, so they tumble and interlock;
                      # singles are set down roughly as served (scans rest on their scanned base)
                      "init_euler": (rng.uniform(0, 2 * np.pi, 3) if a["procedural"] or kind == "pile" else
                                     np.r_[rng.normal(0, 0.3, 2), rng.uniform(0, 2 * np.pi)]).tolist(),
                      "up_axis": a.get("up_axis", "Y"), "color": a.get("color"),
                      "roughness": float(rng.uniform(0.3, 0.8)), "procedural": a["procedural"],
                      "generated": a.get("generated", False), "texture": bool(a["procedural"]),
                      "damping": 0.4 if kind == "pile" else 0.7, "shape": {"kind": kind}})

    rad_of = lambda a: float(np.linalg.norm(a["extent_m"])) / 2       # any rotation fits
    fits_on_heap = lambda size: bool(mounds) and size < 0.6 * mounds[0].rf
    cat_of = {k.lower().replace("_", " "): k for k in by_cat}           # meal names -> asset categories
    if meal["source"] == "random":
        pile_cat = small[rng.integers(len(small))] if small and rng.random() < 0.5 else None
    else:
        pile_cat = cat_of.get(meal["pile"] or "")
    if pile_cat and max_items - len(items) >= 2:
        grp = by_cat[pile_cat]
        k = int(min(rng.integers(2, 6), max_items - len(items)))
        sel = [grp[j] for j in rng.choice(len(grp), k, replace=len(grp) < k)]
        spread = 0.6 * max(max(a["extent_m"][:2]) for a in sel)
        ctr = spot(spread + 0.5 * max(max(a["extent_m"][:2]) for a in sel),
                   on_heap=rng.random() < 0.5 and fits_on_heap(max(max(a["extent_m"]) for a in sel)))
        if ctr is not None:
            z = support_z(ctr) + 0.002        # a pile: stacked by bounding radius, settles into contact
            for a in sel:
                z += rad_of(a)
                add(a, ctr + rng.normal(0, spread / 3, 2), z, "pile")
                z += rad_of(a) + 0.002
    room = max_items - len(items)
    if meal["source"] == "random":
        lo_n = 0 if items else (2 if real_dish else 1)      # a scanned dish holds a meal, not one egg
        n_single = int(rng.integers(min(lo_n, room), min(3 if items else 4, room) + 1)) if room > 0 else 0
        picks = [None] * n_single
    else:                                                    # the meal's own items, those we have scans of
        picks = [by_cat[cat_of[k]] for k in meal["scans"] if k in cat_of][:max(room, 0)]
        if not items and not picks:                         # nothing of the meal fits / exists: one random item
            picks = [None]
    for grp in picks:
        if grp is None:
            src = real if real and rng.random() < 0.8 else [a for a in pool if a["procedural"]]
            a = src[rng.integers(len(src))]
        else:
            a = grp[rng.integers(len(grp))]
        # sinks in (synth.mpm); big scans go beside the heap instead of burying it
        xy = spot(0.5 * max(a["extent_m"][:2]), on_heap=rng.random() < 0.5 and fits_on_heap(max(a["extent_m"])))
        if xy is not None:
            add(a, xy, support_z(xy) + rad_of(a) + 0.002, "single")
    for i, it in enumerate(items):
        it["id"] = i + 1                      # mask label; 0 = background

    K, w, h = intrinsics(rng, res_scale)
    poses, times = orbit_poses(rng, n_views, centre=np.array([0, 0, c.H * 0.5]))
    spec.update({
        "meal": meal,
        "items": items,
        "distractors": distractors(rng, out, c.R + c.rim_w, assets) if rng.random() < 0.7 else [],
        "table": table_look(rng, surfaces),
        "lighting": lighting(rng, hdris),
        # render scene-linear; synth.camera.develop meters and tone-maps per scan like a phone
        # (fixed curves either washed colours out (Filmic) or blew out bright scenes (Standard))
        "look": {"hdr": True},
        "camera": {"K": K.tolist(), "width": w, "height": h, "res_scale": res_scale,
                   "c2w": [p.tolist() for p in poses], "timestamps": times,
                   # a phone's small aperture: full-frame-equivalent f-number (render.py), focus on the dish
                   "dof": {"fstop": float(rng.uniform(8, 16)), "focus": [0.0, 0.0, float(c.H * 0.5)]}},
        "world_to_arkit": world_to_arkit(rng).tolist(),
    })
    (out / "spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))
    return spec
