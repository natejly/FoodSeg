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
import zlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import (Container, Mound, box_mesh, ellipsoid_mesh, lathe_solid, mesh_volume, salad_sheets, sample_container,
                       split_angles, topping_mesh)
from .looks import PIECES, kind_of, look, physics, shape
from .density import density_for
from .meals import BOWL_ONLY, FOODS, SCAN_FOR_FILL, SCAN_P, SIDES, WHOLE_RAW, sample_meal

# whole fruit that may lie on a fruit plate
FRUIT_PLATE_OK = {"apple", "orange", "banana", "peach", "pear", "plum", "kiwi", "mango"}
# serving size of a scanned soft-food portion, as a scale on the scan (default 0.85-1.1)
PORTION_SCALE = {"fried rice": (0.72, 0.88), "stew beef": (0.75, 0.92), "pasta mixed dishes": (0.8, 0.98),
                 "lasagna": (0.85, 1.0), "mashed potato": (0.9, 1.15), "mac": (0.9, 1.1), "rice": (0.9, 1.15),
                 "omelet": (0.85, 1.0), "beans": (0.8, 1.0), "yogurt": (0.9, 1.1)}
from .spec import FILLS as V0_FILLS, intrinsics, orbit_poses, world_to_arkit

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
    vol_ml = mesh_volume(V, F) * 1e6
    rho = density_for(name, shape.get("kind"))       # as-served bulk density, g/mL (synth.density)
    return {"name": name, "category": name, "source": source, "asset_id": None, "mesh": mesh, "fmt": "npz",
            "to_m": 1.0, "scale": 1.0, "volume_ml": vol_ml, "mass_g": float(vol_ml * rho), "mass_source": "density",
            "density_g_ml": float(rho), "dynamic": False,
            "fill": None, "color": list(col), "roughness": rough, "procedural": True, "generated": False,
            "texture": True, "shape": shape,
            # appearance and MPM material by consistency (synth.looks, synth.mpm)
            "look": look(name, rng), "physics": physics(name), "pieces": PIECES.get(name, [])}


STEEL = [(0.62, 0.62, 0.64), (0.70, 0.70, 0.72), (0.5, 0.5, 0.52), (0.04, 0.04, 0.045)]   # last: matte black cutlery
GARISH_DISHES = ("turquo", "yellowblue", "ruby", "deep_colors", "dragon", "scirocco", "good_earth", "029_plate")      # candy colours: not real dinnerware
REAL_PROPS = {"knife": ("032_knife",), "fork": ("030_fork",), "spoon": ("031_spoon",),
              "mug": ("mug", "cup"), "board": ("cutting_board",)}


def _merge(*parts) -> tuple[np.ndarray, np.ndarray]:
    """Union of closed meshes (overlapping is fine for rendering a static prop)."""
    Vs, Fs, n = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def _shift(m, dx=0.0, dy=0.0, dz=0.0):
    return m[0] + [dx, dy, dz], m[1]


def _spoon(rng):
    L, w = rng.uniform(0.15, 0.19), rng.uniform(0.032, 0.045)
    bowl = ellipsoid_mesh(0.3 * L / 2 * 1.0, w / 2, 0.0035, n=24)
    handle = box_mesh(0.7 * L, rng.uniform(0.009, 0.013), 0.0025)
    return _merge(_shift(bowl, dx=0.35 * L), _shift(handle, dx=-0.15 * L))


def _fork(rng):
    L, w = rng.uniform(0.17, 0.2), rng.uniform(0.026, 0.032)
    handle = box_mesh(0.65 * L, rng.uniform(0.008, 0.012), 0.0025)
    head = box_mesh(0.12 * L, w, 0.0025)
    tines = [_shift(box_mesh(0.22 * L, 0.0028, 0.0025), dx=0.3 * L, dy=y)
             for y in np.linspace(-w / 2 + 0.0014, w / 2 - 0.0014, 4)]
    return _merge(_shift(handle, dx=-0.17 * L), _shift(head, dx=0.15 * L), *tines)


def _knife(rng):
    L = rng.uniform(0.2, 0.235)
    blade = box_mesh(0.48 * L, rng.uniform(0.017, 0.022), 0.0015)
    handle = box_mesh(0.5 * L, rng.uniform(0.011, 0.015), rng.uniform(0.006, 0.01))
    return _merge(_shift(blade, dx=0.26 * L), _shift(handle, dx=-0.25 * L))


def _chopsticks(rng):
    parts = []
    for y, tilt in ((-0.011, 0.0), (0.011, rng.uniform(-0.04, 0.04))):
        V, F = lathe_solid(np.array([[0, 0], [0.0045, 0], [0.0045, 0.12], [0.0028, 0.235], [0, 0.235]]), 16)
        V = np.stack([V[:, 2], V[:, 1] + y, V[:, 0]], 1)              # lie along x, resting on the table
        V[:, 0] -= 0.12
        V[:, 1] += tilt * V[:, 0]
        V[:, 2] += 0.0045
        parts.append((V, F[:, ::-1].copy()))
    return _merge(*parts)


def _glass(rng):
    r, h = rng.uniform(0.031, 0.038), rng.uniform(0.09, 0.13)
    r2 = r * rng.uniform(0.85, 1.0)
    return lathe_solid(np.array([[0, 0], [r2, 0], [r, h], [r - 0.0025, h], [r2 - 0.0025, 0.009], [0, 0.009]]), 48)


def _napkin(rng) -> tuple[np.ndarray, np.ndarray]:
    """A paper / cloth napkin: a thin sheet with soft folds and curled corners (open mesh, rendered two-sided)."""
    w, h = rng.uniform(0.1, 0.17), rng.uniform(0.07, 0.13)
    nx, ny = 40, 28
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, nx), np.linspace(-0.5, 0.5, ny))
    z = np.zeros_like(u)
    for _ in range(4):
        k, ph, ang = rng.uniform(5, 18), rng.uniform(0, 6.3), rng.uniform(0, np.pi)
        z += rng.uniform(0.0004, 0.0016) * np.sin(k * (u * np.cos(ang) + v * np.sin(ang)) + ph)
    z += 0.003 * (np.maximum(np.abs(u) - 0.38, 0) + np.maximum(np.abs(v) - 0.38, 0)) / 0.12 * rng.uniform(0.2, 1.0)
    z += 0.0015 + rng.uniform(0, 0.003) * np.maximum(0, 1 - np.abs(u - rng.uniform(-0.3, 0.3)) / 0.05)   # a fold ridge
    V = np.stack([u * w, v * h, z], -1).reshape(-1, 3)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return V, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def distractors(rng: np.random.Generator, out: Path, r_clear: float, assets: list[dict] | None = None,
                style: str = "home") -> list[dict]:
    """0-3 pieces of tableware / clutter on the table outside radius r_clear of the dish axis (label 253).
    Real scans (YCB / GSO mugs, YCB cutlery, a cutting board) a quarter of the time, else procedural
    stainless cutlery, glass, chopsticks, napkins, mugs in plain colours; the red-handled YCB cutlery
    is desaturated so it does not repeat across scenes."""
    if style == "cafeteria":
        pool, p_none = ["fork", "knife", "napkin", "spoon", "phone", "shaker", "bottle", "plate2"], 0.7
    else:
        pool, p_none = ["fork", "knife", "spoon", "mug", "glass", "napkin", "chopsticks", "board", "phone", "shaker", "bottle",
                        "plate2"], 0.45
    if rng.random() < p_none:
        return []
    wts = np.array([{"board": 0.4, "chopsticks": 0.5, "plate2": 0.5, "shaker": 0.5, "bottle": 0.6}.get(k, 1.0) for k in pool])
    kinds = rng.choice(pool, size=min(int(rng.integers(1, 4)), len(pool)), replace=False, p=wts / wts.sum())
    props = [a for a in (assets or []) if a.get("role") == "clutter"]
    res, ang = [], rng.uniform(0, 2 * np.pi)
    # where each prop lies: props of a set sit in unrelated places (a fork on one side, a glass behind), mostly on a
    # side not shared with the others, at varying distances, with a loose yaw (not always tangent to the dish)
    sides = rng.permutation(8)[:len(kinds)] * (2 * np.pi / 8) + rng.uniform(-0.3, 0.3, len(kinds))
    for i, k in enumerate(kinds):
        k = str(k)
        real = [a for a in props if any(t in a["id"].lower() for t in REAL_PROPS.get(k, ()))]
        if real and rng.random() < (0.5 if k in ("mug", "board") else 0.85):
            a = real[rng.integers(len(real))]
            half = 0.5 * max(a["extent_m"][:2])
            dist = r_clear + (0.5 * min(a["extent_m"][:2]) if k != "mug" else half) + rng.uniform(0.02, 0.06)
            ang_i = float(sides[i])
            dist += rng.uniform(0.0, 0.07)
            yaw = ang_i + np.pi / 2 + rng.normal(0, 0.45) if (k != "mug" and rng.random() < 0.75) else rng.uniform(0, 2 * np.pi)
            res.append({"name": k, "mesh": a["path"], "fmt": a["fmt"], "up_axis": a.get("up_axis", "Z"),
                        "asset_id": a["id"], "location": [float(dist * np.cos(ang_i)), float(dist * np.sin(ang_i)), 0.0],
                        "euler": [0.0, 0.0, float(yaw)], "sat": float(rng.uniform(0.0, 0.08)) if k in ("knife", "fork", "spoon")
                        else (float(rng.uniform(0.0, 0.5)) if "ycb" in a["id"] else 1.0)})
            continue
        col, metal, rough, extra, reach = None, 0.0, 0.3, {}, 0.03
        if k == "mug":
            r, hgt = rng.uniform(0.035, 0.045), rng.uniform(0.08, 0.10)
            V, F = lathe_solid(np.array([[0, 0], [r, 0], [r, hgt], [r - 0.004, hgt], [r - 0.004, 0.006], [0, 0.006]]))
            col = [(0.8, 0.8, 0.78), (0.75, 0.72, 0.62), (0.05, 0.05, 0.05), (0.2, 0.25, 0.3), (0.55, 0.3, 0.2)][
                int(rng.choice(5, p=[0.45, 0.15, 0.15, 0.15, 0.10]))]
            reach = r + rng.uniform(0.02, 0.08)
        elif k == "glass":
            V, F = _glass(rng)
            col, rough, extra, reach = (0.9, 0.9, 0.9), 0.02, {"glass": True}, 0.035 + rng.uniform(0.03, 0.08)
        elif k == "napkin":
            V, F = _napkin(rng)
            col = [(0.82, 0.82, 0.8), (0.75, 0.7, 0.58), (0.45, 0.47, 0.5), (0.8, 0.5, 0.45)][int(rng.choice(4, p=[0.5, 0.2, 0.2, 0.1]))]
            rough, reach = 0.95, rng.uniform(0.08, 0.13)
        elif k == "phone":
            V, F = box_mesh(rng.uniform(0.067, 0.078), rng.uniform(0.14, 0.16), rng.uniform(0.0075, 0.009))
            col = [(0.01, 0.01, 0.012), (0.55, 0.55, 0.57), (0.02, 0.03, 0.08), (0.8, 0.78, 0.74)][int(rng.choice(4, p=[0.6, 0.15, 0.15, 0.1]))]
            rough, reach = 0.18, rng.uniform(0.05, 0.1)
        elif k == "shaker":
            r, hgt = rng.uniform(0.015, 0.022), rng.uniform(0.06, 0.1)
            V, F = lathe_solid(np.array([[0, 0], [r, 0], [r * 1.05, hgt * 0.8], [r * 0.7, hgt * 0.95], [r * 0.7, hgt], [0, hgt]]), 32)
            col, rough, extra, reach = (0.9, 0.9, 0.9), 0.05, {"glass": True, "glass_tint": [0.92, 0.95, 0.93]}, 0.04 + rng.uniform(0.02, 0.1)
        elif k == "bottle":
            r, hgt = rng.uniform(0.027, 0.036), rng.uniform(0.2, 0.28)
            V, F = lathe_solid(np.array([[0, 0], [r, 0], [r, hgt * 0.55], [r * 0.85, hgt * 0.7], [r * 0.38, hgt * 0.78], [r * 0.34, hgt],
                                         [0, hgt]]), 40)
            tint = [(0.1, 0.3, 0.12), (0.25, 0.12, 0.04), (0.9, 0.95, 0.95)][int(rng.choice(3, p=[0.3, 0.3, 0.4]))]
            col, rough, extra, reach = tint, 0.03, {"glass": True, "glass_tint": list(tint)}, 0.05 + rng.uniform(0.03, 0.12)
        elif k == "plate2":                        # another dish (empty / pushed aside): mostly out of frame
            c2 = sample_container(rng, str(rng.choice(["plate", "bowl"], p=[0.7, 0.3])))
            V, F = c2.mesh()
            col2, fam = dish_colour(rng, style)
            col, rough = tuple(col2), 0.3
            extra = {"ceramic": {"R": float(c2.R), "rim_w": float(c2.rim_w), "color": col2, "roughness": float(rng.uniform(0.15, 0.4)),
                                 "glaze": glaze(rng, fam, style)}}
            reach = c2.R + c2.rim_w + rng.uniform(0.01, 0.07)
        elif k == "chopsticks":
            V, F = _chopsticks(rng)
            col, rough, reach = (0.5, 0.32, 0.15) if rng.random() < 0.6 else (0.03, 0.03, 0.03), 0.45, 0.04
        elif k == "board":
            V, F = box_mesh(rng.uniform(0.28, 0.4), rng.uniform(0.17, 0.24), 0.018)
            col, rough, reach = (0.45, 0.30, 0.16), 0.55, 0.12
        else:
            V, F = {"knife": _knife, "fork": _fork, "spoon": _spoon}[k](rng)
            col, metal, rough = STEEL[int(rng.choice(4, p=[0.4, 0.35, 0.2, 0.05]))], 1.0, float(rng.uniform(0.25, 0.5))
            reach = 0.05
        V = V - [0, 0, V[:, 2].min()]
        dist = r_clear + reach + rng.uniform(0.01, 0.1)
        a = float(sides[i])
        mesh = f"distractor_{i}.npz"
        np.savez_compressed(out / mesh, V=V.astype(np.float32), F=F.astype(np.int32))
        yaw = a + np.pi / 2 + rng.normal(0, 0.45) if (k in ("knife", "fork", "spoon", "chopsticks") and rng.random() < 0.75) \
            else rng.uniform(0, 2 * np.pi)                      # cutlery roughly tangent, sometimes anyhow
        res.append({"name": k, "mesh": mesh, "location": [float(dist * np.cos(a)), float(dist * np.sin(a)), 0.0],
                    "euler": [0.0, 0.0, float(yaw)], "color": [float(v) for v in col], "metallic": metal,
                    "roughness": rough, **extra})
    return res


# ---- dishes ------------------------------------------------------------------

# Dinnerware body colours (linear RGB, weight): white / ivory stoneware about 3 in 4, a few muted mid tones
# and darker glazes; nothing saturated (the old random blue / green / red / purple plates were a clear tell).
DISH_WHITE = [((0.80, 0.80, 0.79), 30), ((0.78, 0.80, 0.82), 12), ((0.80, 0.77, 0.68), 18), ((0.73, 0.69, 0.57), 8),
              ((0.50, 0.50, 0.50), 6)]
DISH_MID = [((0.45, 0.40, 0.32), 4), ((0.35, 0.38, 0.34), 3), ((0.55, 0.52, 0.45), 3)]
DISH_DARK = [((0.03, 0.03, 0.035), 5), ((0.05, 0.06, 0.075), 3), ((0.10, 0.06, 0.04), 3), ((0.03, 0.05, 0.10), 2),
             ((0.20, 0.09, 0.05), 2), ((0.11, 0.12, 0.08), 2)]
RIM_COLOURS = [(0.02, 0.04, 0.12), (0.08, 0.08, 0.08), (0.35, 0.27, 0.10), (0.22, 0.22, 0.23), (0.22, 0.05, 0.04),
               (0.06, 0.16, 0.08), (0.12, 0.2, 0.3)]
SAUCES = [(0.30, 0.05, 0.02), (0.12, 0.06, 0.02), (0.45, 0.22, 0.04), (0.55, 0.42, 0.12), (0.35, 0.30, 0.18)]
CRUMBS = [(0.25, 0.14, 0.06), (0.4, 0.28, 0.12), (0.12, 0.2, 0.05), (0.35, 0.08, 0.04), (0.5, 0.4, 0.2)]


def _pick(rng, table):
    w = np.array([t[1] for t in table], float)
    return table[int(rng.choice(len(table), p=w / w.sum()))][0]


def dish_colour(rng: np.random.Generator, style: str = "home") -> tuple[list[float], str]:
    """(linear-RGB body colour, family 'white' | 'mid' | 'dark'); cafeteria dishes are white."""
    u = rng.random()
    if style == "cafeteria":
        fam, tab = ("white", DISH_WHITE[:3] + DISH_WHITE[4:]) if u < 0.93 else ("mid", DISH_MID)
    elif u < 0.75:
        fam, tab = "white", DISH_WHITE
    elif u < 0.85:
        fam, tab = "mid", DISH_MID
    else:
        fam, tab = "dark", DISH_DARK
    col = np.asarray(_pick(rng, tab), float) * rng.normal(1.0, 0.025, 3)
    col = np.clip(col, 0.01, 0.9)
    mx, mn = col.max(), col.min()
    cap = 0.8 if fam == "dark" else 0.3               # (max - min) / max: ceramics are not candy colours
    if mx > 0 and (mx - mn) / mx > cap:
        col = mx - (mx - col) * cap * mx / (mx - mn)
    return [float(v) for v in col], fam


def glaze(rng: np.random.Generator, family: str = "white", style: str = "home") -> dict:
    """Procedural dish finish (render.ceramic_material): body type (porcelain / speckled stoneware /
    melamine), glaze unevenness, micro-scratch wear, outline-following rim patterns (thin lines, a band,
    a ring of dots), and in some scenes sauce smears and a few crumbs on the dish (dish label)."""
    cafe = style == "cafeteria"
    body = str(rng.choice(["porcelain", "stoneware", "melamine"], p=[0.6, 0.15, 0.25] if cafe else [0.55, 0.38, 0.07]))
    g: dict = {"body": body, "speckle": 0.0, "pattern": None, "smears": [], "crumbs": None}
    if body == "porcelain":
        g.update(coat=float(rng.uniform(0.3, 0.7) if cafe else rng.uniform(0.4, 0.85)), coat_rough=float(rng.uniform(0.1, 0.22)), base_rough=float(rng.uniform(0.2, 0.4)),
                 sss=float(rng.uniform(0.0, 0.04)), uneven=float(rng.uniform(0.01, 0.04)))
    elif body == "stoneware":
        g.update(coat=float(rng.uniform(0.3, 0.8)), coat_rough=float(rng.uniform(0.12, 0.28)), base_rough=float(rng.uniform(0.3, 0.55)),
                 sss=0.0, uneven=float(rng.uniform(0.04, 0.1)),
                 speckle=float(rng.uniform(0.15, 0.4)) if rng.random() < 0.5 else 0.0)
    else:                                                    # melamine / hard plastic
        g.update(coat=float(rng.uniform(0.0, 0.3)), coat_rough=float(rng.uniform(0.1, 0.25)), base_rough=float(rng.uniform(0.3, 0.45)),
                 sss=0.0, uneven=float(rng.uniform(0.005, 0.02)))
    g["wear"] = float(rng.uniform(0.5, 1.0) if cafe else rng.uniform(0.15, 0.9))
    p_pat = 0.08 if cafe else (0.25 if family == "white" else 0.12)
    if family == "dark" and rng.random() < 0.6:             # glaze break: lighter, rolled rim edge
        g["pattern"] = {"type": "band", "color": [float(v) for v in np.clip(rng.uniform(1.8, 3.5) * np.array(rng.uniform(0.6, 1, 3)) * 0.12, 0, 0.5)],
                        "w": float(rng.uniform(0.004, 0.012)), "soft": 0.005}
    elif rng.random() < p_pat:
        col = [float(v) for v in RIM_COLOURS[rng.integers(len(RIM_COLOURS))]]
        t = str(rng.choice(["lines", "band", "dots"], p=[0.45, 0.3, 0.25]))
        if t == "lines":
            g["pattern"] = {"type": "lines", "color": col, "w": float(rng.uniform(0.0015, 0.004)),
                            "off": [float(rng.uniform(0.003, 0.012))] + ([float(rng.uniform(0.016, 0.026))] if rng.random() < 0.4 else []),
                            "soft": 0.0006}
        elif t == "band":
            g["pattern"] = {"type": "band", "color": col, "w": float(rng.uniform(0.008, 0.03)), "soft": 0.001}
        else:
            g["pattern"] = {"type": "dots", "color": col, "w": float(rng.uniform(0.003, 0.006)),
                            "off": float(rng.uniform(0.008, 0.016)), "k": int(rng.integers(30, 90))}
    if rng.random() < (0.35 if cafe else 0.42):      # sauce smears and wipes (a used plate)
        for _ in range(int(rng.integers(1, 4))):
            g["smears"].append({"color": [float(v) for v in SAUCES[rng.integers(len(SAUCES))]],
                                "alpha": float(rng.uniform(0.2, 0.55)), "scale": float(rng.uniform(25, 70)),
                                "thr": float(rng.uniform(0.56, 0.66)), "gate": float(rng.uniform(0.15, 0.85)),
                                "seed": float(rng.uniform(0, 100))})
    if rng.random() < 0.4:
        g["crumbs"] = {"color": [float(v) for v in CRUMBS[rng.integers(len(CRUMBS))]], "density": float(rng.uniform(0.5, 1.0)),
                       "seed": float(rng.uniform(0, 100)), "rim": bool(rng.random() < 0.5)}
    if rng.random() < 0.8:                           # greasy finger smudges and thumb prints, mostly near the rim
        g["smudge"] = float(rng.uniform(0.3, 1.0))
        g["prints"] = [{"c": [float(np.cos(a) * rng.uniform(0.6, 0.98)), float(np.sin(a) * rng.uniform(0.6, 0.98))],
                        "r": float(rng.uniform(0.008, 0.013)), "seed": float(rng.uniform(0, 100))}
                       for a in rng.uniform(0, 2 * np.pi, int(rng.integers(1, 4)))]
    return g


# ---- tables and mats --------------------------------------------------------

TAN_WOODS = ("plywood", "bamboo_veneer", "okoume_veneer", "white_maple_veneer", "ash_veneer", "oak_veneer_01",
             "oak_veneer_04")
TABLES = [   # plain procedural laminate (render.laminate_material): colour range low, high, roughness range, weight
    ((0.55, 0.55, 0.52), (0.85, 0.85, 0.82), (0.25, 0.5), 5),          # light laminate / white-grey counter
    ((0.40, 0.36, 0.30), (0.6, 0.55, 0.46), (0.3, 0.5), 2),            # beige / tan
    ((0.03, 0.03, 0.03), (0.12, 0.12, 0.12), (0.3, 0.7), 1),           # dark laminate
    ((0.12, 0.07, 0.035), (0.30, 0.17, 0.08), (0.35, 0.6), 2),         # stained wood (muted brown)
    ((0.20, 0.20, 0.21), (0.45, 0.45, 0.46), (0.2, 0.5), 1.5),         # grey stone / concrete
    ((0.30, 0.31, 0.28), (0.5, 0.5, 0.45), (0.7, 0.95), 1),            # cloth: grey-green linen
]
P_PBR = {"home": 0.7, "cafeteria": 0.55}
KIND_P = {"wood": 0.5, "cloth": 0.3, "stone": 0.2}


def _pbr(rng, t: dict, sat=(0.5, 0.95), tint_sd: float = 0.04, flat=(0.5, 0.9)) -> dict:
    """A photographed surface at its real tile size, rotated / offset, tinted a little and partly
    desaturated (stock wood is far more saturated than a real worn tabletop)."""
    return {"dir": t["dir"], "kind": t["kind"], "tile_m": float(t["tile_m"]), "rotation": float(rng.uniform(0, np.pi)),
            "offset": rng.uniform(0, 3, 2).tolist(), "tint": rng.normal(1.0, tint_sd, 3).clip(0.85, 1.15).tolist(),
            "brightness": float(rng.uniform(0.75, 1.1)), "sat": float(rng.uniform(*sat)),
            "grunge": float(rng.uniform(0.0, 0.5)),
            # stock textures have far more fine contrast than a worn, phone-photographed table: pull the colour toward
            # the tile's mean ("flat") and soften the normal map
            "mean": _tile_mean(t["dir"]), "flat": float(rng.uniform(*flat)), "nstr": float(rng.uniform(0.2, 0.8))}


@lru_cache(maxsize=None)
def _tile_mean(d: str) -> list[float]:
    """Mean linear-RGB colour of a surface's diffuse map."""
    from PIL import Image
    a = np.asarray(Image.open(Path(d) / "diffuse.jpg").convert("RGB").resize((16, 16), Image.BILINEAR), np.float32) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return [float(v) for v in lin.reshape(-1, 3).mean(0)]


STAIN_COL = [(0.25, 0.14, 0.07), (0.3, 0.2, 0.1), (0.12, 0.1, 0.08), (0.3, 0.1, 0.05)]


def table_extras(rng: np.random.Generator, t: dict, style: str) -> dict:
    """Surface history and extent of a tabletop (render.table_wear): coffee rings / spills, crumbs, scratches,
    finger smudges in the roughness, sometimes a lacquered / glass top that mirrors the room; a finite table
    (an edge and the room's floor past it) in half of the scenes."""
    cloth = (t.get("pbr") or {}).get("kind") == "cloth"
    cafe = style == "cafeteria"
    W: dict = {"stains": [], "crumbs": None, "scratch": 0.0, "smudge": 0.0, "gloss": None, "seed": float(rng.uniform(0, 100)),
               "scratch_rot": float(rng.uniform(0, np.pi))}
    for _ in range(int(rng.choice(4, p=[0.35, 0.3, 0.2, 0.15]))):
        ring = rng.random() < 0.55
        W["stains"].append({"kind": "ring" if ring else "blob", "c": [float(v) for v in rng.uniform(-0.4, 0.4, 2)],
                            "r": float(rng.uniform(0.03, 0.045) if ring else rng.uniform(0.015, 0.05)),
                            "dark": float(rng.uniform(0.2, 0.6)), "seed": float(rng.uniform(0, 100)),
                            "col": [float(v) for v in STAIN_COL[rng.integers(len(STAIN_COL))]]})
    if rng.random() < (0.6 if cafe else 0.45):
        W["crumbs"] = {"density": float(rng.uniform(0.3, 1.0)), "scale": float(rng.uniform(110, 200)), "seed": float(rng.uniform(0, 100)),
                       "color": [float(v) for v in [(0.5, 0.4, 0.22), (0.3, 0.2, 0.1), (0.6, 0.5, 0.35)][rng.integers(3)]]}
    if not cloth:
        W["scratch"] = float(rng.uniform(0.3, 1.0)) if rng.random() < 0.6 else 0.0
        if rng.random() < 0.2:
            W["gloss"] = {"coat": 1.0, "rough": float(rng.uniform(0.02, 0.12))}
    W["smudge"] = float(rng.uniform(0.3, 1.0)) if rng.random() < 0.6 else 0.0
    if (t.get("pbr") or {}).get("kind") == "stone":        # matte stone / plaster: no streaky scratches or smears, soft normals
        W["scratch"], W["smudge"] = 0.0, 0.0
        t["pbr"]["nstr"] = min(t["pbr"]["nstr"], 0.18)
    t["wear"] = W
    if not t.get("pbr") and rng.random() < 0.12:          # a crumpled sheet / cloth: dark tarp, grey backdrop, creased linen
        dark = rng.random() < 0.5
        base = rng.uniform(0.03, 0.12) if dark else rng.uniform(0.2, 0.45)
        t["color"] = [float(base * v) for v in (1.0, rng.uniform(0.97, 1.03), rng.uniform(0.93, 1.03))]
        t["roughness"] = float(rng.uniform(0.35, 0.7))
        t["crumple"] = {"scale": float(rng.uniform(4, 11)), "strength": float(rng.uniform(0.5, 1.4)), "seed": float(rng.uniform(0, 100))}
    if rng.random() < 0.5:
        sx = float(rng.uniform(1.0, 1.8))
        sy = float(rng.uniform(0.8, 1.0) * sx)
        t.update(size_m=sx, size_y_m=sy, centre=[float(rng.uniform(-1, 1) * max(sx / 2 - 0.4, 0) * 0.8), float(rng.uniform(-1, 1) * max(sy / 2 - 0.4, 0) * 0.8)],
                 floor=[float(v) for v in rng.uniform(0.03, 0.25) * np.array([1.0, rng.uniform(0.85, 1.0), rng.uniform(0.7, 1.0)])])
    return t


def table_look(rng: np.random.Generator, surfaces: list[dict] | None = None, style: str = "home") -> dict:
    return table_extras(rng, _table_look(rng, surfaces, style), style)


def _table_look(rng: np.random.Generator, surfaces: list[dict] | None = None, style: str = "home") -> dict:
    """The tabletop: a photographed PBR surface (wood, stone, laminate, linen) at real-world scale with de-tiled
    lookup (render.pbr_table), else a plain procedural one. Cafeteria style: light tan wood or light laminate."""
    if surfaces and rng.random() < P_PBR[style if style in P_PBR else "home"]:
        if style == "cafeteria":
            pool = [t for t in surfaces if Path(t["dir"]).name in TAN_WOODS] or surfaces
            t = pool[rng.integers(len(pool))]
            sat = (0.5, 0.8)
        else:
            kinds = sorted({t["kind"] for t in surfaces})
            w = np.array([KIND_P.get(k, 0.1) for k in kinds])
            kind = kinds[int(rng.choice(len(kinds), p=w / w.sum()))]
            pool = [t for t in surfaces if t["kind"] == kind]
            t = pool[rng.integers(len(pool))]
            sat = (0.3, 0.75) if kind != "cloth" else (0.0, 0.5)
        return {"pbr": _pbr(rng, t, sat), "color": [0.5, 0.5, 0.5], "roughness": 0.5, "size_m": 4.0}
    w = np.array([t[3] for t in TABLES], float)
    lo, hi, rough, _ = TABLES[int(rng.choice(len(TABLES), p=w / w.sum()))]
    col = np.asarray(lo) + rng.uniform() * (np.asarray(hi) - np.asarray(lo))          # one brightness, fixed hue
    return {"color": [float(v) for v in col * rng.normal(1.0, 0.015, 3)], "roughness": float(rng.uniform(*rough)), "size_m": 4.0}


def mat_look(rng: np.random.Generator, surfaces: list[dict] | None, style: str, r_out: float) -> dict | None:
    """A flat board / tray / placemat / napkin liner under the dish (a thin slab, label 0). It always
    reaches past the dish outline r_out."""
    p = 0.7 if style == "cafeteria" else 0.25
    if rng.random() > p:
        return None
    need = 2.0 * r_out + 0.06
    kind = str(rng.choice(["tray", "board"], p=[0.45, 0.55])) if style == "cafeteria" else \
        str(rng.choice(["board", "placemat", "paper", "slate"], p=[0.5, 0.3, 0.03, 0.17]))
    w = max(need * rng.uniform(1.0, 1.15), rng.uniform(0.34, 0.46))
    h = max(need * rng.uniform(1.0, 1.1), rng.uniform(0.26, 0.34))
    if kind == "paper":
        w, h = need * rng.uniform(1.05, 1.3), need * rng.uniform(1.05, 1.3)
    m = {"kind": kind, "size": [float(w), float(h)], "yaw": float(rng.uniform(0, np.pi)),
         "thickness": 0.0012 if kind != "paper" else 0.0006}
    if kind == "tray":                                  # cafeteria tray: tan / grey / brown plastic
        m.update(color=[float(v) for v in [(0.35, 0.26, 0.16), (0.22, 0.22, 0.22), (0.12, 0.08, 0.05), (0.45, 0.38, 0.28)][
            int(rng.choice(4, p=[0.35, 0.3, 0.15, 0.2]))] * rng.normal(1, 0.04, 3)], roughness=float(rng.uniform(0.35, 0.55)))
    elif kind == "paper":
        m.update(color=[0.72, 0.72, 0.70], roughness=0.85)
    elif kind == "slate":
        m.update(color=[float(v) for v in rng.uniform(0.04, 0.09) * np.array([1.0, 1.02, 1.05])], roughness=float(rng.uniform(0.4, 0.8)))
    else:
        want = ("wood",) if kind == "board" else ("cloth",)
        pool = [t for t in (surfaces or []) if t["kind"] in want]
        if kind == "board":
            pool = [t for t in pool if Path(t["dir"]).name in TAN_WOODS] or pool
        if not pool:
            m.update(color=[0.4, 0.3, 0.2] if kind == "board" else [0.6, 0.58, 0.52], roughness=0.6)
        else:
            m["pbr"] = _pbr(rng, pool[rng.integers(len(pool))], (0.5, 0.9) if kind == "board" else (0.0, 0.4))
            m["color"], m["roughness"] = [0.5, 0.5, 0.5], 0.5
    if kind == "tray" or (style == "cafeteria" and rng.random() < 0.3):     # raised, darker lip all round
        m["rim"] = {"h": float(rng.uniform(0.008, 0.016)), "w": float(rng.uniform(0.008, 0.016)),
                    "color": [float(v) for v in np.asarray(m["color"]) * rng.uniform(0.3, 0.7)] if "pbr" not in m else [0.04, 0.04, 0.04]}
    m["wear"] = {"stains": [], "crumbs": None, "scratch": float(rng.uniform(0.3, 1.0)) if rng.random() < 0.6 else 0.0,
                 "smudge": float(rng.uniform(0.3, 1.0)) if rng.random() < 0.6 else 0.0, "gloss": None,
                 "seed": float(rng.uniform(0, 100)), "scratch_rot": float(rng.uniform(0, np.pi))}
    return m


P_RIG = 0.85          # share of the cafeteria-style scenes captured on the Nutrition5k rig (render.build_rig)


def rig_look(rng: np.random.Generator, surfaces: list[dict] | None, c, r_out: float) -> dict:
    """The Nutrition5k capture rig seen from above (measured on realsense_overhead frames, 0.75 mm / px): a 0.37 x 0.35 m glass
    pane in a ~1 cm black metal frame; under the glass a 0.115 m wide plywood strip along the long side with a pair of
    screws at each end, dark cross braces and a pale floor far below; a white table all round."""
    half = c.R + c.rim_w                                  # the dish stays on the glass
    W = max(float(rng.uniform(0.355, 0.38)), 2 * half + 0.045)
    H = max(float(rng.uniform(0.335, 0.36)), 2 * half + 0.045)
    fw = float(rng.uniform(0.010, 0.014))
    off = [float(rng.uniform(-1, 1) * max(W / 2 - half - fw - 0.01, 0.0)),
           float(rng.uniform(-1, 1) * max(H / 2 - half - fw - 0.01, 0.0))]
    woods = [t for t in (surfaces or []) if Path(t["dir"]).name in TAN_WOODS]
    if woods:
        strip = {"pbr": _pbr(rng, woods[rng.integers(len(woods))], (0.95, 1.0), flat=(0.2, 0.5)), "color": [0.5, 0.5, 0.5], "roughness": 0.5}
        strip["pbr"]["rotation"] = float(rng.choice([0.0, np.pi / 2]))          # grain along the strip
    else:
        strip = {"color": [float(v) for v in np.array([0.55, 0.40, 0.23]) * rng.uniform(0.85, 1.1, 3)], "roughness": 0.55}
    wear = table_extras(rng, {}, "cafeteria")["wear"]
    wear["gloss"] = None
    g = float(rng.uniform(0.6, 0.9))
    return {"pane": [W, H], "frame_w": fw, "frame_h": float(rng.uniform(0.001, 0.004)), "centre": [-off[0], -off[1]],
            "yaw": float(rng.uniform(0, 2 * np.pi)), "strip": strip, "strip_w": float(rng.uniform(0.11, 0.12)),
            "strip_x": float(rng.uniform(-0.06, 0.06)), "frame_color": [0.015, 0.015, 0.017],
            "glass_tint": [g * 0.93, g, g * 0.97], "refl": float(rng.uniform(0.01, 0.05)),
            "floor": [float(v) for v in rng.uniform(0.4, 0.65) * np.array([0.95, 1.0, 1.02])], "floor_depth": float(rng.uniform(0.18, 0.4)),
            "braces": [[float(rng.uniform(0.4, 1.1) * s), float(rng.uniform(-0.25, 0.25)), float(rng.uniform(0.006, 0.012))]
                       for s in (1, -1)[: int(rng.integers(1, 3))]],
            "surround": [float(v) for v in rng.uniform(0.62, 0.85) * np.array([1.0, 1.0, rng.uniform(0.96, 1.02)])],
            "surround_rough": float(rng.uniform(0.35, 0.55)), "wear": wear}


def push_outside_rig(rig: dict, dist: list[dict], r_clear: float) -> None:
    """Props (cutlery, napkins) lie on the white table round the frame, not on the glass: move each one radially out
    until it clears the pane + frame by its own reach."""
    W, H = rig["pane"]
    fw, (cx, cy), yaw = rig["frame_w"], rig["centre"], rig["yaw"]
    ca, sa = np.cos(yaw), np.sin(yaw)
    for d in dist:
        p = np.asarray(d["location"][:2], float)
        u = p / max(np.linalg.norm(p), 1e-6)
        for _ in range(60):
            q = np.array([p[0] - cx, p[1] - cy])
            lx, ly = ca * q[0] + sa * q[1], -sa * q[0] + ca * q[1]
            if abs(lx) > W / 2 + fw + 0.07 or abs(ly) > H / 2 + fw + 0.07:
                break
            p = p + 0.01 * u
        d["location"][0], d["location"][1] = float(p[0]), float(p[1])


# ---- lighting -----------------------------------------------------------------

# Mean colour of each curated HDRI / its green (measured on the 2k maps): the strongly orange ones
# (fireplace, warm_bar, ...) are white-balanced towards neutral, as a phone's AWB would.
HDRI_CAST = {
    "aft_lounge": (1.2, 0.7), "anniversary_lounge": (1.25, 0.65), "cabin": (1.32, 0.58), "en_suite": (1.46, 0.55),
    "fireplace": (2.26, 0.15), "hotel_room": (1.26, 0.8), "kiara_interior": (0.84, 1.43), "lythwood_lounge": (1.32, 0.53),
    "lythwood_room": (1.29, 0.66), "warm_bar": (2.39, 0.32), "warm_restaurant_night": (2.18, 0.28),
    "wooden_lounge": (1.27, 0.76), "country_club": (1.12, 0.9), "combination_room": (0.92, 1.1),
    "solitude_interior": (0.94, 1.17), "veranda": (0.96, 1.13), "studio_small_03": (0.87, 1.13),
}
STUDIO_HDRIS = ("brown_photostudio_02", "photo_studio_01", "studio_small_03", "provence_studio", "white_home_studio")
CAFE_HDRIS = ("small_empty_room_1", "small_empty_room_3", "old_room", "white_home_studio", "photo_studio_01")


def _hdri_name(p: str) -> str:
    return Path(p).parent.name


def _wb(rng, name: str) -> list[float]:
    """Per-channel multiplier that removes a share of the HDRI's own colour cast (luminance kept)."""
    r, b = HDRI_CAST.get(name, (1.0, 1.0))
    k = float(rng.uniform(0.7, 1.0))
    w = np.array([r, 1.0, b]) ** -k
    return [float(v) for v in w / (w @ [0.2126, 0.7152, 0.0722])]


def _kelvin_tint(rng: np.random.Generator, lo: float = -1.0, hi: float = 1.0, spread: float = 0.16) -> list[float]:
    """Light colour on the warm (tungsten, +) - cool (overcast sky, -) axis."""
    t = rng.uniform(lo, hi) * spread
    return [float(1 + t), float(1 + 0.15 * t), float(1 - 1.7 * t)]


def _unit(az: float, el: float) -> list[float]:
    return [float(np.cos(el) * np.cos(az)), float(np.cos(el) * np.sin(az)), float(np.sin(el))]


def _hard_light(rng: np.random.Generator, kind: str) -> dict:
    """A hard directional key: a window (sun / clear sky through glass; low, from the side) or a ceiling spot (high);
    its shadows are crisp (angle_deg is the source's angular diameter). Optionally patterned by a window frame
    (mullions) or blinds, and partly shadowed by a hand / the photographer (L["hand"], render.py)."""
    if kind == "window":
        az, el = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(18, 52))
        ang, strength = rng.uniform(0.8, 5.0), rng.uniform(2.5, 7.5)
        col = _kelvin_tint(rng, -0.8, 0.7)
    else:
        az, el = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(60, 85))
        ang, strength = rng.uniform(2.0, 8.0), rng.uniform(2.0, 5.5)
        col = _kelvin_tint(rng, -0.2, 1.0)
    d = {"dir": _unit(az, el), "strength": float(strength), "angle_deg": float(ang), "color": col, "gobo": None}
    if kind == "window" and rng.random() < 0.3:
        d["gobo"] = {"kind": str(rng.choice(["mullion", "blinds"], p=[0.6, 0.4])), "dist": float(rng.uniform(1.0, 1.6)),
                     "rot": float(rng.uniform(-0.35, 0.35)), "pitch": float(rng.uniform(0.04, 0.09)),
                     "off": [float(v) for v in rng.uniform(-0.25, 0.25, 2)], "bars": [int(rng.integers(1, 3)), int(rng.integers(1, 3))]}
    return d


def _hand(rng: np.random.Generator, key_dir) -> dict:
    """A hand / forearm (and sometimes a phone) held between the key light and the table, moving with the camera:
    render.add_hard_light keyframes it per camera pose, over the table point `r` m from the dish axis on the camera's
    side (rotated by `dth`), `t` m up the ray towards the light, so its shadow falls partly over the scene."""
    return {"dir": [float(v) for v in key_dir], "r": float(rng.uniform(0.08, 0.3)), "dth": float(rng.uniform(-0.9, 0.9)),
            "t": float(rng.uniform(0.28, 0.5)), "yaw": float(rng.uniform(0, np.pi)),
            "size": [float(rng.uniform(0.07, 0.11)), float(rng.uniform(0.03, 0.05)), float(rng.uniform(0.02, 0.035))],
            "phone": bool(rng.random() < 0.5)}


def lighting(rng: np.random.Generator, hdris: list[str] | None, style: str = "home") -> dict:
    """Home: a real indoor HDRI (room bounce, reflections), strongly tinted ones white-balanced, plus man-made
    light: ceiling light, window, small lamp, and in over half the scenes a HARD key (a window's sun / a ceiling
    spot: crisp, correctly sized shadows; sometimes window mullion / blind patterns) with the HDRI dimmed to
    fill; a minority of scenes have a hand / photographer's shadow over part of the table. Light colours run from
    warm tungsten to cool daylight. Cafeteria: overhead panels and LED downlights in a plain bright room.
    Extra lights are in L["lights"] (render.py): {type AREA | POINT, location, energy_w, size_m, color};
    L["sun"] is the hard key {dir, strength, angle_deg, color, gobo}, L["hand"] its occluder."""
    lights = []
    if style == "cafeteria":
        pool = [h for h in (hdris or []) if _hdri_name(h) in CAFE_HDRIS]
        h = pool[rng.integers(len(pool))] if pool and rng.random() < 0.5 else None
        L = {"hdri": h, "hdri_strength": float(rng.uniform(0.1, 0.3)), "hdri_rotation": float(rng.uniform(0, 2 * np.pi)),
             "ambient": float(rng.uniform(0.3, 0.6)), "ambient_tint": [float(v) for v in rng.normal(1, 0.02, 3)],
             "hdri_wb": _wb(rng, _hdri_name(h)) if h else None, "sun": None, "hand": None}
        for j in range(1 + int(rng.random() < 0.6)):
            down = rng.random() < 0.4                       # LED downlight: small, crisp shadows
            lights.append({"type": "AREA", "location": [float(rng.uniform(-0.5, 0.5) + 0.9 * j), float(rng.uniform(-0.5, 0.5)),
                                                        float(rng.uniform(1.5, 2.3))],
                           "energy_w": float(rng.uniform(40, 110) if down else rng.uniform(60, 160)),
                           "size_m": float(rng.uniform(0.15, 0.4) if down else rng.uniform(0.6, 1.4)),
                           "color": [float(v) for v in (0.97, 1.0, 1.03 + rng.uniform(-0.03, 0.05))]})
        if rng.random() < 0.25:
            L["sun"] = _hard_light(rng, "window")
            L["sun"]["gobo"] = None
            L["hdri_strength"] *= 0.7
        if rng.random() < 0.2:
            L["hand"] = _hand(rng, (L["sun"] or {"dir": _unit(0, 1.2)})["dir"])
        L["lights"], L["key"] = lights, None
        return L
    studios = [h for h in (hdris or []) if _hdri_name(h) in STUDIO_HDRIS]
    rooms = [h for h in (hdris or []) if _hdri_name(h) not in STUDIO_HDRIS]
    h = None
    if hdris:
        h = studios[rng.integers(len(studios))] if studios and rng.random() < 0.1 else rooms[rng.integers(len(rooms))]
    L = {"hdri": h, "hdri_strength": float(rng.uniform(0.45, 1.2)), "hdri_rotation": float(rng.uniform(0, 2 * np.pi)),
         "ambient": float(rng.uniform(0.15, 0.5)), "ambient_tint": light_tint(rng, 0.06),
         "hdri_wb": _wb(rng, _hdri_name(h)) if h else None, "sun": None, "hand": None}
    hard = rng.random() < 0.6
    if hard:
        L["sun"] = _hard_light(rng, "window" if rng.random() < 0.65 else "spot")
        L["hdri_strength"] *= float(rng.uniform(0.35, 0.8))   # the room is the fill
        L["ambient"] *= 0.6
    if rng.random() < (0.3 if hard else 0.12):
        L["hand"] = _hand(rng, L["sun"]["dir"] if hard else _unit(rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(55, 80))))
    if rng.random() < (0.25 if hard else 0.5):        # ceiling light: warm to neutral
        lights.append({"type": "AREA", "location": [float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8)), float(rng.uniform(1.4, 2.2))],
                       "energy_w": float(rng.uniform(12, 50)), "size_m": float(rng.uniform(0.15, 0.7)),
                       "color": _kelvin_tint(rng, 0.0, 1.0, 0.1)})
    if (rng.random() < 0.4 and not hard) or not h:   # a softer window / daylight source at the side
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(1.0, 1.8)
        lights.append({"type": "AREA", "location": [float(d * np.cos(a)), float(d * np.sin(a)), float(rng.uniform(0.5, 1.3))],
                       "energy_w": float(rng.uniform(20, 90)), "size_m": float(rng.uniform(0.5, 1.2)),
                       "color": _kelvin_tint(rng, -1.0, 0.0, 0.1)})
    if rng.random() < 0.25:       # small bright lamp: crisp contact shadows
        lights.append({"type": "POINT", "location": [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(0.8, 1.6))],
                       "energy_w": float(rng.uniform(15, 60)), "size_m": float(rng.uniform(0.02, 0.08)),
                       "color": _kelvin_tint(rng, 0.2, 1.0, 0.12)})
    L["lights"], L["key"] = lights, None
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
        rad = float(rng.uniform(0.016, 0.03))
        r0 = float(rng.uniform(0.1, 0.75) * rf)
        lo, hi = fa(np.array([r0]))[0], fb(np.array([r0]))[0]
        t0 = rng.uniform(lo, hi)
        bx, by = r0 * np.cos(t0) + 1.25 * rad * np.cos(ph), r0 * np.sin(t0) + 1.25 * rad * np.sin(ph)
        br, bt = np.hypot(bx, by), np.arctan2(by, bx)
        start = fa(br)
        if ((br < 0.92 * rf) & (np.mod(bt - start, 2 * np.pi) < fb(br) - start)).all():
            return (centre[0] + r0 * np.cos(t0), centre[1] + r0 * np.sin(t0)), rad
    return None


# how many copies of a scanned category make a served portion: (lo, hi); default by size
COPIES = {"bacon": (2, 3), "pancake": (2, 3), "carrot": (2, 4), "green beans": (4, 8), "green bean": (4, 8),
          "asparagus": (3, 6), "chicken tender": (2, 3), "chicken wings": (2, 4), "sausages": (1, 3),
          "hashbrown": (1, 2), "cookies": (2, 4), "biscuits": (1, 3), "brownies": (2, 3), "sushi": (4, 6),
          "boiled egg": (1, 2), "gyoza": (3, 5), "shrimp": (3, 6), "edamame": (3, 6), "kiwi": (2, 4),
          "strawberry": (3, 6), "melons": (4, 8), "tofu": (4, 8), "tortilla chips": (5, 9), "crab cake": (1, 2),
          "french fry": (1, 2), "chicken nugget": (4, 7), "blueberries": (8, 16), "grapes": (1, 2),
          "cherry tomato": (3, 6), "tomato slice": (2, 4), "broccoli": (2, 4), "brussels sprout": (4, 7),
          "roasted potato": (5, 9), "cucumber": (3, 6), "meatball": (3, 5), "falafel": (3, 6),
          "almonds": (6, 12), "olive": (4, 8), "waffle": (1, 2), "yeast bread": (1, 2), "tortilla": (1, 2)}


COPIES["pizza"] = (1, 2)


def n_copies(rng: np.random.Generator, cat: str, ext: float, serving=None) -> int:
    """Copies of a scan in one portion: the asset QA's `serving_copies` (int or [lo, hi]) when given."""
    if serving is not None:
        lo, hi = (serving, serving) if np.isscalar(serving) else (serving[0], serving[-1])
        return max(1, int(rng.integers(int(lo), int(hi) + 1)))
    lo, hi = COPIES.get(cat) or ((6, 12) if ext < 0.035 else (3, 6) if ext < 0.065 else (1, 1))
    return int(rng.integers(lo, hi + 1))


def _pack(rng: np.random.Generator, radii: list[float], R_lim: float, gap: float = 0.004, tries: int = 300):
    """Non-overlapping circles (largest first) of the given radii inside a disc of radius R_lim;
    all radii shrink 10 % per failed attempt; returns (centres, radii)."""
    radii = list(radii)
    for _ in range(10):
        cs: list[np.ndarray] = []
        for r in radii:
            for _ in range(tries):
                d, a = max(R_lim - r, 0) * np.sqrt(rng.uniform(0.15, 1)), rng.uniform(0, 2 * np.pi)
                p = np.array([d * np.cos(a), d * np.sin(a)])
                if all(np.linalg.norm(p - q) >= r + rq + gap for q, rq in zip(cs, radii)):
                    cs.append(p)
                    break
            else:
                break
        if len(cs) == len(radii):
            return cs, radii
        radii = [0.9 * r for r in radii]
    return [np.zeros(2)] * len(radii), radii


MIN_FILL = 0.35          # projected footprint of the food as a share of the dish well
POUR = ("french fry", "tortilla chips")     # long-ish pieces that are still poured in a heap
PILES_TOTAL = 18         # pile pieces in a whole scene (on top of the max_items budget)
PILE_MAX = 14            # pieces in one same-category pile (labels stay far below 252..255)
# measurability (synth.manifest: >= 50 fused points at 24 views, 1.5 mm voxels): an item needs a visible
# surface of a few cm^2, so tiny items (blueberry, chickpea, almond) are scaled up a little (<= x1.35) and
# pieces that stay below MIN_FOOT_M2 are not served; pile pieces are small (a pile of 8 cm cucumbers buries the plate)
MIN_FOOT_M2 = 3.5e-4
MIN_FOOT_FLOOR_M2 = 2.5e-4
MAX_UPSCALE = 1.35
PILE_PIECE_MAX_M = 0.055
CROWD_COVER = (1.7, 1.2)   # share of the well the scans' footprints may add up to: (no soft food, soft food)
CROWD_N = 20                # pieces (scans + piles + foods) on a plate with a 0.085 m well, scaled with the well area


def sample_scene_hard(rng: np.random.Generator, assets: list[dict], out: Path, scene_id: str,
                      n_views: int = 24, res_scale: float = 1.0, max_items: int = 14,
                      hdris: list[str] | None = None, surfaces: list[dict] | None = None,
                      n5k: dict | None = None) -> dict:
    """One hard scene spec. A plate whose food covers less than MIN_FILL of the dish (after the
    in-scene top-up of extra scans / piles) is redrawn, up to 3 times (`redraws` in the meal log)."""
    for attempt in range(4):
        spec = _sample_scene_hard(rng, assets, out, scene_id, n_views, res_scale, max_items, hdris, surfaces, n5k)
        if spec["fill_fraction"] >= MIN_FILL or attempt == 3:
            break
    spec["meal"]["redraws"] = attempt
    (out / "spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))
    return spec


def _sample_scene_hard(rng: np.random.Generator, assets: list[dict], out: Path, scene_id: str,
                       n_views: int = 24, res_scale: float = 1.0, max_items: int = 14,
                       hdris: list[str] | None = None, surfaces: list[dict] | None = None,
                       n5k: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    # what is served (synth.meals): a cuisine template, a real Nutrition5k dish, or random
    meal = sample_meal(rng, n5k)
    style = "cafeteria" if rng.random() < 0.25 else "home"       # Nutrition5k-like: white plate on a tan board under ceiling light
    want = meal["dish"]
    if want == "any":
        want = "bowl" if any(f in BOWL_ONLY for f in meal["heap"] or []) else ("plate" if rng.random() < 0.75 else "bowl")
    c = sample_container(rng, want, p_square=0.55 if style == "cafeteria" else 0.25)
    V, F = c.mesh()
    np.savez_compressed(out / "container.npz", V=V.astype(np.float32), F=F.astype(np.int32))
    dcol, dfam = dish_colour(rng, style)
    gl = glaze(rng, dfam, style)
    spec: dict = {"scene_id": scene_id, "preset": "hard", "style": style, "container": {
        **c.to_json(), "mesh": "container.npz", "capacity_ml": c.capacity_ml(),
        "color": dcol, "roughness": gl["base_rough"], "glaze": gl}}
    items: list[dict] = []

    # 0. sometimes a real scanned dish (GSO / YCB). Heaps are built to fit the procedural
    # dish's analytic surface, so real-dish scenes hold only physics-dropped items; `c`
    # becomes an approximate stand-in used for placement radii and the off-dish check.
    dishes = [a for a in assets if a.get("role") == "container" and _dish_kind(a) == c.kind
              and not any(g_ in a["id"].lower() for g_ in GARISH_DISHES)]
    # a meal with a heap needs the procedural dish (heaps follow its analytic surface)
    p_real = 0.2 if meal["source"] == "random" else (0.0 if meal["heap"] else 0.5)
    if style == "cafeteria":
        p_real = 0.0
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

    # 0b. what is on offer. Scans are planned as groups of copies (a served portion: a handful
    # of berries, 3 strips of bacon, a few nuggets), so a plate is filled rather than dotted with
    # one item; the dish is then sized to the meal and every requested item is placed (never
    # dropped): big scans are scaled to fit, a full plate gets items resting on one another.
    fruit_meal = meal.get("cuisine") == "fruit"

    def servable(a: dict) -> bool:
        """Not a QA reject; no whole raw produce on a plate unless it is a fruit plate (assets_v4 has
        `serve`; v3 falls back to the meals.WHOLE_RAW hand list by category and extent)."""
        if a.get("qa_exclude"):
            return False
        if a.get("volume_source") == "voxel_column" or (a.get("volume_source") == "voxel_fill" and a["volume_ml"] < 6):
            return False                                  # non-watertight scan: spec volume != simulated volume
        sv = a.get("serve")
        if sv:
            return sv != "whole_produce" or fruit_meal
        cat = str(a["category"]).lower().replace("_", " ")
        if cat not in WHOLE_RAW or a["procedural"]:
            return True
        if fruit_meal and cat in FRUIT_PLATE_OK:
            return True
        return max(a["extent_m"]) < WHOLE_RAW[cat]

    pool = [a for a in assets if a.get("role", "food") == "food" and a.get("volume_ok", a["watertight"])
            and not a.get("container_shaped") and servable(a)]
    real = [a for a in pool if not a["procedural"]]
    by_cat: dict[str, list[dict]] = {}
    for a in real:
        by_cat.setdefault(a["category"], []).append(a)
    meal_cat: dict[str, list[dict]] = {}                                 # meal names -> any food asset (scans and
    for a_ in pool:                                                      # primitives: chickpea, olive, cheese cube)
        meal_cat.setdefault(a_["category"].lower().replace("_", " "), []).append(a_)

    def resolve_cat(name: str) -> list[dict] | None:
        k = str(name).lower().replace("_", " ")
        if k in meal_cat:
            return meal_cat[k]
        near = [a_ for kk, v in meal_cat.items() if k in kk or kk in k for a_ in v]
        return near or None

    foods: list = []
    if meal["source"] == "random":
        if rng.random() < 0.95:
            foods = _similar(rng, int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])), c.kind == "bowl")
    else:
        foods = [FILL_BY_NAME[n] for n in meal["heap"] if n in FILL_BY_NAME
                 and (c.kind == "bowl" or n not in BOWL_ONLY)][:3]
    if real_dish is not None:
        foods = []
    if meal["source"] == "random":
        greens_list = [GREENS[rng.integers(len(GREENS))]] if rng.random() < 0.35 else []
    else:                                           # every leafy ingredient of the meal (n5k: up to 3), else the one
        names = list(dict.fromkeys(meal.get("greens_all") or ([meal["greens"]] if meal["greens"] else [])))
        greens_list = [next((g for g in GREENS if g[0] == n_), GREENS[3]) for n_ in names]   # [3] mixed greens
        greens_list = list({g[0]: g for g in greens_list}.values())
    greens = greens_list[0] if greens_list else None
    gm = meal.get("grams") or {}

    groups: list[dict] = []
    requested: list[str] = []
    unresolved: list[str] = []
    unfulfilled: list[dict] = []                     # requested by the meal, not in the scene: {name, reason}

    def add_group(name: str, copies: int | None = None, assets_: list[dict] | None = None, **extra) -> None:
        requested.append(str(name))
        grp = assets_ or resolve_cat(name)
        if not grp:
            unresolved.append(str(name))
            unfulfilled.append({"name": str(name), "reason": "no_asset"})
            return
        key = str(name).lower().replace("_", " ")
        ext = float(np.median([max(a_["extent_m"]) for a_ in grp]))
        pile = copies is None and (ext < 0.065 or key in POUR)
        if pile and key not in POUR:                   # a pile is poured from small pieces
            small_ = [a_ for a_ in grp if max(a_["extent_m"]) <= PILE_PIECE_MAX_M]
            if small_:
                grp = small_
                ext = float(np.median([max(a_["extent_m"]) for a_ in grp]))
        if copies is None and "scale" not in extra:    # tiny items: enough visible surface to be measured
            foot_ = float(np.median([np.prod(sorted(a_["extent_m"])[-2:]) for a_ in grp]))
            up_ = float(np.clip(np.sqrt(MIN_FOOT_M2 / max(foot_, 1e-9)), 1.0, MAX_UPSCALE))
            grp = [a_ for a_ in grp if np.prod(sorted(a_["extent_m"])[-2:]) * up_ ** 2 >= MIN_FOOT_FLOOR_M2]
            if not grp:
                unfulfilled.append({"name": str(name), "reason": "too_small"})
                return
            if up_ > 1.0:
                extra = {**extra, "scale": up_}
        n = copies if copies is not None else n_copies(rng, key, ext, grp[0].get("serving_copies"))
        grams_ = (gm.get("scans") or {}).get(key) if copies is None else None
        masses = [a_["mass_g"] for a_ in grp if a_.get("mass_g")]
        if grams_ and masses:                          # Nutrition5k grams -> number of pieces
            n = int(np.clip(round(grams_ / float(np.median(masses))), 1, PILE_MAX if pile else 8))
        elif pile and n > 1 and rng.random() < 0.5:    # piles run from a few pieces to a heap of 25
            n = int(np.clip(round(np.exp(rng.uniform(np.log(2), np.log(PILE_MAX)))), 2, PILE_MAX))
        n = min(n, PILE_MAX if pile else 8)
        groups.append({"cat": str(name), "assets": [grp[j] for j in rng.integers(len(grp), size=n)], "ext": ext,
                       "pile": pile and n > 1, **extra})

    # scanned portions in place of procedural mounds: real rice / mash / mac / pasta / stew / omelet
    # scans are rigid but photographic; with p = SCAN_P a soft food of the meal is served as one
    portions: list[dict] = []
    if foods and real_dish is None and meal["source"] != "random":
        keep = []
        for f_ in foods:
            cat_ = SCAN_FOR_FILL.get(f_[0])
            grp_ = meal_cat.get(cat_) if cat_ else None
            if grp_ and rng.random() < SCAN_P:
                a_ = grp_[rng.integers(len(grp_))]
                lo_, hi_ = PORTION_SCALE.get(cat_, (0.85, 1.1))
                portions.append({"fill": f_[0], "category": a_["category"], "asset": a_["id"],
                                 "scale": float(rng.uniform(lo_, hi_))})
                add_group(cat_, 1, [a_], scale=portions[-1]["scale"], portion=f_[0])
            else:
                keep.append(f_)
        foods = keep
        if portions and not foods:
            meal["layout"] = None

    small = sorted(k for k, v in by_cat.items() if np.median([max(a["extent_m"]) for a in v]) < 0.07)
    if meal["source"] == "random":
        if small and rng.random() < 0.5:
            add_group(small[rng.integers(len(small))])
        n_single = int(rng.integers(1 if foods else 2, 4))
        for _ in range(n_single):
            src = real if real and rng.random() < 0.8 else ([a_ for a_ in pool if a_["procedural"]] or pool)
            a_ = src[rng.integers(len(src))]
            add_group(a_["category"], 1 if max(a_["extent_m"]) > 0.06 else None, [a_])
    else:
        if meal["pile"]:
            add_group(meal["pile"])
        for k_ in meal["scans"]:
            add_group(k_)
    if not groups and not foods and greens is None:        # nothing of the meal exists: a few random items
        for _ in range(2):
            a_ = (real or pool)[rng.integers(len(real or pool))]
            add_group(a_["category"], 1 if max(a_["extent_m"]) > 0.06 else None, [a_])
    reserve = 2 * len(foods) + len(greens_list)             # soft pieces + toppings + leaves
    budget = max(max_items - reserve, 3)
    trimmed = 0
    singles = lambda: [g for g in groups if not g["pile"]]  # same-category piles have their own (PILE_MAX) budget
    while sum(len(g["assets"]) for g in singles()) > budget:
        big = max(singles(), key=lambda g: len(g["assets"]))
        if len(big["assets"]) > 1:
            big["assets"].pop()
            trimmed += 1
        else:                                               # drop the smallest (garnish), never the main item
            gone = min(singles(), key=lambda g: g["ext"])
            groups.remove(gone)
            unfulfilled.append({"name": gone["cat"], "reason": "budget"})
            trimmed += 1
    piles_ = lambda: [g for g in groups if g["pile"]]
    while sum(len(g["assets"]) for g in piles_()) > PILES_TOTAL:       # keeps the physics drop affordable
        big = max(piles_(), key=lambda g: len(g["assets"]))
        if len(big["assets"]) <= 2:
            break
        big["assets"].pop()
        trimmed += 1
    foot = lambda a_: 0.7 * float(np.prod(sorted(a_["extent_m"])[-2:]))     # resting footprint, m^2
    # footprint of the scans on the dish; a small-item pile is poured in a heap, it covers ~half its items' sum
    A_s = sum(foot(a_) * g.get("scale", 1.0) ** 2 * (0.5 if g["pile"] else 1.0)
              for g in groups for a_ in g["assets"])

    # size a procedural plate to the meal: a snack gets a smaller plate, a big spread a larger one
    n_foods = len(foods)
    if real_dish is None and c.kind == "plate":
        W = c.well_r
        k = 1.0
        if n_foods == 0 and greens is None and A_s > 0:
            k = float(np.sqrt(A_s / (np.pi * rng.uniform(0.6, 0.95))) / W)     # real plates are 50-85 % covered (scan footprints overestimate irregular items)
        elif A_s > 0.35 * np.pi * W * W:
            k = float(np.sqrt(A_s / (0.35 * np.pi)) / W)
        if gm.get("heap") or gm.get("scans") or gm.get("toppings"):
            # Nutrition5k grams: the dish scales with the volume actually served
            v_ml = sum(g_ / density_for(n_, "piece") for n_, g_ in (gm.get("heap") or {}).items()) \
                + sum(g_ / density_for(n_, "topping") for n_, g_ in (gm.get("toppings") or {}).items()) \
                + sum(g_ / density_for(n_, "leaves") for n_, g_ in (gm.get("greens") or {}).items()) \
                + sum(a_["volume_ml"] * g.get("scale", 1.0) ** 3 for g in groups for a_ in g["assets"])
            k = max(k, float(np.clip((v_ml / 350.0) ** 0.4, 0.75, 1.45)))
        k = float(np.clip(c.R * np.clip(k, 0.6, 1.45), 0.08, 0.14) / c.R)
        if abs(k - 1) > 0.02:
            c = replace(c, R=c.R * k, well_r=c.well_r * k, foot_r=c.foot_r * k)
            V, F = c.mesh()
            np.savez_compressed(out / "container.npz", V=V.astype(np.float32), F=F.astype(np.int32))
            spec["container"].update(c.to_json())
            spec["container"]["capacity_ml"] = c.capacity_ml()

    # crowding: more pieces than the dish can show bury each other (and the small ones become unmeasurable).
    # Trim copies from the biggest multi-piece group (then the smallest garnish) until the scans' footprints
    # and the piece count fit the dish; fuller plates stay: the caps are those of a well-filled plate.
    if real_dish is None:
        r_well = c.well_r if c.kind == "plate" else c.r_at_height(c.floor + 0.5 * (c.H - c.floor))
        area_w = np.pi * r_well ** 2
        cap_cov = CROWD_COVER[1 if (foods or greens_list) else 0] * area_w
        cap_n = int(np.clip(round(CROWD_N * area_w / (np.pi * 0.085 ** 2)), 8, 1.2 * CROWD_N))
        res_n = 2 * len(foods) + len(greens_list)
        cover = lambda: sum(foot(a_) * g.get("scale", 1.0) ** 2 * (0.4 if g["pile"] else 1.0)
                            for g in groups for a_ in g["assets"])
        while groups and (cover() > cap_cov or sum(len(g["assets"]) for g in groups) + res_n > cap_n):
            multi = [g for g in groups if len(g["assets"]) > (3 if g["pile"] else 2) and "portion" not in g]
            if multi:
                max(multi, key=lambda g: len(g["assets"]) * g["ext"]).get("assets").pop()
            elif len(groups) > 1:
                gone = min(groups, key=lambda g: g["ext"] * g.get("scale", 1.0))
                groups.remove(gone)
                unfulfilled.append({"name": gone["cat"], "reason": "crowded"})
            else:
                break
            trimmed += 1
        for g in groups:
            g["pile"] = g["pile"] and len(g["assets"]) > 1
        A_s = cover()

    heap: list = []                          # (item id, Mound, wedge start, wedge end)
    topping_sites: list[tuple[np.ndarray, float]] = []      # (xy, radius) of toppings: nothing is dropped on them

    def heap_fractions() -> np.ndarray:
        """Volume share of each soft food: Nutrition5k grams / bulk density when every food has grams
        (a little jitter), else a random split."""
        n_ = len(foods)
        gv = [(gm.get("heap") or {}).get(f_[0]) for f_ in foods]
        if n_ > 1 and all(gv):
            v = np.array([g_ / density_for(f_[0], "piece") for g_, f_ in zip(gv, foods)])
            v = v * rng.lognormal(0, 0.08, n_)
            return v / v.sum()
        return rng.dirichlet(np.ones(n_) * 4)

    # 1. soft foods. Bowl: one heap split into touching wedges (re-partitioned by synth.mpm). Plate: either
    # that, or (usually) each food its own mound at its own place on the dish, like a served plate
    n_pieces = len(foods)
    regions = False
    if foods:
        if c.kind == "bowl":
            lev = c.floor + rng.uniform(0.5, 0.85) * (c.H - c.floor)
            rf, centre = c.r_at_height(lev), (0.0, 0.0)
        else:
            lev = None
            regions = rng.random() < (0.5 if meal.get("layout") else 0.8)
            if not regions:
                rf = c.well_r * rng.uniform(0.7, 0.95)        # a served portion fills the well
                t = rng.uniform(0, 2 * np.pi)
                off = rng.uniform(0, c.well_r - rf)
                centre = (float(off * np.cos(t)), float(off * np.sin(t)))
        if regions:
            W = c.well_r
            scan_cov = A_s / (np.pi * W * W)
            total = max(0.3, rng.uniform(0.6, 0.9) - 0.85 * scan_cov)   # fraction of the well the soft foods cover
            fr = heap_fractions() if n_pieces > 1 else np.ones(1)
            radii = [float(np.clip(W * np.sqrt(total * f), 0.025, 0.8 * W)) for f in fr]
            order = np.argsort(radii)[::-1]
            cents, radii_o = _pack(rng, [radii[i] for i in order], 1.02 * W)
            rfs, centres = [0.0] * n_pieces, [None] * n_pieces
            for j, i in enumerate(order):
                rfs[i], centres[i] = radii_o[j], (float(cents[j][0]), float(cents[j][1]))
            sets = [(rfs[k], centres[k], *split_angles(rng, 1, rfs[k])[:2], None) for k in range(n_pieces)]
        else:
            fns = split_angles(rng, n_pieces, rf)
            sets = [(rf, centre, fns[k], fns[k + 1], True) for k in range(n_pieces)]
        # the served shape (synth.mpm then lets it flow / slump / heap by consistency): liquids
        # level out, purees mound, grains heap; near a seam both sides ease to the heap's mean
        # profile, so a dome of rice meets a pool of curry, not a cliff
        shapes = [shape(name, c.kind, sets[k][0], rng) for k, (name, _, _) in enumerate(foods)]
        shared = (float(np.mean([s["mound"] for s in shapes])), float(np.mean([s["p"] for s in shapes])))
        for k, ((name, col, rough), sh) in enumerate(zip(foods, shapes)):
            rf_k, ctr_k, fa_k, fb_k, joined = sets[k]
            mnd = Mound(rf=rf_k, level=lev, centre=ctr_k, seed=int(rng.integers(1 << 31)), mound=sh["mound"],
                        p=sh["p"], wobble=sh["wobble"], relief=sh["relief"][0], relief_scale=sh["relief"][1],
                        shared=shared if (joined and n_pieces > 1) else None, edges=(fa_k, fb_k),
                        blend=float(rng.uniform(0.01, 0.02)))
            items.append(_procedural(name, f"piece_{k}.npz", *mnd.piece(c, fa_k, fb_k), out, col, rough,
                                     "procedural_piece", {"kind": "piece", "n_pieces": 1 if regions else n_pieces,
                                                          "rf": rf_k, "level": lev, "mound": mnd.mound,
                                                          "centre": ctr_k}, rng))
            heap.append((len(items), mnd, fa_k, fb_k))
        if n_pieces > 1 and not regions:
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
                            "fractions": heap_fractions().tolist(),
                            "top_thick": float(rng.uniform(0.005, 0.012)), "top_r": float(rng.uniform(0.45, 0.85) * rf)}

        # 2. toppings lying on a piece
        for item_id, mnd, fa, fb in heap:
            menu = TOPPINGS if meal["source"] == "random" else [t for t in TOPPINGS if t[0] in meal["toppings"]]
            p_top = 0.35 if meal["source"] == "random" else 0.55
            for j_ in range(2 if (menu and rng.random() < 0.25) else 1):      # a second blob on some wedges
                site = _topping_site(rng, mnd.rf, mnd.centre, fa, fb) if menu and rng.random() < p_top else None
                if site is None:
                    continue
                name, col, rough = menu[rng.integers(len(menu))]
                thick_t = float(rng.uniform(0.007, 0.012))                   # >= 3 particles deep, thick enough to be seen / measured
                seed_t = int(rng.integers(1 << 31))
                Vt, Ft = topping_mesh(c, mnd, site[0], site[1], thick_t, seed=seed_t)
                # the blob is re-laid on the *settled* surface of the food below by synth.mpm.settle (it is not simulated:
                # as a particle blob it splashed off the heap and floated); the parameters are kept for that
                items.append(_procedural(name, f"topping_{item_id}_{j_}.npz", Vt, Ft, out, col, rough, "procedural_topping",
                                         {"kind": "topping", "on_item": item_id, "radius": site[1], "thick": thick_t,
                                          "seed": seed_t, "centre": [float(site[0][0]), float(site[0][1])]}, rng))
                topping_sites.append((np.asarray(site[0], float), site[1]))

    fit_r = c.well_r if c.kind == "plate" else c.r_at_height(c.floor + 0.5 * (c.H - c.floor))
    mounds = [m for _, m, _, _ in heap]
    placed: list[tuple[np.ndarray, float, float]] = []     # (xy, radius, top z) of everything dropped so far

    def support_z(xy: np.ndarray, r_xy: float = 0.0) -> float:
        """Height to drop something at xy from: the dish / heap surface or an item already there."""
        if real_dish:                         # surface unknown analytically: drop from above the rim
            z = c.H + 0.003
        else:
            z = float(c.z_in(np.hypot(*xy)))
            for m in mounds:
                z = max(z, float(m.top(c, xy[0], xy[1])))
                if np.hypot(xy[0] - m.centre[0], xy[1] - m.centre[1]) < m.rf:
                    z = max(z, float(m.top(c, xy[0], xy[1])) + 0.009)     # clear any topping
        for p, r, zt in placed:
            if np.linalg.norm(xy - p) < r + 0.6 * r_xy:
                z = max(z, zt)
        return z

    def clearance(xy: np.ndarray, r_xy: float) -> float:
        return min([np.linalg.norm(xy - p) - r - r_xy for p, r, _ in placed]
                   + [np.linalg.norm(xy - np.asarray(m.centre)) - m.rf - 0.25 * r_xy for m in mounds] + [1.0])

    def free_spot(r_xy: float, lim: float, tries: int = 60, compact: bool = False, tol: float = 0.25,
                  accept_best: float | None = None) -> np.ndarray | None:
        """A spot on the dish clear of the heaps and of what is placed (items may overlap neighbours by
        up to tol x r_xy and lean on them). compact: the spot that touches what is already there and
        stays near the middle, so a meal is one served cluster, not items scattered over an empty plate.
        When nothing is that free the roomiest spot found is returned if its overlap is within
        accept_best x r_xy (None: always), else None."""
        best = (-9.0, None)
        feas = []
        if compact:
            pts = [p for p, _, _ in placed] + [np.asarray(m.centre) for m in mounds]
            anchor = np.mean(pts, axis=0) if pts else np.zeros(2)
            cands = []
            for _ in range(100):
                rr = max(lim - r_xy, 0) * np.sqrt(rng.random())
                th = rng.uniform(0, 2 * np.pi)
                xy = np.array([rr * np.cos(th), rr * np.sin(th)])
                cl = clearance(xy, r_xy)
                if cl >= -(tol + 0.05) * r_xy:              # may lean on a neighbour, never bury it
                    cands.append((max(cl, 0) + 0.25 * np.linalg.norm(xy - anchor) + rng.uniform(0, 0.01), xy))
            if cands:
                cands.sort(key=lambda t: t[0])
                return cands[rng.integers(min(3, len(cands)))][1]
        for _ in range(tries):
            rr = max(lim - r_xy, 0) * np.sqrt(rng.random())
            th = rng.uniform(0, 2 * np.pi)
            xy = np.array([rr * np.cos(th), rr * np.sin(th)])
            cl = clearance(xy, r_xy)
            if cl >= -tol * r_xy:
                feas.append(xy)
            if cl > best[0]:
                best = (cl, xy)
        if feas:
            return feas[rng.integers(len(feas))]
        if best[1] is not None and (accept_best is None or best[0] >= -accept_best * r_xy):
            return best[1]
        return None

    def spot(r_xy: float, lim: float, compact: bool = False) -> np.ndarray | None:
        """free_spot with retries: a smaller footprint and more overlap with what is placed (the
        physics drop resolves it: items settle on one another) before giving up (None)."""
        for shrink, tol, acc in ((1.0, 0.25, 0.8), (0.8, 0.45, 1.5), (0.65, 0.8, 6.0)):
            xy = free_spot(r_xy * shrink, lim, compact=compact, tol=tol, accept_best=acc)
            if xy is not None:
                return xy
        return None

    def heap_spot() -> np.ndarray:
        w = np.array([m.rf ** 2 for m in mounds])
        m = mounds[rng.choice(len(mounds), p=w / w.sum())]
        for _ in range(20):
            a, rr = rng.uniform(0, 2 * np.pi), 0.5 * m.rf * np.sqrt(rng.random())
            xy = np.asarray(m.centre) + rr * np.array([np.cos(a), np.sin(a)])
            if all(np.linalg.norm(xy - t_) > r_t + 0.03 for t_, r_t in topping_sites):
                break
        return xy

    occ: list[tuple[np.ndarray, float]] = []         # (xy, equal-area radius) of every food footprint on the dish

    def fill_frac() -> float:
        """Share of the dish well (disc fit_r) covered by the food seen from above: heaps, leaves,
        and every dropped piece as a disc of its resting-footprint area."""
        gx = np.linspace(-fit_r, fit_r, 101)
        GX, GY = np.meshgrid(gx, gx)
        disc = np.hypot(GX, GY) < fit_r
        cov = np.zeros_like(disc)
        for m_ in mounds:
            cov |= np.hypot(GX - m_.centre[0], GY - m_.centre[1]) < m_.rf
        for xy_, r_ in occ:
            cov |= np.hypot(GX - xy_[0], GY - xy_[1]) < r_
        return float((cov & disc).sum() / disc.sum())

    # 2b. leafy greens: thin leaves dropped as cloth (render.py), one item per portion (up to 3 kinds,
    # as in a mixed Nutrition5k dish); beside the heaps or heaped over one of them
    for gi, (name, col, rough) in enumerate(greens_list):
        if len(items) >= max_items + 2:
            break
        big_heap = bool(mounds) and max(m.rf for m in mounds) >= 0.05
        salad_spot = heap_spot() if (big_heap and rng.random() < 0.3) else spot(0.05, min(0.6 * fit_r, 0.35 * fit_r + 0.05))
        if salad_spot is None:
            unfulfilled.append({"name": name, "reason": "no_fit"})
            continue
        ring = [salad_spot] + [salad_spot + 0.03 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        top = max(support_z(p) for p in ring) + 0.003
        n_leaves = int(np.clip(round(rng.integers(8, 19) * min(1.0, (fit_r / 0.11) ** 2)), 6, 18))   # fewer leaves on a small dish: they slide off
        Vl, Fl = salad_sheets(rng, n_leaves, salad_spot, top + 0.01, 0.012,   # tight column: stays on
                              "arugula" if name == "arugula" else "mixed")
        mesh_name = "salad.npz" if gi == 0 else f"salad_{gi}.npz"
        np.savez_compressed(out / mesh_name, V=Vl.astype(np.float32), F=Fl.astype(np.int32))
        thick = 0.0006
        a, b, cc = Vl[Fl[:, 0]], Vl[Fl[:, 1]], Vl[Fl[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1).sum()
        items.append({"name": name, "category": name, "source": "procedural_leaves", "asset_id": None,
                      "mesh": mesh_name, "fmt": "npz", "to_m": 1.0, "scale": 1.0,
                      "volume_ml": float(area * thick * 1e6),     # replaced by the simulated solid's volume
                      "mass_g": None, "mass_source": "density", "density_g_ml": density_for(name, "leaves"),
                      "dynamic": False, "fill": None, "color": list(col), "roughness": rough,
                      "procedural": True, "generated": False, "texture": True,
                      "cloth": {"thickness_m": thick, "bending": float(rng.uniform(0.1, 1.0)), "grid": [12, 5]},
                      "shape": {"kind": "leaves", "n_leaves": n_leaves}})
        placed.append((np.asarray(salad_spot), 0.05, top + 0.02))
        occ.append((np.asarray(salad_spot), 0.045))

    # 3. discrete items dropped onto whatever is there: each group is a cluster of copies; they
    # start at their own spots (a high drop bounces off the dish) and pile / lean on what is there
    def add(a: dict, xy: np.ndarray, z: float, kind: str, s: float, extra: dict | None = None) -> None:
        if a.get("mass_g"):                           # a scan's mass scales with its volume
            mass_g, msrc = float(a["mass_g"]) * s ** 3, "asset"
        else:                                         # primitives / unweighed scans: volume x bulk density
            mass_g, msrc = float(a["volume_ml"] * s ** 3 * density_for(a["category"], "piece")), "density"
        items.append({"name": a["name"], "category": a["category"], "source": a["source"], "asset_id": a["id"],
                      "mesh": a["path"], "fmt": a["fmt"], "to_m": a["to_m"], "scale": s,
                      "volume_ml": a["volume_ml"] * s ** 3, "mass_g": mass_g, "mass_source": msrc, "dynamic": True, "fill": None,
                      "init_location": [float(xy[0]), float(xy[1]), float(z)],
                      # pile pieces are poured: any orientation, so they tumble and interlock;
                      # singles are set down roughly as served (scans rest on their scanned base)
                      "init_euler": (rng.uniform(0, 2 * np.pi, 3) if a["procedural"] or kind == "pile" else
                                     np.r_[rng.normal(0, 0.3, 2), rng.uniform(0, 2 * np.pi)]).tolist(),
                      "up_axis": a.get("up_axis", "Y"), "color": a.get("color"),
                      "roughness": float(rng.uniform(0.3, 0.8)), "procedural": a["procedural"],
                      "generated": a.get("generated", False), "texture": bool(a["procedural"]),
                      "damping": 0.4 if kind == "pile" else 0.7, "shape": {"kind": kind, **(extra or {})}})
        foot_a = 0.7 * float(np.prod(sorted(a["extent_m"])[-2:])) * s ** 2         # resting footprint, m^2
        occ.append((np.asarray(xy[:2], float), float(np.sqrt(foot_a / np.pi)) * (0.8 if kind == "pile" else 1.0)))

    lim_ext = (1.7 if c.kind == "plate" else 2.0) * fit_r         # largest item footprint that sits on the dish
    rad_of = lambda a, s: s * float(np.linalg.norm(a["extent_m"])) / 2       # any rotation fits

    def place_group(g: dict) -> None:
        ext_max = max(max(a_["extent_m"]) for a_ in g["assets"])
        s = g.get("scale", 1.0)
        if ext_max * s > lim_ext:
            s = float(np.clip(lim_ext / ext_max, 0.5, s))
        n = len(g["assets"])
        extra = {"portion": g["portion"]} if g.get("portion") else None      # a scan served instead of a mound
        r_i = float(np.median([0.5 * max(a_["extent_m"][:2]) for a_ in g["assets"]])) * s
        lim = 0.88 * fit_r
        if g["pile"]:                                   # a small-item pile: poured in one place, packed
            # a pile is at most 2 layers deep (a taller heap buries its middle) and sits in free space: when the
            # dish has no room for all pieces the pile is poured smaller, not on top of what is placed
            cl_r = lambda n_: min(r_i * np.sqrt(max(n_ / 2.0, 1.0)) * 1.45, 0.5 * fit_r)
            ctr, n_try = None, n
            sizes = sorted({n, max(int(0.75 * n), 2), max(int(0.5 * n), 2), min(n, 3), 2}, reverse=True)
            tiers = [(0.2, 0.6), (0.2, 0.6), (0.45, 1.5), (0.8, 6.0), (0.8, 6.0)]
            for k_, n_try in enumerate(sizes):
                r_cl = cl_r(n_try)
                on_heap = bool(mounds) and rng.random() < 0.15 and r_cl < 0.5 * max(m.rf for m in mounds)
                if on_heap:
                    ctr = heap_spot()
                else:
                    tol_, acc_ = tiers[min(k_ + (len(tiers) - len(sizes)), len(tiers) - 1)]
                    ctr = free_spot(r_cl, lim, compact=True, tol=tol_, accept_best=acc_)
                if ctr is not None:
                    break
            if ctr is None:
                unfulfilled.append({"name": g["cat"], "reason": "no_fit"})
                return
            if n_try < n:
                del g["assets"][n_try:]
                n = n_try
            base = support_z(ctr, r_cl)
            per = max(2, int(np.ceil(n / 2)))
            for i, a_ in enumerate(g["assets"]):
                rad = rad_of(a_, s)
                add(a_, ctr + rng.normal(0, r_cl / 2.2, 2), base + rad + (i // per) * 1.6 * rad + 0.002, "pile", s)
            placed.append((ctr, r_cl, base + 2.2 * r_i * (1 + n // per)))
        else:                                           # larger items: side by side, overlapping a little
            on_heap = bool(mounds) and rng.random() < 0.2 and r_i < 0.45 * max(m.rf for m in mounds)
            prev = None
            for a_ in g["assets"]:
                r_a = 0.5 * max(a_["extent_m"][:2]) * s
                if prev is None:
                    xy = heap_spot() if on_heap else spot(r_a, lim, compact=True)
                    if xy is None:
                        unfulfilled.append({"name": g["cat"], "reason": "no_fit"})
                        return
                else:
                    ang = rng.uniform(0, 2 * np.pi)
                    xy = prev[0] + (prev[1] + r_a) * rng.uniform(0.45, 0.9) * np.array([np.cos(ang), np.sin(ang)])
                    if np.hypot(*xy) > lim - 0.5 * r_a:                     # keep it on the dish
                        xy = xy * (lim - 0.5 * r_a) / np.hypot(*xy)
                rad = rad_of(a_, s)
                z = support_z(xy, r_a) + rad + 0.002
                add(a_, xy, z, "single", s, extra)
                placed.append((xy, r_a, z + 0.8 * rad))
                prev = (xy, r_a)

    # requested scans first, biggest item first, then the small-item piles / garnish
    key_ = lambda g: -g["ext"] * len(g["assets"]) ** 0.5
    for g in sorted([g for g in groups if not g["pile"]], key=key_) + sorted([g for g in groups if g["pile"]], key=key_):
        place_group(g)
    # minimum fill: a sparse dish gets another scan / pile from the meal (templates: its own slots, else a side)
    topups = 0
    while fill_frac() < MIN_FILL and topups < 3 and meal["source"] != "random" and real_dish is None:
        topups += 1
        pool_ = [k_ for k_ in list(meal["scans"]) + ([meal["pile"]] if meal["pile"] else []) + SIDES
                 if resolve_cat(k_)]
        if not pool_:
            break
        n0 = len(groups)
        add_group(pool_[rng.integers(len(pool_))])
        if len(groups) > n0:
            place_group(groups[-1])
    # toppings of a meal with no soft food (a sauce over scans): a thin puddle on the dish, laid down in
    # the free space left by the scans (one dropped on it would bury it: v12 pesto was invisible)
    puddle = None
    if not mounds and real_dish is None and meal["source"] != "random" and meal.get("toppings"):
        menu_ = [t_ for t_ in TOPPINGS if t_[0] in meal["toppings"]]
        if menu_:
            name, col, rough = menu_[rng.integers(len(menu_))]
            rad0 = min(float(rng.uniform(0.03, 0.045)), 0.3 * fit_r)
            lim_p = 0.8 * fit_r
            xy = None
            for rad in (rad0, 0.8 * rad0, max(0.65 * rad0, 0.022)):
                for _ in range(60):
                    xy_ = rng.uniform(-1, 1, 2) * lim_p
                    if np.hypot(*xy_) <= lim_p - rad and clearance(xy_, 1.1 * rad) >= 0.0:
                        xy = xy_
                        break
                if xy is not None:
                    break
            if xy is None:
                unfulfilled.append({"name": name, "reason": "no_fit"})
            else:
                class _Dish:                          # topping_mesh lays the sheet on the dish surface
                    def top(self, c_, x_, y_):
                        return c_.z_in(np.hypot(x_, y_))
                Vt, Ft = topping_mesh(c, _Dish(), (float(xy[0]), float(xy[1])), rad, float(rng.uniform(0.005, 0.008)),
                                      seed=int(rng.integers(1 << 31)))
                items.append(_procedural(name, "topping_dish.npz", Vt, Ft, out, col, rough, "procedural_topping",
                                         {"kind": "topping", "on_item": None, "on": "dish", "radius": rad}, rng))
                placed.append((xy, 1.0 * rad, c.floor + 0.008))
                topping_sites.append((np.asarray(xy, float), rad))
                occ.append((xy, rad))
                puddle = name

    fill_fraction = fill_frac()
    # (legacy) footprint estimate (union of discs of everything on the dish / dish well)
    gx = np.linspace(-fit_r, fit_r, 101)
    GX, GY = np.meshgrid(gx, gx)
    disc = np.hypot(GX, GY) < fit_r
    cov = np.zeros_like(disc)
    for m_ in mounds:
        cov |= np.hypot(GX - m_.centre[0], GY - m_.centre[1]) < m_.rf
    for it in items:
        if "init_location" in it:
            a_ = next((x for x in assets if x["id"] == it["asset_id"]), None)
            if a_ is not None:
                r_ = 0.5 * max(a_["extent_m"][:2]) * it["scale"] * (0.8 if it["shape"]["kind"] == "pile" else 1.0)
                cov |= np.hypot(GX - it["init_location"][0], GY - it["init_location"][1]) < r_
    spec_meal_log = {"requested": len(requested), "dropped": unresolved, "meal_unfulfilled": unfulfilled,
                     "trimmed_copies": trimmed, "topups": topups, "dish_puddle": puddle,
                     "n_groups": len(groups), "scan_portions": portions,
                     "pile_sizes": [len(g["assets"]) for g in groups if g["pile"]],
                     "est_coverage": float((cov & disc).sum() / disc.sum())}
    assert len(items) < 252, "labels 252..255 are reserved (distractor 253, container 254)"
    for i, it in enumerate(items):
        it["id"] = i + 1                      # mask label; 0 = background

    K, w, h = intrinsics(rng, res_scale)
    r_out = (c.R + c.rim_w) * (2 ** (0.5 - 1 / c.square) if c.square > 0 else 1.0)    # dish outline (corners of a square plate)
    poses, times = orbit_poses(rng, n_views, centre=np.array([0, 0, c.H * 0.5]), dish_r=r_out)
    meal.update(spec_meal_log)
    spec.update({
        "meal": meal,
        "fill_fraction": fill_fraction,
        "items": items,
        "distractors": distractors(rng, out, r_out, assets, style),
        "table": table_look(rng, surfaces, style),
        "mat": mat_look(rng, surfaces, style, r_out),
        "lighting": lighting(rng, hdris, style),
        # render scene-linear; synth.camera.develop meters and tone-maps per scan like a phone
        # (fixed curves either washed colours out (Filmic) or blew out bright scenes (Standard))
        "look": {"hdr": True},
        "camera": {"K": K.tolist(), "width": w, "height": h, "res_scale": res_scale,
                   "c2w": [p.tolist() for p in poses], "timestamps": times,
                   # a phone's small aperture: full-frame-equivalent f-number (render.py), focus on the dish
                   "dof": {"fstop": float(rng.uniform(8, 16)), "focus": [0.0, 0.0, float(c.H * 0.5)]}},
        "world_to_arkit": world_to_arkit(rng).tolist(),
    })
    # settings drawn from a stream of their own (the food above and the camera are the same as without them)
    side = np.random.default_rng([zlib.crc32(scene_id.encode()), 7])
    if style == "cafeteria" and side.random() < P_RIG:
        spec["rig"] = rig_look(side, surfaces, c, r_out)
        spec["mat"] = None
        push_outside_rig(spec["rig"], spec["distractors"], r_out)
    (out / "spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))
    return spec
