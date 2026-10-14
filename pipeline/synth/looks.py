"""How procedural foods look and behave, by consistency.

One noise texture on a smooth dome made every fill (rice, oatmeal, chili, yogurt) look like
the same sand. Real fills differ in kind:
  grain    rice, quinoa: a heap of distinct grains, crevices between them, matte;
  porridge oatmeal: flakes suspended in a wet, glossy, lumpy paste that levels out;
  puree    yogurt, mashed potato: smooth, swirled by the spoon, translucent;
  liquid   stews, chili, curry, sauces: glossy, level surface, chunks (beans, lentils) in it;
  curd     scrambled eggs, mac and cheese: soft rounded lumps.

`look(name, rng)` gives render.py's material parameters (synth.render.food_material);
`shape(...)` the heap profile (liquids level out, purees mound) and surface relief;
`physics(name)` the MPM material it is simulated with (synth.mpm).
"""
from __future__ import annotations

import numpy as np

KIND = {
    "white rice": "grain", "fried rice": "grain", "brown rice": "grain", "quinoa": "grain",
    "oatmeal": "porridge",
    "mashed potato": "puree", "yogurt": "puree", "sour cream": "puree", "melted cheese": "puree",
    "lentil stew": "liquid", "chili": "liquid", "curry sauce": "liquid", "tomato sauce": "liquid",
    "gravy": "liquid", "salsa": "liquid", "pesto": "liquid", "guacamole": "liquid",
    "mac and cheese": "curd", "scrambled eggs": "curd", "egg whites": "curd",
}

# MPM constitutive model per consistency (synth.mpm), SI units. Values are softer / less
# viscous than the real foods where the real ones would need a tiny explicit time step; what
# matters is the behaviour within ~1 s: liquids level and flow round items, oatmeal and yogurt
# slump slowly, grains keep an angle of repose and slide, mash and curds hold a dent.
PHYSICS = {
    "grain": {"model": "sand", "E": 6e4, "nu": 0.3, "rho": 850, "phi_deg": 38},
    "porridge": {"model": "fluid", "K": 4e4, "visc": 10.0, "rho": 1050},
    "liquid": {"model": "fluid", "K": 4e4, "visc": 0.8, "rho": 1050},
    "puree": {"model": "plastic", "E": 3e4, "nu": 0.35, "rho": 1050, "theta_c": 0.012, "theta_s": 0.004},
    "curd": {"model": "plastic", "E": 4e4, "nu": 0.3, "rho": 1000, "theta_c": 0.02, "theta_s": 0.007},
}
PHYSICS_BY_NAME = {   # within a kind
    # thick: a yield stress holds a spoonful's shape (as a fluid it spread into a pancake)
    "yogurt": {"model": "plastic", "E": 2e4, "nu": 0.4, "rho": 1050, "theta_c": 0.008, "theta_s": 0.003},
    "sour cream": {"model": "plastic", "E": 2e4, "nu": 0.4, "rho": 1050, "theta_c": 0.01, "theta_s": 0.004},
    "chili": {"model": "fluid", "K": 4e4, "visc": 2.5, "rho": 1100},
    "lentil stew": {"model": "fluid", "K": 4e4, "visc": 2.0, "rho": 1100},
    "guacamole": PHYSICS["puree"], "pesto": {"model": "fluid", "K": 4e4, "visc": 3.0, "rho": 1050},
}
ITEM_PHYSICS = {"model": "rigid", "rho": 1000}   # scans: rigid bodies coupled to the continuum

# Real small pieces scattered over a food's settled surface (render.scatter_pieces), so grains,
# flakes and beans have geometry and silhouette instead of painted bumps. Per food a list of
# (shape, per m^2 of upward-facing surface, size L x W x H mm, colour or None = the food's own
# material, fraction of the piece's height sunk below the surface).
PIECES = {
    "white rice": [("rice", 9e4, (5.5, 2.2, 1.9), None, 0.35)],
    "brown rice": [("rice", 9e4, (5.0, 2.3, 2.0), None, 0.35)],
    "fried rice": [("rice", 8e4, (5.5, 2.2, 1.9), None, 0.35), ("pea", 2.5e3, (7, 7, 7), (0.18, 0.42, 0.06), 0.4),
                   ("cube", 2.5e3, (5, 5, 5), (0.85, 0.32, 0.05), 0.4), ("flake", 2e3, (9, 7, 2), (0.95, 0.78, 0.2), 0.3)],
    "quinoa": [("quinoa", 3e5, (2.2, 2.2, 1.2), None, 0.3)],
    "oatmeal": [("flake", 3.5e4, (7, 5.5, 1.0), None, 0.6)],
    "chili": [("bean", 5e3, (13, 7.5, 6), (0.28, 0.05, 0.04), 0.55)],
    "lentil stew": [("lentil", 6e4, (4.5, 4.5, 1.8), (0.45, 0.28, 0.1), 0.5)],
    "mac and cheese": [("macaroni", 1.4e4, (14, 6, 6), None, 0.45)],
    "salsa": [("cube", 1.5e4, (5, 5, 5), (0.75, 0.08, 0.04), 0.5), ("cube", 6e3, (4, 4, 4), (0.9, 0.88, 0.75), 0.5)],
}


def physics(name: str) -> dict:
    return PHYSICS_BY_NAME.get(name, PHYSICS[kind_of(name)])

# chunks suspended in a liquid: (colour, cell size range m, coverage)
CHUNKS = {
    "chili": ((0.14, 0.035, 0.02), (0.006, 0.009), 0.45),       # kidney beans, mince
    "lentil stew": ((0.42, 0.28, 0.10), (0.003, 0.005), 0.6),
    "salsa": ((0.7, 0.08, 0.04), (0.003, 0.006), 0.5),          # diced tomato / onion
    "pesto": ((0.12, 0.22, 0.04), (0.0015, 0.003), 0.4),
    "guacamole": ((0.30, 0.42, 0.10), (0.003, 0.006), 0.4),
    "curry sauce": ((0.85, 0.6, 0.2), (0.008, 0.012), 0.25),    # potato / chicken pieces
}
# coloured bits mixed into grains: (colour, fraction of grains)
SPECKLES = {
    # fried rice: peas / carrot / egg are real pieces now (PIECES), not painted cells
    "quinoa": [((0.35, 0.08, 0.05), 0.08), ((0.12, 0.08, 0.05), 0.05)],
    "brown rice": [((0.30, 0.18, 0.08), 0.15)],
}


def kind_of(name: str) -> str:
    return KIND.get(name, "curd")


def look(name: str, rng: np.random.Generator) -> dict:
    """Material parameters for render.food_material (cells = Voronoi grains / flakes / chunks
    in object space, metres; lumps / swirl = low-frequency relief)."""
    k = kind_of(name)
    u = lambda a, b: float(rng.uniform(a, b))
    L = {"kind": k, "cells": None, "speckles": SPECKLES.get(name, []), "chunk_color": None,
         "lumps": (u(40, 80), 0.15), "swirl": None, "tint_var": 0.06, "coat": 0.0, "sss": 0.2, "bump": 0.4}
    if k == "grain":
        size = u(0.0015, 0.0025) if name == "quinoa" else u(0.0035, 0.005)
        L.update(cells={"scale": 1 / size, "mode": "dome", "jitter": 0.12, "crevice": 0.4, "distort": 0.3},
                 sss=0.3, bump=0.7)
    elif k == "porridge":
        L.update(cells={"scale": 1 / u(0.005, 0.009), "mode": "flake", "jitter": 0.1, "crevice": 0.06, "distort": 1.0,
                        "cover": u(0.6, 0.85)},
                 lumps=(u(18, 35), 0.6), coat=u(0.2, 0.45), sss=0.45, bump=0.45, tint_var=0.12)
    elif k == "puree":
        glossy = name in ("yogurt", "sour cream", "melted cheese")
        L.update(swirl=(u(12, 30), u(0.4, 0.8)), lumps=(u(25, 50), 0.25), coat=u(0.3, 0.6) if glossy else 0.0,
                 sss=0.6 if glossy else 0.35, bump=0.45, tint_var=0.04)
    elif k == "liquid":
        L.update(coat=u(0.2, 0.4), sss=0.1, bump=0.35, lumps=(u(25, 50), 0.2), swirl=(u(15, 30), 0.15))
        if name in CHUNKS:
            col, (lo, hi), cover = CHUNKS[name]
            L.update(cells={"scale": 1 / u(lo, hi), "mode": "chunk", "jitter": 0.15, "crevice": 0.0,
                            "distort": 0.5, "cover": cover}, chunk_color=list(col), bump=0.55)
    else:  # curd
        size = u(0.005, 0.008) if name == "mac and cheese" else u(0.008, 0.015)
        L.update(cells={"scale": 1 / size, "mode": "dome", "jitter": 0.1, "crevice": 0.3, "distort": 0.8},
                 coat=u(0.2, 0.4) if name == "mac and cheese" else 0.0, sss=0.35, bump=0.55)
    if name in PIECES and L["cells"] and L["cells"]["mode"] in ("chunk", "flake"):
        L["cells"], L["chunk_color"] = None, None     # real beans / flakes are scattered (PIECES): no painted ones
    return L


def shape(name: str, container_kind: str, rf: float, rng: np.random.Generator) -> dict:
    """Heap profile for one piece: mound height (m) above the fill level / plate, flatness p
    (top = mound * (1 - rho^p): large p is a level pool with a rounded edge), relief (m, m)."""
    k = kind_of(name)
    u = lambda a, b: float(rng.uniform(a, b))
    if k == "liquid":
        mound = u(0.0, 0.004) if container_kind == "bowl" else min(rf * u(0.08, 0.18), 0.012)
        return {"mound": mound, "p": u(6, 10), "relief": (u(0.0002, 0.0006), u(0.015, 0.03)), "wobble": 0.0}
    if k == "porridge":
        mound = u(0.002, 0.01) if container_kind == "bowl" else min(rf * u(0.15, 0.35), 0.025)
        return {"mound": mound, "p": u(3, 6), "relief": (u(0.001, 0.002), u(0.01, 0.018)), "wobble": u(0, 0.002)}
    if k == "puree":
        mound = u(0.005, 0.03) if container_kind == "bowl" else min(rf * u(0.3, 0.7), 0.05)
        return {"mound": mound, "p": u(2, 4), "relief": (u(0.0015, 0.003), u(0.012, 0.025)), "wobble": u(0, 0.004)}
    if k == "grain":
        mound = u(0.005, 0.03) if container_kind == "bowl" else min(rf * u(0.35, 0.8), 0.05)
        # grains are shading (render.food_material); geometric relief only for clumps
        return {"mound": mound, "p": u(2, 4), "relief": (u(0.0002, 0.0005), u(0.008, 0.014)), "wobble": u(0, 0.004)}
    mound = u(0.005, 0.025) if container_kind == "bowl" else min(rf * u(0.25, 0.6), 0.04)
    return {"mound": mound, "p": u(2, 4), "relief": (u(0.0008, 0.0015), u(0.01, 0.016)), "wobble": u(0, 0.004)}
