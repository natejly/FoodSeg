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

from . import food_textures

# food -> (real texture group from synth.food_textures, tile scale, relief, strength).
# Colour comes from the spec colour, the photo adds grain / crumb / sauce texture and relief.
TEX = {
    "white rice": ("rice", 1.0), "brown rice": ("rice", 1.0), "quinoa": ("rice", 0.4), "couscous": ("rice", 0.3),
    "fried rice": ("fried_rice", 1.0),
    "mashed potato": ("mash", 1.0), "yogurt": ("applesauce", 1.0), "sour cream": ("applesauce", 1.0),
    "melted cheese": ("applesauce", 1.0), "hummus": ("mash", 1.0), "refried beans": ("mash", 1.0),
    "guacamole": ("guacamole", 1.0),
    "mac and cheese": ("mac", 1.0), "spaghetti": ("pasta", 1.0), "penne in tomato sauce": ("stew", 1.0),
    "scrambled eggs": ("omelet", 1.0), "egg whites": ("omelet", 1.0),
    "oatmeal": ("cottage", 1.0), "risotto": ("cottage", 1.0), "polenta": ("mash", 1.0), "congee": ("cottage", 1.0),
    "chili": ("stew", 1.0), "lentil stew": ("stew", 1.0), "dal": ("stew", 1.0), "chicken curry": ("stew", 1.0),
    "green curry": ("stew", 1.0), "curry sauce": ("stew", 1.0), "baked beans": ("stew", 1.0),
    "black beans": ("stew", 1.0), "gravy": ("stew", 1.0), "tomato sauce": ("stew", 1.0),
    "tomato soup": ("soup", 1.0), "pumpkin soup": ("soup", 1.0), "miso soup": ("soup", 1.0), "ramen": ("soup", 1.0),
    "salsa": ("stew", 1.0), "pesto": ("guacamole", 1.0),
}

KIND = {
    "white rice": "grain", "fried rice": "grain", "brown rice": "grain", "quinoa": "grain",
    "oatmeal": "porridge",
    "mashed potato": "puree", "yogurt": "puree", "sour cream": "puree", "melted cheese": "puree",
    "lentil stew": "liquid", "chili": "liquid", "curry sauce": "liquid", "tomato sauce": "liquid",
    "gravy": "liquid", "salsa": "liquid", "pesto": "liquid", "guacamole": "liquid",
    "mac and cheese": "curd", "scrambled eggs": "curd", "egg whites": "curd",
    # added for diversity (synth.meals.FOODS)
    "couscous": "grain", "granola": "grain", "corn": "grain", "mixed vegetables": "grain",
    "congee": "porridge", "risotto": "porridge", "polenta": "porridge",
    "hummus": "puree", "refried beans": "puree",
    "penne in tomato sauce": "liquid", "dal": "liquid", "chicken curry": "liquid", "green curry": "liquid",
    "black beans": "liquid", "baked beans": "liquid", "tomato soup": "liquid", "pumpkin soup": "liquid",
    "miso soup": "liquid", "ramen": "liquid", "cereal with milk": "liquid",
    "spaghetti": "curd",
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
    # thin: soups, broths, milk
    **{n: {"model": "fluid", "K": 4e4, "visc": 0.3, "rho": 1020}
       for n in ("tomato soup", "pumpkin soup", "miso soup", "ramen", "cereal with milk")},
    "black beans": {"model": "fluid", "K": 4e4, "visc": 2.0, "rho": 1100},
    "baked beans": {"model": "fluid", "K": 4e4, "visc": 2.0, "rho": 1100},
    "penne in tomato sauce": {"model": "fluid", "K": 4e4, "visc": 2.5, "rho": 1050},
    "spaghetti": {"model": "plastic", "E": 3e4, "nu": 0.3, "rho": 1000, "theta_c": 0.015, "theta_s": 0.005},
}
ITEM_PHYSICS = {"model": "rigid", "rho": 1000}   # scans: rigid bodies coupled to the continuum

# Real small pieces scattered over a food's settled surface (render.scatter_pieces), so grains,
# flakes and beans have geometry and silhouette instead of painted bumps. Per food a list of
# (shape, per m^2 of upward-facing surface, size L x W x H mm, colour or None = the food's own
# material, fraction of the piece's height sunk below the surface).
PIECES = {
    "white rice": [("rice", 9e4, (5.5, 2.2, 1.9), None, 0.35), ("rice", 2.5e4, (5.5, 2.2, 1.9), None, 0.05)],
    "brown rice": [("rice", 9e4, (5.0, 2.3, 2.0), None, 0.35), ("rice", 2.5e4, (5.0, 2.3, 2.0), None, 0.05)],
    "fried rice": [("rice", 8e4, (5.5, 2.2, 1.9), None, 0.35), ("pea", 2.5e3, (7, 7, 7), (0.18, 0.42, 0.06), 0.4),
                   ("cube", 2.5e3, (5, 5, 5), (0.85, 0.32, 0.05), 0.4), ("flake", 2e3, (9, 7, 2), (0.95, 0.78, 0.2), 0.3)],
    "quinoa": [("quinoa", 3e5, (2.2, 2.2, 1.2), None, 0.3)],
    "oatmeal": [("flake", 3.5e4, (7, 5.5, 1.0), None, 0.6)],
    "chili": [("bean", 5e3, (13, 7.5, 6), (0.28, 0.05, 0.04), 0.55)],
    "lentil stew": [("lentil", 4e4, (4.5, 4.5, 1.8), (0.4, 0.24, 0.08), 0.6)],
    "mac and cheese": [("macaroni", 1.4e4, (14, 6, 6), None, 0.45)],
    "salsa": [("cube", 1.5e4, (5, 5, 5), (0.75, 0.08, 0.04), 0.5), ("cube", 6e3, (4, 4, 4), (0.9, 0.88, 0.75), 0.5)],
    # synth.meals.FOODS
    "spaghetti": [("noodle", 1.2e4, (70, 2.2, 2.2), None, 0.3)],
    "ramen": [("noodle", 1.0e4, (80, 2.5, 2.5), (0.85, 0.7, 0.35), 0.4), ("cube", 600, (12, 3, 3), (0.2, 0.42, 0.08), 0.2)],
    "penne in tomato sauce": [("penne", 6e3, (26, 8, 8), (0.88, 0.7, 0.36), 0.4)],
    "couscous": [("quinoa", 4e5, (1.6, 1.6, 1.0), None, 0.3)],
    "dal": [("lentil", 2.4e4, (4, 4, 1.6), (0.62, 0.4, 0.08), 0.4)],
    # chicken pieces: rounded lumps stained by the sauce, mostly submerged (pale cubes read as paper)
    "chicken curry": [("chunk", 3e3, (13, 11, 9), (0.6, 0.36, 0.16), 0.6)],
    "green curry": [("chunk", 2.5e3, (13, 11, 9), (0.7, 0.62, 0.4), 0.6), ("pea", 2e3, (8, 8, 8), (0.2, 0.4, 0.05), 0.4)],
    "black beans": [("bean", 1.5e4, (8, 5.5, 4.5), (0.04, 0.03, 0.04), 0.45)],
    "baked beans": [("bean", 1.5e4, (9, 6, 5), (0.72, 0.42, 0.22), 0.45)],
    "congee": [("rice", 3e4, (5.5, 2.4, 2.0), None, 0.8)],
    "risotto": [("rice", 9e4, (6, 2.8, 2.4), None, 0.35)],
    "miso soup": [("cube", 1.5e3, (8, 8, 8), (0.9, 0.88, 0.8), 0.45), ("cube", 1.5e3, (10, 3, 3), (0.2, 0.42, 0.08), 0.2)],
    "cereal with milk": [("flake", 6e3, (16, 13, 2), (0.82, 0.55, 0.18), 0.2)],
    "granola": [("cluster", 2.5e4, (9, 7, 5), None, 0.35)],
    "corn": [("cube", 3e4, (7, 6, 4), (0.95, 0.75, 0.12), 0.35)],
    "mixed vegetables": [("pea", 1.2e4, (7, 7, 7), (0.18, 0.42, 0.06), 0.35), ("cube", 1.2e4, (7, 7, 7), (0.85, 0.32, 0.05), 0.35),
                         ("cube", 1e4, (6, 5, 4), (0.95, 0.75, 0.12), 0.35), ("cube", 6e3, (12, 5, 5), (0.25, 0.45, 0.1), 0.35)],
}


# Realism extras per food (round 2), consumed by render.food_material / contact_film:
#   oil:    (colour multiplier of the oil-stained paste, coverage 0..1) glossy separated-oil patches
#   flecks: [(linear rgb, fraction of cells, cell size m)] herbs / spice / pepper bits in the surface
#   brown:  (colour multiplier, coverage) uneven cooking / browning on the exposed peaks
#   film:   thin sauce film at the food / plate boundary: (mean reach m, drip count)
EXTRAS = {
    "chicken curry": {"oil": ((1.25, 0.75, 0.25), 0.4), "brown": ((0.7, 0.55, 0.45), 0.3),
                      "flecks": [((0.10, 0.22, 0.05), 0.05, 0.0025), ((0.45, 0.07, 0.015), 0.04, 0.0020),
                                 ((0.9, 0.7, 0.3), 0.02, 0.0018)], "film": (0.006, 3)},
    "green curry": {"oil": ((1.3, 1.0, 0.3), 0.35), "brown": ((0.75, 0.75, 0.55), 0.2),
                    "flecks": [((0.10, 0.24, 0.05), 0.07, 0.0025), ((0.5, 0.09, 0.02), 0.03, 0.0018),
                               ((0.85, 0.8, 0.5), 0.03, 0.0015)], "film": (0.006, 3)},
    "curry sauce": {"oil": ((1.25, 0.8, 0.3), 0.4), "brown": ((0.75, 0.6, 0.4), 0.2),
                    "flecks": [((0.05, 0.12, 0.02), 0.05, 0.0025)], "film": (0.006, 2)},
    "dal": {"oil": ((1.3, 0.9, 0.3), 0.35), "brown": ((0.65, 0.5, 0.35), 0.3),
            "flecks": [((0.10, 0.04, 0.015), 0.05, 0.002), ((0.10, 0.24, 0.05), 0.05, 0.0025),
                       ((0.5, 0.06, 0.01), 0.03, 0.0018)], "film": (0.005, 2)},
    "lentil stew": {"oil": ((1.2, 0.85, 0.35), 0.25), "brown": ((0.65, 0.55, 0.4), 0.3),
                    "flecks": [((0.04, 0.12, 0.02), 0.04, 0.0025)], "film": (0.005, 2)},
    "chili": {"oil": ((1.5, 0.7, 0.3), 0.35), "brown": ((0.6, 0.5, 0.45), 0.35),
              "flecks": [((0.03, 0.01, 0.005), 0.06, 0.002), ((0.5, 0.08, 0.02), 0.04, 0.002),
                         ((0.04, 0.12, 0.02), 0.02, 0.002)], "film": (0.005, 2)},
    "tomato sauce": {"oil": ((1.4, 0.8, 0.5), 0.3), "flecks": [((0.04, 0.12, 0.02), 0.03, 0.002)], "film": (0.006, 2)},
    "gravy": {"oil": ((1.2, 1.0, 0.7), 0.25), "flecks": [((0.02, 0.015, 0.01), 0.03, 0.0015)], "film": (0.006, 2)},
    "oatmeal": {"brown": ((0.7, 0.5, 0.3), 0.4), "flecks": [((0.22, 0.09, 0.03), 0.04, 0.0015), ((0.35, 0.12, 0.06), 0.02, 0.003)],
                "film": (0.004, 1)},
    "risotto": {"oil": ((1.15, 1.0, 0.7), 0.3), "brown": ((0.78, 0.6, 0.4), 0.25),
                "flecks": [((0.05, 0.16, 0.03), 0.05, 0.002), ((0.04, 0.03, 0.02), 0.03, 0.0012)], "film": (0.005, 2)},
    "congee": {"oil": ((1.1, 1.0, 0.8), 0.2), "flecks": [((0.04, 0.16, 0.03), 0.03, 0.002), ((0.2, 0.1, 0.05), 0.02, 0.0015)],
               "film": (0.004, 1)},
    "polenta": {"oil": ((1.1, 0.9, 0.4), 0.3), "brown": ((0.8, 0.6, 0.3), 0.25), "flecks": [((0.04, 0.03, 0.02), 0.03, 0.0012)],
                "film": (0.004, 1)},
    "hummus": {"oil": ((1.2, 1.0, 0.5), 0.45), "brown": ((0.9, 0.35, 0.12), 0.25),
               "flecks": [((0.04, 0.14, 0.03), 0.03, 0.002), ((0.55, 0.1, 0.02), 0.04, 0.0015)], "film": (0.004, 1)},
    "refried beans": {"oil": ((1.3, 0.9, 0.5), 0.3), "brown": ((0.6, 0.5, 0.4), 0.3),
                      "flecks": [((0.02, 0.012, 0.008), 0.04, 0.0015)], "film": (0.004, 1)},
    "mashed potato": {"oil": ((1.15, 1.0, 0.4), 0.35), "brown": ((0.85, 0.7, 0.5), 0.15),
                      "flecks": [((0.02, 0.02, 0.015), 0.04, 0.0012), ((0.05, 0.16, 0.03), 0.03, 0.002)], "film": (0.004, 1)},
    "yogurt": {"flecks": [((0.04, 0.14, 0.03), 0.01, 0.002)], "film": (0.003, 1)},
    "sour cream": {"film": (0.003, 1)},
    "guacamole": {"brown": ((0.6, 0.55, 0.3), 0.25), "flecks": [((0.5, 0.05, 0.03), 0.03, 0.0025), ((0.05, 0.2, 0.03), 0.03, 0.002)],
                  "film": (0.003, 1)},
    "mac and cheese": {"oil": ((1.2, 0.9, 0.3), 0.3), "brown": ((0.65, 0.45, 0.2), 0.3),
                       "flecks": [((0.03, 0.025, 0.02), 0.02, 0.0012)], "film": (0.004, 1)},
    "scrambled eggs": {"oil": ((1.15, 1.0, 0.45), 0.3), "brown": ((0.8, 0.55, 0.2), 0.3),
                       "flecks": [((0.04, 0.14, 0.02), 0.02, 0.002)], "film": (0.003, 1)},
    "baked beans": {"oil": ((1.4, 0.8, 0.4), 0.25), "film": (0.005, 2)},
    "penne in tomato sauce": {"oil": ((1.4, 0.8, 0.4), 0.3), "flecks": [((0.04, 0.14, 0.02), 0.04, 0.002)], "film": (0.005, 2)},
    "tomato soup": {"oil": ((1.4, 0.8, 0.4), 0.2), "flecks": [((0.04, 0.14, 0.02), 0.02, 0.002)]},
    "pumpkin soup": {"oil": ((1.2, 0.9, 0.5), 0.2), "flecks": [((0.04, 0.14, 0.02), 0.02, 0.002)]},
    "white rice": {"brown": ((0.9, 0.82, 0.6), 0.2)},
    "brown rice": {"brown": ((0.8, 0.7, 0.55), 0.3)},
    "fried rice": {"oil": ((1.15, 0.85, 0.4), 0.35), "brown": ((0.65, 0.5, 0.3), 0.35),
                   "flecks": [((0.05, 0.16, 0.03), 0.03, 0.002)]},
}
# wetness (spatially varying gloss): cavity = gloss gained in crevices, patch = coverage of glossy
# puddles / sheen, dark = darkening and saturation of the paste where sauce pools, dry = peaks' roughness add
WET = {
    "liquid": {"cavity": 1.0, "patch": 0.55, "dark": 0.4, "dry": 0.12},
    "porridge": {"cavity": 0.8, "patch": 0.4, "dark": 0.3, "dry": 0.1},
    "puree": {"cavity": 0.6, "patch": 0.3, "dark": 0.2, "dry": 0.1},
    "curd": {"cavity": 0.6, "patch": 0.3, "dark": 0.25, "dry": 0.1},
    "grain": {"cavity": 0.4, "patch": 0.2, "dark": 0.15, "dry": 0.05},
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
        L["swirl"] = None          # sine bands on a pool read as brush strokes (liquids level out)
        if name in CHUNKS:
            col, (lo, hi), cover = CHUNKS[name]
            L.update(cells={"scale": 1 / u(lo, hi), "mode": "chunk", "jitter": 0.15, "crevice": 0.0,
                            "distort": 0.5, "cover": cover}, chunk_color=list(col), bump=0.55)
    else:  # curd
        size = u(0.005, 0.008) if name == "mac and cheese" else u(0.008, 0.015)
        L.update(cells={"scale": 1 / size, "mode": "dome", "jitter": 0.1, "crevice": 0.3, "distort": 0.8},
                 coat=u(0.2, 0.4) if name == "mac and cheese" else 0.0, sss=0.35, bump=0.55)
    if name in TEX:
        t = food_textures.pick(TEX[name][0], rng)
        if t:
            L["tex"] = {"albedo": t[0], "height": t[1], "tile_m": t[2], "scale": TEX[name][1],
                        "seed": int(rng.integers(1 << 30)), "chroma": 0.4,
                        "strength": {"puree": 0.6, "liquid": 0.3, "porridge": 0.6}.get(k, 1.0),
                        "relief": u(1.0, 1.8) if k in ("grain", "curd") else u(0.5, 1.0) * (0.3 if k == "liquid" else 1.0)}
            L["tint_var"] = min(L["tint_var"], 0.04)
    L["wet"] = dict(WET[k])
    L["wet"]["patch_scale"] = 30.0 + (sum(map(ord, name)) * 7) % 40       # no rng draw: keeps spec streams unchanged
    x = EXTRAS.get(name, {})
    L["oil"], L["brown"] = x.get("oil"), x.get("brown")
    L["flecks"] = [(c_, f_ * 0.6, s_) for c_, f_, s_ in x.get("flecks", [])]
    L["film"] = x.get("film")
    L["smooth_iter"] = {"liquid": 14, "porridge": 8, "puree": 5, "curd": 3, "grain": 3}[k]   # marching-cubes terraces
    L["relief_mm"] = {"liquid": 0.0, "porridge": 1.2, "puree": 0.8, "curd": 1.2, "grain": 1.5}[k]
    L["ao"] = True                       # darken the paste between real grains
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
