"""Real food colours from Nutrition5k overhead photos, per dominant ingredient.

Dishes whose mass is >= SHARE one ingredient give an (almost) single-food photo: the food pixels (not the white
plate: L > PLATE_L; not the neutral table / shadows: chroma < MIN_C; central crop only) are summarised in Lab per
dish, then per ingredient (median over dishes).  Used by synth.colour_audit as the 'plausible real colour' target and
by tools that clamp scan textures.  Only coloured foods are meaningful (white rice / cauliflower are lost to the
plate mask).

    python -m synth.n5k_colours [--share 0.7] [--min-dishes 3]   # -> data/synth/n5k_food_colours.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N5K = Path("/home/kyle/Desktop/FoodSeg/data/datasets/Nutrition5k")
OUT = Path(__file__).with_name("n5k_food_colours.json")
PLATE_L, MIN_C = 88.0, 11.0


def dish_ingredients(row: list[str]) -> list[tuple[str, float]]:
    """metadata row -> [(ingredient name, grams)] (6 dish columns, then 7 per ingredient)."""
    out = []
    for i in range(6, len(row) - 6, 7):
        try:
            out.append((row[i + 1].strip(), float(row[i + 2])))
        except ValueError:
            break
    return out


def dominant(ings: list[tuple[str, float]], share: float) -> str | None:
    tot = sum(g for _, g in ings)
    if tot <= 0:
        return None
    n, g = max(ings, key=lambda x: x[1])
    return n if g / tot >= share else None


def food_pixels_lab(rgb: np.ndarray) -> np.ndarray:
    from synth.colour_audit import chroma, srgb_to_lab
    h, w = rgb.shape[:2]
    c = rgb[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)].reshape(-1, 3)
    lab = srgb_to_lab(c[::7])
    keep = (lab[:, 0] < PLATE_L) & (lab[:, 0] > 12) & (chroma(lab) > MIN_C)
    return lab[keep]


def build(share: float, min_dishes: int) -> dict:
    from PIL import Image
    from synth.colour_audit import chroma, hue_deg
    by = defaultdict(list)
    for csvf in sorted((N5K / "metadata").glob("dish_metadata_cafe*.csv")):
        for row in csv.reader(open(csvf)):
            d = dominant(dish_ingredients(row), share)
            if d:
                by[d].append(row[0])
    res = {}
    for name, dishes in by.items():
        if len(dishes) < min_dishes:
            continue
        meds = []
        for dish in dishes[:40]:
            p = N5K / "imagery" / "realsense_overhead" / dish / "rgb.png"
            if not p.exists():
                continue
            lab = food_pixels_lab(np.asarray(Image.open(p).convert("RGB")))
            if len(lab) > 800:
                meds.append(np.percentile(lab, [10, 50, 90], axis=0))
        if len(meds) >= min_dishes:
            m = np.array(meds)                     # (dishes, 3 pct, 3 ch)
            med = np.median(m[:, 1], 0)
            res[name] = dict(n=len(meds), L=float(med[0]), a=float(med[1]), b=float(med[2]),
                             C=float(np.hypot(med[1], med[2])), hue=float(hue_deg(np.array([0, med[1], med[2]]))),
                             L_lo=float(np.median(m[:, 0, 0])), L_hi=float(np.median(m[:, 2, 0])),
                             C_hi=float(np.median(np.hypot(m[:, 2, 1], m[:, 2, 2]))),
                             C_max=float(np.percentile(np.hypot(m[:, 1, 1], m[:, 1, 2]), 90)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.7)
    ap.add_argument("--min-dishes", type=int, default=3)
    a = ap.parse_args()
    r = build(a.share, a.min_dishes)
    json.dump(r, open(OUT, "w"), indent=1)
    for k, v in sorted(r.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{k:24s} n={v['n']:3d} L={v['L']:5.1f} a={v['a']:6.1f} b={v['b']:6.1f} C={v['C']:5.1f} h={v['hue']:4.0f}")


if __name__ == "__main__":
    main()
