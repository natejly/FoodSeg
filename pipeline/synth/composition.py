"""Real meal compositions (Nutrition5k) -> what the synthetic generator can render.

    python -m synth.composition ../data/datasets/Nutrition5k --assets ../data/synth/assets.jsonl

Nutrition5k lists every ingredient of 5,006 cafeteria dishes with grams. An ingredient of
>= 10 g counts as a visible component. Each common ingredient is mapped to:
    ("asset", [categories])   MetaFood3D scan categories (names normalised: lower case, spaces)
    ("fill", name)            a procedural fill in synth.spec.FILLS (amorphous, takes the dish shape)
    ("topping", name)         a procedural topping in synth.spec_hard.TOPPINGS
    ("skip", why)             not visible as its own item (oil, dressing)
    ("gap", kind)             common in real meals, nothing to render it with yet
The report says how much of real meals the generator covers and which gaps matter most;
`sample_dish` draws a real dish's visible components for scene sampling.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

VISIBLE_G = 10.0

A = lambda *c: ("asset", list(c))
FILL = lambda n: ("fill", n)
TOP = lambda n: ("topping", n)
GAP = lambda k: ("gap", k)

# The ~90 most frequent visible ingredients (>= 10 g), about 80 % of all visible components.
INGREDIENTS: dict[str, tuple] = {
    "broccoli": A("broccoli"), "chicken": A("chicken breast", "chicken thighs", "chicken tender"),
    "mixed greens": GAP("leafy"), "cauliflower": A("cauliflower"), "scrambled eggs": FILL("scrambled eggs"),
    "carrot": A("carrot"), "cheese pizza": A("pizza"), "berries": A("blueberries", "strawberry"),
    "bacon": A("bacon"), "olives": A("olive"), "cucumbers": A("cucumber"), "arugula": GAP("leafy"),
    "fish": A("breaded fish", "salmon grill"), "cherry tomatoes": A("tomato"), "white rice": FILL("white rice"),
    "spinach (raw)": GAP("leafy"), "sweet potato": A("sweet potato"), "pineapple": A("pineapple"),
    "beef": A("steak", "stew beef"), "cantaloupe": A("melons"), "brussels sprouts": A("brussels sprout"),
    "egg whites": FILL("egg whites"), "roasted potatoes": A("roasted potato"), "olive oil": ("skip", "oil"),
    "caesar salad": GAP("leafy"), "pork": A("pork chop", "pork rib"), "brown rice": FILL("brown rice"),
    "sausage": A("sausages"), "bell peppers": A("bell pepper"), "wheat berry": FILL("brown rice"),
    "apple": A("apple"), "squash": A("squash"), "tomatoes": A("tomato", "tomato slice"), "tofu": A("tofu"),
    "corn on the cob": A("corn stick"), "green beans": A("green beans", "green bean"), "grapes": A("grapes"),
    "almonds": A("almonds"), "salmon": A("salmon grill"), "potatoes": A("baked potato"),
    "mushroom": A("fried mushrooms"), "lettuce": GAP("leafy"), "onions": A("onion"), "watermelon": A("melons"),
    "quinoa": FILL("quinoa"), "yam": A("sweet potato"), "kale": GAP("leafy"),
    "chicken thighs": A("chicken thighs"), "orange": A("orange"), "asparagus": A("asparagus"),
    "pizza": A("pizza"), "tortilla": A("tortilla"), "avocado": A("avocado"), "millet": FILL("quinoa"),
    "bok choy": GAP("leafy"), "steak": A("steak"), "honeydew melons": A("melons"),
    "strawberries": A("strawberry"), "chard": GAP("leafy"), "ham": GAP("sliced meat"),
    "oatmeal": FILL("oatmeal"), "mustard greens": GAP("leafy"), "eggplant": A("roasted potato"),
    "corn": GAP("small pile"), "cheese": TOP("melted cheese"), "zucchini": A("zucchini"),
    "chicken apple sausage": A("sausages"), "nuts": A("almonds", "peanuts"), "lemon": A("lemon"),
    "pepperoni pizza": A("pizza"), "kiwi": A("kiwi"), "chicken breast": A("chicken breast"),
    "hash browns": A("hashbrown"), "pears": A("pear"), "banana": A("banana"),
    "potato chips": A("tortilla chips"), "salsa": TOP("salsa"), "cabbage": A("cabbage", "coleslaw"),
    "bread": A("yeast bread"), "shrimp": A("shrimp"), "radishes": GAP("small pile"),
    "turkey bacon": A("bacon"), "parmesan cheese": TOP("melted cheese"), "noodles": A("pasta mixed dishes"),
    "chickpeas": A("chickpea"), "country rice": FILL("white rice"), "greek yogurt": FILL("yogurt"),
    "eggs": A("egg"), "mashed potatoes": FILL("mashed potato"),
}


def norm(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def load_dishes(n5k: Path) -> list[list[tuple[str, float]]]:
    dishes = []
    for f in sorted((n5k / "metadata").glob("dish_metadata_cafe*.csv")):
        for r in csv.reader(open(f)):
            dishes.append([(r[k + 1], float(r[k + 2])) for k in range(6, len(r) - 6, 7)])
    return dishes


def resolve(asset_cats: set[str]) -> dict[str, tuple]:
    """Drop asset categories we don't have; an ingredient with none left becomes a gap."""
    out = {}
    for ing, m in INGREDIENTS.items():
        if m[0] == "asset":
            have = [c for c in m[1] if c in asset_cats]
            out[ing] = ("asset", have) if have else GAP("missing asset")
        else:
            out[ing] = m
    return out


def sample_dish(rng: np.random.Generator, dishes: list, mapping: dict, renderable_only: bool = True):
    """A real dish's visible components [(ingredient, grams, mapping)], optionally resampling
    until every component can be rendered."""
    for _ in range(1000):
        d = dishes[rng.integers(len(dishes))]
        vis = [(n, g, mapping.get(n, GAP("unmapped"))) for n, g in d if g >= VISIBLE_G]
        vis = [v for v in vis if v[2][0] != "skip"]
        if vis and (not renderable_only or all(v[2][0] in ("asset", "fill", "topping") for v in vis)):
            return vis
    raise RuntimeError("no renderable dish found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n5k")
    ap.add_argument("--assets", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.assets)]
    cats = {norm(a["category"]) for a in rows                                 # what spec_hard samples
            if a.get("volume_ok", a["watertight"]) and not a.get("container_shaped")}
    mapping = resolve(cats)
    dishes = load_dishes(Path(args.n5k))
    comp, kinds, gaps = Counter(), Counter(), Counter()
    full = 0
    for d in dishes:
        vis = [(n, g) for n, g in d if g >= VISIBLE_G]
        ok = True
        for n, _ in vis:
            m = mapping.get(n, GAP("unmapped"))
            if m[0] == "skip":
                continue
            comp[m[0]] += 1
            if m[0] == "gap":
                ok = False
                gaps[m[1]] += 1
        full += ok and bool(vis)
    tot = sum(comp.values())
    print(f"{len(dishes)} dishes, {tot} visible components (>= {VISIBLE_G:g} g, excluding oil etc.)")
    for k in ("asset", "fill", "topping", "gap"):
        print(f"  {k:<8} {comp[k]:5d}  {100 * comp[k] / tot:5.1f} %")
    print(f"dishes renderable in full: {full} ({100 * full / len(dishes):.1f} %)")
    print("gaps by kind (components):", ", ".join(f"{k} {v}" for k, v in gaps.most_common()))


if __name__ == "__main__":
    main()
