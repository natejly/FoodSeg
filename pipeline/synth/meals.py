"""What goes on the dish: meal compositions for scene sampling (synth.spec_hard).

Random combinations gave scenes nobody eats (oatmeal with pork ribs) and a cafeteria-only
prior (Nutrition5k) skews Western. A scene's food now comes from one of
  template  a cuisine-specific meal (curry and rice, bento, ramen, Mexican plate, breakfast...)
            with alternatives per slot, so composition is plausible and cuisines are balanced;
  n5k       a real Nutrition5k dish's visible components (synth.composition.sample_dish);
  random    the old independent draws (kept for hard cases: similar colours, odd mixes).
`sample_meal` returns names that spec_hard resolves: heap foods (procedural fills, see
FOODS + spec.FILLS), toppings (spec_hard.TOPPINGS), leafy greens (spec_hard.GREENS) and
scanned-asset categories (normalised: lower case, spaces) for piles and single items.
"""
from __future__ import annotations

import numpy as np

# Procedural foods added for diversity: name -> (base colour, linear RGB; roughness).
# Kind, physics and real pieces (noodles, beans, tofu cubes...) are in synth.looks.
FOODS = {
    "spaghetti": ((0.82, 0.62, 0.28), 0.45), "penne in tomato sauce": ((0.48, 0.09, 0.03), 0.3),
    "couscous": ((0.8, 0.62, 0.32), 0.7), "dal": ((0.72, 0.45, 0.08), 0.3),
    "chicken curry": ((0.55, 0.22, 0.05), 0.3), "green curry": ((0.42, 0.45, 0.14), 0.3),
    "hummus": ((0.72, 0.58, 0.36), 0.45), "refried beans": ((0.3, 0.17, 0.08), 0.5),
    "black beans": ((0.07, 0.05, 0.05), 0.35), "baked beans": ((0.55, 0.18, 0.05), 0.3),
    "polenta": ((0.88, 0.65, 0.15), 0.5), "congee": ((0.85, 0.83, 0.76), 0.35),
    "risotto": ((0.82, 0.74, 0.52), 0.4), "tomato soup": ((0.55, 0.1, 0.03), 0.2),
    "pumpkin soup": ((0.82, 0.42, 0.07), 0.25), "miso soup": ((0.55, 0.4, 0.2), 0.15),
    "ramen": ((0.6, 0.42, 0.18), 0.15), "cereal with milk": ((0.9, 0.9, 0.86), 0.2),
    "granola": ((0.55, 0.36, 0.14), 0.7), "corn": ((0.92, 0.72, 0.12), 0.45),
    "mixed vegetables": ((0.35, 0.3, 0.12), 0.5),
}
# Too runny for a plate: only served in bowls.
BOWL_ONLY = {"tomato soup", "pumpkin soup", "miso soup", "ramen", "cereal with milk", "congee"}

# Templates: dish ("plate" / "bowl" / "any"); heap = slots of (alternatives, p include);
# toppings / greens / pile / scans: alternatives (scans: slots of (categories, p include)).
T = lambda name, cuisine, dish="any", heap=(), toppings=(), greens=(), p_greens=0.0, pile=(), p_pile=0.0, scans=(), \
    layout=None, top=None: \
    dict(name=name, cuisine=cuisine, dish=dish, heap=list(heap), toppings=list(toppings), greens=list(greens),
         p_greens=p_greens, pile=list(pile), p_pile=p_pile, scans=list(scans), layout=layout, top=top)
# layout: how the heap foods share the dish (synth.mpm.relayout), alternatives; top: the food poured
# / spooned over the others when the layout is "layers"
TEMPLATES = [
    # South Asian
    T("curry and rice", "south asian", heap=[(["white rice"], 1), (["chicken curry", "dal", "green curry"], 1),
                                              (["dal"], 0.3)], scans=[(["tortilla"], 0.4)], layout=["sides", "layers"]),
    T("dal bowl", "south asian", "bowl", heap=[(["dal"], 1), (["white rice"], 0.6)], toppings=["sour cream"]),
    # East / Southeast Asian
    T("bento", "east asian", "plate", heap=[(["white rice"], 1)], layout=["sides"],
      scans=[(["salmon grill", "chicken thighs"], 0.8), (["gyoza", "sushi"], 0.6), (["edamame", "broccoli"], 0.6),
             (["boiled egg"], 0.3)]),
    T("ramen", "east asian", "bowl", heap=[(["ramen"], 1)], scans=[(["boiled egg"], 0.6)], greens=["spinach (raw)"],
      p_greens=0.3),
    T("miso and rice", "east asian", "bowl", heap=[(["miso soup", "congee"], 1)], scans=[(["tofu"], 0.3)]),
    T("fried rice plate", "east asian", heap=[(["fried rice"], 1)],
      scans=[(["chicken wings", "shrimp", "gyoza"], 0.7), (["broccoli", "edamame"], 0.5)]),
    T("sushi plate", "east asian", "plate", pile=["sushi"], p_pile=1.0, scans=[(["edamame"], 0.4)]),
    T("green curry", "southeast asian", "bowl", heap=[(["green curry"], 1), (["white rice"], 0.7)],
      layout=["sides", "layers"], top="green curry"),
    # Latin American
    T("mexican plate", "latin american", "plate", heap=[(["white rice"], 0.8), (["refried beans", "black beans"], 0.9)],
      toppings=["salsa", "guacamole", "sour cream"],
      scans=[(["taco", "burrito", "quesadilla"], 0.8), (["tortilla chips", "nachos"], 0.4)]),
    T("burrito bowl", "latin american", "bowl", heap=[(["white rice", "brown rice"], 1), (["black beans"], 0.8),
                                                         (["corn"], 0.5)],
      toppings=["salsa", "guacamole", "sour cream"], greens=["lettuce"], p_greens=0.5, scans=[(["avocado"], 0.3)],
      layout=["patches", "sides"]),
    # Mediterranean / Middle Eastern
    T("mezze", "middle eastern", "plate", heap=[(["hummus"], 1), (["couscous"], 0.6)],
      pile=["falafel"], p_pile=0.7, scans=[(["olive", "cherry tomato"], 0.6), (["cucumber"], 0.4)],
      greens=["mixed greens", "arugula"], p_greens=0.4),
    T("pasta", "italian", heap=[(["penne in tomato sauce", "spaghetti", "mac and cheese"], 1)],
      toppings=["pesto", "melted cheese", "tomato sauce"], greens=["arugula"], p_greens=0.3,
      scans=[(["meatball"], 0.4), (["yeast bread", "bread roll"], 0.3)]),
    T("risotto or polenta", "italian", heap=[(["risotto", "polenta"], 1)], toppings=["melted cheese", "pesto"],
      scans=[(["shrimp", "fried mushrooms"], 0.4)]),
    # Western
    T("eggs breakfast", "western breakfast", "plate", heap=[(["scrambled eggs", "egg whites"], 1)],
      scans=[(["bacon", "sausages"], 0.8), (["yeast bread", "hashbrown", "biscuits"], 0.6), (["avocado", "tomato slice"], 0.3)],
      pile=["blueberries", "strawberry", "grapes"], p_pile=0.3),
    T("pancakes", "western breakfast", "plate", scans=[(["pancake", "waffle", "french toast"], 1), (["bacon"], 0.4)],
      pile=["blueberries", "strawberry"], p_pile=0.6),
    T("porridge bowl", "western breakfast", "bowl", heap=[(["oatmeal"], 1)],
      pile=["blueberries", "strawberry"], p_pile=0.7, scans=[(["banana", "almonds"], 0.3)]),
    T("yogurt and granola", "western breakfast", "bowl", heap=[(["yogurt"], 1), (["granola"], 0.9)],
      pile=["blueberries", "strawberry"], p_pile=0.6, layout=["layers"], top="granola"),
    T("cereal", "western breakfast", "bowl", heap=[(["cereal with milk"], 1)], pile=["blueberries"], p_pile=0.3),
    T("meat and two veg", "western dinner", "plate", heap=[(["mashed potato"], 0.8), (["mixed vegetables", "corn"], 0.5)],
      toppings=["gravy"], scans=[(["steak", "pork chop", "meatloaf", "chicken breast", "salmon grill"], 1),
                                 (["green bean", "broccoli", "carrot", "asparagus"], 0.7)]),
    T("southern plate", "western dinner", "plate", heap=[(["mac and cheese"], 0.8), (["baked beans"], 0.5)],
      scans=[(["chicken wings", "chicken tender", "pork rib"], 1), (["corn stick", "biscuits"], 0.5)]),
    T("fish and chips", "western dinner", "plate", heap=[(["baked beans", "mixed vegetables"], 0.5)],
      scans=[(["breaded fish"], 1), (["french fry"], 1)]),
    T("soup and bread", "western dinner", "bowl", heap=[(["tomato soup", "pumpkin soup", "lentil stew"], 1)],
      toppings=["sour cream", "pesto"], scans=[(["bread roll", "yeast bread"], 0.5)]),
    T("chili bowl", "western dinner", "bowl", heap=[(["chili"], 1), (["white rice"], 0.3)], layout=["layers", "sides"],
      top="chili",
      toppings=["sour cream", "melted cheese"], scans=[(["corn stick"], 0.3)]),
    T("salad bowl", "western lunch", "any", greens=["mixed greens", "lettuce", "spinach (raw)", "kale", "arugula"],
      p_greens=1.0, heap=[(["quinoa", "couscous"], 0.4)],
      scans=[(["cherry tomato", "cucumber"], 0.8), (["chicken breast", "boiled egg", "cheese cube"], 0.6),
             (["avocado", "chickpea"], 0.4)]),
    T("sandwich lunch", "western lunch", "plate", scans=[(["sandwich", "burger", "peanut butter and jelly sandwiches"], 1),
                                                        (["french fry", "tortilla chips", "coleslaw"], 0.5)]),
    T("kids plate", "western lunch", "plate", heap=[(["mac and cheese", "mashed potato"], 0.6)],
      scans=[(["chicken nugget", "frankfurter sandwich", "pizza"], 1), (["carrot", "apple"], 0.4)]),
    # snacks / desserts
    T("fruit plate", "fruit", "plate", pile=["grapes", "blueberries", "strawberry"], p_pile=0.7,
      scans=[(["melons", "pineapple", "mango", "kiwi", "orange", "apple", "banana", "peach", "pear"], 1),
             (["melons", "pineapple", "kiwi", "papaya"], 0.6)]),
    T("dessert", "dessert", "plate", scans=[(["cake", "brownies", "pie", "cake slice", "cinnamon roll", "doughnut",
                                              "muffin", "cookies"], 1), (["ice cream"], 0.3)], pile=["strawberry"], p_pile=0.3),
]

SOURCE_P = {"template": 0.45, "n5k": 0.3, "random": 0.25}


def _pick(rng, options):
    return options[rng.integers(len(options))] if options else None


def sample_meal(rng: np.random.Generator, n5k: dict | None = None) -> dict:
    """{source, name, cuisine, dish, heap [fills], toppings [names], greens name | None,
    pile category | None, scans [categories]}; empty lists where the meal has none."""
    src = rng.choice(list(SOURCE_P), p=list(SOURCE_P.values()))
    if src == "n5k" and not n5k:
        src = "template"
    if src == "random":
        return {"source": "random", "name": "random", "cuisine": None, "dish": "any", "heap": None,
                "toppings": None, "greens": None, "pile": None, "scans": None}
    if src == "template":                          # cuisine first, so no cuisine dominates by template count
        cuisine = _pick(rng, sorted({t["cuisine"] for t in TEMPLATES}))
        t = _pick(rng, [t for t in TEMPLATES if t["cuisine"] == cuisine])
        heap = []
        for opts, p in t["heap"]:
            if rng.random() < p:
                f = _pick(rng, [o for o in opts if o not in heap])
                if f:
                    heap.append(f)
        scans = [_pick(rng, opts) for opts, p in t["scans"] if rng.random() < p]
        return {"source": "template", "name": t["name"], "cuisine": t["cuisine"], "dish": t["dish"], "heap": heap,
                "layout": t["layout"], "top": t["top"],
                "toppings": list(t["toppings"]), "greens": _pick(rng, t["greens"]) if rng.random() < t["p_greens"] else None,
                "pile": _pick(rng, t["pile"]) if rng.random() < t["p_pile"] else None, "scans": scans}
    from .composition import sample_dish
    vis = sample_dish(rng, n5k["dishes"], n5k["mapping"], renderable_only=False)
    heap, toppings, scans, greens = [], [], [], None
    for ing, grams, m in vis:
        if m[0] == "fill" and m[1] not in heap and len(heap) < 3:
            heap.append(m[1])
        elif m[0] == "topping":
            toppings.append(m[1])
        elif m[0] == "asset":
            scans.append(_pick(rng, m[1]))
        elif m == ("gap", "leafy"):
            greens = ing                           # spec_hard falls back to mixed greens
    return {"source": "n5k", "name": " + ".join(n for n, _, _ in vis), "cuisine": "cafeteria (n5k)", "dish": "any",
            "heap": heap, "toppings": toppings, "greens": greens, "pile": None, "scans": scans}
