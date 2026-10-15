"""Typical bulk density (g/mL) of served food by category, to turn a GT volume into a GT mass.

Used when an item has `mass_g: None` (procedural foods; scanned assets already carry a mass).
Values are *as-served* densities (cooked, including air between pieces), rounded from USDA
FoodData Central / FNDDS household-measure weights (g per cup / 236.6 mL) and USDA SR "volume
per gram" tables. Lookup is by normalised category name, longest-substring match, then a
shape-kind fallback, then DEFAULT.
"""
from __future__ import annotations

DEFAULT = 0.8

DENSITY: dict[str, float] = {
    # grains / starches (FNDDS cup weights: cooked white rice 158 g/cup -> 0.67-0.8 packed)
    "white rice": 0.80, "brown rice": 0.80, "fried rice": 0.75, "rice": 0.80, "risotto": 0.90,
    "quinoa": 0.78, "couscous": 0.72, "spaghetti": 0.60, "ramen": 0.55, "mac and cheese": 0.95,
    "oatmeal": 1.00, "congee": 1.00, "polenta": 1.02, "granola": 0.45, "corn": 0.75,
    "mashed potato": 1.00,           # USDA 210 g/cup
    # legumes
    "black beans": 0.90, "refried beans": 1.05, "lentil stew": 0.95, "hummus": 1.05, "beans": 0.90,
    # sauces, soups, dairy (liquids ~1.0)
    "soup": 1.00, "pumpkin soup": 1.00, "chili": 0.98, "chicken curry": 1.00, "green curry": 1.00,
    "curry sauce": 1.03, "gravy": 1.02, "pesto": 0.95, "salsa": 1.00, "sour cream": 1.01,
    "melted cheese": 1.05, "yogurt": 1.03, "cheese cube": 1.10,
    # protein
    "scrambled eggs": 0.95, "egg whites": 1.03, "tofu": 1.05, "chicken": 1.05, "salmon": 1.05,
    "sausage": 1.05, "falafel": 0.95, "sushi": 1.10,
    # leafy / vegetables (loose, uncompacted: USDA 1 cup raw spinach 30 g -> 0.13; lettuce 0.12-0.2)
    "spinach": 0.15, "arugula": 0.15, "kale": 0.20, "lettuce": 0.15, "mixed greens": 0.18,
    "broccoli": 0.40, "green bean": 0.45, "bell pepper": 0.55, "cherry tomato": 0.90, "olive": 0.95,
    "fried mushrooms": 0.60,
    # fruit pieces
    "strawberry": 0.60, "blueberries": 0.65, "grapes": 0.60, "melons": 0.65,
    # nuts / baked / snacks
    "almonds": 0.62, "peanuts": 0.60, "brownies": 0.55, "quickbread": 0.35, "yeast bread": 0.30,
    "tortilla chips": 0.15, "burrito": 0.85, "hashbrown": 0.60, "chicken nugget": 0.85,
}
SHAPE_DEFAULT = {"leaves": 0.17, "topping": 1.0, "pile": 0.7, "piece": 0.9, "single": 0.9}


def _norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("(raw)", "").strip()


def density_for(category: str, shape_kind: str | None = None) -> float:
    c = _norm(category)
    if c in DENSITY:
        return DENSITY[c]
    hits = [k for k in DENSITY if k in c or c in k]
    if hits:
        return DENSITY[max(hits, key=len)]
    return SHAPE_DEFAULT.get(shape_kind or "", DEFAULT)


def mass_from_volume(item: dict) -> tuple[float, str]:
    """(mass_g, source): the item's own mass if it has one, else volume * density."""
    if item.get("mass_g"):
        return float(item["mass_g"]), item.get("mass_source") or "asset"
    kind = (item.get("shape") or {}).get("kind")
    return float(item["volume_ml"]) * density_for(item["category"], kind), "density"
