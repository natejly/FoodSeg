"""Food-asset QA: audit assets_v3.jsonl and write a cleaned, annotated assets_v4.jsonl.

    python -m synth.asset_qa --in ../data/synth/assets_v3.jsonl --out ../data/synth/assets_v4.jsonl \
        --qa-dir ../data/synth/qa_v4 --tex-dir ../data/synth/assets_v4_tex [--blender PATH] [--no-thumbs]

What it does (all rules and tables are in this file so the result is reproducible):

1. Real-size check.  EXPECTED holds, per category, the plausible max dimension (cm) of ONE served asset, whether
   the asset is a single piece ("item") or a heap / bowl-sized portion, how it is served and how many pieces make
   a typical serving.  MetaFood3D scans are in metres already (all `to_m` = 1, the unit that matches the xlsx
   reference volume) and every scan's convex-hull volume agrees with its xlsx volume, so there are no x10 / x2.54
   unit mistakes to repair; what remains are (a) items that are simply scanned large (strawberry 7.5 cm, kiwi,
   croissant, lime ...) which are RESCALED to the category's upper bound via `to_m` (`qa_scale`, extent / volume /
   mass scale with it, originals kept in `to_m_orig` / `volume_ml_orig` / `mass_g_orig`), and (b) implausible assets
   which are excluded.  The xlsx weight / volume are cross-checked against each other (density) and against the
   mesh convex hull (a volume cannot exceed it): an impossible volume is replaced by mass / typical density, an
   impossible mass by volume * typical density (`qa_flags` mass_fixed / volume_fixed).
2. Appearance check.  Each asset is rendered to a thumbnail (Blender workbench, true texture colours; thumbnails
   were also inspected by eye, the per-asset verdicts are the EXCLUDE / *_FIX tables).  Excluded: containers /
   plates / cups / bags baked into the scan, packaging and wrappers, stray geometry (floating pieces), missing /
   magenta texture, odd colours, multi-object clusters.  Fixed: textures whose red and blue channels are swapped
   (blue brownies / egg) and the neon-yellow exposure cast of fried foods are rewritten to
   OUT_TEX_DIR/<id>/ (a symlink to the original OBJ + copied MTL + corrected texture); `path` points there.
3. Plating role.  `serve`: plated | whole_produce | snack | dessert | platter (a whole roast chicken, rack of ribs,
   whole loaf / cake: a centrepiece, not a per-person item).  `plate_ok` False for whole_produce and platter assets
   (fruit-bowl / produce / tray context only); `whole_context` is "tray" for grab-and-go fruit (apple, banana,
   orange ...) and "bowl" for produce that only belongs in a fruit bowl / produce display (pineapple, melon,
   cabbage, onion, ...).  `serving_copies` is the typical number of pieces of this asset in one serving.

New fields: qa_exclude, qa_reason, qa_flags, qa_scale, to_m_orig, mass_g_orig, volume_ml_orig, kind
(item | portion | procedural), quality (scan | procedural), expected_size_cm, serve, plate_ok, whole_context,
serving_copies, container_baked, tex_fix, tex_source.  Excluded assets stay in the file with qa_exclude true.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------------------
# 1 / 3. expected size, kind, serving.  key | lo_cm | hi_cm | kind (I item, P portion / heap) | serve | copies [| ctx]
# lo / hi: plausible max dimension of one asset as served.  serve: plated whole_produce snack dessert platter.
# ---------------------------------------------------------------------------------------------------------
EXPECTED_TXT = """
almond(bowl)|8|16|P|snack|1
almonds|1.5|12|P|snack|1
apple|6.5|9.5|I|whole_produce|1|tray
applesauce|7|13|P|plated|1
asparagus|12|28|I|plated|5
avocado|5.5|11|I|plated|1
bacon|8|17|I|plated|3
bagel|9.5|13|I|plated|1
baked potato|8|14|I|plated|1
banana|15|23|I|whole_produce|1|tray
beans|6|16|P|plated|1
bell pepper|3|11|I|plated|3
biscuits|5|10|I|plated|2
blueberries|1|7|P|plated|15
blueberry(bowl)|8|16|P|snack|1
boiled egg|4|6|I|plated|1
bread roll|5|12|I|plated|1
breaded fish|8|17|I|plated|1
broccoli|4|12|I|plated|3
brownies|5|9|I|dessert|2
brussels sprout|2|3.8|I|plated|5
burger|8|13.5|I|plated|1
burrito|12|22|I|plated|1
cabbage|5|18|I|plated|1
cake|8|13|I|dessert|1
cake slice|5|10|I|dessert|1
carrot|5|22|I|plated|1
cauliflower|5|12|I|plated|3
cereal bars|6|12|I|snack|1
cheese cube|1.5|3|I|snack|6
cherry tomato|2|3.5|I|plated|6
chicken breast|9|16|I|plated|1
chicken nugget|3.5|6.5|I|plated|5
chicken tender|8|13|I|plated|3
chicken thighs|8|14|I|plated|2
chicken wings|5|12|I|plated|3
chickpea|0.8|1.4|I|plated|15
chocolate|3|16|I|dessert|1
cinnamon roll|5|13|I|dessert|1
coleslaw|9|16|P|plated|1
cookies|4.5|10|I|dessert|2
corn dog|15|22|I|plated|1
corn stick|15|22|I|plated|1
corn(bowl)|8|17|P|plated|1
cottage cheese|7|13|P|plated|1
crab|8|22|I|plated|1
crab cake|5|9|I|plated|2
croissant|9|16|I|plated|1
cucumber|3|14|I|plated|4
doughnut|7.5|11|I|dessert|1
edamame|4|12|P|plated|1
egg|4.5|6.5|I|plated|1
falafel|3.5|6|I|plated|4
frankfurter sandwich|14|23|I|plated|1
french fry|6|20|P|plated|1
french toast|11|16|I|plated|2
fried mushrooms|3|5.5|I|plated|5
fried egg|7|13|I|plated|1
fried rice|9|17|P|plated|1
ginger|3|10|I|plated|1
grapes|4|14|P|plated|1
green bean|8|16|I|plated|8
green beans|8|16|P|plated|1
guacamole|7|12|P|plated|1
gyoza|6|9|I|plated|4
hashbrown|9|15|I|plated|1
ice cream|8|15|P|dessert|1
kiwi|4.5|7.5|I|whole_produce|1|tray
lasagna|10|20|P|plated|1
lemon|5.5|8|I|whole_produce|1|bowl
lime|4|5.5|I|whole_produce|1|bowl
lobster|12|22|I|plated|1
lychee|2.5|3.5|I|whole_produce|1|bowl
mac|8|16|P|plated|1
mango|9|13.5|I|whole_produce|1|bowl
mashed potato|9|14|P|plated|1
meatball|2.5|5|I|plated|4
meatloaf|7|13|I|plated|1
melons|3|6.5|I|plated|5
muffin|5|10|I|dessert|1
nachos|15|26|P|plated|1
nutrition bars|8|12|I|snack|1
olive|1.5|3|I|plated|4
omelet|15|23|I|plated|1
onion|5|11|I|whole_produce|1|bowl
orange|6.5|9|I|whole_produce|1|tray
pancake|9|18|I|plated|2
papaya|4|22|I|plated|1
pasta mixed dishes|14|24|P|plated|1
peach|5.5|8.5|I|whole_produce|1|tray
peanut butter and jelly sandwich|10|14|I|plated|1
peanuts|2|14|P|snack|1
pear|7|11|I|whole_produce|1|tray
pie|9|12|I|dessert|1
pineapple|6|30|I|plated|1
pizza|12|25|I|plated|1
plum|4|6.5|I|whole_produce|1|tray
pomegranate|8|12|I|whole_produce|1|bowl
pork chop|11|18|I|plated|1
pork rib|8|31|I|plated|1
potato|5|9|I|plated|3
pudding|6|9|P|dessert|1
quesadilla|12|17|I|plated|1
quickbread|7|16|I|plated|1
rice|7|17|P|plated|1
roasted potato|2|3.5|I|plated|6
salmon|6|15|I|plated|1
sandwich|11|18|I|plated|1
sausages|5|16|I|plated|2
shrimp|2.5|9|I|plated|6
soup|9|13|P|plated|1
squash|1.5|3|I|plated|6
steak|9|19|I|plated|1
stew beef|10|23|P|plated|1
strawberry|2.5|4.8|I|plated|5
sushi|3.5|6.5|I|plated|6
sweet potato|8|16|I|plated|1
taco|11|17|I|plated|2
tofu|2.5|9|I|plated|6
tomato|3|11|I|whole_produce|1|bowl
tomato slice|3.5|9|I|plated|3
tortilla|15|22|I|plated|1
tortilla chips|3|16|I|snack|8
tuna salad|5|14|P|plated|1
waffle|9|14|I|plated|1
whole chicken|16|24|I|platter|1
yeast bread|8|16|I|plated|1
yogurt|7|9|P|dessert|1
zucchini|3|5|I|plated|6
"""


def _parse_expected(txt: str = EXPECTED_TXT) -> dict[str, dict]:
    out = {}
    for line in txt.strip().splitlines():
        p = line.split("|")
        out[p[0]] = dict(lo=float(p[1]), hi=float(p[2]), kind={"I": "item", "P": "portion"}[p[3]], serve=p[4],
                         copies=int(p[5]), ctx=p[6] if len(p) > 6 else None)
    return out


EXPECTED = _parse_expected()
KEY_ALIAS = {"peanut butter and jelly sand": "peanut butter and jelly sandwich", "pasta mixed dishes": "pasta mixed dishes",
             "pasta mixed": "pasta mixed dishes"}


def cat_key(r: dict) -> str:
    c = str(r["category"]).replace("_", " ").strip().lower()
    c = KEY_ALIAS.get(c, c)
    if c in EXPECTED:
        return c
    for k in EXPECTED:            # "Pasta_mixed_dishes" / "Peanut butter and jelly sand..." / plural spellings
        if c.startswith(k) or k.startswith(c):
            return k
    return c


# whole produce / platter / whole-cake overrides per asset (key = "<category>_<object>" or the online id)
SERVE_OVERRIDE = {
    **{k: ("whole_produce", "bowl") for k in (
        "Avocado_avocado_7", "Avocado_avocado_8", "Avocado_avocado_9", "polyhaven_food_avocado_01",
        "Cabbage_cabbage_5", "Cabbage_new_cabbage_1", "Cabbage_new_cabbage_3",
        "Onion_new_onion_1", "Onion_new_onion_5", "polyhaven_yellow_onion",
        "Pineapple_new_pineapple_1", "Pineapple_new_pineapple_2",
        "Papaya_new_pappaya_1", "Papaya_new_pappaya_2",
        "Mango_new_mango_1", "Mango_new_mango_2", "Mango_new_mango_5",
        "Cucumber_new_cucumber_1", "Cucumber_new_cucumber_2", "Cucumber_new_cucumber_3",
        "Cauliflower_new_cauliflower_1",
        "Carrot_carrot_1", "Carrot_carrot_2", "Carrot_carrot_6", "Carrot_carrot_7",
        "polyhaven_sweet_potato", "Tomato_tomato_4",
        "Tomato_new_tomato_1", "Tomato_new_tomato_2", "Tomato_new_tomato_3", "Tomato_new_tomato_4")},
    **{k: ("platter", None) for k in (
        "Whole_Chicken_chicken_1", "Whole_Chicken_chicken_2", "Whole_Chicken_chicken_3", "Whole_Chicken_chicken_5",
        "Pork_Rib_pork_rib_2", "Pork_Rib_pork_rib_3",
        "Yeast_bread_yeast_bread_4", "Yeast_bread_yeast_bread_5", "Yeast_bread_yeast_bread_6")},
    "polyhaven_carrot_cake": ("platter", None),            # whole 23 cm cakes: slices only on a plate
    "polyhaven_strawberry_chocolate_cake": ("platter", None),
}
# kiwi halves / slices and half avocados are plated; sweet potato scans are baked, raw polyhaven is not
PLATED_KIWI = ("Kiwi_new_kiwi_3", "Kiwi_new_kiwi_4")

# ---------------------------------------------------------------------------------------------------------
# 2. appearance verdicts (from the thumbnails).  pattern ("<category>_<object>" fnmatch) -> (code, reason)
# ---------------------------------------------------------------------------------------------------------
_CONT = "container"
EXCLUDE = {
    # bowl / cup / tub / plate / box / bag / tray baked into the scan
    "Applesauce_*": (_CONT, "plastic cup / pouch with label baked in"),
    "Asparagus_new_asparagus_3": (_CONT, "lid / plate baked in"),
    "Bacon_new_bacon_3": (_CONT, "bowl baked in"), "Bacon_new_bacon_4": (_CONT, "bowl baked in"),
    "Beans_*": (_CONT, "bowl / ramekin / plate baked in"),
    "Broccoli_new_broccoli_2": (_CONT, "bowl baked in"),
    "Carrot_carrot_4": (_CONT, "bowl baked in"), "Carrot_carrot_5": (_CONT, "ramekin baked in"),
    "Carrot_new_carrot_1": (_CONT, "bowl baked in"),
    "Chocolate_new_chocolate_3": (_CONT, "bowl baked in"),
    "Coleslaw_new_coleslaw_[1-4]": (_CONT, "bowl / tub baked in"),
    "Cottage cheese_*": (_CONT, "retail tub with label"),
    "Crab_new_crab_2": (_CONT, "bowl baked in"),
    "Edamame_new_edamame_5": (_CONT, "bowl baked in"),
    "French_Fry_fries_1": (_CONT, "bowl baked in"), "French_Fry_new_fries_3": (_CONT, "branded fry carton baked in"),
    "French_Fry_new_fries_4": (_CONT, "paper tray baked in"), "French_Fry_new_waffle_fry_2": (_CONT, "branded fry carton baked in"),
    "Fried_rice_*": (_CONT, "bowl / dish baked in"),
    "Green_beans_green_beans_1": (_CONT, "bowl baked in"), "Green_beans_new_green_beans_*": (_CONT, "bowl baked in"),
    "Guacamole_*": (_CONT, "retail tub / bowl baked in"),
    "Hashbrown_new_hashbrown_1": (_CONT, "paper bag with print baked in"), "Hashbrown_new_hashbrown_4": (_CONT, "cup baked in"),
    "Ice_cream_ice_cream_1": (_CONT, "cup baked in"), "Ice_cream_new_ice_cream_[1-3]": (_CONT, "cup baked in"),
    "Lasagna_new_lasagna_3": (_CONT, "plate baked in, 27 cm"),
    "Mac_new_mac_and_cheese_1": (_CONT, "bowl baked in"), "Mac_new_mac_and_cheese_4": (_CONT, "bowl baked in"),
    "Nachos_new_nachos_3": (_CONT, "plate baked in"), "Nachos_new_nachos_4": (_CONT, "plate baked in"),
    "Pasta_mixed_dishes_new_pasta_and_mixed_dishes_[34]": (_CONT, "casserole dish baked in"),
    "Pasta_mixed_dishes_pasta1": (_CONT, "casserole dish baked in"),
    "Peanuts_new_peanut_6": (_CONT, "bowl baked in"),
    "Pudding_*": (_CONT, "plastic cup baked in"),
    "Rice_new_rice_3": (_CONT, "ramekin baked in"), "Rice_new_white_and_brown_rice_*": (_CONT, "bowl baked in"),
    "Shrimp_shrimp_7": (_CONT, "bowl baked in"),
    "Soup_*": (_CONT, "bowl / cup baked in"),
    "Stew_Beef_new_beef_stew_[124]": (_CONT, "bowl baked in"), "Stew_Beef_stew_1": (_CONT, "plastic tray baked in"),
    "Sweet Potato_new_sweet_potato_5": (_CONT, "bowl baked in"),
    "Tomato_tomato_3": (_CONT, "bowl baked in"),
    "Tuna_salad_new_tuna_salad_[12]": (_CONT, "bowl baked in"),
    "Waffle_new_waffle_1": (_CONT, "plate baked in, 27 cm"),
    "Yogurt_*": (_CONT, "retail cup with label"),
    "*(bowl)_*": (_CONT, "'(bowl)' scan: the bowl is part of the mesh"),
    # packaging / wrapper
    "Chocolate_new_chocolate_4": ("packaging", "branded wrapper baked in"),
    "Burrito_burrito_2": ("packaging", "grey foil-like wrap"),
    # bad geometry
    "Bagel_bagel_4": ("stray_geometry", "floating geometry (bbox 27 cm around a 10 cm bagel)"),
    "Biscuits_biscuit_4": ("stray_geometry", "floating geometry (bbox 24 cm around a 6 cm biscuit)"),
    "Blueberries_blueberry_5": ("stray_geometry", "floating geometry (bbox 23 cm around a 1.5 cm berry)"),
    "Shrimp_shrimp_5": ("stray_geometry", "floating geometry (bbox 22 cm around a 2 cm shrimp)"),
    "Quickbread_quickbread_4": ("stray_geometry", "floating geometry (bbox 26 cm around a 5 cm slice)"),
    "French_Fry_fries_2": ("stray_geometry", "thin sheet / floating fries, mass mismatch"),
    "Mac_mac3": ("bad_mesh", "2.8 cm fragment, not a serving"),
    "Carrot_new_carrot_2": ("bad_mesh", "7 ml fragment but 25 g mass, shapeless"),
    "Pork_Rib_pork_rib_1": ("bad_mesh", "4 cm fragment, mass 309 g (rack of ribs)"),
    "Beans_baked_bean_4": ("bad_mesh", "1 cm fragment"),
    # texture problems
    "Tortilla_Chips_new_chips_3": ("bad_texture", "texture file missing: renders magenta"),
    "Shrimp_shrimp_6": ("odd_colour", "texture correction turns it magenta"),
    "Chicken_tender_tender_4": ("odd_colour", "purple-brown texture"),
    "Cauliflower_new_cauliflower_[345]": ("odd_colour", "yellow cast, correction cannot reach cream"),
    "Quickbread_quickbread_2": ("bad_texture", "dark star-shaped texture holes"),
    "Egg_boiledegg1": ("odd_colour", "cyan / white texture, R-B swap does not give an egg"),
    "Egg_new_egg_1": ("odd_colour", "blue / yellow gradient texture, not an egg"),
    "Egg_new_egg_3": ("odd_colour", "saturated orange texture, not an egg"),
    "Peach_new_peach_1": ("odd_colour", "purple / orange blotched texture"),
    "Quesadilla_quesadilla_1": ("odd_colour", "iridescent blue / purple texture patches"),
    "Rice_rice_3": ("odd_colour", "pink rice ball"),
    "Waffle_new_waffle_[45]": ("odd_colour", "grey-blue desaturated texture"),
    # clusters / implausible scale
    "polyhaven_hamburger_buns": ("cluster", "3 buns in one 40 cm mesh"),
    "polyhaven_food_pears_asian_01": ("cluster", "3 pears in one 17 cm mesh"),
    "polyhaven_food_ginger_01": ("scale", "57 cm root, volume from hull only"),
    "Grapes_new_grape_1": ("scale", "28 cm bunch (412 ml), not a serving"),
}

# R <-> B swapped textures (verified on thumbnails: blue brownies / blue egg turn brown / cream when swapped)
SWAP_FIX = ("Brownies_brownies_1", "Brownies_brownies_2")
# neon-yellow exposure cast: pull the texture's mean colour to the named target (sRGB mean of a golden fried
# crust / bread / cream cauliflower ... as seen in the non-cast scans of the same food)
CAST_TARGETS = {"fried": (0.64, 0.42, 0.20), "bread": (0.74, 0.54, 0.30), "cream": (0.80, 0.76, 0.58),
                "sausage": (0.60, 0.30, 0.18), "shrimp": (0.90, 0.60, 0.42), "chip": (0.80, 0.62, 0.30)}
CAST_FIX = {
    **{f"Chicken_Nugget_nugget_{i}": "fried" for i in range(1, 7)},
    **{f"Chicken_tender_tender_{i}": "fried" for i in (3, 4, 5)},
    **{f"Chicken_thighs_thigh_{i}": "fried" for i in (5, 6, 7)},
    "Chicken_wings_wings_4": "fried", "Breaded_Fish_breaded_fish_1": "fried", "Breaded_Fish_breaded_fish_2": "fried",
    "Breaded_Fish_breaded_fish_3": "fried", "French_Fry_new_waffle_fry_1": "fried",
    "Hashbrown_new_hashbrown_3": "fried",
    "Bagel_bagel_3": "bread", "Quickbread_quickbread_1": "bread", "Quickbread_quickbread_2": "bread",
    "Quickbread_quickbread_3": "bread",
    "Cauliflower_new_cauliflower_3": "cream", "Cauliflower_new_cauliflower_4": "cream",
    "Cauliflower_new_cauliflower_5": "cream",
    "Sausages_new_sausage_2": "sausage", "Sausages_new_sausage_3": "sausage", "Sausages_new_sausage_5": "sausage",
    "Shrimp_shrimp_1": "shrimp", "Shrimp_shrimp_6": "shrimp",
    "Tortilla_Chips_new_chips_1": "chip", "Tortilla_Chips_new_chips_2": "chip", "Tortilla_Chips_new_chips_4": "chip",
    "Tortilla_Chips_tortilla_corn_chip_4": "chip",
}
CAST_STRENGTH = 1.0


def akey(r: dict) -> str:
    """Override key: '<category>_<object>' for scans, the id otherwise."""
    return f"{r['category']}_{r['object']}" if r["source"] == "metafood3d" else r["id"]


def match_pat(table: dict, key: str):
    if key in table:
        return table[key]
    for pat, v in table.items():
        if any(ch in pat for ch in "*[?") and fnmatch.fnmatchcase(key, pat):
            return v
    return None


# ---------------------------------------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------------------------------------
def _geo_one(r: dict):
    import trimesh
    if r["source"] != "metafood3d":
        return r["id"], None
    try:
        m = trimesh.load(r["path"], force="mesh", process=False)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        V = m.vertices
        lo, hi = np.percentile(V, 0.5, 0), np.percentile(V, 99.5, 0)
        o = {"ext": m.extents.tolist(), "rext": (hi - lo).tolist(), "hull_ml": float(m.convex_hull.volume * 1e6),
             "wt": bool(m.is_watertight)}
        lab = trimesh.graph.connected_components(m.edges_sorted, nodes=np.arange(len(V)), min_len=1)
        vl = np.zeros(len(V), int)
        for k, c in enumerate(lab):
            vl[c] = k
        cnt = np.bincount(vl[m.faces[:, 0]], minlength=len(lab))
        o["ncomp_big"] = int((cnt > 0.03 * len(m.faces)).sum())
        o["main_frac"] = float(cnt.max() / len(m.faces))
        return r["id"], o
    except Exception as e:  # noqa: BLE001
        return r["id"], {"err": str(e)}


def geometry(recs: list[dict], cache: Path, workers: int) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    from multiprocessing import Pool
    with Pool(workers) as p:
        out = dict(p.map(_geo_one, recs, chunksize=2))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def texture_path(obj_path: str) -> tuple[str | None, str | None]:
    """(texture file, mtl file) of an OBJ (first map_Kd)."""
    import re
    d = os.path.dirname(obj_path)
    mt = sorted(f for f in os.listdir(d) if f.endswith(".mtl"))
    if not mt:
        return None, None
    tex = None
    for line in open(os.path.join(d, mt[0]), errors="ignore"):
        m = re.match(r"\s*map_Kd\s+(.*)", line)
        if m:
            tex = os.path.join(d, m.group(1).strip())
            break
    return tex, os.path.join(d, mt[0])


def write_fixed_asset(r: dict, out_dir: Path, swap: bool, cast: str | None) -> str:
    """OUT/<id>/{symlink OBJ, MTL, corrected texture}; returns the new OBJ path."""
    from PIL import Image
    src_obj = Path(r["path"])
    tex, mtl = texture_path(r["path"])
    d = out_dir / r["id"]
    d.mkdir(parents=True, exist_ok=True)
    obj = d / src_obj.name
    if obj.is_symlink() or obj.exists():
        obj.unlink()
    obj.symlink_to(src_obj)
    shutil.copy(mtl, d / Path(mtl).name)
    a = np.asarray(Image.open(tex).convert("RGB")).astype(np.float32) / 255.0
    if swap:
        a = a[..., ::-1]
    if cast:
        lin = a ** 2.2
        fg = a.mean(2) > 0.06
        mu = lin[fg].mean(0)
        g = np.clip((np.asarray(CAST_TARGETS[cast]) ** 2.2) / np.maximum(mu, 1e-4), 0.3, 3.0)
        a = np.clip(lin * (1 + CAST_STRENGTH * (g - 1)), 0, 1) ** (1 / 2.2)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(d / Path(tex).name, quality=92)
    return str(obj)


# ---------------------------------------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------------------------------------
def render_thumbs(recs: list[dict], out_dir: Path, blender: str, n_proc: int = 8) -> None:
    todo = [dict(id=r["id"], path=r["path"], fmt=r["fmt"], color=r.get("color")) for r in recs
            if not (out_dir / f"{r['id']}.jpg").exists()]
    if not todo:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_jobs"
    tmp.mkdir(exist_ok=True)
    procs = []
    for k in range(n_proc):
        jp = tmp / f"jobs_{k}.json"
        jp.write_text(json.dumps(todo[k::n_proc]))
        procs.append(subprocess.Popen([blender, "-b", "--python", str(HERE / "asset_qa_bpy.py"), "--", str(jp),
                                       str(out_dir), "192"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------------------------------------
def audit(recs: list[dict], geo: dict, tex_dir: Path | None) -> list[dict]:
    out = [dict(r) for r in recs]
    food = [r for r in out if r.get("role") in (None, "food")]

    # category densities from consistent (mass, xlsx volume) pairs
    dens = defaultdict(list)
    for r in food:
        if r["source"] == "metafood3d" and r.get("mass_g") and r.get("ref_volume_ml", 0) > 1:
            dens[cat_key(r)].append(r["mass_g"] / r["ref_volume_ml"])
    med = {k: float(np.median(v)) for k, v in dens.items() if len(v) >= 3}
    all_med = float(np.median([x for v in dens.values() for x in v]))

    for r in out:
        r["qa_exclude"], r["qa_reason"], r["qa_flags"] = False, None, []
        if r.get("role") not in (None, "food"):
            continue
        key = akey(r)
        ck = cat_key(r)
        E = EXPECTED.get(ck)
        r["quality"] = "procedural" if r["source"] == "primitive" else "scan"
        r["kind"] = "procedural" if r["source"] == "primitive" else (E["kind"] if E else "item")
        r["container_baked"] = False
        if E is None:
            r["qa_flags"].append("no_expected_size")
            E = dict(lo=0, hi=1e9, kind="item", serve="plated", copies=1, ctx=None)
        r["expected_size_cm"] = [E["lo"], E["hi"]]
        serve, ctx, copies = E["serve"], E["ctx"], E["copies"]
        if key in PLATED_KIWI:
            serve, ctx = "plated", None
        ov = match_pat(SERVE_OVERRIDE, key)
        if ov:
            serve, ctx = ov
        if key.startswith("Carrot_") and not ov:
            serve, ctx = "plated", None
        r["serve"], r["whole_context"] = serve, ctx
        r["plate_ok"] = serve not in ("whole_produce", "platter")
        r["serving_copies"] = 1 if r["kind"] == "portion" or ov else copies

        g = geo.get(r["id"])
        # --- exclusions (visual verdicts first, then automatic geometry rules)
        ex = match_pat(EXCLUDE, key)
        if ex is None and g and "err" not in g and max(g["ext"]) / max(g["rext"]) > 1.9:
            ex = ("stray_geometry", f"floating geometry: bbox {max(g['ext']) * 100:.0f} cm vs body {max(g['rext']) * 100:.0f} cm")
        if "(" in str(r["category"]):
            ex = ex or (_CONT, "'(bowl)' scan: the bowl is part of the mesh")
        if ex:
            r["qa_exclude"], r["qa_reason"] = True, f"{ex[0]}: {ex[1]}"
            r["qa_flags"].append(ex[0])
            r["container_baked"] = ex[0] == _CONT
            continue

        # --- mass / volume sanity (scans)
        if r["source"] == "metafood3d" and g and "err" not in g:
            ref, mass, hull = r.get("ref_volume_ml"), r.get("mass_g"), g["hull_ml"]
            d0 = float(np.clip(med.get(ck, all_med), 0.4, 1.1))      # typical g/ml of this food (corrupt categories clipped)
            vol = r["volume_ml"]
            new_vol = None
            if vol <= 1 and mass:                                     # negative / ~0 xlsx volume
                new_vol = min(hull, mass / d0)
            elif ref and ref > 5 and ref > 1.5 * hull and r["volume_source"] != "mesh":   # larger than its own convex hull
                new_vol = min(hull, mass / d0) if mass else hull * 0.9
            elif mass and vol > 1 and not 0.06 < mass / vol < 1.8:   # impossible density: which of the two is wrong?
                sib = [x["mass_g"] for x in food if x is not r and cat_key(x) == ck and x.get("mass_g")]
                mass_ok = len(sib) >= 3 and 0.5 < mass / np.median(sib) < 2.0
                if mass_ok and mass / d0 <= hull:                      # mass agrees with its siblings, volume does not
                    new_vol = mass / d0
                else:
                    r["mass_g_orig"], r["mass_g"] = mass, round(vol * d0, 1)
                    r["qa_flags"].append("mass_fixed")
            if new_vol is not None:
                r["volume_ml_orig"] = vol
                r["volume_ml"] = float(new_vol)
                r["volume_source"] = "qa_hull_mass"
                r["qa_flags"].append("volume_fixed")

        # --- real-size check on the (possibly stray-free) max dimension
        size = max(r["extent_m"]) * 100
        lo, hi = E["lo"], E["hi"]
        if r["kind"] == "portion":
            if size > hi * 1.35:
                r["qa_exclude"], r["qa_reason"] = True, f"scale: portion {size:.0f} cm vs plausible <= {hi:g} cm"
                r["qa_flags"].append("scale")
                continue
        elif r["source"] != "primitive" and r["serve"] != "platter":      # centrepieces keep their (large) size
            s = None
            if size > hi * 1.1:
                s = hi / size
            elif size < lo * 0.85 and lo > 0:
                s = lo / size
            if s is not None:
                if 0.55 <= s <= 1.8:
                    r["to_m_orig"] = r["to_m"]
                    r["to_m"] = r["to_m"] * s
                    r["qa_scale"] = round(s, 4)
                    r["extent_m"] = [x * s for x in r["extent_m"]]
                    r["volume_ml_orig"] = r.get("volume_ml_orig", r["volume_ml"])
                    r["volume_ml"] = r["volume_ml"] * s ** 3
                    if r.get("mass_g"):
                        r["mass_g_orig"] = r.get("mass_g_orig", r["mass_g"])
                        r["mass_g"] = round(r["mass_g"] * s ** 3, 1)
                    r["qa_flags"].append("rescaled")
                else:
                    r["qa_exclude"], r["qa_reason"] = True, f"scale: {size:.1f} cm vs plausible {lo:g}-{hi:g} cm (x{s:.2f} too far)"
                    r["qa_flags"].append("scale")
                    continue

        # --- texture fixes
        swap, cast = key in SWAP_FIX, CAST_FIX.get(key)
        if (swap or cast) and tex_dir is not None:
            r["tex_source"] = r["path"]
            r["path"] = write_fixed_asset(r, tex_dir, swap, cast)
            r["tex_fix"] = "swap_rb" if swap else f"cast:{cast}"
            r["qa_flags"].append("tex_fixed")
        if r["serve"] in ("whole_produce", "platter"):
            r["qa_flags"].append(r["serve"])
        if r["quality"] == "procedural":
            r["qa_flags"].append("procedural")
    return out


# ---------------------------------------------------------------------------------------------------------
# contact sheets + report
# ---------------------------------------------------------------------------------------------------------
FLAG_CODE = {"container": "BOX", "packaging": "PKG", "stray_geometry": "STRAY", "bad_mesh": "MESH", "bad_texture": "TEX",
             "odd_colour": "COL", "cluster": "CLUS", "scale": "SIZE", "rescaled": "RS", "tex_fixed": "FIX",
             "volume_fixed": "VOL", "mass_fixed": "MASS", "whole_produce": "WHOLE", "platter": "PLAT",
             "procedural": "PROC"}


def contact_sheets(recs: list[dict], thumbs: Path, thumbs_fixed: Path | None, out_dir: Path) -> list[Path]:
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = sorted((r for r in recs if r.get("role") in (None, "food")),
                   key=lambda r: (str(r["category"]).lower(), r["id"]))
    W, TH, cols, per = 148, 148, 11, 88
    paths = []
    for si in range(0, len(cells), per):
        chunk = cells[si:si + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * W, rows * (TH + 24)), (28, 28, 30))
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            x, y = (k % cols) * W, (k // cols) * (TH + 24)
            fp = (thumbs_fixed / f"{r['id']}.jpg") if thumbs_fixed and r.get("tex_fix") else None
            tp = fp if fp is not None and fp.exists() else thumbs / f"{r['id']}.jpg"
            try:
                im = Image.open(tp).convert("RGB").resize((TH - 4, TH - 4))
            except Exception:  # noqa: BLE001
                im = Image.new("RGB", (TH - 4, TH - 4), (90, 0, 90))
            col = (200, 40, 40) if r["qa_exclude"] else ((235, 150, 30) if r["qa_flags"] and any(
                f in r["qa_flags"] for f in ("rescaled", "tex_fixed", "volume_fixed", "mass_fixed")) else (50, 170, 70))
            if r.get("serve") in ("whole_produce", "platter") and not r["qa_exclude"]:
                col = (70, 120, 230)
            dr.rectangle([x, y, x + W - 2, y + TH - 2], outline=col, width=2)
            sheet.paste(im, (x + 2, y + 2))
            nm = f"{str(r['category'])[:11]} {str(r.get('object') or r['id'])[-9:]}"
            dr.text((x + 2, y + TH), nm, fill=(255, 255, 0))
            codes = [FLAG_CODE[f] for f in r["qa_flags"] if f in FLAG_CODE]
            txt = f"{max(r['extent_m']) * 100:.1f}cm " + (f"x{r['qa_scale']:.2f} " if r.get("qa_scale") else "") + " ".join(codes[:3])
            dr.text((x + 2, y + TH + 11), txt[:26], fill=(255, 255, 255) if not r["qa_exclude"] else (255, 120, 120))
        p = out_dir / f"contact_{si // per:02d}.jpg"
        sheet.save(p, quality=82)
        paths.append(p)
    return paths


def report(recs: list[dict]) -> tuple[str, dict]:
    food = [r for r in recs if r.get("role") in (None, "food")]
    usable = [r for r in food if not r["qa_exclude"]]
    fl = Counter(f for r in food for f in r["qa_flags"])
    reasons = Counter(r["qa_reason"].split(":")[0] for r in food if r["qa_exclude"])
    serve = Counter(r["serve"] for r in usable)
    per_cat = defaultdict(lambda: [0, 0])
    for r in food:
        per_cat[str(r["category"])][0 if not r["qa_exclude"] else 1] += 1
    lines = [f"food assets: {len(food)}  usable: {len(usable)}  excluded: {len(food) - len(usable)}",
             f"usable by serve: {dict(serve)}",
             f"usable plate_ok: {sum(r['plate_ok'] for r in usable)}  whole_produce/platter only: {sum(not r['plate_ok'] for r in usable)}",
             f"exclusion reasons: {dict(reasons)}", f"flag counts: {dict(fl)}",
             f"categories with >=1 usable: {sum(1 for v in per_cat.values() if v[0])} / {len(per_cat)}",
             f"categories with 0 usable (lost): {sorted(k for k, v in per_cat.items() if not v[0])}", "",
             "rescaled: " + ", ".join(f"{akey(r)} x{r['qa_scale']}" for r in food if r.get("qa_scale")), "",
             "texture fixed: " + ", ".join(f"{akey(r)}[{r['tex_fix']}]" for r in food if r.get("tex_fix")), "",
             "mass_fixed: " + ", ".join(f"{akey(r)} {r['mass_g_orig']}->{r['mass_g']}" for r in food if "mass_fixed" in r["qa_flags"]), "",
             "volume_fixed: " + ", ".join(f"{akey(r)} {r['volume_ml_orig']:.0f}->{r['volume_ml']:.0f}" for r in food if "volume_fixed" in r["qa_flags"]),
             "", "per category (usable / excluded):"]
    for k in sorted(per_cat, key=str.lower):
        lines.append(f"  {k:32s} {per_cat[k][0]:3d} / {per_cat[k][1]:3d}")
    data = dict(food=len(food), usable=len(usable), flags=dict(fl), reasons=dict(reasons), serve=dict(serve),
                per_category={k: dict(usable=v[0], excluded=v[1]) for k, v in per_cat.items()})
    return "\n".join(lines), data


# ---------------------------------------------------------------------------------------------------------
# Objaverse photogrammetry scans (synth.fetch_assets --sources objaverse) -> assets_v5.jsonl
#
#   python -m synth.asset_qa --objaverse --in ../data/synth/assets_v4.jsonl --out ../data/synth/assets_v5.jsonl \
#       --obj-sources ../data/datasets/objaverse/sources.jsonl --qa-dir ../data/synth/qa_v5 [--no-thumbs]
#
# v5 = v4 (rows copied unchanged) + one row per downloaded Objaverse model.  Objaverse models are NOT metric:
# the scale comes from OBJ_EXPECTED_TXT (typical size of ONE served item of the category; the model's largest
# extent is set to the geometric middle of [lo, hi]) -> `to_m`, `scale_source: "category_norm"`.  The category is
# decided from the model's name / description (NAME_RULES; LVIS tags are far too noisy: "zucchini" contains rice,
# "radish" a pomegranate), then every thumbnail was inspected and the verdicts are in OBJ_VERDICT.
# glTF is Y up: extent_m is in Blender's Z-up order, `fmt` is "gltf" (also for .glb files, so that
# synth.collision / synth.mpm apply the Y-up -> Z-up conversion; render.load_gltf and Blender read .glb by suffix).
# ---------------------------------------------------------------------------------------------------------
# key | lo_cm | hi_cm | kind | serve | copies [| ctx]   (overrides / additions to EXPECTED for Objaverse assets)
OBJ_EXPECTED_TXT = """
apricot|4|5.5|I|whole_produce|1|tray
mandarin|5|7.5|I|whole_produce|1|tray
grapefruit|9|12.5|I|whole_produce|1|bowl
fig|4|6.5|I|whole_produce|1|tray
pineapple whole|24|32|I|whole_produce|1|bowl
watermelon|22|32|I|whole_produce|1|bowl
melon|12|18|I|whole_produce|1|bowl
papaya whole|14|25|I|whole_produce|1|bowl
mango slice|7|12|I|plated|2
kiwi slice|4.5|7|I|plated|3
lime slice|4|6|I|plated|3
lemon slice|5|8|I|plated|3
grapefruit slice|8|11|I|plated|2
coconut|11|15|I|whole_produce|1|bowl
walnut|2.8|4|I|snack|5
potato raw|5|10|I|whole_produce|1|bowl
sweet potato raw|10|18|I|whole_produce|1|bowl
cucumber whole|15|22|I|whole_produce|1|bowl
zucchini whole|15|22|I|whole_produce|1|bowl
butternut squash|14|22|I|whole_produce|1|bowl
carrot whole|14|20|I|whole_produce|1|bowl
broccoli head|9|14|I|whole_produce|1|bowl
cauliflower head|11|16|I|whole_produce|1|bowl
cabbage head|15|22|I|whole_produce|1|bowl
artichoke|8|12|I|whole_produce|1|bowl
bell pepper whole|8|12|I|whole_produce|1|bowl
chili pepper|6|12|I|whole_produce|1|bowl
eggplant|14|22|I|whole_produce|1|bowl
mushroom|4|9|I|whole_produce|1|bowl
corn|15|22|I|plated|1
beet|6|9|I|whole_produce|1|bowl
radish|7|12|I|whole_produce|1|bowl
baguette|35|50|I|platter|1
bread loaf|17|26|I|platter|1
bread slice|9|12|I|plated|2
pain au chocolat|9|13|I|plated|1
pretzel|11|16|I|plated|1
flatbread|16|26|I|plated|1
pastry|7|13|I|dessert|1
cupcake|6|9|I|dessert|1
cream puff|5|8|I|dessert|2
whole cake|16|24|I|platter|1
whole pie|20|27|I|platter|1
whole pizza|26|35|I|platter|1
cracker|8|15|I|snack|2
spring roll|9|16|I|plated|1
wrap|15|24|I|plated|1
sub sandwich|15|25|I|plated|1
toast|11|15|I|plated|1
grilled fish|12|28|I|plated|1
"""
OBJ_EXPECTED = {**EXPECTED, **_parse_expected(OBJ_EXPECTED_TXT)}

# (regex on the lower-cased model name -> (category in our naming, key into OBJ_EXPECTED)); first match wins, dishes
# before fruit ("lemon meringue pie" is a pie).  A second pass tries the description and tags with the same rules.
_R = [
    (r"sushi|nigiri|maki\b|golden roll", "sushi"),
    (r"pain au chocolat", "pain au chocolat"),
    (r"croissant|curason", "croissant"),
    (r"baguet", "baguette"),
    (r"pretzel|brezel|bretzel", "pretzel"),
    (r"bagel", "bagel"),
    (r"pizza", "whole pizza"),
    (r"slice of sourdough|sliced bread$|bread slice", "bread slice"),
    (r"melonpan|メロンパン|danish|cinnamon swirl|bebe pastry|meat bun|macaron|\bpastry\b", "pastry"),
    (r"roll\b|bun\b|semmel|br[oö]tchen|mohnzopf", "bread roll"),
    (r"challah|bread|loaf", "bread loaf"),
    (r"muffin", "muffin"), (r"cupcake", "cupcake"), (r"cream puff", "cream puff"),
    (r"doughnut|donut|oliebol", "doughnut"),
    (r"brownie", "brownies"),
    (r"crispbread|cracker", "cracker"), (r"cookie|biscuit", "cookies"),
    (r"tiramisu|gateau|gâteau|ガトーショコラ|piece of cake|cake slice", "cake slice"),
    (r"pie\b|tart\b|tarte|galette|pistachio cake|pistache|ピスターシュ", "whole pie"),
    (r"cake|kuchen|colomba|shortcake", "whole cake"),
    (r"waffle|whaffle", "waffle"), (r"pancake|panqueca|crepe", "pancake"), (r"omelet|omelette", "omelet"),
    (r"taco", "taco"), (r"quesadilla", "quesadilla"), (r"burrito", "burrito"),
    (r"kebab|shawarma|wrap\b", "wrap"), (r"burger|big mac", "burger"),
    (r"toast hawaii", "toast"), (r"sandwich|turkey melt|\bsub\b", "sandwich"),
    (r"spring roll|egg roll", "spring roll"), (r"pita|flatbread", "flatbread"),
    (r"spaghetti|pasta", "pasta mixed dishes"),
    (r"salmon|sardine|trout", "grilled fish"), (r"steak|boeuf|beef", "steak"), (r"sausage|kielbasa", "sausages"),
    (r"chicken", "whole chicken"), (r"crab", "crab"), (r"egg\b", "egg"), (r"chocolate", "chocolate"),
    (r"grapefruit|citrus maxima|paleo", "grapefruit"),
    (r"mikan|mandarin|tangerine|clementine|iyokan|dekopon|ponkan|ポンカン|amanatsu|dekokponn|siranuhi|sumo", "mandarin"),
    (r"orange|naranja", "orange"), (r"lemon|limone|lim[aã]o|lemmon", "lemon"), (r"lime\b", "lime"),
    (r"pineapple|ananas|piña|pina!", "pineapple"), (r"apricot", "apricot"),
    (r"peach|nectarine|nektarin|saucer", "peach"), (r"apple", "apple"),
    (r"pear|pera\b|la france|ラ・フランス|kosui|poire|truval", "pear"),
    (r"kiwi", "kiwi"), (r"\bfig\b", "fig"), (r"avocado|palta|\bavo\b|avacado", "avocado"), (r"banan", "banana"),
    (r"mango", "mango"), (r"papaya", "papaya"), (r"strawberr|ichigo|fraise", "strawberry"),
    (r"grape|muscat", "grapes"), (r"pomegran", "pomegranate"), (r"watermelon|suika|water melon", "watermelon"),
    (r"melon", "melon"), (r"coconut", "coconut"), (r"walnut|nuez", "walnut"),
    (r"sweet potato|yam\b|yakiimo|batata", "sweet potato"), (r"potato", "potato"), (r"onion|zwiebel", "onion"),
    (r"tomato", "tomato"), (r"cucumber|pepino", "cucumber"), (r"zucchini|squash", "zucchini"),
    (r"butternut", "butternut squash"), (r"carrot", "carrot"), (r"broccoli|romanesco|broccolo", "broccoli"),
    (r"cauliflower|culiflower|カリフラワー", "cauliflower"), (r"cabbage|kohl|kochsalat", "cabbage"),
    (r"artichoke", "artichoke"), (r"paprika|bell pepper|capsicum|sweet long pepper|green pepper", "bell pepper"),
    (r"chil+i|fresno|red pepper", "chili pepper"), (r"eggplant|aubergine", "eggplant"),
    (r"mushroom|bolete|boletus", "mushroom"), (r"corn|milho|maize", "corn"),
    (r"beterraba|beet", "beet"), (r"radish", "radish"),
]
NAME_RULES = [(re.compile(p, re.I), c) for p, c in _R]
# category -> key into OBJ_EXPECTED (default: the category itself); slices / halves by name
NORM_KEY = {"pineapple": "pineapple whole", "papaya": "papaya whole", "potato": "potato raw",
            "sweet potato": "sweet potato raw", "cucumber": "cucumber whole", "zucchini": "zucchini whole",
            "carrot": "carrot whole", "broccoli": "broccoli head", "cauliflower": "cauliflower head",
            "cabbage": "cabbage head", "bell pepper": "bell pepper whole"}
SLICE_RE = re.compile(r"slice|half|halves|cut |sliced", re.I)
SLICEABLE = {"kiwi": "kiwi slice", "lime": "lime slice", "lemon": "lemon slice", "mango": "mango slice",
             "grapefruit": "grapefruit slice", "whole pizza": "pizza", "whole pie": "pie", "whole cake": "cake slice"}
# models that are not food / not a single plateable item, from the name alone (before any rule)
NOT_FOOD = re.compile(
    r"\b(rock|stone|fossil|gorget|bead|brick|towel|hummer|doggie|nokia|flower|coral|vase|jar|campbell|tropicana|"
    r"nisshin|raoh|can|bowl|plate|molde|alabastron|statue|sculpture|candle|pillow|squeaky|toy|pysanka|easter|auk|"
    r"merganser|agaric|seedpod|pod|spider|snail|tesia|piranha|leaves|leaf|cactus|acorn|celery|tree|hand|echinoid|"
    r"clypeaster|eupatagus|jicama|ceramic|truck|seed|custard|rotten|crumbled|carved|failed?|roast pig|christmas fruit|"
    r"raisins?|comparison|pot|tower|decorative|zipper|ear1|lars|cuisine|bento|duomo|natto)\b"
    r"|\(fail\)|bad photoscan|low ?poly|low tria|lowpoli", re.I)

FLAT_CATS = {"pizza", "whole pizza", "bread slice", "cookies", "waffle", "pancake", "whole pie", "cracker", "flatbread",
             "omelet", "whole cake", "cake slice", "pastry", "wrap", "toast", "doughnut", "mango slice", "kiwi slice",
             "lime slice", "lemon slice", "grapefruit slice", "sandwich"}
LOWPOLY_FACES = 1500         # below this: retopologised game-ready / stylised, not a scan
MIN_FACES_NOTE = 5000         # below: flagged `lowish` only

# visual verdicts from the thumbnails.  uid prefix (8 chars) -> category | "!code: reason" (exclude) | "~category"
# (keep, but the name rules gave a wrong / no category).  Filled in by hand after viewing qa_v5/sheets.
OBJ_VERDICT: dict[str, str] = {
    "16c995f4": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "3a55aabc": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "0ee4f66b": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "449c75ac": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "effa9692": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "af514ae1": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "b6dd89da": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "dcefb9b9": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e3fbea5e": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e8bfd53d": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "f7fdd802": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "90769b46": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "acf21b61": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "da0f3082": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "33d98be6": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "260a3301": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "c1137353": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "74bd33d4": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "a922e2c7": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "271f05a2": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "0e013d28": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "4b2990b0": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e7d3f8fd": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "25f6eb11": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "108fbe18": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "9753fcce": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "b5709832": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "89f8295a": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "0a741a4d": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "f959f5e9": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "26225729": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "3916a122": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "0e6df6c6": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "260d3873": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "6ec3ad35": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "d5b66f6c": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "fa508767": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "29832125": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "077d22b7": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "4f0dc7ef": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "8334ff04": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "feb91b2c": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "611af31c": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "192b9d5b": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "a1b7c76f": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "af3cc997": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "ecc33fdd": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "5b42cc3b": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "7d1cac05": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "7fc9800c": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "8327c239": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "9b2530fa": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "b43ef95c": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "b4a41eae": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "bb3f527e": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "c01b7abb": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "cfcd4baf": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e9044276": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "f76a0a6c": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "6fcaca01": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "09af8624": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "0e21faea": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "523b1fa6": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "c4423e51": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e0b03a1b": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "b3a8b0a9": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e9fa19e6": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "1d3ed42e": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "7aeeaa23": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "b91d421a": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "848ad9fa": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "d5afc8ea": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "f6ef4190": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "8dc2b2b3": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "e27442a7": "!container: plate / board / bowl / tray / mat / wrapping paper baked into the scan",
    "dd1fbf71": "!packaging: sandwich wrapper baked in",
    "507ddc56": "!packaging: Campbell's can",
    "0400821f": "!packaging: retail pack",
    "d3b46595": "!not_food: succulent plant",
    "0a1ee08e": "!not_food: decorative artichoke statue",
    "46f65621": "!not_food: sandstone",
    "53a51432": "!not_food: crab specimen",
    "fb448aab": "!not_food: crab specimen",
    "fdebdabc": "!not_food: crab specimen",
    "48f5e505": "!bad_mesh: messy fragments",
    "dc7d087c": "!bad_mesh: broken render",
    "9eb45ccb": "!damaged: bitten apple",
    "4b88c020": "!bad_mesh: shapeless lump",
    "5907e9c9": "!bad_mesh: stick-like fragment",
    "aa7e5d9e": "!novelty: drawn-on face",
    "bccf270c": "!novelty: drawn-on face",
    "3b73d7d5": "!cluster: two halves in one mesh",
    "56b71e48": "!damaged: lemon with a hole",
    "bb650241": "!damaged: deformed lemon",
    "b75c5534": "!bad_mesh: hand in the scan",
    "c384f15c": "!cluster: two pears",
    "d3f5697b": "!not_plateable: bitter melon half",
    "dc883f9b": "!bad_mesh: messy muffin",
    "550f0b8f": "!no_category: pumpkin",
    "62954f4d": "!no_category: pumpkin",
    "5809e2f6": "!scale: mini pineapple",
    "fbe5dfc5": "!odd_colour: near-black pineapple",
    "4f629ada": "!odd_colour: grey corn",
    "fda2d273": "!odd_colour: near-black croissant",
    "c2e916b6": "!odd_colour: dark brown roll",
    "5ca6da60": "!bad_mesh: shapeless patty",
    "b9f516c5": "!bad_mesh: partial scan, torn edges",
    "9fe0f5f3": "!odd_colour: pale green roll",
    "a00977e4": "!novelty: decorated bunny cupcake",
    "7b81d735": "!not_plateable: watermelon half",
    "3c718d25": "!not_plateable: squash half",
    "bfdc0641": "!not_plateable: squash half",
    "105da0b0": "!cluster: fruit pack",
    "ab86d68d": "!unclear: cut bun / orange?",
    "8e7f843c": "!damaged: shrivelled avocado",
    "4dcb9fcf": "~flatbread",
    "5846b28f": "~flatbread",
    "db4af5ff": "~flatbread",
    "6ec22c4c": "~cracker",
    "4e5abbed": "~lime",
    "7f64d677": "~lime",
    "5adb70c1": "~lemon slice",
    "d9c7f3d0": "~watermelon",
    "0ac51fc2": "~kiwi slice",
    "16d913f9": "~kiwi slice",
    "a075f9b3": "~sub sandwich",
    "ae4982aa": "~cake slice",
    "c96ec380": "~cake slice",
    "b4fb5eca": "~pastry",
    "c65f5d6c": "~pastry",
    "3be54641": "~pastry",
    "af77f3a0": "~pastry",
    "e20c3851": "~pastry",
    "c276f133": "~whole pie",
    "1897a2f4": "~bread roll",
    "2656ffb5": "~butternut squash",
    "2c0a2a23": "~orange",
    "459556bb": "~corn",
    "a2650310": "~sandwich",
    "c0877c26": "~strawberry",
    "c7aa636f": "~banana",
    "d8400af9": "~cabbage", "cfe9b27e": "!container: plate / stand baked in", "5fb8bc0a": "~whole pie",
    "efb5f769": "~chili pepper",
    "79af8a0b": "~mandarin",
}


def glb_meta(path: str) -> dict:
    """Cheap GLB header parse: texture / vertex-colour / extension info."""
    import struct
    with open(path, "rb") as f:
        head = f.read(20)
        if head[:4] != b"glTF":
            return {"err": "not glb"}
        ln, kind = struct.unpack("<II", head[12:20])
        js = json.loads(f.read(ln))
    prims = [p for m in js.get("meshes", []) for p in m.get("primitives", [])]
    # base colour map: metallic-roughness baseColorTexture or (older Sketchfab exports) specular-glossiness diffuseTexture
    base = [m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
            or m.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {}).get("diffuseTexture")
            for m in js.get("materials", [])]
    img = [i for i in js.get("images", [])]
    return {"n_prims": len(prims), "n_mat": len(js.get("materials", [])), "n_base_tex": sum(b is not None for b in base),
            "n_images": len(img), "vcol": any("COLOR_0" in p.get("attributes", {}) for p in prims),
            "ext": js.get("extensionsUsed", []), "n_nodes": len(js.get("nodes", []))}


def _geo_obj(job):
    """Geometry of one GLB in Blender's Z-up frame (glTF x, y, z -> x, -z, y), raw (unscaled) units."""
    uid, path = job
    import trimesh
    from scipy import ndimage
    o: dict = {"meta": glb_meta(path)}
    try:
        sc = trimesh.load(path, force="scene", skip_materials=True)
        parts = sc.dump(concatenate=False) if hasattr(sc, "dump") else [sc]
        Vs, Fs, off = [], [], 0
        for p in parts:
            if not hasattr(p, "faces") or len(p.faces) == 0:         # point clouds / line paths in the scene
                continue
            Vs.append(np.asarray(p.vertices, np.float64))
            Fs.append(np.asarray(p.faces, np.int64) + off)
            off += len(p.vertices)
        if not Vs:
            raise ValueError("no triangle mesh (point cloud only)")
        V = np.concatenate(Vs)
        V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
        m = trimesh.Trimesh(V, np.concatenate(Fs), process=False)
        o["n_faces"] = int(len(m.faces))
        m.merge_vertices()
        lo, hi = np.percentile(m.vertices, 0.5, 0), np.percentile(m.vertices, 99.5, 0)
        o["ext"] = m.extents.tolist()
        o["rext"] = (hi - lo).tolist()
        o["hull"] = float(m.convex_hull.volume)
        o["wt"] = bool(m.is_watertight)
        vol, src = None, None
        if o["wt"]:
            vol, src = abs(float(m.volume)), "mesh"
        else:
            m2 = m.copy()
            trimesh.repair.fill_holes(m2)
            if m2.is_watertight:
                vol, src = abs(float(m2.volume)), "mesh_filled"
        if vol is None or vol < 0.2 * o["hull"]:           # open scans (cut off at the turntable / table)
            mm = m
            if len(mm.faces) > 200000:
                mm = mm.simplify_quadric_decimation(face_count=150000)
            pitch = float(max(mm.extents)) / 96
            vox = mm.voxelized(pitch).fill()
            M = vox.matrix
            v3 = float(M.sum()) * pitch ** 3
            filled = np.maximum.accumulate(M, axis=2) & np.maximum.accumulate(M[:, :, ::-1], axis=2)[:, :, ::-1]
            vcol = float(filled.sum()) * pitch ** 3
            if vol is None or v3 > vol:
                vol, src = v3, "voxel_fill"
            if vol < 0.35 * o["hull"]:
                vol, src = vcol, "voxel_column"
        o["vol"], o["vol_src"] = vol, src
        lab = trimesh.graph.connected_components(m.edges_sorted, nodes=np.arange(len(m.vertices)), min_len=1)
        vl = np.zeros(len(m.vertices), int)
        for k, c in enumerate(lab):
            vl[c] = k
        cnt = np.bincount(vl[m.faces[:, 0]], minlength=len(lab))
        o["ncomp_big"] = int((cnt > 0.03 * len(m.faces)).sum())
        o["main_frac"] = float(cnt.max() / len(m.faces))
    except Exception as e:  # noqa: BLE001
        o["err"] = f"{type(e).__name__}: {e}"
    return uid, o


def geometry_obj(recs: list[dict], cache: Path, workers: int) -> dict:
    cur = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted((r for r in recs if r["id"] not in cur), key=lambda r: -(r.get("glb_bytes") or 0))
    if todo:
        from multiprocessing import Pool
        with Pool(workers, maxtasksperchild=1) as p:
            for k, (uid, o) in enumerate(p.imap_unordered(_geo_obj, [(r["id"], r["path"]) for r in todo])):
                cur[uid] = o
                if k % 20 == 19:
                    cache.write_text(json.dumps(cur))
                    print(f"geometry {k + 1}/{len(todo)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cur))
    return cur


def obj_category(r: dict) -> tuple[str | None, str | None]:
    """(category, why-not): category from the name rules, else description / tags, else None."""
    name = str(r.get("name") or "")
    if NOT_FOOD.search(name):
        return None, "not_food: " + name[:40]
    for rx, c in NAME_RULES:
        if rx.search(name):
            return c, None
    extra = " ".join([str(r.get("description") or "")[:200]] + list(r.get("tags") or []))
    for rx, c in NAME_RULES:
        if rx.search(extra) and not NOT_FOOD.search(extra):
            return c, None
    return None, "unrecognised_name"


def reorient_flat(path: str, uid: str, ext_zup: np.ndarray, out_dir: Path) -> str | None:
    """A clearly flat food (disc / sheet) whose thin axis is not up in Blender's frame: write a copy of the GLB
    rotated so that it lies flat (glTF is Y up; Blender x = gX, y = -gZ, z = gY).  Returns the new path or None."""
    import trimesh
    t = int(np.argmin(ext_zup))
    if t == 2 or ext_zup[t] > 0.5 * np.delete(ext_zup, t).min():
        return None
    T = trimesh.transformations.rotation_matrix(-np.pi / 2 if t == 1 else np.pi / 2, [1, 0, 0] if t == 1 else [0, 0, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{uid}.glb"
    if not dst.exists():
        sc = trimesh.load(path, force="scene")
        sc.apply_transform(T)
        dst.write_bytes(sc.export(file_type="glb"))
    return str(dst.resolve())


def audit_obj(srcs: list[dict], geo: dict, reorient_dir: Path | None = None) -> list[dict]:
    out = []
    for s in srcs:
        uid = s["id"]
        g = geo.get(uid) or {}
        key8 = uid[:8]
        raw_name = s.get("name")
        verdict = OBJ_VERDICT.get(key8)
        cat, why = obj_category(s)
        if verdict and verdict.startswith("~"):
            cat, why = verdict[1:], None
        elif verdict and not verdict.startswith("!"):
            cat, why = verdict, None
        flags: list[str] = []
        ex = None                                              # (code, reason)
        if verdict and verdict.startswith("!"):
            code, _, reason = verdict[1:].partition(":")
            ex = (code.strip(), reason.strip())
        elif why:
            ex = (why.split(":")[0], why)
        if ex is None and (g.get("err") or "ext" not in g):
            ex = ("bad_mesh", g.get("err", "no geometry"))
        meta = glb_meta(s["path"])
        nf = s.get("face_count") or g.get("n_faces") or 0
        if ex is None and nf < LOWPOLY_FACES:
            ex = ("lowpoly", f"{nf} faces (retopologised / stylised)")
        if ex is None and nf < MIN_FACES_NOTE:
            flags.append("lowish")
        if ex is None and meta.get("n_base_tex", 0) == 0:
            ex = ("untextured", "no base-colour texture" + (" (vertex colours only)" if meta.get("vcol") else ""))
        if ex is None and g["ncomp_big"] > 3 and g["main_frac"] < 0.6:
            ex = ("cluster", f"{g['ncomp_big']} separate big pieces")
        if ex is None and max(g["ext"]) / max(max(g["rext"]), 1e-9) > 1.9:
            ex = ("stray_geometry", "floating geometry far from the body")

        r = dict(id=f"objaverse_{uid}", source="objaverse", name=cat or str(raw_name), category=cat or "unknown",
                 object=uid, path=s["path"], fmt="gltf", role="food", procedural=False, generated=False, color=None,
                 mass_g=None, container_shaped=False, up_axis="Z", licence=s["licence"], attribution=s["attribution"],
                 url=s["url"], uid=uid, lvis_category=s.get("lvis_category"), raw_name=raw_name,
                 description=(s.get("description") or "")[:200], n_faces=int(nf), kind="item", quality="scan")
        r["qa_exclude"], r["qa_reason"], r["qa_flags"] = False, None, flags
        if "ext" in g:
            ext_raw = np.asarray(g["ext"])
            body = np.asarray(g["rext"])                        # 0.5 .. 99.5 % extent: the body without strays
            V = g["vol"]
            if cat == "bread loaf" and sorted(body)[2] > 3.0 * max(sorted(body)[1], 1e-9):
                cat = r["category"] = r["name"] = "baguette"          # long thin loaf
            key = SLICE_RE.search(str(raw_name)) and cat in SLICEABLE and SLICEABLE[cat] or cat
            if key != cat:
                r["category"] = r["name"] = key
            nk = NORM_KEY.get(key, key)
            E = OBJ_EXPECTED.get(nk) or (OBJ_EXPECTED.get(r["category"]))
            if E is None:
                E = dict(lo=0, hi=1e9, kind="item", serve="plated", copies=1, ctx=None)
                if ex is None:
                    ex = ("no_norm", f"no size norm for {r['category']!r}")
            size_units = float(max(body) if max(ext_raw) / max(max(body), 1e-9) > 1.25 else max(ext_raw))
            target_cm = float(np.sqrt(E["lo"] * E["hi"])) if E["hi"] < 1e8 else 0.0
            to_m = target_cm / 100 / size_units if size_units > 0 else 1.0
            r.update(to_m=to_m, scale_source="category_norm", size_target_cm=round(target_cm, 2),
                     extent_m=(ext_raw * to_m).tolist(), volume_ml=float(V * to_m ** 3 * 1e6),
                     volume_source=g["vol_src"], volume_ok=True, watertight=bool(g["wt"]),
                     hull_ml=float(g["hull"] * to_m ** 3 * 1e6), raw_extent_units=ext_raw.tolist(),
                     raw_max_units=size_units, expected_size_cm=[E["lo"], E["hi"]], norm_key=nk, kind=E["kind"],
                     serve=E["serve"], whole_context=E["ctx"], plate_ok=E["serve"] not in ("whole_produce", "platter"),
                     serving_copies=1 if E["kind"] == "portion" else E["copies"])
            if g["vol_src"] not in ("mesh", "mesh_filled"):
                flags.append("volume_" + g["vol_src"])
            if V < 0.12 * g["hull"]:
                flags.append("thin_shell")
            if r["category"] in FLAT_CATS and ext_raw[2] > 0.75 * min(ext_raw[0], ext_raw[1]):
                flags.append("tilted")                      # a flat food (pizza, slice of bread ...) not lying flat in the file
                if not r["qa_exclude"] and ex is None and reorient_dir is not None:
                    fixed = reorient_flat(s["path"], uid, ext_raw, reorient_dir)
                    if fixed:
                        r["path_orig"], r["path"] = s["path"], fixed
                        r["extent_m"] = [r["extent_m"][i] for i in (0, 2, 1)] if np.argmin(ext_raw) == 1 else \
                            [r["extent_m"][i] for i in (2, 1, 0)]
                        flags.append("reoriented")
            if E["serve"] in ("whole_produce", "platter"):
                flags.append(E["serve"])
        else:
            r.update(to_m=1.0, scale_source="none", extent_m=[0.0, 0.0, 0.0], volume_ml=0.0, volume_source="none",
                     volume_ok=False, watertight=False, expected_size_cm=[0, 0], serve="plated", whole_context=None,
                     plate_ok=True, serving_copies=1)
        if ex:
            r["qa_exclude"], r["qa_reason"] = True, f"{ex[0]}: {ex[1]}"
            flags.append(ex[0])
        out.append(r)
    return out


def contact_sheets_obj(recs: list[dict], thumbs: Path, out_dir: Path, per: int = 60, cols: int = 10) -> list[Path]:
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = sorted(recs, key=lambda r: (str(r["category"]), r["uid"]))
    W, TH = 164, 164
    paths = []
    for si in range(0, len(cells), per):
        chunk = cells[si:si + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * W, rows * (TH + 34)), (28, 28, 30))
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            x, y = (k % cols) * W, (k // cols) * (TH + 34)
            try:
                im = Image.open(thumbs / f"{r['id']}.jpg").convert("RGB").resize((TH - 4, TH - 4))
            except Exception:  # noqa: BLE001
                im = Image.new("RGB", (TH - 4, TH - 4), (90, 0, 90))
            col = (200, 40, 40) if r["qa_exclude"] else ((70, 120, 230) if r["serve"] in ("whole_produce", "platter")
                                                         else (50, 170, 70))
            dr.rectangle([x, y, x + W - 2, y + TH - 2], outline=col, width=2)
            sheet.paste(im, (x + 2, y + 2))
            dr.text((x + 2, y + TH), f"{r['uid'][:8]} {str(r['category'])[:14]}", fill=(255, 255, 0))
            dr.text((x + 2, y + TH + 11), f"{max(r['extent_m']) * 100:.0f}cm f={r['n_faces'] // 1000}k " +
                    " ".join(f for f in r["qa_flags"][:2]), fill=(255, 255, 255) if not r["qa_exclude"] else (255, 120, 120))
            dr.text((x + 2, y + TH + 22), str(r["raw_name"])[:26], fill=(170, 200, 255))
        p = out_dir / f"obj_{si // per:02d}.jpg"
        sheet.save(p, quality=84)
        paths.append(p)
    return paths


def report_obj(recs: list[dict]) -> str:
    use = [r for r in recs if not r["qa_exclude"]]
    reasons = Counter(r["qa_reason"].split(":")[0] for r in recs if r["qa_exclude"])
    per = defaultdict(lambda: [0, 0])
    for r in recs:
        per[r["category"]][1 if r["qa_exclude"] else 0] += 1
    lines = [f"objaverse models: {len(recs)}  kept: {len(use)}  excluded: {len(recs) - len(use)}",
             f"exclusion reasons: {dict(reasons)}", f"kept by serve: {dict(Counter(r['serve'] for r in use))}",
             f"kept plate_ok: {sum(r['plate_ok'] for r in use)}", f"licences (kept): {dict(Counter(r['licence'] for r in use))}",
             f"flags (kept): {dict(Counter(f for r in use for f in r['qa_flags']))}",
             f"volume_source (kept): {dict(Counter(r['volume_source'] for r in use))}", "", "per category (kept / excluded):"]
    for k in sorted(per):
        lines.append(f"  {k:24s} {per[k][0]:3d} / {per[k][1]:3d}")
    return "\n".join(lines)


def main_objaverse(args) -> None:
    v4 = [json.loads(line) for line in Path(args.inp).read_text().splitlines() if line.strip()]
    srcs = [json.loads(line) for line in Path(args.obj_sources).read_text().splitlines() if line.strip()]
    srcs = [s for s in srcs if s["source"] == "objaverse" and os.path.exists(s.get("path", ""))]
    qa = Path(args.qa_dir)
    qa.mkdir(parents=True, exist_ok=True)
    geo = geometry_obj(srcs, qa / "cache_geo_obj.json", args.workers)
    rows = audit_obj(srcs, geo, Path(args.reorient_dir))
    if not args.no_thumbs:
        render_thumbs([dict(id=r["id"], path=r["path"], fmt="gltf") for r in rows], qa / "thumbs_obj", args.blender,
                      n_proc=args.thumb_procs)
        paths = contact_sheets_obj(rows, qa / "thumbs_obj", qa / "sheets_obj")
        kept = contact_sheets_obj([r for r in rows if not r["qa_exclude"]], qa / "thumbs_obj", qa / "sheets_obj_kept")
        print("kept-only contact sheets:", *kept, sep="\n  ")
        print("contact sheets:", *paths, sep="\n  ")
    with open(args.out, "w") as f:
        for r in v4 + rows:
            f.write(json.dumps(r) + "\n")
    txt = report_obj(rows)
    (qa / "report_obj.txt").write_text(txt + "\n")
    print(txt)
    print(f"wrote {args.out}: {len(v4)} v4 rows + {len(rows)} objaverse rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../data/synth/assets_v3.jsonl")
    ap.add_argument("--out", default="../data/synth/assets_v4.jsonl")
    ap.add_argument("--qa-dir", default="../data/synth/qa_v4", help="thumbnails, contact sheets, report, caches")
    ap.add_argument("--tex-dir", default="../data/synth/assets_v4_tex", help="corrected-texture asset folders")
    ap.add_argument("--blender", default=os.path.expanduser("~/blender/blender-4.2.1-linux-x64/blender"))
    ap.add_argument("--no-thumbs", action="store_true", help="skip Blender thumbnails / contact sheets")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--objaverse", action="store_true", help="v4 + Objaverse scans -> assets_v5 (see the Objaverse section)")
    ap.add_argument("--obj-sources", default="../data/datasets/objaverse/sources.jsonl")
    ap.add_argument("--thumb-procs", type=int, default=6)
    ap.add_argument("--reorient-dir", default="../data/datasets/objaverse/reoriented",
                    help="GLB copies of flat foods that were not lying flat")
    args = ap.parse_args()
    if args.objaverse:
        return main_objaverse(args)

    recs = [json.loads(line) for line in Path(args.inp).read_text().splitlines() if line.strip()]
    qa = Path(args.qa_dir)
    qa.mkdir(parents=True, exist_ok=True)
    geo = geometry(recs, qa / "cache_geo.json", args.workers)
    out = audit(recs, geo, Path(args.tex_dir))
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    txt, data = report(out)
    (qa / "report.txt").write_text(txt + "\n")
    (qa / "report.json").write_text(json.dumps(data, indent=1))
    print(txt.split("per category")[0])
    if not args.no_thumbs:
        render_thumbs([dict(r) for r in recs], qa / "thumbs", args.blender)
        fixed = [dict(r) for r in out if r.get("tex_fix")]
        render_thumbs(fixed, qa / "thumbs_fixed", args.blender, n_proc=4)
        paths = contact_sheets(out, qa / "thumbs", qa / "thumbs_fixed", qa / "sheets")
        print("contact sheets:", *paths, sep="\n  ")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
