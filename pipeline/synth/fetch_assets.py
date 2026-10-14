"""Download real scanned food and tableware models from open sources (no account needed).

    python -m synth.fetch_assets --out ../data/datasets/online [--list] [--sources polyhaven ycb gso]

Sources (licences checked 2026-10-13; see docs/dataset_curation.md):
  polyhaven  food scans, CC0, glTF at 2k textures       https://polyhaven.com (api.polyhaven.com)
  ycb        YCB objects, CC BY 4.0, google_16k OBJ       http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com
             (tableware: bowl, mug, plate, fork, spoon, knife; food: plastic fruit replicas)
  gso        Google Scanned Objects, CC BY 4.0, OBJ zip   https://app.gazebosim.org/GoogleResearch
             (plates, bowls, mugs, cups, ramekins, saucers)
  hdri       Poly Haven indoor HDRIs (homes, cafes, restaurants, a few studios), CC0, 2k .hdr:
             the room light and reflections of a meal photographed indoors
  surface    Poly Haven PBR texture sets for tabletops (wood, stone, laminate, cloth), CC0, 2k jpg
             (colour, roughness, OpenGL normal)

Each model lands in OUT/<source>/<id>/ with its files as published; OUT/sources.jsonl gets one
line per model (source, id, role food / container / clutter / hdri / surface, licence, attribution,
url), merged with the lines already there.
Only meshes and textures are fetched (no .blend files, which can carry scripts).
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "FoodSeg-thesis-asset-fetch/1.0"}

POLYHAVEN_SKIP = {"long_life_food", "russian_food_cans_01"}           # packaged goods, not plated food
POLYHAVEN_ROLE = {"wooden_cutting_board": "clutter"}
YCB = {  # id: role
    "024_bowl": "container", "025_mug": "clutter", "029_plate": "container", "030_fork": "clutter",
    "031_spoon": "clutter", "032_knife": "clutter",
    "011_banana": "food", "012_strawberry": "food", "013_apple": "food", "014_lemon": "food",
    "015_peach": "food", "016_pear": "food", "017_orange": "food", "018_plum": "food",
}
YCB_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/{id}_google_16k.tgz"
GSO_QUERIES = {"plate": "container", "bowl": "container", "ramekin": "container", "saucer": "container",
               "mug": "clutter", "cup": "clutter"}
GSO_API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
# name search also returns plant pots, pet bowls, shoes ("Gold Cup"), appliances and test targets
GSO_EXCLUDE = ("plant", "planter", "dog", "shoe", "loafer", "venetian", "boat", "coffeemaker", "kcup",
               "frother", "calibration", "target", "heightmap", "bradshaw", "sea_to_summit", "salad_on_a_plate",
               "ecoforms", "cole_hardware_saucer")


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


# Indoor HDRIs where people eat (Poly Haven ids; checked 2026-10-14): rooms, cafes, restaurants,
# plus a few plain studios for even light. Not venues (ballrooms, chapels) or empty sheds.
HDRIS = ["aft_lounge", "anniversary_lounge", "bush_restaurant", "cabin", "cayley_interior", "combination_room",
         "comfy_cafe", "country_club", "en_suite", "fireplace", "hotel_room", "kiara_interior", "lythwood_lounge",
         "lythwood_room", "old_room", "reading_room", "relax_inn_seaview_suite", "small_empty_room_1",
         "small_empty_room_3", "solitude_interior", "studio_kominka_01", "veranda", "warm_bar", "warm_restaurant",
         "warm_restaurant_night", "white_home_studio", "wooden_lounge", "brown_photostudio_02", "studio_small_03",
         "photo_studio_01", "ferndale_studio_05", "provence_studio"]
# Tabletops: kind -> Poly Haven texture ids
SURFACES = {
    "wood": ["wood_table_001", "kitchen_wood", "coated_pine", "coated_pine_02", "plywood", "oak_veneer_02",
             "oak_veneer_03", "oak_veneer_05", "white_oak_veneer", "american_walnut_veneer", "black_walnut_veneer_01",
             "european_walnut_veneer_04", "grey_oak_veneer_01", "washed_grey_oak_veneer", "bamboo_veneer",
             "mocha_oak_veneer", "smoked_walnut_veneer"],
    "stone": ["marble_01"],          # grey_cartago_*: floor tiles with seams and orange veins, not tabletops
    "laminate": ["laminate_floor_02"],
    "cloth": ["gingham_check", "fabric_pattern_05", "fabric_pattern_07", "waffle_pique_cotton", "book_pattern",
              "stretch_poplin"],
}
SURFACE_MAPS = {"Diffuse": "diffuse.jpg", "Rough": "rough.jpg", "nor_gl": "normal_gl.jpg"}


def hdris(res: str = "2k") -> list[dict]:
    out = []
    for aid in HDRIS:
        f = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))["hdri"][res]["hdr"]
        out.append({"source": "polyhaven_hdri", "id": aid, "name": aid, "role": "hdri", "licence": "CC0",
                    "attribution": "Poly Haven", "url": f"https://polyhaven.com/a/{aid}",
                    "_files": {f"{aid}_{res}.hdr": f["url"]}})
    return out


def surfaces(res: str = "2k") -> list[dict]:
    out = []
    for kind, ids in SURFACES.items():
        for aid in ids:
            f = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
            if "Diffuse" not in f:                    # patterned fabrics come in colourways: the first
                f["Diffuse"] = f[sorted(k for k in f if k.startswith("col"))[0]]
            info = json.loads(get(f"https://api.polyhaven.com/info/{aid}"))
            out.append({"source": "polyhaven_surface", "id": aid, "name": aid, "role": "surface", "kind": kind,
                        "licence": "CC0", "attribution": "Poly Haven", "url": f"https://polyhaven.com/a/{aid}",
                        "dimensions_mm": info.get("dimensions"),        # real-world size of one texture tile
                        "_files": {name: f[m][res]["jpg"]["url"] for m, name in SURFACE_MAPS.items()}})
    return out


def polyhaven(res: str = "2k") -> list[dict]:
    idx = json.loads(get("https://api.polyhaven.com/assets?type=models&categories=food"))
    out = []
    for aid, info in sorted(idx.items()):
        if aid in POLYHAVEN_SKIP:
            continue
        files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))["gltf"][res]["gltf"]
        out.append({"source": "polyhaven", "id": aid, "name": info["name"], "role": POLYHAVEN_ROLE.get(aid, "food"),
                    "licence": "CC0", "attribution": ", ".join(info.get("authors", {})),
                    "url": f"https://polyhaven.com/a/{aid}", "dimensions_mm": info.get("dimensions"),
                    "_files": {f"{aid}.gltf": files["url"], **{k: v["url"] for k, v in files["include"].items()}}})
    return out


def ycb() -> list[dict]:
    return [{"source": "ycb", "id": i, "name": i.split("_", 1)[1], "role": r, "licence": "CC BY 4.0",
             "attribution": "Calli et al., The YCB Object and Model Set", "url": YCB_URL.format(id=i),
             "_tgz": YCB_URL.format(id=i)} for i, r in YCB.items()]


def gso() -> list[dict]:
    seen, out = set(), []
    for q, role in GSO_QUERIES.items():
        for m in json.loads(get(f"{GSO_API}?per_page=100&q={q}")):
            n = m["name"]
            if n in seen or q not in n.lower() or any(x in n.lower() for x in GSO_EXCLUDE):
                continue
            seen.add(n)
            out.append({"source": "gso", "id": n, "name": m.get("description", n), "role": role,
                        "licence": m.get("license_name", "CC BY 4.0"), "attribution": "Google Scanned Objects",
                        "url": f"https://app.gazebosim.org/GoogleResearch/fuel/models/{n}",
                        "_zip": f"{GSO_API}/{n}.zip", "size_bytes": m.get("filesize")})
    return out


def fetch(rec: dict, out: Path) -> None:
    d = out / rec["source"] / rec["id"]
    if (d / ".done").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    if "_files" in rec:
        for rel, url in rec["_files"].items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(get(url))
    elif "_tgz" in rec:
        with tarfile.open(fileobj=io.BytesIO(get(rec["_tgz"])), mode="r:gz") as t:
            t.extractall(d, members=[m for m in t.getmembers()
                                     if m.isfile() and not m.name.startswith("/") and ".." not in m.name])
    else:
        with zipfile.ZipFile(io.BytesIO(get(rec["_zip"]))) as z:
            for m in z.infolist():
                if not m.is_dir() and not m.filename.startswith("/") and ".." not in m.filename:
                    z.extract(m, d)
    gso_texture_fix(d)
    (d / ".done").write_text("ok")


def gso_texture_fix(d: Path) -> None:
    """GSO's model.mtl says `map_Kd texture.png` (next to the mesh) but the zip keeps it in
    materials/textures/; without a copy Blender renders the dish magenta (missing image)."""
    tex = d / "materials" / "textures" / "texture.png"
    if tex.exists() and (d / "meshes").is_dir() and not (d / "meshes" / "texture.png").exists():
        (d / "meshes" / "texture.png").write_bytes(tex.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", default=["polyhaven", "ycb", "gso", "hdri", "surface"])
    ap.add_argument("--list", action="store_true", help="only print what would be fetched")
    args = ap.parse_args()
    out = Path(args.out)
    srcs = {"polyhaven": polyhaven, "ycb": ycb, "gso": gso, "hdri": hdris, "surface": surfaces}
    recs = [r for s in args.sources for r in srcs[s]()]
    for r in recs:
        print(f"{r['source']:<9} {r['role']:<9} {r['id']}")
    print(f"{len(recs)} models")
    if args.list:
        return
    out.mkdir(parents=True, exist_ok=True)
    index = out / "sources.jsonl"
    lines = {}                                     # (source, id) -> record; keeps other sources' lines
    if index.exists():
        for line in index.read_text().splitlines():
            r = json.loads(line)
            lines[(r["source"], r["id"])] = r
    for r in recs:
        try:
            fetch(r, out)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {r['source']}/{r['id']}: {e}")
            continue
        lines[(r["source"], r["id"])] = {k: v for k, v in r.items() if not k.startswith("_")}
    index.write_text("".join(json.dumps(r) + "\n" for r in lines.values()))
    print("done ->", out)


if __name__ == "__main__":
    main()
