"""Food asset index: one JSON line per mesh with metric scale, exact volume and (if known) mass.

    python -m synth.assets --out ../data/synth/assets.jsonl \
        [--metafood3d ../data/datasets/MetaFood3D] [--omni ../data/datasets/OmniObject3D/raw_scans] \
        [--mesh-dir NAME=DIR ...]

Every source is indexed the same way: find ROOT/<category>/<object>/*.obj, load with
trimesh, pick units (from the dataset's reference volume when it has one, else from
the extent: food is < 30 cm, so a max extent > 1 means millimetres), and compute the
volume. MetaFood3D rows are joined with its nutrition xlsx (mass, FNDDS code,
macros, reference volume). Open meshes are hole-filled; if still open the volume
falls back to the dataset's reference volume, else the convex hull (an upper bound,
`volume_ok: false`, which spec.py skips unless --allow-open). `volume_source`
records which: mesh / mesh_filled / ref / hull.

Procedural primitives (ellipsoids, boxes) are always added so the generator runs
with no downloads; they are written to OUT_DIR/primitives/*.npz.

Record: {id, source, name, category, object, path, fmt, to_m, volume_ml, mass_g,
         watertight, extent_m, procedural, generated, [fndds_code, kcal, ..., ref_volume_ml]}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from .geometry import box_mesh, ellipsoid_mesh, mesh_volume

MESH_EXT = (".obj", ".ply", ".glb", ".gltf", ".stl")
UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}

# name, shape, size ranges (m), base colour (linear RGB) -- appearance only
PRIMITIVES = [
    ("meatball", "ellipsoid", [(0.012, 0.02)] * 3, (0.25, 0.12, 0.06)),
    ("boiled egg", "ellipsoid", [(0.02, 0.024), (0.015, 0.018), (0.015, 0.018)], (0.85, 0.82, 0.75)),
    ("potato", "ellipsoid", [(0.025, 0.04), (0.02, 0.03), (0.015, 0.025)], (0.55, 0.40, 0.20)),
    ("bread roll", "ellipsoid", [(0.03, 0.045), (0.025, 0.035), (0.015, 0.025)], (0.60, 0.35, 0.12)),
    ("cherry tomato", "ellipsoid", [(0.01, 0.014)] * 3, (0.70, 0.05, 0.03)),
    ("chicken breast", "ellipsoid", [(0.05, 0.07), (0.025, 0.035), (0.008, 0.015)], (0.80, 0.65, 0.45)),
    ("tofu", "box", [(0.02, 0.035), (0.02, 0.035), (0.015, 0.025)], (0.90, 0.88, 0.78)),
    ("cake slice", "box", [(0.05, 0.08), (0.03, 0.05), (0.03, 0.05)], (0.45, 0.25, 0.12)),
    ("cheese cube", "box", [(0.015, 0.025)] * 3, (0.90, 0.70, 0.25)),
    # added for Nutrition5k gaps (synth/composition.py); appended so the ones above keep their sizes
    ("olive", "ellipsoid", [(0.009, 0.012), (0.007, 0.009), (0.007, 0.009)], (0.12, 0.10, 0.04)),
    ("chickpea", "ellipsoid", [(0.0045, 0.006)] * 3, (0.75, 0.58, 0.30)),
    ("brussels sprout", "ellipsoid", [(0.011, 0.016), (0.011, 0.016), (0.010, 0.014)], (0.25, 0.40, 0.08)),
    ("roasted potato", "box", [(0.015, 0.028)] * 3, (0.70, 0.45, 0.15)),
    ("squash", "box", [(0.015, 0.025)] * 3, (0.85, 0.45, 0.05)),
    ("zucchini", "ellipsoid", [(0.014, 0.02), (0.014, 0.02), (0.003, 0.005)], (0.55, 0.65, 0.30)),
    ("orange", "ellipsoid", [(0.034, 0.04)] * 3, (0.85, 0.35, 0.02)),
    ("lemon", "ellipsoid", [(0.032, 0.038), (0.025, 0.029), (0.025, 0.029)], (0.90, 0.75, 0.05)),
    ("pear", "ellipsoid", [(0.03, 0.035), (0.03, 0.035), (0.042, 0.05)], (0.60, 0.65, 0.20)),
]


def _rec(**kw) -> dict:
    base = dict(mass_g=None, procedural=False, generated=False, watertight=True, color=None, object=None,
                volume_source="mesh", volume_ok=True, container_shaped=False)
    base.update(kw)
    return base


def build_primitives(out_dir: Path, n_per: int = 6, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for name, shape, ranges, color in PRIMITIVES:
        for i in range(n_per):
            dims = [rng.uniform(*r) for r in ranges]
            V, F = ellipsoid_mesh(*dims) if shape == "ellipsoid" else box_mesh(*dims)
            aid = f"prim_{name.replace(' ', '_')}_{i}"
            p = out_dir / f"{aid}.npz"
            np.savez_compressed(p, V=V.astype(np.float32), F=F.astype(np.int32))
            recs.append(_rec(id=aid, source="primitive", name=name, category=name, path=str(p.resolve()),
                             fmt="npz", to_m=1.0, volume_ml=mesh_volume(V, F) * 1e6,
                             extent_m=(V.max(0) - V.min(0)).tolist(), procedural=True, color=color))
    return recs


def _clean(cat: str) -> str:
    """'Almond(bowl)' -> 'almond', 'chicken_wing' -> 'chicken wing'."""
    return cat.split("(")[0].replace("_", " ").strip().lower()


def _key(root: Path, p: Path) -> tuple[str, str]:
    rel = p.relative_to(root)
    return rel.parts[0], (rel.parts[1] if len(rel.parts) > 2 else p.stem)


def _norm(key: tuple[str, str]) -> tuple[str, str]:
    """Folder names and xlsx rows differ in case and underscores ('almond_2' / 'Almond_2', 'Banana3')."""
    return tuple(re.sub(r"[^a-z0-9]", "", k.lower()) for k in key)


def _index_one(job) -> dict | None:
    p, root, source, units, ref = job
    import trimesh
    try:
        m = trimesh.load(p, force="mesh", process=True)
    except Exception as e:  # noqa: BLE001
        print(f"skip {p}: {e}")
        return None
    if len(m.faces) == 0:
        return None
    # scans split vertices along texture seams; merge them so the surface can close
    m.merge_vertices(merge_tex=True, merge_norm=True)
    wt = bool(m.is_watertight)
    vsrc = "mesh"
    if not wt:
        trimesh.repair.fill_holes(m)
        wt, vsrc = bool(m.is_watertight), "mesh_filled"
    raw_vol = abs(m.volume) if wt else abs(m.convex_hull.volume)
    if units != "auto":
        to_m = UNITS[units]
    elif ref:   # the unit that agrees with the dataset's own volume (mL)
        to_m = min(UNITS.values(), key=lambda u: abs(np.log(raw_vol * u ** 3 * 1e6 / ref)))
    else:
        to_m = 1e-3 if m.extents.max() > 1.0 else 1.0
    vol = raw_vol * to_m ** 3 * 1e6
    if not wt:   # still open: the dataset's own (closed-mesh) volume if it has one, else hull = upper bound
        vol, vsrc = (ref, "ref") if ref else (vol, "hull")
    cat, obj = _key(root, p)
    return _rec(id=f"{source}_{cat}_{obj}", source=source, name=_clean(cat), category=cat, object=obj,
                path=str(p.resolve()), fmt=p.suffix.lower()[1:], to_m=to_m, volume_ml=float(vol),
                volume_source=vsrc, volume_ok=vsrc != "hull", watertight=wt,
                # MetaFood3D "X(bowl)" scans are the bowl's contents (bowl removed), so their
                # underside is bowl-shaped; spec.py keeps them out of random placement for now
                container_shaped="(" in cat,
                extent_m=(m.extents * to_m).tolist(), n_faces=int(len(m.faces)))


def index_mesh_dir(root: Path, source: str, meta: dict[tuple[str, str], dict] | None = None,
                   units: str = "auto", workers: int = 8) -> list[dict]:
    """Index every mesh under ROOT/<category>/<object>/ (MetaFood3D 3D_Mesh and OmniObject3D
    raw_scans both look like this). `meta` maps (category, object) to extra fields; its
    `ref_volume_ml` also decides the units when units='auto'."""
    from multiprocessing import Pool
    meta = {_norm(k): v for k, v in (meta or {}).items()}
    files = sorted(Path(d) / f for d, _, fs in os.walk(root, followlinks=True)   # datasets are often symlinked
                   for f in fs if Path(f).suffix.lower() in MESH_EXT)
    jobs = [(p, root, source, units, meta.get(_norm(_key(root, p)), {}).get("ref_volume_ml")) for p in files]
    with Pool(workers) as pool:
        recs = [r for r in pool.imap(_index_one, jobs, chunksize=4) if r]
    for r in recs:
        r.update(meta.get(_norm((r["category"], r["object"])), {}))
    n_meta = sum(_norm((r["category"], r["object"])) in meta for r in recs)
    print(f"{source}: {len(recs)} meshes, {sum(r['watertight'] for r in recs)} watertight, "
          f"{n_meta} joined with metadata ({len(meta)} metadata rows)")
    return recs


# synth.fetch_assets ids -> food category; None = not usable as plated food
ONLINE_CATEGORY = {
    "bananas": None,                          # a 60 cm display bunch
    "carrot_cake": "cake", "croissant": "croissant", "food_apple_01": "apple", "food_avocado_01": "avocado",
    "food_ginger_01": "ginger", "food_kiwi_01": "kiwi", "food_lime_01": "lime", "food_lychee_01": "lychee",
    "food_pears_asian_01": "pear", "food_pomegranate_01": "pomegranate", "hamburger_buns": "bread roll",
    "lemon": "lemon", "strawberry_chocolate_cake": "cake", "sweet_potato": "sweet potato",
    "yellow_onion": "onion", "011_banana": "banana", "012_strawberry": "strawberry", "013_apple": "apple",
    "014_lemon": "lemon", "015_peach": "peach", "016_pear": "pear", "017_orange": "orange", "018_plum": "plum",
}


def online_mesh(d: Path) -> Path | None:
    """The one textured mesh of a synth.fetch_assets model folder."""
    for pat in ("*.gltf", "textured.obj", "model.obj"):
        hits = sorted(d.rglob(pat))
        if hits:
            return hits[0]
    return None


def index_online(root: Path) -> list[dict]:
    """Index synth.fetch_assets downloads (OUT/sources.jsonl). Food gets volume like any scan;
    containers and clutter are recorded with their `role` so food sampling skips them.
    glTF is Y up (Blender's importer converts it); YCB / GSO OBJ are Z up; all in metres."""
    import trimesh
    recs = []
    for line in open(root / "sources.jsonl"):
        s = json.loads(line)
        p = online_mesh(root / s["source"] / s["id"])
        cat = ONLINE_CATEGORY.get(s["id"], s["name"]) if s["role"] == "food" else s["id"]
        if p is None or cat is None:
            continue
        m = trimesh.load(p, force="mesh")
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
        wt = bool(m.is_watertight)
        ext = m.extents if p.suffix != ".gltf" else m.extents[[0, 2, 1]]        # -> Z up
        recs.append(_rec(id=f"{s['source']}_{s['id']}", source=s["source"], name=cat, category=cat,
                         object=s["id"], path=str(p.resolve()), fmt=p.suffix.lower()[1:], to_m=1.0,
                         up_axis="Z", role=s["role"], watertight=wt,
                         volume_ml=float(abs(m.volume) * 1e6) if wt else float(m.convex_hull.volume * 1e6),
                         volume_source="mesh_filled" if wt else "hull",
                         volume_ok=wt or s["role"] != "food", extent_m=ext.tolist(), n_faces=int(len(m.faces)),
                         licence=s["licence"], attribution=s["attribution"], url=s["url"]))
    print(f"online: {len(recs)} models ({sum(r['role'] == 'food' for r in recs)} food, "
          f"{sum(r['role'] == 'food' and r['volume_ok'] for r in recs)} with exact volume)")
    return recs


def load_metafood3d_meta(xlsx: Path) -> dict[tuple[str, str], dict]:
    """complete_dataset_nutrition_v2.xlsx: Object_name (category folder), Food_Type (object
    folder), FNDDS Food Code, Weight (g), Energy (Kcal), Protein/Fat/Carbs (g), Volume (mL)."""
    import openpyxl
    rows = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0].iter_rows(values_only=True)
    col = {str(h).strip(): i for i, h in enumerate(next(rows)) if h}
    out = {}
    for r in rows:
        if not r or r[col["Object_name"]] is None:
            continue
        g = {k: r[i] for k, i in col.items()}
        num = lambda k: float(g[k]) if isinstance(g.get(k), (int, float)) else None  # noqa: E731
        out[(str(g["Object_name"]).strip(), str(g["Food_Type"]).strip())] = {
            "mass_g": num("Weight (g)"), "fndds_code": g.get("FNDDS Food Code"), "kcal": num("Energy (Kcal)"),
            "protein_g": num("Protein (g)"), "fat_g": num("Fat (g)"), "carb_g": num("Carbs (g)"),
            "ref_volume_ml": num("Volume")}
    return out


def load_index(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="assets.jsonl to write; primitives go next to it")
    ap.add_argument("--metafood3d", default=None, help="MetaFood3D root (3D_Mesh/ + nutrition xlsx)")
    ap.add_argument("--omni", default=None, help="OmniObject3D raw_scans root (food categories only)")
    ap.add_argument("--mesh-dir", action="append", default=[], help="NAME=DIR, any other mesh folder")
    ap.add_argument("--online", default=None, help="synth.fetch_assets output (Poly Haven / YCB / GSO)")
    ap.add_argument("--n-prim", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = build_primitives(out.parent / "primitives", n_per=args.n_prim)
    if args.metafood3d:
        root = Path(args.metafood3d)
        xlsx = sorted(root.glob("*nutrition*.xlsx"))
        meta = load_metafood3d_meta(xlsx[-1]) if xlsx else None
        mf = index_mesh_dir(root / "3D_Mesh", "metafood3d", meta, workers=args.workers)
        for r in mf:
            r["up_axis"] = "Z"     # Revopoint turntable scans: Z up, resting on z ~ 0
        recs += mf
    if args.omni:
        recs += index_mesh_dir(Path(args.omni), "omniobject3d", workers=args.workers)
    if args.online:
        recs += index_online(Path(args.online))
    for spec in args.mesh_dir:
        name, d = spec.split("=", 1)
        recs += index_mesh_dir(Path(d), name, workers=args.workers)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by_src: dict[str, int] = {}
    for r in recs:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"wrote {len(recs)} assets to {out}: {by_src}")


if __name__ == "__main__":
    main()
