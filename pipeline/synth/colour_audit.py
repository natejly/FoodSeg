"""Colour audit of rendered synthetic scans: finds neon / black / grey / hue-shifted foods.

For every food item of every scan under <root>/scans (rgb/*.jpg + gt/seg_ids/*.png + gt.json, and
<root>/specs/<sid>/spec.json for the procedural colours) the rendered pixels of the item (eroded masks, the
best-covered frames) are summarised in CIE Lab and compared with a target:

  * scanned MetaFood3D assets: the REAL photo pixels of the same object (`_colour_cache/<id>.R.npz`, the masked
    frames synth.scan_colour calibrates against), pooled over the category when the object has none;
  * procedural foods: the spec colour (sRGB -> Lab); no distribution, so only chroma / hue / lightness ratios.

Flags (see `classify`): neon (chroma far above what the real photos of that food reach), black (lightness far below
the real food's), grey (chroma collapsed for a coloured food), hue (hue rotated by more than HUE_TOL degrees).

    python -m synth.colour_audit <root> [--out audit.json] [--sheet worst.jpg] [--top 24]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

DATA = Path("/home/kyle/Desktop/FoodSeg/data")
CACHE = DATA / "synth" / "_colour_cache"
ASSETS = DATA / "synth" / "assets_v6.jsonl"

NEON_MARGIN = 14.0     # rendered chroma p90 above the real food's p95 by this much = neon
NEON_ABS = 78.0        # ... or above this absolute Lab chroma, whatever the food
BLACK_RATIO = 0.55     # rendered median L below this fraction of the real median L (and below BLACK_ABS)
BLACK_ABS = 26.0
GREY_RATIO = 0.35      # rendered median chroma below this fraction of the real median chroma ...
GREY_REAL_MIN = 16.0   # ... when the real food is clearly coloured
HUE_TOL = 40.0         # degrees
HUE_MIN_CHROMA = 14.0  # hue is meaningless below this chroma
ERODE = 3
MIN_PX = 600


# ----------------------------------------------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------------------------------------------
def srgb_to_lab(rgb) -> np.ndarray:
    """sRGB, uint8 (0..255) or float (0..1), shape (...,3) -> Lab (D65)."""
    a = np.asarray(rgb)
    a = a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab) -> np.ndarray:
    """Lab (D65) -> sRGB in [0, 1], clipped per channel."""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(xyz @ M.T, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def linear_to_srgb(x) -> np.ndarray:
    """Linear light (what Blender's Base Color takes; synth.meals.FOODS / primitives store linear RGB) -> sRGB."""
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def chroma(lab) -> np.ndarray:
    lab = np.asarray(lab)
    return np.hypot(lab[..., 1], lab[..., 2])


def hue_deg(lab) -> np.ndarray:
    lab = np.asarray(lab)
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0


def hue_diff(h1: float, h2: float) -> float:
    """Smallest absolute angle between two hues in degrees."""
    d = abs(float(h1) - float(h2)) % 360.0
    return min(d, 360.0 - d)


def summarise(lab: np.ndarray) -> dict:
    """(N,3) Lab samples -> compact stats (means, L percentiles, chroma percentiles, circular-mean hue)."""
    lab = np.asarray(lab, np.float64).reshape(-1, 3)
    c = chroma(lab)
    a, b = lab[:, 1].mean(), lab[:, 2].mean()
    return dict(n=int(len(lab)), L=float(lab[:, 0].mean()), a=float(a), b=float(b),
                L_p5=float(np.percentile(lab[:, 0], 5)), L_p50=float(np.percentile(lab[:, 0], 50)),
                L_p95=float(np.percentile(lab[:, 0], 95)), C_p50=float(np.percentile(c, 50)),
                C_p90=float(np.percentile(c, 90)), C_p95=float(np.percentile(c, 95)),
                hue=float(hue_deg(np.array([0, a, b]))))


def classify(r: dict, t: dict) -> list[str]:
    """Flags for rendered stats `r` against target stats `t` (both from `summarise`; a procedural target may
    only carry L_p50 / C_p50 / C_p95 / hue)."""
    flags = []
    t_c95 = t.get("C_p95", t["C_p50"])
    if r["C_p90"] > max(t_c95 + NEON_MARGIN, 0) and r["C_p90"] > 45 or r["C_p90"] > NEON_ABS:
        flags.append("neon")
    if r["L_p50"] < BLACK_RATIO * t["L_p50"] and r["L_p50"] < BLACK_ABS:
        flags.append("black")
    if t["C_p50"] > GREY_REAL_MIN and r["C_p50"] < GREY_RATIO * t["C_p50"]:
        flags.append("grey")
    if r["C_p50"] > HUE_MIN_CHROMA and t["C_p50"] > HUE_MIN_CHROMA and hue_diff(r["hue"], t["hue"]) > HUE_TOL:
        flags.append("hue")
    return flags


def delta_e(r: dict, t: dict) -> float:
    """Lab distance between mean colours; dE_ab ignores lightness (the render's light differs from the photo's)."""
    return float(np.linalg.norm([r["L"] - t["L"], r["a"] - t["a"], r["b"] - t["b"]]))


def delta_e_ab(r: dict, t: dict) -> float:
    return float(math.hypot(r["a"] - t["a"], r["b"] - t["b"]))


# ----------------------------------------------------------------------------------------------------------
# broken textures
# ----------------------------------------------------------------------------------------------------------
STREAK_RATIO = 3.0     # vertical / horizontal mean gradient of a scan-line smear (healthy atlases: 1.0-1.7)


def streak_ratio(gray: np.ndarray) -> float:
    """Mean |d/dy| over mean |d/dx| of a 2-D luminance array: a few MetaFood3D atlases are horizontal scan-line
    smears (salmon_grill_4, waffle_5, breaded_fish_4, cabbage_5) and render as flat grey / brown slabs."""
    g = np.asarray(gray, np.float32)
    return float(np.abs(np.diff(g, axis=0)).mean() / max(np.abs(np.diff(g, axis=1)).mean(), 1e-3))


def write_assets_excluding_streaks(src: Path, dst: Path) -> list[str]:
    """Copy an assets jsonl, setting qa_exclude (reason 'streak_texture') on rows whose texture is a smear."""
    from PIL import Image
    bad, rows = [], []
    for line in open(src):
        a = json.loads(line)
        if a.get("source") == "metafood3d" and not a.get("qa_exclude"):
            tex = sorted(Path(a["path"]).parent.glob("*.jpg"))
            if tex:
                g = np.asarray(Image.open(tex[0]).convert("L").resize((512, 512)))
                if streak_ratio(g) > STREAK_RATIO:
                    a["qa_exclude"], a["qa_reason"] = True, "streak_texture"
                    a["qa_flags"] = list(a.get("qa_flags") or []) + ["streak_texture"]
                    bad.append(a["id"])
        rows.append(a)
    with open(dst, "w") as f:
        for a in rows:
            f.write(json.dumps(a) + "\n")
    return bad


# ----------------------------------------------------------------------------------------------------------
# targets
# ----------------------------------------------------------------------------------------------------------
_cat_pool: dict = {}


def _real_lab(asset_id: str, category: str | None) -> np.ndarray | None:
    f = CACHE / f"{asset_id}.R.npz"
    if f.exists():
        R = np.load(f)["R"]
        if len(R) > 500:
            return R
    key = (category or "").lower()
    if key not in _cat_pool:
        parts = []
        for g in CACHE.glob("metafood3d_*.R.npz"):
            stem = g.name[:-6]
            rest = stem[len("metafood3d_"):]
            if rest.lower().startswith(key + "_") and key:
                R_ = np.load(g)["R"]
                if len(R_):
                    parts.append(R_)
        _cat_pool[key] = np.concatenate(parts) if parts and sum(map(len, parts)) > 500 else None
    return _cat_pool[key]


# white / pale foods: the plate mask (L < PLATE_L, chroma > MIN_C) keeps only their shadowed, tinted parts, so the
# Nutrition5k statistics are biased for them
N5K_PALE = {"white rice", "cauliflower", "egg whites", "cottage cheese", "yogurt", "frozen yogurt", "tofu", "eggs", "oatmeal"}
N5K_ALIAS = {"boiled egg": None, "salmon grill": "salmon", "chicken nugget": "chicken", "fried mushroom": "mushroom", "melon": None,
             "cherry tomato": "cherry tomatoes", "french fr": "potatoes", "strawberry": "strawberries",
             "sushi": None, "pasta": None, "bread": None}
_n5k: dict | None = None


def n5k_key(name: str, keys) -> str | None:
    """Nutrition5k ingredient whose real-photo colour stands for a food name (alias, exact, then the longest
    key contained in the name or containing it; singular / plural insensitive)."""
    n = (name or "").lower().strip()
    for k, v in N5K_ALIAS.items():
        if k in n:
            return v if v is None or v in keys else None
    sing = lambda w: (w[:-2] if w.endswith("oes") else w[:-1] if w.endswith("s") and not w.endswith("ss") else w)   # noqa: E731
    ns = " ".join(sing(w) for w in n.split())
    best = None
    for k in keys:
        ks = " ".join(sing(w) for w in re.sub(r"\(.*?\)", "", k).split())
        if ks and (ks == ns or ks in ns):
            if best is None or len(ks) > len(best[0]):
                best = (ks, k)
    return best[1] if best else None


def n5k_target(name: str) -> dict | None:
    global _n5k
    if _n5k is None:
        f = Path(__file__).with_name("n5k_food_colours.json")
        _n5k = json.load(open(f)) if f.exists() else {}
        _n5k.pop("deprecated", None)
    k = n5k_key(name, [x for x in _n5k if x not in N5K_PALE])
    if k is None:
        return None
    v = _n5k[k]
    return dict(L=v["L"], a=v["a"], b=v["b"], L_p50=v["L"], C_p50=v["C"], C_p95=v["C_hi"], hue=v["hue"], n5k=k)


def target_for(item: dict, asset: dict | None) -> tuple[dict | None, str]:
    """(target stats, kind) of one spec item: Nutrition5k real photos of that food when known, else the capture
    photos of the scan, else the spec colour."""
    t5 = n5k_target(item.get("name") or "")
    if t5 is not None:
        return t5, "n5k"
    if item.get("color") is not None and (asset is None or asset.get("procedural", item.get("procedural"))):
        lab = srgb_to_lab(linear_to_srgb(item["color"]))      # spec colours are linear RGB (Blender Base Color)
        c = float(chroma(lab))
        # the render is lit and shaded: lightness only known to a factor, so give a broad L, chroma = colour's
        return dict(L=float(lab[0]), a=float(lab[1]), b=float(lab[2]), L_p50=float(lab[0]), C_p50=c,
                    C_p95=c, hue=float(hue_deg(lab))), "spec_colour"
    aid = item.get("asset_id")
    if aid:
        R = _real_lab(aid, (asset or {}).get("category") or item.get("category"))
        if R is not None:
            return summarise(R), "real_photos"
    return None, "none"


# ----------------------------------------------------------------------------------------------------------
# rendered pixels
# ----------------------------------------------------------------------------------------------------------
def item_pixels(root: Path, sid: str, n_frames: int = 5, per_frame: int = 6000, seed: int = 0):
    """{item_id: (Lab samples, [(frame, bbox, npx)])} from the best-covered frames of one scan."""
    from PIL import Image
    from scipy.ndimage import binary_erosion
    rng = np.random.default_rng(seed)
    sd = root / "scans" / sid
    segs = sorted((sd / "gt" / "seg_ids").glob("*.png"))
    if not segs:
        return {}
    counts = {}
    for p in segs[::2]:
        m = np.asarray(Image.open(p))
        ids, n = np.unique(m, return_counts=True)
        counts[p.stem] = dict(zip(ids.tolist(), n.tolist()))
    out: dict = {}
    all_ids = {i for c in counts.values() for i in c if 0 < i < 250}
    for iid in sorted(all_ids):
        best = sorted(counts, key=lambda s: -counts[s].get(iid, 0))[:n_frames]
        samples, frames = [], []
        for stem in best:
            if counts[stem].get(iid, 0) < MIN_PX:
                continue
            m = np.asarray(Image.open(sd / "gt" / "seg_ids" / f"{stem}.png")) == iid
            e = binary_erosion(m, iterations=ERODE)
            if e.sum() < MIN_PX // 2:
                e = m
            im = np.asarray(Image.open(sd / "rgb" / f"{stem}.jpg").convert("RGB"))
            px = im[e]
            px = px[px.max(1) < 250]                       # clipped speculars
            if len(px) < 100:
                continue
            if len(px) > per_frame:
                px = px[rng.choice(len(px), per_frame, replace=False)]
            samples.append(srgb_to_lab(px))
            ys, xs = np.where(m)
            frames.append((stem, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), int(m.sum())))
        if samples:
            out[iid] = (np.concatenate(samples), frames)
    return out


def load_assets() -> dict:
    d = {}
    with open(ASSETS) as f:
        for line in f:
            a = json.loads(line)
            d[a["id"]] = a
    return d


def audit_root(root: Path, assets: dict | None = None, scans: list[str] | None = None) -> list[dict]:
    assets = assets if assets is not None else load_assets()
    rows = []
    for sd in sorted((root / "scans").iterdir()):
        sid = sd.name
        if scans and sid not in scans:
            continue
        sp = root / "specs" / sid / "spec.json"
        gtp = sd / "gt.json"
        if not gtp.exists():
            continue
        items = {it["id"]: it for it in json.load(open(gtp))["items"]}
        spec_items = {it["id"]: it for it in (json.load(open(sp))["items"] if sp.exists() else [])}
        for iid, (lab, frames) in item_pixels(root, sid).items():
            gi, si = items.get(iid, {}), spec_items.get(iid, {})
            item = {**gi, **{k: v for k, v in si.items() if k in ("color", "look")}}
            item["asset_id"] = gi.get("asset_id") or si.get("asset_id")
            asset = assets.get(item["asset_id"]) if item["asset_id"] else None
            r = summarise(lab)
            t, kind = target_for(item, asset)
            row = dict(scan=sid, item=iid, name=gi.get("name"), asset_id=item["asset_id"], source=gi.get("source"),
                       render=r, target=t, target_kind=kind, frames=frames[:1])
            if t:
                row.update(dE=delta_e(r, t), dE_ab=delta_e_ab(r, t), flags=classify(r, t))
            else:
                row.update(dE=None, dE_ab=None, flags=["no_target"] + (["neon"] if r["C_p90"] > NEON_ABS else []))
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------------------
def summary(rows: list[dict]) -> dict:
    de = [r["dE_ab"] for r in rows if r["dE_ab"] is not None]
    flagged = [r for r in rows if set(r["flags"]) & {"neon", "black", "grey", "hue"}]
    cnt = {k: sum(k in r["flags"] for r in rows) for k in ("neon", "black", "grey", "hue")}
    return dict(n_items=len(rows), n_flagged=len(flagged), flag_counts=cnt,
                median_dE_ab=float(np.median(de)) if de else None, mean_dE_ab=float(np.mean(de)) if de else None)


def sheet(root: Path, rows: list[dict], path: Path, top: int = 24, cell: int = 300) -> None:
    from PIL import Image, ImageDraw
    worst = sorted([r for r in rows if r["dE_ab"] is not None], key=lambda r: -(len(r["flags"]) * 100 + r["dE_ab"]))[:top]
    cols = 6
    rws = (len(worst) + cols - 1) // cols
    img = Image.new("RGB", (cols * cell, rws * (cell + 28)), (30, 30, 30))
    dr = ImageDraw.Draw(img)
    for k, r in enumerate(worst):
        stem, (x0, y0, x1, y1), _ = r["frames"][0]
        im = Image.open(root / "scans" / r["scan"] / "rgb" / f"{stem}.jpg").convert("RGB")
        pad = int(0.15 * max(x1 - x0, y1 - y0))
        im = im.crop((max(x0 - pad, 0), max(y0 - pad, 0), x1 + pad, y1 + pad))
        im.thumbnail((cell, cell))
        x, y = (k % cols) * cell, (k // cols) * (cell + 28)
        img.paste(im, (x, y + 28))
        dr.text((x + 3, y + 2), f"{r['scan']}#{r['item']} {str(r['name'])[:22]}", fill=(255, 255, 255))
        dr.text((x + 3, y + 14), f"dEab {r['dE_ab']:.0f} {','.join(r['flags'])}", fill=(255, 200, 80))
    img.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="scan root, or 'assets' with --src / --dst to write a streak-free assets jsonl")
    ap.add_argument("--src", default=str(ASSETS))
    ap.add_argument("--dst", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--scans", nargs="*")
    a = ap.parse_args()
    if a.root == "assets":
        print(write_assets_excluding_streaks(Path(a.src), Path(a.dst)))
        return
    root = Path(a.root)
    rows = audit_root(root, scans=a.scans)
    s = summary(rows)
    print(json.dumps(s, indent=1))
    for r in sorted(rows, key=lambda r: -(r["dE_ab"] or 0))[:a.top]:
        print(f"{r['scan']}#{r['item']:<2} {str(r['name'])[:20]:<20} dEab={r['dE_ab'] or 0:5.1f} "
              f"L={r['render']['L_p50']:5.1f} C50={r['render']['C_p50']:5.1f} C90={r['render']['C_p90']:5.1f} "
              f"h={r['render']['hue']:5.0f} | tgt L={r['target']['L_p50'] if r['target'] else 0:5.1f} "
              f"C50={r['target']['C_p50'] if r['target'] else 0:5.1f} h={r['target']['hue'] if r['target'] else 0:5.0f} "
              f"[{r['target_kind']}] {','.join(r['flags'])}")
    if a.out:
        json.dump(dict(summary=s, items=rows), open(a.out, "w"), indent=1, default=float)
    if a.sheet:
        sheet(root, rows, Path(a.sheet), a.top)


if __name__ == "__main__":
    main()
