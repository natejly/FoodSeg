"""Real-photo food texture bank for procedural foods (rice beds, mash, curries, purees...).

MetaFood3D scans carry real photographic colour, but their UV atlases are shredded islands, so
instead each scan is rendered top-down (points sampled on the surface, nearest to the camera
wins) into a colour image + height map at a fixed 0.2 mm / pixel. From that we cut windows that
are pure food (no background, no steep walls), flatten the baked lighting (divide by a heavy
blur: what remains is per-channel albedo *ratio* around 1 = grain / crumb / sauce texture, with
the scan's own colour variation), fill holes, and make each tile seamless (offset blend).

  python -m synth.food_textures [--groups Rice,Yogurt] [--force]

writes  data/synth/food_textures/<group>/<k>.jpg   (ratio / 2, RGB, mean ~ 128 -> x2 in shader)
        data/synth/food_textures/<group>/<k>_h.jpg (height, high-passed, mean 128)
        data/synth/food_textures/index.json        group -> [{"albedo", "height", "tile_m"}]
Scan top views are cached in data/synth/food_textures/_cache/ (npz).

render.food_material multiplies the food's spec colour by the ratio texture (object-space
metres, tile_m wide, two rotations / scales blended so no repetition shows) and bumps by height.
`pick(group, rng)` is used by looks.look().
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/kyle/Desktop/FoodSeg/data/synth")
BANK = ROOT / "food_textures"
ASSETS = ROOT / "assets_v3.jsonl"
PIX = 0.0002        # metres per pixel of the top views
WIN = 256           # tile size, pixels (5.12 cm)

# texture group -> MetaFood3D categories (id substrings may restrict: "Rice:new_white")
GROUPS = {
    "rice": ["Rice"], "fried_rice": ["Fried_rice"], "mash": ["Mashed_Potato"], "mac": ["Mac"],
    "pasta": ["Pasta_mixed_dishes"], "yogurt": ["Yogurt"], "pudding": ["Pudding"],
    "guacamole": ["Guacamole"], "applesauce": ["Applesauce"], "soup": ["Soup"],
    "stew": ["Stew_Beef"], "beans": ["Beans"], "omelet": ["Omelet"], "cottage": ["Cottage cheese"],
    "coleslaw": ["Coleslaw"], "tuna": ["Tuna_salad"], "lasagna": ["Lasagna"],
}


# hand-curated tile indices (<group>/<k>.jpg) to leave out of the index: cup rims, labels, text,
# smeared hole fills (review with a contact sheet after any rebuild; indices are deterministic)
DROP = {"yogurt": "all", "pudding": "all", "applesauce": {2}, "guacamole": {4}, "soup": {1, 3, 4}, "cottage": {1},
        "omelet": {6}, "fried_rice": {0}, "pasta": {4, 5, 6, 7}}


def pick(group: str, rng, bank: Path = BANK):
    """(albedo path, height path, tile_m) of a random tile of the group, or None."""
    try:
        idx = json.load(open(bank / "index.json"))
    except OSError:
        return None
    tiles = idx.get(group)
    if not tiles:
        return None
    t = tiles[int(rng.integers(len(tiles)))]
    return str(bank / t["albedo"]), str(bank / t["height"]), float(t["tile_m"])


def topview(path: str, up: str = "Z", res: int = 1400, n: int = 9_000_000, seed: int = 0):
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    img = np.asarray(m.visual.material.image.convert("RGB"))
    uv = np.asarray(m.visual.uv)
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces)
    ax = {"X": 0, "Y": 1, "Z": 2}[up]
    V = V[:, [i for i in range(3) if i != ax] + [ax]]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    P = A[fi] + u * (B[fi] - A[fi]) + v * (C[fi] - A[fi])
    T = uv[F[fi, 0]] * (1 - u - v) + uv[F[fi, 1]] * u + uv[F[fi, 2]] * v
    c = np.median(V, 0)
    ij = np.floor((P[:, :2] - c[:2]) / PIX + res / 2).astype(int)
    ok = (ij >= 0).all(1) & (ij < res).all(1)
    ij, P, T = ij[ok], P[ok], T[ok]
    idx = ij[:, 1] * res + ij[:, 0]
    o = np.argsort(P[:, 2])
    idx, P, T = idx[o], P[o], T[o]
    last = np.full(res * res, -1)
    last[idx] = np.arange(len(idx))                      # highest z written last wins
    k = last >= 0
    h = np.full(res * res, np.nan, np.float32)
    col = np.zeros((res * res, 3), np.uint8)
    h[k] = P[last[k], 2]
    tx = np.clip((T[last[k], 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)
    ty = np.clip(((1 - T[last[k], 1]) * img.shape[0]).astype(int), 0, img.shape[0] - 1)
    col[k] = img[ty, tx]
    return col.reshape(res, res, 3)[::-1].copy(), h.reshape(res, res)[::-1].copy()


def fill_holes(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
    return a[iy, ix]


def windows(valid: np.ndarray, h: np.ndarray, k: int = 3, win: int = WIN):
    """Up to k non-overlapping windows that are (nearly) all food and not steep."""
    from scipy import ndimage as ndi
    hz = np.where(valid, h, np.nanmedian(h))
    hz = ndi.gaussian_filter(hz, 8.0)
    gy, gx = np.gradient(hz)
    steep = np.hypot(gx, gy) / PIX > 1.2      # macro slope > ~50 deg
    bad = (~valid | steep).astype(np.float32)
    box = ndi.uniform_filter(bad, win, mode="constant", cval=1.0)
    # prefer windows with texture contrast (not smeared)
    out = []
    score = box.copy()
    r = win // 2
    for _ in range(k):
        y, x = np.unravel_index(np.argmin(score), score.shape)
        if score[y, x] > 0.15:
            break
        out.append((y - r, x - r))
        score[max(0, y - win): y + win, max(0, x - win): x + win] = 9
    return out


def make_tile(col: np.ndarray, h: np.ndarray, valid: np.ndarray):
    """Flatten lighting, fill holes, make seamless. Returns (ratio RGB float, height float)."""
    from scipy import ndimage as ndi
    f = col.astype(np.float32) / 255.0 + 1e-3
    for i in range(3):
        f[..., i] = fill_holes(f[..., i], valid) if (~valid).any() else f[..., i]
    hh = fill_holes(np.nan_to_num(h, nan=0.0), valid) if (~valid).any() else h
    sig = f.shape[0] / 10
    low = np.stack([ndi.gaussian_filter(f[..., i], sig, mode="reflect") for i in range(3)], -1)
    # lighting is a luminance effect: flatten luminance fully, keep most of the colour variation
    lum = f.mean(-1, keepdims=True)
    lum_low = low.mean(-1, keepdims=True)
    chroma = ((f / lum) / (low / lum_low)) ** 0.6
    ratio = lum / lum_low * np.clip(chroma, 0.92, 1.09)
    hl = hh - ndi.gaussian_filter(hh, sig, mode="reflect")
    hl = hl / (hl.std() + 1e-6)
    # seamless: blend with the half-shifted copy, weight 0 at every edge
    n = f.shape[0]
    w1 = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    w = (w1[:, None] * w1[None, :])[..., None]
    sh = (n // 2, n // 2)
    ratio = w * ratio + (1 - w) * np.roll(ratio, sh, (0, 1))
    hl = (w[..., 0] * hl + (1 - w[..., 0]) * np.roll(hl, sh, (0, 1)))
    ratio = ratio / ratio.mean((0, 1), keepdims=True)
    return ratio, hl


def build(groups=None, force: bool = False, per_mesh: int = 2) -> dict:
    from PIL import Image
    (BANK / "_cache").mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(ASSETS)]
    idx_path = BANK / "index.json"
    index = json.load(open(idx_path)) if idx_path.exists() else {}
    for g, cats in GROUPS.items():
        if groups and g not in groups:
            continue
        index[g] = []
        (BANK / g).mkdir(exist_ok=True)
        k = 0
        for r in recs:
            if r["category"] not in cats or r.get("fmt") != "obj":
                continue
            cache = BANK / "_cache" / (r["id"] + ".npz")
            try:
                if cache.exists() and not force:
                    z = np.load(cache)
                    col, h = z["col"], z["h"]
                else:
                    col, h = topview(r["path"], r.get("up_axis", "Z"))
                    np.savez_compressed(cache, col=col, h=h)
            except Exception as e:          # no texture / odd mesh
                print("skip", r["id"], e)
                continue
            valid = ~np.isnan(h)
            from scipy import ndimage as ndi
            valid = ndi.binary_closing(valid, iterations=2) | valid
            h2 = np.where(np.isnan(h), np.nanmedian(h), h)
            valid = valid & (col.sum(-1) > 12)
            for (y, x) in windows(valid, h2, per_mesh):
                sl = (slice(y, y + WIN), slice(x, x + WIN))
                ratio, hl = make_tile(col[sl], h2[sl], valid[sl])
                a = (np.clip(ratio / 2, 0, 1) * 255).astype(np.uint8)
                b = (np.clip(hl * 0.18 + 0.5, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(a).save(BANK / g / f"{k}.jpg", quality=92)
                Image.fromarray(b).save(BANK / g / f"{k}_h.jpg", quality=92)
                index[g].append({"albedo": f"{g}/{k}.jpg", "height": f"{g}/{k}_h.jpg", "tile_m": WIN * PIX,
                                 "src": r["id"]})
                k += 1
        drop = DROP.get(g, set())
        index[g] = [t for i, t in enumerate(index[g]) if drop != "all" and i not in drop]
        print(g, k, "tiles,", len(index[g]), "kept")
        json.dump(index, open(idx_path, "w"), indent=1)
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build([g for g in a.groups.split(",") if g] or None, a.force)
