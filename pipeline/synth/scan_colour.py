"""Colour-calibrate MetaFood3D scan textures against the real capture frames of the SAME objects.

Why: scan textures are photogrammetry bakes (phone camera + automatic exposure/white balance + baked lighting),
rendered by us under different light -> saturated strawberries, fluorescent omelets, orange shrimp...  MetaFood3D
ships ~20 real photos per object (`data/metafood3d/frames/<Category>__<object>/<k>.jpg` + `<k>_m.png`, a tight
255/0 object mask).  The mask pixels are what the scan *should* look like on a table.

Method (all in CIE Lab, D65, sRGB input):
  T = area-weighted samples of the scan's surface colour (random points on the triangles, UV -> texture lookup,
      weight 0.1 + 0.9*up-facing so that the underside, never seen from a camera above, counts little)
  R = pixels of the eroded masks in all frames of the object (specular-clipped pixels dropped)
  fit: Gaussian optimal transport T -> R in Lab, i.e. a 3x3 matrix + offset (A = S_T^-1/2 (S_T^1/2 S_R S_T^1/2)^1/2
       S_T^-1/2), covariances ridge-regularised and the eigenvalues of A clamped to [A_MIN, A_MAX] so a noisy fit
       cannot blow up the texture; followed by an L-only monotone tone curve (quantile map of the residual, blended
       with weight TONE_W and slope clamped) for skewed brightness distributions.  Lab (not RGB) so the matrix is
       perceptual: lightness and chroma move independently of hue.
  highlights/shadows: baked speculars (very bright, low-chroma) are rolled off above the texture's own P_HI of L
       and the darkest P_LO is lifted a little (baked AO would otherwise double with our own shading).
  objects without frames -> fit pooled over all objects of the SAME category that have frames (their T and R samples
       concatenated, applied as x -> muR + A (x - muT)); categories without any frames -> global correction (median
       L gain / chroma gain only, no hue shift).

Outputs: <out>/<id>/<obj symlink, mtl copy, calibrated texture (<=2048 px, JPEG q92)>, and assets_v6.jsonl = v5 with
`path` swapped for MetaFood3D rows and `colour_calib` = {mode, n_frames, dE_before, dE_after, ...}.
Input textures are read from the v5 row's path, so the hand fixes of asset_qa (BGR swap, cast fixes) are kept and
refined, not redone.

    python -m synth.scan_colour stats   # cache T/R Lab samples per object
    python -m synth.scan_colour fit     # per-object / category / global parameters + error table
    python -m synth.scan_colour write   # textures + assets_v6.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA = Path("/home/kyle/Desktop/FoodSeg/data")
FRAMES = DATA / "metafood3d" / "frames"
V5 = DATA / "synth" / "assets_v5.jsonl"
V6 = DATA / "synth" / "assets_v6.jsonl"
V6_TEX = DATA / "synth" / "assets_v6_tex"
CACHE = DATA / "synth" / "_colour_cache"

MAX_TEX = 2048
N_T = 40000            # surface samples per scan
N_R_FRAME = 3000       # max pixels per frame
ERODE = 3
RIDGE = 6.0            # Lab^2 added to covariance diagonals
A_MIN, A_MAX = 0.55, 1.8
TONE_W = 0.6
P_HI, P_LO = 97.0, 2.0
HI_ROLL = 0.5          # slope of L above the P_HI knee
LO_LIFT = 0.25


# ----------------------------------------------------------------------------------------------------------
# colour helpers
# ----------------------------------------------------------------------------------------------------------
def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or float [0,1] (...,3) -> Lab float32; chunked."""
    from skimage.color import rgb2lab as f
    a = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
    sh = a.shape
    flat = a.reshape(-1, 3)
    out = np.empty_like(flat)
    for i in range(0, len(flat), 1 << 20):
        out[i:i + (1 << 20)] = f(flat[i:i + (1 << 20)][None])[0]
    return out.reshape(sh)


_M = np.array([[3.2406255, -1.5372080, -0.4986286], [-0.9689307, 1.8757561, 0.0415175],
               [0.0557101, -0.2040211, 1.0569959]], np.float32)
GAMUT_CHROMA = 0.0
_WP = np.array([0.95047, 1.0, 1.08883], np.float32)


def _lab2lin(lab: np.ndarray) -> np.ndarray:
    fy = (lab[:, 0] + 16) / 116
    fx, fz = fy + lab[:, 1] / 500, fy - lab[:, 2] / 200
    f = np.stack([fx, fy, fz], 1)
    xyz = np.where(f > 0.20689655, f ** 3, (f - 16 / 116) / 7.787) * _WP
    return xyz @ _M.T


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """Lab -> sRGB [0,1].  Out-of-gamut colours: chroma is reduced by GAMUT_CHROMA (0..1) of the way to the gamut
    surface at constant L/hue, the remainder is clipped per channel.  Full chroma-reduction (1.0) was tried and
    bleaches bright saturated foods (omelet, nuggets, waffle) toward white, pure clipping (0.0) is closest to the
    real photos' saturated look, so the default is 0."""
    sh = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float32)
    out = np.empty(flat.shape, np.float32)
    for i in range(0, len(flat), 1 << 20):
        x = flat[i:i + (1 << 20)].copy()
        x[:, 0] = np.clip(x[:, 0], 0, 100)
        lin = _lab2lin(x)
        bad = (lin.min(1) < -0.002) | (lin.max(1) > 1.002)
        if bad.any() and GAMUT_CHROMA > 0:
            lo, hi = np.zeros(bad.sum(), np.float32), np.ones(bad.sum(), np.float32)
            xb = x[bad]
            for _ in range(10):
                mid = (lo + hi) / 2
                y = xb.copy()
                y[:, 1:] *= mid[:, None]
                l2 = _lab2lin(y)
                ok = (l2.min(1) >= -0.002) & (l2.max(1) <= 1.002)
                lo = np.where(ok, mid, lo)
                hi = np.where(ok, hi, mid)
            xb[:, 1:] *= (1 - GAMUT_CHROMA + GAMUT_CHROMA * lo)[:, None]
            x[bad] = xb
            lin = _lab2lin(x)
        lin = np.clip(lin, 0, 1)
        out[i:i + (1 << 20)] = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(out, 0, 1).reshape(sh)


def de76(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1).mean())


def dist_error(x: np.ndarray, y: np.ndarray, nq: int = 41) -> dict:
    """x, y: (N,3) Lab samples.  mean-colour dE, and the quantile-profile distance (mean over channels'
    quantiles of the Euclidean per-quantile difference) which also sees spread/skew."""
    q = np.linspace(2, 98, nq)
    qx = np.percentile(x, q, axis=0)
    qy = np.percentile(y, q, axis=0)
    return dict(dE_mean=de76(x.mean(0), y.mean(0)), dE_q=float(np.linalg.norm(qx - qy, axis=1).mean()),
                dL=float(x[:, 0].mean() - y[:, 0].mean()),
                dC=float(np.hypot(x[:, 1], x[:, 2]).mean() - np.hypot(y[:, 1], y[:, 2]).mean()))


# ----------------------------------------------------------------------------------------------------------
# frame lookup
# ----------------------------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def frame_dirs(rec: dict) -> Path | None:
    """The frames directory of a v5 MetaFood3D row: `<Category>__<mesh dir name>` (= rec['object']; the .obj file
    name inside the mesh dir can differ, e.g. new_carrot_3/carrot_3.obj, so it is NOT used).  Case/separator
    insensitive; a unique match differing only by plural 's' (new_almonds_3 / new_almond_3) is accepted, a
    different object of the same category (carrot_1 vs new_carrot_1) never is."""
    idx = {_norm(d.name): d for d in FRAMES.iterdir()}
    cat, obj = rec["category"], rec.get("object")
    if not obj:
        return None
    k = _norm(cat + "__" + obj)
    if k in idx:
        return idx[k]
    ks = k.replace("s", "")
    hits = [d for kk, d in idx.items() if kk.replace("s", "") == ks]
    return hits[0] if len(hits) == 1 else None


def real_samples(d: Path, rng: np.random.Generator) -> np.ndarray:
    """Lab samples of the masked (eroded) pixels of all frames of a directory."""
    from PIL import Image
    from scipy.ndimage import binary_erosion
    out = []
    for jp in sorted(d.glob("*.jpg")):
        mp = d / (jp.stem + "_m.png")
        if not mp.exists():
            continue
        im = np.asarray(Image.open(jp).convert("RGB"))
        mk = np.asarray(Image.open(mp)) > 0
        if mk.ndim == 3:
            mk = mk.any(-1)
        if mk.shape != im.shape[:2]:
            continue
        if mk.sum() < 400:
            continue
        e = binary_erosion(mk, iterations=ERODE)
        if e.sum() < 150:
            e = mk
        px = im[e]
        px = px[px.min(1) < 246]               # clipped speculars / blown highlights
        if len(px) < 100:
            continue
        if len(px) > N_R_FRAME:
            px = px[rng.choice(len(px), N_R_FRAME, replace=False)]
        out.append(rgb2lab(px))
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


# ----------------------------------------------------------------------------------------------------------
# texture surface samples
# ----------------------------------------------------------------------------------------------------------
def load_texture(path: str):
    from PIL import Image
    from synth.asset_qa import texture_path
    tex, _ = texture_path(path)
    if tex is None or not os.path.exists(tex):
        return None, None
    im = Image.open(tex).convert("RGB")
    if max(im.size) > MAX_TEX:
        s = MAX_TEX / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im, tex


def tex_samples(rec: dict, img, rng: np.random.Generator) -> np.ndarray | None:
    import trimesh
    me = trimesh.load(rec["path"], process=False, force="mesh")
    uv = getattr(me.visual, "uv", None)
    if uv is None or len(me.faces) == 0:
        return None
    V, F = np.asarray(me.vertices), np.asarray(me.faces)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    nz = n[:, 2] / np.maximum(2 * area, 1e-12)          # up axis is Z in every v5 MetaFood3D row
    w = area * (0.1 + 0.9 * np.clip(0.5 + 0.5 * nz, 0, 1))
    if w.sum() <= 0:
        return None
    fi = rng.choice(len(F), N_T, p=w / w.sum())
    r1, r2 = rng.random(N_T), rng.random(N_T)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    b = np.stack([1 - r1 - r2, r1, r2], 1)
    puv = (np.asarray(uv)[F[fi]] * b[:, :, None]).sum(1)
    tx = np.asarray(img)
    h, wd = tx.shape[:2]
    x = np.clip((puv[:, 0] % 1.0) * wd, 0, wd - 1).astype(int)
    y = np.clip((1 - (puv[:, 1] % 1.0)) * h, 0, h - 1).astype(int)
    return rgb2lab(tx[y, x])


# ----------------------------------------------------------------------------------------------------------
# fit / apply
# ----------------------------------------------------------------------------------------------------------
def _sqrtm(S):
    e, v = np.linalg.eigh(S)
    return (v * np.sqrt(np.maximum(e, 1e-9))) @ v.T


def _invsqrtm(S):
    e, v = np.linalg.eigh(S)
    return (v / np.sqrt(np.maximum(e, 1e-9))) @ v.T


def fit(T: np.ndarray, R: np.ndarray) -> dict:
    """Gaussian OT T -> R in Lab (+ L tone curve).  Returns JSON-able params."""
    muT, muR = T.mean(0), R.mean(0)
    ST = np.cov(T.T) + RIDGE * np.eye(3)
    SR = np.cov(R.T) + RIDGE * np.eye(3)
    iT, sT = _invsqrtm(ST), _sqrtm(ST)
    A = iT @ _sqrtm(sT @ SR @ sT) @ iT
    e, v = np.linalg.eigh((A + A.T) / 2)
    A = (v * np.clip(e, A_MIN, A_MAX)) @ v.T
    p = dict(muT=muT.tolist(), muR=muR.tolist(), A=A.tolist(), tone=None)
    # L tone curve on the residual after the affine part
    Ta = apply_affine(T, p)
    qs = np.linspace(3, 97, 25)
    src = np.percentile(Ta[:, 0], qs)
    dst = np.percentile(R[:, 0], qs)
    # blend, enforce strictly increasing output and slope limits
    out = src + TONE_W * (dst - src)
    for i in range(1, len(out)):
        d_src = max(src[i] - src[i - 1], 1e-3)
        out[i] = np.clip(out[i], out[i - 1] + 0.6 * d_src, out[i - 1] + 1.5 * d_src)
    p["tone"] = [src.tolist(), out.tolist()]
    return p


def apply_affine(lab: np.ndarray, p: dict) -> np.ndarray:
    return np.asarray(p["muR"]) + (lab - np.asarray(p["muT"])) @ np.asarray(p["A"]).T


def apply_tone(L: np.ndarray, tone) -> np.ndarray:
    if tone is None:
        return L
    src, dst = np.asarray(tone[0]), np.asarray(tone[1])
    y = np.interp(L, src, dst)
    lo, hi = L < src[0], L > src[-1]
    y[lo] = dst[0] + (L[lo] - src[0])
    y[hi] = dst[-1] + (L[hi] - src[-1])
    return y


def apply(lab: np.ndarray, p: dict, ref: np.ndarray | None = None) -> np.ndarray:
    """Calibrate Lab values (any shape (...,3)) with params p.  Highlight roll-off / shadow lift are relative to
    the percentiles of `ref` (the texture's own samples; default: lab itself)."""
    sh = lab.shape
    x = lab.reshape(-1, 3)
    if "gain" in p:                                        # global fallback: L gain + chroma gain only
        y = x.copy()
        y[:, 0] = np.clip(y[:, 0] * p["gain"][0], 0, 100)
        y[:, 1:] *= p["gain"][1]
    else:
        y = apply_affine(x, p)
        y[:, 0] = apply_tone(y[:, 0], p["tone"])
    r = apply_affine(ref.reshape(-1, 3), p) if ("gain" not in p and ref is not None) else y
    if "gain" not in p:
        r = r.copy()
        r[:, 0] = apply_tone(r[:, 0], p["tone"])
        hi, lo = np.percentile(r[:, 0], P_HI), np.percentile(r[:, 0], P_LO)
        m = y[:, 0] > hi
        y[m, 0] = hi + HI_ROLL * (y[m, 0] - hi)
        m = y[:, 0] < lo
        y[m, 0] = lo + (1 - LO_LIFT) * (y[m, 0] - lo)
    y[:, 0] = np.clip(y[:, 0], 0, 100)
    return y.reshape(sh)


def apply_clipped(lab: np.ndarray, p: dict, ref: np.ndarray) -> np.ndarray:
    """apply() followed by the sRGB gamut clip the written texture gets (what a render would actually see)."""
    return rgb2lab(lab2rgb(apply(lab, p, ref)))


def apply_image(img, p: dict, ref: np.ndarray):
    from PIL import Image
    a = np.asarray(img)
    lab = rgb2lab(a)
    out = lab2rgb(apply(lab, p, ref))
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------------------------------------
def _stats_one(rec: dict):
    """T (slow: OBJ parse) cached as <id>.npz; R (fast) as <id>.R.npz, recomputed if the frame dir changes."""
    rid = rec["id"]
    try:
        tf, rf = CACHE / f"{rid}.npz", CACHE / f"{rid}.R.npz"
        rng = np.random.default_rng(zlib.crc32(rid.encode()))
        if not tf.exists():
            img, _ = load_texture(rec["path"])
            if img is None:
                return rid, "notex"
            T = tex_samples(rec, img, rng)
            if T is None:
                return rid, "nouv"
            np.savez_compressed(tf, T=T.astype(np.float16))
        fd = frame_dirs(rec)
        if not rf.exists() or str(np.load(rf)["fd"]) != str(fd or ""):
            R = real_samples(fd, rng) if fd else np.zeros((0, 3), np.float32)
            np.savez_compressed(rf, R=R.astype(np.float16), fd=str(fd or ""))
        return rid, "ok"
    except Exception as e:  # noqa: BLE001
        return rid, f"ERR {e}"


def load_v5():
    return [json.loads(l) for l in open(V5)]


def cmd_stats(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load_v5() if r["source"] == "metafood3d"]
    with Pool(args.workers) as p:
        res = p.map(_stats_one, recs, chunksize=2)
    from collections import Counter
    print(Counter(s.split()[0] for _, s in res))
    for i, s in res:
        if s.startswith(("ERR", "notex", "nouv")):
            print(i, s)


def load_cache(rid):
    T = np.load(CACHE / f"{rid}.npz")["T"].astype(np.float32)
    z = np.load(CACHE / f"{rid}.R.npz")
    return T, z["R"].astype(np.float32), str(z["fd"])


# categories without any capture frames -> real-photo statistics of the closest visual category that has some
ALIAS = {
    "Shrimp": ["Lobster", "Crab"], "Blueberries": ["Blueberry(bowl)"], "Cookies": ["Biscuits"],
    "Brownies": ["Chocolate", "Cake"], "Sandwich": ["Peanut butter and jelly sandwiches"],
    "Baked_Potato": ["Sweet Potato"], "French_toast": ["Pancake", "Waffle"], "Pork_Chop": ["Steak", "Chicken_thighs"],
    "Pork_Rib": ["Chicken_wings", "Steak"], "Sushi": ["Salmon_Grill", "Rice"], "Whole_Chicken": ["Chicken_thighs"],
    "Quickbread": ["Muffin", "Cake"], "Frankfurter_Sandwich": ["Corn dog", "Burger"],
}


def compute_params(recs: list[dict]) -> dict:
    """{id: {mode, p}} for every MetaFood3D row with a cached texture.
    object   : the object's own frames (fit(T_obj, R_obj))
    category : no frames for this object -> matrix fitted on the pooled T and R of the OTHER objects of its category
               (leave-one-out dE 22.5 -> 14.1; matching the object's own T to the pooled R is no better than the
               category-mean colour, 17.3: real colour varies by dE ~17 between capture sessions of one category,
               the pooled T->R matrix also carries the typical session exposure/white balance)
    alias    : category has no frames at all -> matrix from the pooled T of that category to the pooled real
               pixels of ALIAS[category] (a visually similar category)
    global   : last resort, L/chroma gain only"""
    C = {}
    for r in recs:
        if (CACHE / f"{r['id']}.npz").exists() and (CACHE / f"{r['id']}.R.npz").exists():
            T, R, fd = load_cache(r["id"])
            C[r["id"]] = (r, T, R)
    bycat = defaultdict(list)
    for rid, (r, T, R) in C.items():
        if len(R) >= 500:
            bycat[r["category"]].append(rid)
    P = {}
    for rid in bycat_all(bycat):
        r, T, R = C[rid]
        P[rid] = dict(mode="object", p=fit(T, R))
    gl, gc = [], []
    for rid in P:
        r, T, R = C[rid]
        gl.append(R[:, 0].mean() / max(T[:, 0].mean(), 1))
        gc.append(np.hypot(R[:, 1], R[:, 2]).mean() / max(np.hypot(T[:, 1], T[:, 2]).mean(), 1))
    glob = dict(gain=[float(np.median(gl)), float(np.median(gc))])
    pool = {}

    def rpool(cats):
        k = tuple(cats)
        if k not in pool:
            pool[k] = np.concatenate([C[i][2] for c in cats for i in bycat.get(c, [])] or [np.zeros((0, 3), np.float32)])
        return pool[k]

    tpool = defaultdict(list)
    for rid, (r, T, R) in C.items():
        tpool[r["category"]].append(T)
    catfit = {}
    for rid, (r, T, R) in C.items():
        if rid in P:
            continue
        cat = r["category"]
        al = [c for c in ALIAS.get(cat, []) if bycat.get(c)]
        if bycat.get(cat):
            if cat not in catfit:
                catfit[cat] = fit(np.concatenate([C[i][1] for i in bycat[cat]]), rpool([cat]))
            P[rid] = dict(mode="category", p=catfit[cat])
        elif al:
            k = ("alias", cat)
            if k not in catfit:
                catfit[k] = fit(np.concatenate(tpool[cat]), rpool(al))
            P[rid] = dict(mode="alias", p=catfit[k])
        else:
            P[rid] = dict(mode="global", p=glob)
    return dict(P=P, C=C, glob=glob, bycat=bycat)


def bycat_all(bycat):
    for ids in bycat.values():
        yield from ids


def evaluate(S: dict) -> list[dict]:
    rows = []
    for rid, d in S["P"].items():
        r, T, R = S["C"][rid]
        if len(R) < 500:
            continue
        before = dist_error(T, R)
        after = dist_error(apply_clipped(T, d["p"], T), R)
        rows.append(dict(id=rid, cat=r["category"], mode=d["mode"], before=before, after=after))
    return rows


def loo_category(S: dict) -> list[dict]:
    """Honest error of the category fallback: matrix fitted on the pooled T/R of the OTHER objects of the category,
    tested on this object's own real frames."""
    out = []
    for cat, ids in S["bycat"].items():
        if len(ids) < 2:
            continue
        for rid in ids:
            o = [i for i in ids if i != rid]
            p = fit(np.concatenate([S["C"][i][1] for i in o]), np.concatenate([S["C"][i][2] for i in o]))
            r, T, R = S["C"][rid]
            out.append(dict(id=rid, cat=cat, before=dist_error(T, R), after=dist_error(apply_clipped(T, p, T), R)))
    return out


def split_half(S: dict) -> list[dict]:
    """Per-object fit on odd frames' samples, tested on even ones (R is stored frame-major: 3000/frame cap)."""
    out = []
    rng = np.random.default_rng(0)
    for rid, d in S["P"].items():
        if d["mode"] != "object":
            continue
        r, T, R = S["C"][rid]
        idx = rng.permutation(len(R))
        a, b = R[idx[: len(R) // 2]], R[idx[len(R) // 2:]]
        # frame-level split is stricter: contiguous halves (frames are in numeric-string order, views interleave)
        h = len(R) // 2
        a, b = R[:h], R[h:]
        p = fit(T, a)
        out.append(dict(id=rid, cat=r["category"], before=dist_error(T, b), after=dist_error(apply_clipped(T, p, T), b)))
    return out


def summarize(rows: list[dict], title: str) -> dict:
    if not rows:
        return {}
    f = lambda k, s: float(np.mean([x[s][k] for x in rows]))  # noqa: E731
    d = dict(n=len(rows), dE_mean_before=f("dE_mean", "before"), dE_mean_after=f("dE_mean", "after"),
             dE_q_before=f("dE_q", "before"), dE_q_after=f("dE_q", "after"))
    print(f"{title}: n={d['n']}  mean-colour dE {d['dE_mean_before']:.1f} -> {d['dE_mean_after']:.1f}   "
          f"quantile-profile dE {d['dE_q_before']:.1f} -> {d['dE_q_after']:.1f}")
    return d


def cmd_fit(args):
    recs = [r for r in load_v5() if r["source"] == "metafood3d"]
    S = compute_params(recs)
    summarize(evaluate(S), "in-sample (per-object fit, own frames)")
    summarize(split_half(S), "held-out frames (fit on half the pixels/frames, test on rest)")
    summarize(loo_category(S), "category fallback, leave-one-object-out")
    rows = evaluate(S)
    bycat = defaultdict(list)
    for x in rows:
        bycat[x["cat"]].append(x)
    print("per category (before -> after, mean-colour dE):")
    for c, xs in sorted(bycat.items(), key=lambda kv: -np.mean([x["before"]["dE_mean"] for x in kv[1]])):
        print(f"  {c:28s} n={len(xs):2d}  {np.mean([x['before']['dE_mean'] for x in xs]):5.1f} -> "
              f"{np.mean([x['after']['dE_mean'] for x in xs]):5.1f}   q {np.mean([x['before']['dE_q'] for x in xs]):5.1f}"
              f" -> {np.mean([x['after']['dE_q'] for x in xs]):5.1f}")
    print("global gains", S["glob"])


def cmd_write(args):
    recs = load_v5()
    mf = [r for r in recs if r["source"] == "metafood3d"]
    S = compute_params(mf)
    ev = {x["id"]: x for x in evaluate(S)}
    V6_TEX.mkdir(parents=True, exist_ok=True)
    jobs = [(r, S["P"][r["id"]], S["C"][r["id"]][1]) for r in mf if r["id"] in S["P"]
            and (not args.only or re.search(args.only, r["id"]))]
    with Pool(args.workers) as p:
        res = dict(p.map(_write_one, jobs, chunksize=2))
    n = defaultdict(int)
    if args.only:
        print("partial write:", len(res), "textures; assets_v6.jsonl not rewritten")
        return
    with open(V6, "w") as f:
        for r in recs:
            if r["source"] == "metafood3d":
                rid = r["id"]
                if rid in res and res[rid] is not None:
                    d = S["P"][rid]
                    e = ev.get(rid)
                    r = dict(r, path_v5=r["path"], path=res[rid])
                    r["colour_calib"] = dict(mode=d["mode"], dE_before=e and round(e["before"]["dE_mean"], 2),
                                             dE_after=e and round(e["after"]["dE_mean"], 2),
                                             dEq_before=e and round(e["before"]["dE_q"], 2),
                                             dEq_after=e and round(e["after"]["dE_q"], 2))
                    n[d["mode"]] += 1
                else:
                    r = dict(r, colour_calib=dict(mode="skipped"))
                    n["skipped"] += 1
            f.write(json.dumps(r) + "\n")
    print(dict(n))


def _write_one(job):
    r, d, T = job
    try:
        from synth.asset_qa import texture_path
        img, tex = load_texture(r["path"])
        _, mtl = texture_path(r["path"])
        out = V6_TEX / r["id"]
        out.mkdir(parents=True, exist_ok=True)
        obj = out / Path(r["path"]).name
        if obj.is_symlink() or obj.exists():
            obj.unlink()
        # keep the file's real location: resolve v4 symlinks so v6 does not chain
        obj.symlink_to(Path(r["path"]).resolve())
        shutil.copy(mtl, out / Path(mtl).name)
        res = apply_image(img, d["p"], T)
        tname = Path(tex).name
        res.save(out / Path(tname).with_suffix(".jpg"), quality=92, subsampling=0)
        if Path(tname).suffix.lower() != ".jpg":          # MTL must point to the .jpg we wrote
            txt = (out / Path(mtl).name).read_text(errors="ignore")
            txt = txt.replace(tname, Path(tname).with_suffix(".jpg").name)
            (out / Path(mtl).name).write_text(txt)
        return r["id"], str(obj)
    except Exception as e:  # noqa: BLE001
        print("ERR", r["id"], e)
        return r["id"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "fit", "write"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="write: regex on asset id; textures only, no jsonl")
    a = ap.parse_args()
    {"stats": cmd_stats, "fit": cmd_fit, "write": cmd_write}[a.stage](a)


if __name__ == "__main__":
    main()
