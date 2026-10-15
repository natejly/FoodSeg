"""How distinguishable are our renders from real food photos? (a "realism probe")

    python -m synth.realism --synth ../data/synth/hard_v10 ../data/synth/hard_v9 \
        --real n5k mf3d --out ../results/synth/realism [--mode patch image] [--style cafeteria]

Why: "looks real" is otherwise an eyeball judgement. Here a linear classifier on frozen
DINOv2 features tries to tell render from photo; held-out accuracy / AUC of 50 % / 0.5
means this probe cannot (it is a lower bound on detectability: a stronger probe or a
fine-tuned net would do better). The probe's most confident "synthetic" renders are
saved as a contact sheet, so the output is also a to-do list of what looks fakest.

Two modes (--mode, default both), each run per (synthetic batch x viewpoint x real set)
plus a pooled row over all given batches:

  patch (default; the one to rank rendering quality with)
      Equal-sized local patches (--patch-px 224 working pixels, --patches-per-image 8)
      centred on FOOD pixels on both sides, so the probe sees texture / shading / noise /
      colour of the food and not the layout of the scene.
        * synthetic food = seg labels 1..252; real MetaFood3D food = the *_m.png mask;
          real Nutrition5k food = a colour heuristic (detect_food_n5k: the bright,
          unsaturated plate component nearest the image centre is found, holes are filled,
          and non-plate-coloured pixels inside it are food; plate/tray pixels are not).
        * scale matching: real images stay at native resolution (640x480 / 448x336); the
          synthetic frame is resampled so its reference object has the same pixel size as
          the real one: dish bbox -> 380 px (N5k plates fill ~0.8 of the 480 px frame) /
          food bbox -> each real image's own food bbox is rescaled to the real-set median
          food bbox (MF3D), applied identically on both sides. Factor clipped to 0.4-2.5.
        * a patch needs >= --min-food (0.25) food pixels, placed greedily at random with
          centres >= patch/2 apart, inside the frame (no padding). Patch food coverage is
          reported for both sides; it is not equalised, so check it is similar.
        * CV groups = synth scan / real dish or capture dir (all patches and frames of
          one scene stay in one fold). Scores are also averaged per image -> image level.
  image
      Whole-dish crops (the original behaviour; see below). Confounded by scene content.
        top-down : synth frames 6,7 vs Nutrition5k realsense_overhead rgb.png (real
                   cafeteria dishes). Synth crop = dish bbox (+15 %), real crop = central
                   square (the dish fills ~85 % of it, so both show "plate fills frame").
        orbit    : synth frames 0-5 (~45 deg) vs MetaFood3D capture frames (single foods on
                   black cloth with fiducials). Crop = square of 2x the food bbox.
  Nothing real from an iPad exists locally, so these two are the nearest domains.

Real-vs-real baseline (--real-pairs A:B [--n-images N]): the identical pipeline with real set
A as class 1 and real set B as class 0, e.g. `--real-pairs n5k:mf3d mf3d_a:mf3d_b`. Defines
the ceiling for "indistinguishable": a synthetic set that is no more separable than one
real dataset is from another is as close as a photo set from another source. mf3d_a/b split
food categories, n5k_a/b split dishes (a hash of the id). Patch mode uses one common patch
size (96 px) on both sides; image mode uses each set's usual crop (N5k central square,
MF3D food bbox x2), so n5k:mf3d keeps the content confound of the synthetic comparison.

Matching / confound control:
  * synth is resampled to the real set's resolution (image mode: whole frame to native
    width; patch mode: as above), cropped, resized to --size+32, JPEG re-encoded (q=90) on
    BOTH sides, then resized to --size;
  * real set is subsampled to the same number of images as the synthetic set (seeded, the
    real list is shuffled once so smaller sets are nested in larger ones);
  * CV is StratifiedGroupKFold (5 folds x 3 seeds, scores averaged), standardised features,
    L2 logistic regression (C=0.1, balanced class weights);
  * --style NAME keeps only synthetic scenes whose spec.json has "style": NAME (e.g.
    cafeteria, to compare only against Nutrition5k); scenes without the key are dropped.
Backbone: timm `vit_base_patch14_dinov2.lvd142m` (DINOv2 ViT-B/14, from the HF cache,
offline). Feature = [CLS, mean patch token] (1536-d), fp16 on GPU if >2 GB free else CPU.

Metrics (all from out-of-fold probe scores = logit of P(synthetic); S = synth, R = real):
  accuracy, auc        saturate at 1.0 when the sets are trivially separable
  gap                  mean score(S) - mean score(R) (log-odds units; 0 = indistinguishable).
                       Keeps moving after AUC hits 1.0 - the number to rank batches with
  p_synth / p_real     mean predicted P(synthetic) on synth / real items (ideal: both 0.5)
  afc2                 human-style 2AFC proxy: for every (synth, real) pair an observer that
                       sees the probe's score plus Gaussian internal noise (sd HUMAN_SIGMA =
                       5 logits, --human-sigma) picks the "more synthetic" one; P(correct)
                       = mean Phi(dscore / (sqrt2 * sigma)). 0.5 = guessing; a larger sigma
                       models a weaker observer. Monotone in gap, so it is a rescaling of it
  centroid gap         same, for a nearest-centroid probe (one fixed synth-minus-real mean
                       direction, standardised-feature units, out-of-fold, with CI). A fitted
                       1536-d hyperplane saturates (gap ~11 logits for any batch at these
                       sample sizes); the centroid gap keeps resolving differences. Compare
                       it to the control's |centroid gap| (~1-2.5)
  dprime, kid          d' of the scores; KID (unbiased cubic-kernel MMD^2, smaller = closer;
                       +- = spread over random 100-sample subsets, not a group CI)
  each reported at patch level and image level (scores averaged over an image's patches),
  with 95 % bootstrap CIs resampling scene groups (synth scans and real dishes separately),
  and a real-vs-real control (real groups split at random in halves, 5 splits, same n):
  that control must read ~0.5 / gap ~0 else the probe is learning something spurious.
Outputs in --out/<synthname>__<mode>__<view>__<real>/ : report.json, most_fake.jpg,
most_real.jpg, pairs.jpg; and --out/report.json aggregating all comparisons.
Caveats: real sets differ from our scenes in content (cafeteria tray / black cloth with
markers), so even patch mode is not content-free; N5k food mask is a heuristic (misses
white / pale foods, may include tray edges); small synthetic sets give wide CIs. Compare
numbers between batches (same real sample, same settings) rather than reading the absolute.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA = Path(os.environ.get("FOODSEG_DATA", "/home/kyle/Desktop/FoodSeg/data"))
N5K = DATA / "datasets/Nutrition5k/imagery/realsense_overhead"
MF3D = DATA / "metafood3d/frames"
TOP_FRAMES, ORBIT_FRAMES = (6, 7), (0, 1, 2, 3, 4, 5)
NATIVE = {"n5k": (640, 480), "mf3d": (448, 336)}   # (w, h) synth is resampled to
N5K_DISH_PX = 380                                  # patch mode: dish bbox side after scaling
JPEG_Q = 90
HUMAN_SIGMA = 5.0                                  # 2AFC proxy: observer internal noise sd (logits)
PATCH_PX = {"n5k": 224, "mf3d": 96}                # default patch side (working px) per real set
SEG_DISH, SEG_FOOD_MAX = 254, 253                  # seg: 254 dish, 1..252 food items


# ---------------------------------------------------------------- crops
def square_crop(im: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    """Square crop centred on (cx, cy); side is clamped and the window shifted to stay inside
    the frame (edge padding made smeared streaks that the probe latched onto)."""
    w, h = im.size
    side = int(round(min(max(side, 16), w, h)))
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return im.crop((x0, y0, x0 + side, y0 + side))


def bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def finish(crop: Image.Image, size: int) -> Image.Image:
    """Common tail for both sides: resize, JPEG re-encode, resize to the model size."""
    crop = crop.convert("RGB").resize((size + 32, size + 32), Image.BICUBIC)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=JPEG_Q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((size, size), Image.BICUBIC)


def synth_crop(rgb: Path, seg_path: Path, view: str, real: str, size: int):
    im = Image.open(rgb).convert("RGB")
    seg = np.load(seg_path)["seg"]
    s = im.size[0] / seg.shape[1]
    if view == "top":
        bb = bbox(seg == SEG_DISH) or bbox((seg > 0) & (seg < SEG_FOOD_MAX))
        pad = 1.15
    else:
        bb = bbox((seg > 0) & (seg < SEG_FOOD_MAX))
        pad = 2.0
    if bb is None:
        return None
    nw, nh = NATIVE[real]
    k = nw / im.size[0]                       # resample to the real set's native resolution
    im = im.resize((nw, round(im.size[1] * k)), Image.LANCZOS)
    x0, y0, x1, y1 = [v * s * k for v in bb]
    return finish(square_crop(im, (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * pad), size)


def real_crop(path: Path, real: str, size: int):
    im = Image.open(path).convert("RGB")
    if real == "n5k":
        w, h = im.size
        return finish(square_crop(im, w / 2, h / 2, min(w, h)), size)
    m = path.with_name(path.stem + "_m.png")
    bb = bbox(np.asarray(Image.open(m)) > 0) if m.exists() else None
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    return finish(square_crop(im, (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 2.0), size)


# ---------------------------------------------------------------- food patches
def detect_food_n5k(im: Image.Image) -> np.ndarray:
    """Heuristic food mask for a Nutrition5k overhead frame (bool, full res).
    Plate = bright unsaturated (HSV S<0.2, V>0.5) connected component weighted towards the
    image centre; its holes are filled and what inside is NOT plate-coloured is food.
    Misses white/pale foods; the plate component may leak into the white tray background
    (harmless: only non-plate-coloured pixels count as food)."""
    from scipy import ndimage as ndi
    w, h = im.size
    s = 4
    sm = im.resize((w // s, h // s), Image.BILINEAR)
    hsv = np.asarray(sm.convert("HSV"), np.float32) / 255
    plateish = (hsv[..., 1] < 0.2) & (hsv[..., 2] > 0.5)
    po = ndi.binary_opening(plateish, np.ones((5, 5)))
    lab, n = ndi.label(po)
    if n == 0:
        return np.zeros((h, w), bool)
    hh, ww = po.shape
    yy, xx = np.mgrid[:hh, :ww]
    wgt = np.exp(-(((xx - ww / 2) / (0.35 * ww)) ** 2 + ((yy - hh / 2) / (0.35 * hh)) ** 2))
    best = 1 + int(np.argmax(ndi.sum(wgt, lab, np.arange(1, n + 1))))
    plate = lab == best
    filled = ndi.binary_fill_holes(ndi.binary_closing(plate, np.ones((9, 9))))
    food = ndi.binary_opening(filled & ~plateish, np.ones((3, 3)))
    return np.asarray(Image.fromarray((food * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0


def sample_patches(img: Image.Image, food: np.ndarray, n: int, px: int, min_food: float,
                   rng: random.Random):
    """Up to n px*px windows fully inside img with >= min_food food-pixel fraction, greedily
    chosen at random with centres >= px/2 apart. Returns [(crop, food_fraction)]."""
    w, h = img.size
    if w < px or h < px:
        return []
    ii = np.pad(food.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    st = 16
    xs, ys = np.arange(0, w - px + 1, st), np.arange(0, h - px + 1, st)
    X, Y = np.meshgrid(xs, ys)
    frac = (ii[Y + px, X + px] - ii[Y, X + px] - ii[Y + px, X] + ii[Y, X]) / float(px * px)
    cand = [(frac[j, i], X[j, i], Y[j, i]) for j in range(len(ys)) for i in range(len(xs))
            if frac[j, i] >= min_food]
    rng.shuffle(cand)
    out, ctr = [], []
    for f, x, y in cand:
        if all(max(abs(x - a), abs(y - b)) >= px // 2 for a, b in ctr):
            ctr.append((x, y))
            out.append((img.crop((int(x), int(y), int(x) + px, int(y) + px)), float(f)))
            if len(out) >= n:
                break
    return out


def clip_scale(k: float) -> float:
    return float(min(max(k, 0.4), 2.5))


def px_for(a, real: str) -> int:
    return a.patch_px or PATCH_PX[real]


def synth_patches(rgb: Path, seg_path: Path, view: str, real: str, a, rng, ref_px: float):
    """Food patches of one synthetic frame, scaled so the reference object (dish for top-down
    / food bbox for orbit) has ref_px pixels (the real-side value)."""
    im = Image.open(rgb).convert("RGB")
    seg = np.load(seg_path)["seg"]
    food = (seg > 0) & (seg < SEG_FOOD_MAX)
    bb = (bbox(seg == SEG_DISH) or bbox(food)) if view == "top" else bbox(food)
    if bb is None:
        return []
    s = im.size[0] / seg.shape[1]
    k = clip_scale(ref_px / (max(bb[2] - bb[0], bb[3] - bb[1]) * s))
    sz = (round(im.size[0] * k), round(im.size[1] * k))
    im = im.resize(sz, Image.LANCZOS)
    fm = np.asarray(Image.fromarray((food * 255).astype(np.uint8)).resize(sz, Image.NEAREST)) > 0
    return sample_patches(im, fm, a.patches_per_image, px_for(a, real), a.min_food, rng)


def real_patches(path: Path, real: str, a, rng, ref_px: float):
    im = Image.open(path).convert("RGB")
    if real == "n5k":
        food = detect_food_n5k(im)
    else:
        m = path.with_name(path.stem + "_m.png")
        if not m.exists():
            return []
        food = np.asarray(Image.open(m)) > 0
        bb = bbox(food)
        if bb is None:
            return []
        k = clip_scale(ref_px / max(bb[2] - bb[0], bb[3] - bb[1]))
        sz = (round(im.size[0] * k), round(im.size[1] * k))
        im = im.resize(sz, Image.LANCZOS)
        food = np.asarray(Image.fromarray((food * 255).astype(np.uint8)).resize(sz, Image.NEAREST)) > 0
    return sample_patches(im, food, a.patches_per_image, px_for(a, real), a.min_food, rng)


def real_ref_px(real: str, items) -> float:
    """Reference size (px) the synthetic frames are scaled to: fixed dish size for N5k;
    median real food-bbox side for MF3D."""
    if real == "n5k":
        return float(N5K_DISH_PX)
    sides = []
    for p, _ in items[:60]:
        bb = bbox(np.asarray(Image.open(p.with_name(p.stem + "_m.png"))) > 0)
        if bb:
            sides.append(max(bb[2] - bb[0], bb[3] - bb[1]))
    return float(np.median(sides)) if sides else 200.0


# ---------------------------------------------------------------- item lists
def spec_style(batch: Path, sid: str):
    p = batch / "specs" / sid / "spec.json"
    try:
        return json.loads(p.read_text()).get("style")
    except (OSError, ValueError):
        return None


def synth_items(batch: Path, view: str, style: str | None = None):
    """[(rgb, seg, group)] for frames of one viewpoint that have both rgb and raw seg
    (and, with `style`, whose spec.json style matches)."""
    out, ok = [], {}
    for rgb in sorted(glob.glob(str(batch / "scans/*/rgb/*.jpg"))):
        rgb = Path(rgb)
        sid, idx = rgb.parent.parent.name, int(rgb.stem)
        if idx not in (TOP_FRAMES if view == "top" else ORBIT_FRAMES):
            continue
        if style is not None:
            if sid not in ok:
                ok[sid] = spec_style(batch, sid) == style
            if not ok[sid]:
                continue
        seg = batch / "raw" / sid / f"frame_{idx:06d}.npz"
        if seg.exists():
            out.append((rgb, seg, f"{batch.name}/{sid}"))
    return out


def real_base(real: str) -> str:
    """'n5k_a' -> 'n5k', 'mf3d_b' -> 'mf3d'."""
    return real.split("_")[0]


def real_part(real: str):
    """None, or 'a' / 'b' for the two disjoint halves of a real set (real-vs-real controls):
    n5k_a/n5k_b split dishes by a hash of the dish id, mf3d_a/mf3d_b split FOOD CATEGORIES
    (so the two halves also differ in content, not just in capture session)."""
    return real.split("_")[1] if "_" in real else None


def in_part(real: str, group: str) -> bool:
    part = real_part(real)
    if part is None:
        return True
    if real_base(real) == "n5k":
        h = zlib.crc32(group.encode()) % 2
    else:
        h = zlib.crc32(group.split("__")[0].encode() + b"cat-split-1") % 2
    return h == (0 if part == "a" else 1)


def real_items(real: str, n: int, rng: random.Random, per_group: int):
    """Real (path, group) list from a seeded shuffle (callers take a prefix of it, so a
    smaller request is nested in a larger one); 2x n for slack when crops/patches fail."""
    if real_base(real) == "n5k":
        dishes = sorted(p for p in N5K.glob("dish_*/rgb.png") if in_part(real, p.parent.name))
        rng.shuffle(dishes)
        return [(p, p.parent.name) for p in dishes[:max(n * 2, 50)]]
    dirs = sorted(d for d in MF3D.iterdir() if d.is_dir() and in_part(real, d.name))
    rng.shuffle(dirs)
    out = []
    for d in dirs:
        fr = sorted(p for p in d.glob("*.jpg") if (d / (p.stem + "_m.png")).exists())
        for p in rng.sample(fr, min(per_group, len(fr))):
            out.append((p, d.name))
        if len(out) >= n * 2:
            break
    return out


# ---------------------------------------------------------------- features
def load_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import timm
    free = 0
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 2**30
    dev = "cuda" if free > 2 else "cpu"
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                          img_size=224).eval().to(dev)
    if dev == "cuda":
        m = m.half()
    return m, dev


def features(model, dev, imgs, size, bs=16):
    import torch
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = np.stack([(np.asarray(im.resize((size, size)), np.float32) / 255 - mean) / std
                          for im in imgs[i:i + bs]]).transpose(0, 3, 1, 2)
            x = torch.from_numpy(x).to(dev)
            if dev == "cuda":
                x = x.half()
            tok = model.forward_features(x)
            f = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- statistics
def probe(X, y, groups, seeds=(0, 1, 2)):
    """Out-of-fold decision scores of a group-CV logistic regression (mean over seeds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    scores = np.zeros(len(y))
    for sd in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(X, y, groups):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            scores[te] += clf.decision_function(X[te]) / len(seeds)
    return scores


def probe_centroid(X, y, groups, seeds=(0, 1, 2)):
    """Out-of-fold scores of a nearest-centroid probe: project onto the (standardised) synth-minus-
    real mean direction w/|w| fitted on the training folds. One fixed direction instead of a
    fitted 1536-d hyperplane, so it does not saturate: gap = held-out distance between the
    class centroids along that direction in standardised-feature units (0 = same mean)."""
    from sklearn.model_selection import StratifiedGroupKFold
    scores = np.zeros(len(y))
    for sd in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(X, y, groups):
            mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
            w = (X[tr][y[tr] == 1] - mu).mean(0) / sg - (X[tr][y[tr] == 0] - mu).mean(0) / sg
            w /= np.linalg.norm(w) + 1e-9
            scores[te] += (((X[te] - mu) / sg) @ w) / len(seeds)
    return scores


def kid(a, b, n_sub=100, n_rep=50, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = np.concatenate([a, b]).mean(0), np.concatenate([a, b]).std(0) + 1e-6
    a, b = (a - mu) / sd, (b - mu) / sd
    d = a.shape[1]
    k = lambda x, y: (x @ y.T / d + 1) ** 3
    m = min(n_sub, len(a), len(b))
    vals = []
    for _ in range(n_rep):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        vals.append((kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def score_metrics(ss, sr):
    """Metrics from probe scores of synthetic (ss) and real (sr) items."""
    from scipy.special import ndtr
    from scipy.stats import rankdata
    n1, n0 = len(ss), len(sr)
    r = rankdata(np.concatenate([ss, sr]))
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    d = ss[:, None] - sr[None, :]
    sig = lambda z: 1 / (1 + np.exp(-z))
    return {"accuracy": float(((ss > 0).mean() + (sr <= 0).mean()) / 2), "auc": float(auc),
            "gap": float(ss.mean() - sr.mean()),
            "p_synth": float(sig(ss).mean()), "p_real": float(sig(sr).mean()),
            "afc2": float(ndtr(d / (np.sqrt(2) * HUMAN_SIGMA)).mean()),
            "dprime": float((ss.mean() - sr.mean()) / np.sqrt((ss.var() + sr.var()) / 2 + 1e-9))}


def bootstrap(ss, gs, sr, gr, n_boot, seed=0):
    """95 % CIs of score_metrics, resampling synth groups and real groups separately."""
    rng = np.random.default_rng(seed)
    gs, gr = np.asarray(gs), np.asarray(gr)
    us, ur = np.unique(gs), np.unique(gr)
    ix_s = {g: np.nonzero(gs == g)[0] for g in us}
    ix_r = {g: np.nonzero(gr == g)[0] for g in ur}
    rows = []
    for _ in range(n_boot):
        i = np.concatenate([ix_s[g] for g in rng.choice(us, len(us))])
        j = np.concatenate([ix_r[g] for g in rng.choice(ur, len(ur))])
        rows.append(score_metrics(ss[i], sr[j]))
    return {k: [float(np.percentile([r[k] for r in rows], 2.5)),
                float(np.percentile([r[k] for r in rows], 97.5))] for k in rows[0]}


def per_image(sc, iid):
    """Average scores over each image's patches -> (scores, image ids in first-seen order)."""
    order = list(dict.fromkeys(iid))
    idx = {k: i for i, k in enumerate(order)}
    tot, cnt = np.zeros(len(order)), np.zeros(len(order))
    for s, k in zip(sc, iid):
        tot[idx[k]] += s
        cnt[idx[k]] += 1
    return tot / cnt, order


# ---------------------------------------------------------------- sheets
def sheet(imgs, labels, path, cols=6, cell=224):
    rows = (len(imgs) + cols - 1) // cols
    W = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
    d = ImageDraw.Draw(W)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        W.paste(im.resize((cell, cell)), (x, y + 16))
        d.text((x + 3, y + 2), lb, fill="black")
    W.save(path, quality=90)


def pairs_sheet(real_imgs, syn_imgs, path, n=12, seed=0, cell=224):
    rng = random.Random(seed)
    r = rng.sample(range(len(real_imgs)), min(n, len(real_imgs)))
    s = rng.sample(range(len(syn_imgs)), min(n, len(syn_imgs)))
    W = Image.new("RGB", (6 * cell, (len(r) // 3 + 1) * (cell + 16)), "white")
    d = ImageDraw.Draw(W)
    for i, (a, b) in enumerate(zip(r, s)):
        x, y = (i % 3) * 2 * cell, (i // 3) * (cell + 16)
        W.paste(real_imgs[a].resize((cell, cell)), (x, y + 16))
        W.paste(syn_imgs[b].resize((cell, cell)), (x + cell, y + 16))
        d.text((x + 3, y + 2), "REAL", fill="green")
        d.text((x + cell + 3, y + 2), "SYNTH", fill="red")
    W.save(path, quality=90)


# ---------------------------------------------------------------- main
def build(syn, real_name, mode, a):
    """Returns (S, R): lists of dicts {img, group, iid, food} built with the same procedure
    on both sides; the real side stops once it has as many images as the synthetic side."""
    view = syn[0][3]
    rng = random.Random(a.seed)
    if a.n_max and len(syn) > a.n_max:
        syn = [syn[i] for i in sorted(rng.sample(range(len(syn)), a.n_max))]
    ritems = real_items(real_name, len(syn), rng, 1 if real_name == "n5k" else 3)
    S, R = [], []
    n_img_s = 0
    if mode == "image":
        for rgb, seg, g, _ in syn:
            c = synth_crop(rgb, seg, view, real_name, a.size)
            if c is not None:
                S.append({"img": c, "group": g, "iid": f"{g.split('/')[-1]}/{rgb.stem}", "food": 1.0})
        n_img_s = len(S)
        for p, g in ritems:
            c = real_crop(p, real_name, a.size)
            if c is not None:
                R.append({"img": c, "group": f"real/{g}", "iid": f"{g}/{p.stem}", "food": 1.0})
            if len(R) >= n_img_s:
                break
    else:
        ref = real_ref_px(real_name, ritems)
        for rgb, seg, g, _ in syn:
            pr = random.Random(zlib.crc32(str(rgb).encode()) + a.seed)
            ps = synth_patches(rgb, seg, view, real_name, a, pr, ref)
            iid = f"{g.split('/')[-1]}/{rgb.stem}"
            n_img_s += bool(ps)
            for c, f in ps:
                S.append({"img": finish(c, a.size), "group": g, "iid": iid, "food": f})
        n_img_r = 0
        for p, g in ritems:
            pr = random.Random(zlib.crc32(str(p).encode()) + a.seed)
            ps = real_patches(p, real_name, a, pr, ref)
            n_img_r += bool(ps)
            for c, f in ps:
                R.append({"img": finish(c, a.size), "group": f"real/{g}", "iid": f"{g}/{p.stem}",
                          "food": f})
            if n_img_r >= n_img_s:
                break
    return S, R


def build_real_side(real: str, mode: str, a, n_images: int, tag: str):
    """Items of one REAL set processed exactly like the real side of build() (same crops /
    food masks / scale matching / patch sampler / JPEG tail), `n_images` images."""
    base = real_base(real)
    rng = random.Random(a.seed)
    items = real_items(real, n_images, rng, 1 if base == "n5k" else 3)
    out, n_img = [], 0
    ref = real_ref_px(real, items) if (mode == "patch" and base == "mf3d") else float(N5K_DISH_PX)
    for p, g in items:
        iid = f"{tag}/{g}/{p.stem}"
        if mode == "image":
            c = real_crop(p, base, a.size)
            if c is not None:
                out.append({"img": c, "group": f"{tag}/{g}", "iid": iid, "food": 1.0})
                n_img += 1
        else:
            pr = random.Random(zlib.crc32(str(p).encode()) + a.seed)
            ps = real_patches(p, base, a, pr, ref)
            n_img += bool(ps)
            for c, f in ps:
                out.append({"img": finish(c, a.size), "group": f"{tag}/{g}", "iid": iid, "food": f})
        if n_img >= n_images:
            break
    return out


def compare_real(real_a: str, real_b: str, mode: str, model, dev, a):
    """Real-vs-real analogue of compare(): class 1 = real_a, class 0 = real_b, same pipeline.
    Both sides come from independent photo sources, so this is the ceiling for how
    separable two different (honest) photo sets are with this probe. Patch mode uses one
    common patch size (--patch-px, default 96 = the MF3D size) on both sides, each set at
    native resolution (MF3D rescaled to a common food-bbox size, as in the synthetic runs)."""
    if a.patch_px is None:
        a = argparse.Namespace(**{**vars(a), "patch_px": 96})
    S = build_real_side(real_a, mode, a, a.n_images, "A-" + real_a)
    R = build_real_side(real_b, mode, a, a.n_images, "B-" + real_b)
    name = f"realpair__{mode}__{real_a}__vs__{real_b}"
    return analyze(name, S, R, mode, model, dev, a,
                   {"view": "-", "real": f"{real_a} vs {real_b}", "pair": [real_a, real_b]})


def compare(name, syn, real_name, mode, model, dev, a):
    """syn: list of (rgb, seg, group, view). Returns the report dict (+ writes sheets)."""
    view = syn[0][3]
    S, R = build(syn, real_name, mode, a)
    return analyze(name, S, R, mode, model, dev, a, {"view": view, "real": real_name})


def analyze(name, S, R, mode, model, dev, a, meta):
    """Probe + statistics + sheets for two item lists (S = class 1, R = class 0)."""
    view = meta["view"]
    real_name = meta["real"]
    n = min(len(S), len(R))                    # balance patch counts (whole-image order kept)
    S, R = S[:n], R[:n]
    gs = [d["group"] for d in S]
    gr = [d["group"] for d in R]
    if n < 10 or len(set(gs)) < 5 or len(set(gr)) < 5:
        return {"name": name, "skipped": f"too few items (synth {len(S)} in {len(set(gs))} scenes, "
                                         f"real {len(R)} in {len(set(gr))} groups)"}
    fS = features(model, dev, [d["img"] for d in S], a.size)
    fR = features(model, dev, [d["img"] for d in R], a.size)
    X = np.concatenate([fS, fR])
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)            # 1 = synthetic
    sc = probe(X, y, np.array(gs + gr))
    ss, sr = sc[:n], sc[n:]
    k, ks = kid(fS, fR)
    rep = {"name": name, "mode": mode, "view": view, "real": real_name, "style": a.style, **meta,
           "n_synth": n, "n_real": n, "n_synth_groups": len(set(gs)), "n_real_groups": len(set(gr)),
           "n_synth_images": len({d["iid"] for d in S}), "n_real_images": len({d["iid"] for d in R}),
           "food_frac_synth": float(np.mean([d["food"] for d in S])),
           "food_frac_real": float(np.mean([d["food"] for d in R])),
           "kid": k, "kid_std": ks}
    lvl = "patch" if mode == "patch" else "image"
    rep[lvl] = {**score_metrics(ss, sr), "ci95": bootstrap(ss, gs, sr, gr, a.boot, a.seed)}
    if mode == "patch":                                       # also score whole images
        si, sid = per_image(ss, [d["iid"] for d in S])
        ri, rid = per_image(sr, [d["iid"] for d in R])
        gmap = {d["iid"]: d["group"] for d in S + R}
        rep["image"] = {**score_metrics(si, ri),
                        "ci95": bootstrap(si, [gmap[i] for i in sid], ri, [gmap[i] for i in rid],
                                          a.boot, a.seed)}
    cs = probe_centroid(X, y, np.array(gs + gr))
    cm = score_metrics(cs[:n], cs[n:])
    rep["centroid"] = {k2: cm[k2] for k2 in ("gap", "auc", "dprime")}
    cb = bootstrap(cs[:n], gs, cs[n:], gr, a.boot, a.seed)
    rep["centroid"]["ci95"] = {k2: cb[k2] for k2 in ("gap", "auc", "dprime")}
    rep["headline"] = rep[lvl]
    # control: split REAL groups at random into two halves -> must look like chance
    ctrl = []
    ug = sorted(set(gr))
    for salt in range(5):
        half = set(np.random.default_rng(100 + salt).permutation(ug)[:len(ug) // 2])
        yc = np.array([g in half for g in gr]).astype(int)
        if 0 < yc.sum() < n:
            sc_c = probe(fR, yc, np.array(gr), seeds=(0,))
            cc = probe_centroid(fR, yc, np.array(gr), seeds=(0,))
            ctrl.append({**score_metrics(sc_c[yc == 1], sc_c[yc == 0]),
                         "centroid_gap": float(cc[yc == 1].mean() - cc[yc == 0].mean())})
    if ctrl:
        rep["real_vs_real_control"] = {k2: float(np.mean([c[k2] for c in ctrl])) for k2 in ctrl[0]}
        rep["real_vs_real_control"]["gap_abs"] = float(np.mean([abs(c["gap"]) for c in ctrl]))
        rep["real_vs_real_control"]["centroid_gap_abs"] = float(
            np.mean([abs(c["centroid_gap"]) for c in ctrl]))
    rep["config"] = {"size": a.size, "jpeg_q": JPEG_Q, "model": "vit_base_patch14_dinov2.lvd142m",
                     "C": 0.1, "cv": "StratifiedGroupKFold(5) x 3 seeds", "seed": a.seed,
                     "patch_px": a.patch_px or PATCH_PX, "patches_per_image": a.patches_per_image,
                     "min_food": a.min_food, "boot": a.boot}
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-ss)
    top, bot = order[:a.k], order[::-1][:a.k]
    lab = lambda i: f"{ss[i]:+.1f} {S[i]['iid']}"
    sheet([S[i]["img"] for i in top], [lab(i) for i in top], out / "most_fake.jpg")
    sheet([S[i]["img"] for i in bot], [lab(i) for i in bot], out / "most_real.jpg")
    pairs_sheet([d["img"] for d in R], [d["img"] for d in S], out / "pairs.jpg", seed=a.seed)
    rep["most_fake"] = [{"id": S[i]["iid"], "score": float(ss[i])} for i in top]
    rep["most_real"] = [{"id": S[i]["iid"], "score": float(ss[i])} for i in bot]
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def fmt(rep):
    if "skipped" in rep:
        return f"{rep['name']}: SKIPPED {rep['skipped']}"
    h = rep["headline"]
    c = rep.get("real_vs_real_control") or {}
    ci = lambda key: f"[{h['ci95'][key][0]:+.2f},{h['ci95'][key][1]:+.2f}]"
    s = (f"{rep['name']}: n={rep['n_synth']} acc={h['accuracy']:.3f} auc={h['auc']:.3f} "
         f"gap={h['gap']:+.2f}{ci('gap')} afc2={h['afc2']:.3f}{ci('afc2')} "
         f"p_s/p_r={h['p_synth']:.2f}/{h['p_real']:.2f} kid={rep['kid']:.3f}+-{rep['kid_std']:.2f} "
         f"cent_gap={rep['centroid']['gap']:.2f}[{rep['centroid']['ci95']['gap'][0]:.2f},"
         f"{rep['centroid']['ci95']['gap'][1]:.2f}]")
    if rep["mode"] == "patch":
        s += f" | img-level auc={rep['image']['auc']:.3f} gap={rep['image']['gap']:+.2f}"
    if c:
        s += f" | ctrl auc={c['auc']:.3f} |gap|={c['gap_abs']:.2f} afc2={c['afc2']:.3f} " \
             f"|cent_gap|={c['centroid_gap_abs']:.2f}"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synth", nargs="*", default=[], help="batch dirs with scans/ and raw/")
    ap.add_argument("--real", nargs="+", default=["n5k", "mf3d"], choices=["n5k", "mf3d"])
    ap.add_argument("--real-pairs", nargs="*", default=[], metavar="A:B",
                    help="real-vs-real baseline: run the same probe on real set A vs real set B "
                         "(names n5k, mf3d, n5k_a, n5k_b, mf3d_a, mf3d_b; e.g. n5k:mf3d "
                         "mf3d_a:mf3d_b n5k_a:n5k_b). The ceiling for 'indistinguishable'")
    ap.add_argument("--n-images", type=int, default=120, help="images per side for --real-pairs")
    ap.add_argument("--mode", nargs="+", default=["patch", "image"], choices=["patch", "image"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224, help="model input size")
    ap.add_argument("--patch-px", type=int, default=None, help="patch side in working pixels (default 224 for n5k, 96 for mf3d: MF3D food is only ~100 px wide)")
    ap.add_argument("--patches-per-image", type=int, default=8)
    ap.add_argument("--min-food", type=float, default=0.25, help="min food fraction of a patch")
    ap.add_argument("--style", default=None, help="keep only synthetic scenes with spec style==NAME")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap resamples for the CIs")
    ap.add_argument("--n-max", type=int, default=400, help="cap on synthetic frames per comparison")
    ap.add_argument("--k", type=int, default=18, help="images on most_fake / most_real sheets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human-sigma", type=float, default=HUMAN_SIGMA, help="2AFC proxy noise sd (logits)")
    a = ap.parse_args()
    globals()["HUMAN_SIGMA"] = a.human_sigma
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model, dev = load_model()
    print("backbone on", dev)
    batches = [Path(s) for s in a.synth]
    todo = [(b.name, [b]) for b in batches]
    if len(batches) > 1 and a.synth:
        todo.append(("pooled", batches))
    reps = []
    for pair in a.real_pairs:
        ra, rb = pair.split(":")
        for mode in a.mode:
            rep = compare_real(ra, rb, mode, model, dev, a)
            reps.append(rep)
            print(fmt(rep), flush=True)
    for name, bs in todo:
        for real_name in a.real:
            view = "top" if real_name == "n5k" else "orbit"
            syn = [(r, s, g, view) for b in bs for r, s, g in synth_items(b, view, a.style)]
            if not syn:
                print(f"{name}__{view}__{real_name}: no synthetic frames"
                      + (f" with style={a.style!r}" if a.style else ""))
                continue
            for mode in a.mode:
                tag = f"{name}{'-' + a.style if a.style else ''}__{mode}__{view}__{real_name}"
                rep = compare(tag, syn, real_name, mode, model, dev, a)
                reps.append(rep)
                print(fmt(rep), flush=True)
    (Path(a.out) / "report.json").write_text(json.dumps(reps, indent=1))


if __name__ == "__main__":
    main()
