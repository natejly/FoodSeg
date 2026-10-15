"""How distinguishable are our renders from real food photos? (a "realism probe")

    python -m synth.realism --synth ../data/synth/hard_v10 --synth ../data/synth/hard_v9 \
        --real n5k mf3d --out ../results/synth/realism [--size 224] [--n-max 400]

Why: "looks real" is otherwise an eyeball judgement. Here a linear classifier on frozen
DINOv2 features tries to tell render from photo; held-out accuracy / AUC of 50 % / 0.5
means this probe cannot (it is a lower bound on detectability: a stronger probe or a
fine-tuned net would do better). The probe's most confident "synthetic" renders are
saved as a contact sheet, so the output is also a to-do list of what looks fakest.

Comparisons are matched, one per (synthetic batch x viewpoint x real set), plus a pooled
row over all given batches:
  top-down : synth frames 6,7 (near top-down) vs Nutrition5k realsense_overhead rgb.png
             (real cafeteria dishes). Synth crop = dish bbox (+15 %), real crop = central
             square (the dish fills the tray image), so both show "plate fills frame".
  orbit    : synth frames 0-5 (~45 deg) vs MetaFood3D capture frames (real single foods on
             a black cloth with fiducials, ~40-60 deg). Crop = square of 2x the food bbox
             (synth: seg labels 1..252, real: the *_m.png food mask), i.e. food + context.
Nothing real from an iPad exists locally, so these two are the nearest domains.
Matching / confound control (the point of the exercise):
  * the synth frame is first resampled to the real set's native resolution (640x480 /
    448x336) so sharpness / pixel density is not the giveaway, then cropped, resized to
    --size+32, JPEG re-encoded (q=90) on BOTH sides, then resized to --size;
  * real set is subsampled to the same count as the synthetic set (seeded);
  * CV is StratifiedGroupKFold (5 folds, repeated with 3 seeds, scores averaged) with
    group = synth scan / N5k dish / MF3D capture directory, so frames of one scene never
    straddle train and test. Features are standardised, L2 logistic regression (C=0.1).
  * KID = unbiased MMD^2 with cubic polynomial kernel on the standardised features, over
    random 100-sample subsets (mean +- std), as in Binkowski et al. Smaller = closer.
Backbone: timm `vit_base_patch14_dinov2.lvd142m` (DINOv2 ViT-B/14, ~350 MB, from the HF cache
~/.cache/huggingface/hub; loaded offline). Feature = [CLS, mean patch token] (1536-d), fp16
on GPU when there is >2 GB free, else CPU.
Outputs in --out/<synthname>__<view>__<real>/ : report.json, most_fake.jpg, most_real.jpg,
pairs.jpg; and --out/report.json aggregating every comparison.
Caveats: real sets differ from our scenes in content (cafeteria tray / black cloth with
markers) which inflates accuracy; compare numbers between batches rather than reading the
absolute value, and read the contact sheets.
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
JPEG_Q = 90


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
        bb = bbox(seg == 254) or bbox((seg > 0) & (seg < 253))
        pad = 1.15
    else:
        bb = bbox((seg > 0) & (seg < 253))
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


# ---------------------------------------------------------------- item lists
def synth_items(batch: Path, view: str):
    """[(rgb, seg, group)] for frames of one viewpoint that have both rgb and raw seg."""
    out = []
    for rgb in sorted(glob.glob(str(batch / "scans/*/rgb/*.jpg"))):
        rgb = Path(rgb)
        sid, idx = rgb.parent.parent.name, int(rgb.stem)
        if idx not in (TOP_FRAMES if view == "top" else ORBIT_FRAMES):
            continue
        seg = batch / "raw" / sid / f"frame_{idx:06d}.npz"
        if seg.exists():
            out.append((rgb, seg, f"{batch.name}/{sid}"))
    return out


def real_items(real: str, n: int, rng: random.Random, per_group: int):
    """n real images from at most ceil(n/per_group)-ish groups (dish / capture dir)."""
    if real == "n5k":
        dishes = sorted(N5K.glob("dish_*/rgb.png"))
        return [(p, p.parent.name) for p in rng.sample(dishes, min(n, len(dishes)))]
    dirs = sorted(d for d in MF3D.iterdir() if d.is_dir())
    rng.shuffle(dirs)
    out = []
    for d in dirs:
        fr = [p for p in d.glob("*.jpg") if (d / (p.stem + "_m.png")).exists()]
        for p in rng.sample(fr, min(per_group, len(fr))):
            out.append((p, d.name))
        if len(out) >= n * 1.3:             # slack for crops that fail (empty mask)
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
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(X[tr], y[tr])
            scores[te] += clf.decision_function(X[te]) / len(seeds)
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
def compare(name, syn, real_name, real_n_per_group, model, dev, a):
    """syn: list of (rgb, seg, group, view). Returns the report dict (+ writes sheets)."""
    from sklearn.metrics import roc_auc_score
    view = syn[0][3]
    rng = random.Random(a.seed)
    S, sg, sp = [], [], []
    for rgb, seg, g, _ in syn:
        c = synth_crop(rgb, seg, view, real_name, a.size)
        if c is not None:
            S.append(c), sg.append(g), sp.append(f"{g.split('/')[-1]}/{rgb.stem}")
    if a.n_max and len(S) > a.n_max:
        keep = sorted(rng.sample(range(len(S)), a.n_max))
        S, sg, sp = [[l[i] for i in keep] for l in (S, sg, sp)]
    R, rg = [], []
    for p, g in real_items(real_name, len(S), rng, real_n_per_group):
        c = real_crop(p, real_name, a.size)
        if c is not None:
            R.append(c), rg.append(g)
        if len(R) >= len(S):
            break
    n = min(len(S), len(R))
    if n < 10 or len(set(sg)) < 5:
        return {"name": name, "skipped": f"too few images (synth {len(S)}, real {len(R)})"}
    S, sg, sp, R, rg = S[:n], sg[:n], sp[:n], R[:n], rg[:n]
    fS, fR = features(model, dev, S, a.size), features(model, dev, R, a.size)
    X = np.concatenate([fS, fR])
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)            # 1 = synthetic
    groups = np.array(sg + [f"real/{g}" for g in rg])
    sc = probe(X, y, groups)
    acc = float(((sc > 0) == y).mean())
    auc = float(roc_auc_score(y, sc))
    k, ks = kid(fS, fR)
    # sanity control: two disjoint random halves of the REAL set must score ~50 % / 0.5
    h = np.array([zlib.crc32(g.encode()) % 2 for g in rg])
    ctrl = None
    if 0 < h.sum() < n and len(set(rg)) >= 10:
        yc = h.astype(int)
        sc_c = probe(fR, yc, np.array(rg))
        ctrl = {"accuracy": float(((sc_c > 0) == yc).mean()), "auc": float(roc_auc_score(yc, sc_c))}
    dprime = float((sc[:n].mean() - sc[n:].mean()) / np.sqrt((sc[:n].var() + sc[n:].var()) / 2 + 1e-9))
    rep = {"name": name, "view": view, "real": real_name, "n_synth": n, "n_real": n,
           "n_synth_groups": len(set(sg)), "n_real_groups": len(set(rg)),
           "accuracy": acc, "auc": auc, "kid": k, "kid_std": ks,
           "dprime": dprime, "real_vs_real_control": ctrl, "synth_recall": float((sc[:n] > 0).mean()), "real_recall": float((sc[n:] <= 0).mean()),
           "config": {"size": a.size, "jpeg_q": JPEG_Q, "model": "vit_base_patch14_dinov2.lvd142m",
                      "C": 0.1, "cv": "StratifiedGroupKFold(5) x 3 seeds", "seed": a.seed}}
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    ss = sc[:n]
    order = np.argsort(-ss)
    top = order[:a.k]
    bot = order[::-1][:a.k]
    sheet([S[i] for i in top], [f"{ss[i]:+.1f} {sp[i]}" for i in top], out / "most_fake.jpg")
    sheet([S[i] for i in bot], [f"{ss[i]:+.1f} {sp[i]}" for i in bot], out / "most_real.jpg")
    pairs_sheet(R, S, out / "pairs.jpg", seed=a.seed)
    rep["most_fake"] = [{"id": sp[i], "score": float(ss[i])} for i in top]
    rep["most_real"] = [{"id": sp[i], "score": float(ss[i])} for i in bot]
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synth", nargs="+", required=True, help="batch dirs with scans/ and raw/")
    ap.add_argument("--real", nargs="+", default=["n5k", "mf3d"], choices=["n5k", "mf3d"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-max", type=int, default=400, help="cap on synthetic crops per comparison")
    ap.add_argument("--k", type=int, default=18, help="images on most_fake / most_real sheets")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model, dev = load_model()
    print("backbone on", dev)
    batches = [Path(s) for s in a.synth]
    todo = [(b.name, [b]) for b in batches]
    if len(batches) > 1:
        todo.append(("pooled", batches))
    reps = []
    for name, bs in todo:
        for real_name in a.real:
            view = "top" if real_name == "n5k" else "orbit"
            syn = [(r, s, g, view) for b in bs for r, s, g in synth_items(b, view)]
            if not syn:
                continue
            rep = compare(f"{name}__{view}__{real_name}", syn, real_name,
                          1 if real_name == "n5k" else 3, model, dev, a)
            reps.append(rep)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()
                   if k in ("name", "n_synth", "accuracy", "auc", "kid", "skipped")})
    (Path(a.out) / "report.json").write_text(json.dumps(reps, indent=1))


if __name__ == "__main__":
    main()
