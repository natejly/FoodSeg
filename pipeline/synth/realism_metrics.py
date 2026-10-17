"""Per-content-class realism metrics: CMMD (CLIP) and FD (DINOv2) of synthetic patches vs real
patches, per class, with bootstrap CIs and a paired accept/reject test between batches.

    cd pipeline
    flock <gpu.lock> python -m synth.realism_metrics \
        --synth /abs/data/synth/hard_v9 .../hard_v10 .../hard_v11 .../hard_v12 \
        --real mf3d n5k --baseline hard_v11 --out ../results/synth/realism_metrics
    (features are cached in --out as feat_*.npz and reused on re-run: delete them if patch
    sampling / data changed)

Why per class (research report #3): one global realism number hides which content is fake. Classes
are taken from the synthetic seg + spec: food kind (scan-mf3d/<shape>, procedural/<look>,
topping/<look>, objaverse/..., "mixed" = no item covers >= 60 % of the patch), plus "dish" and
"table"; aggregates food/*, food/scan-mf3d/*, .... Patches are cut like synth.realism's patch mode (scale matched
to the real set, JPEG-re-encoded both sides, coverage >= --cov of the class pixels).

Real references (the real food kinds are not our synthetic kinds, so every synthetic food class
is compared with the whole real food-patch pool):
    mf3d : food (object mask) and table (>15 px away from the mask, i.e. black cloth + fiducials)
    n5k  : food / dish (plate-coloured inside the dish) / table (tray); heuristic regions
           (realism-style colour segmentation; no GT masks exist) -> noisy
So "table" vs mf3d = black cloth is a domain mismatch by construction: use it only as a relative
number between batches, never as an absolute goal.

Metrics (lower = closer to real):
  CMMD   unbiased MMD^2 x 1000, RBF sigma=10 on L2-normalised CLIP ViT-L/14 image embeddings
         (Jayasumana et al. 2024; unbiased so it can be ~0 / slightly negative).
  FD     Frechet distance on DINOv2-B/14 CLS features (224 px), after a PCA to --fd-dim (default
         128) fitted on the real pool (full 768-d covariances are rank-deficient at n~200);
         values are not comparable with ImageNet-FID numbers, only with each other.
Both are evaluated at the SAME sample size per class for all batches (--n-eval cap, reduced to
the smallest batch count of that class, min --min-n) so batches compare fairly; bootstrap =
resample synthetic scans and real captures with replacement (--boot draws), then subsample.
Floor: real-vs-real (two disjoint halves of the real captures), same n.
Paired test (--baseline B): per-draw delta = metric(batch) - metric(B), sharing the real
resample; a class "improves" if the 95 % CI of the delta is entirely < 0, "regresses" if > 0.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import realism as R

PX = {("n5k", "food"): 224, ("n5k", "dish"): 112, ("n5k", "table"): 112,
      ("mf3d", "food"): 96, ("mf3d", "table"): 96}
COV = {"food": 0.25, "dish": 0.85, "table": 0.9}
NPER = {"food": 16, "dish": 8, "table": 8}
FOOD_MAX, TABLE_IDS, DISH_ID = 251, (0, 252), 254
CLIP_ID = "openai/clip-vit-large-patch14"


# ------------------------------------------------------------------ patches
def sample(img, mask, n, px, cov, rng):
    w, h = img.size
    if w < px or h < px:
        return []
    ii = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    xs, ys = np.arange(0, w - px + 1, 16), np.arange(0, h - px + 1, 16)
    X, Y = np.meshgrid(xs, ys)
    fr = (ii[Y + px, X + px] - ii[Y, X + px] - ii[Y + px, X] + ii[Y, X]) / float(px * px)
    cand = [(fr[j, i], int(X[j, i]), int(Y[j, i])) for j in range(len(ys)) for i in range(len(xs)) if fr[j, i] >= cov]
    rng.shuffle(cand)
    out, ctr = [], []
    for f, x, y in cand:
        if all(max(abs(x - a), abs(y - b)) >= px // 2 for a, b in ctr):
            ctr.append((x, y)); out.append((img.crop((x, y, x + px, y + px)), float(f), x, y))
            if len(out) >= n:
                break
    return out


def n5k_regions(im):
    from scipy import ndimage as ndi
    w, h = im.size; s = 4
    hsv = np.asarray(im.resize((w // s, h // s), Image.BILINEAR).convert("HSV"), np.float32) / 255
    plateish = (hsv[..., 1] < 0.2) & (hsv[..., 2] > 0.5)
    po = ndi.binary_opening(plateish, np.ones((5, 5)))
    lab, n = ndi.label(po)
    z = np.zeros((h, w), bool)
    if n == 0:
        return z, z, z
    hh, ww = po.shape; yy, xx = np.mgrid[:hh, :ww]
    wgt = np.exp(-(((xx - ww / 2) / (0.35 * ww)) ** 2 + ((yy - hh / 2) / (0.35 * hh)) ** 2))
    best = 1 + int(np.argmax(ndi.sum(wgt, lab, np.arange(1, n + 1))))
    filled = ndi.binary_fill_holes(ndi.binary_closing(lab == best, np.ones((9, 9))))
    food = ndi.binary_opening(filled & ~plateish, np.ones((3, 3)))
    plate = ndi.binary_erosion(filled & plateish & ~ndi.binary_dilation(food, np.ones((5, 5))), np.ones((5, 5)))
    tray = ndi.binary_erosion(~ndi.binary_dilation(filled, np.ones((7, 7))), np.ones((5, 5)))
    up = lambda m: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
    return up(food), up(plate), up(tray)


def item_label(it):
    src, sh, lk = it.get("source"), (it.get("shape") or {}).get("kind"), (it.get("look") or {}).get("kind")
    if src == "procedural_piece": return f"procedural/{lk}"
    if src == "procedural_topping": return f"topping/{lk}"
    if src == "procedural_leaves": return "procedural/leaves"
    if src == "metafood3d": return f"scan-mf3d/{sh}"
    return f"{src}/{sh}"


def load_spec(batch: Path, sid: str):
    d = json.loads((batch / "specs" / sid / "spec.json").read_text())
    return {it.get("id", k + 1): it for k, it in enumerate(d["items"])}


def frames_for(nviews: int, view: str):
    if nviews >= 24:
        return (18, 23) if view == "top" else (0, 3, 6, 9, 12, 15)
    return (nviews - 2, nviews - 1) if view == "top" else tuple(range(max(nviews - 2, 1)))


def synth_records(batch: Path, real: str, ref_px: float):
    view = "top" if real == "n5k" else "orbit"
    recs, imgs = [], []
    for sd in sorted(batch.glob("scans/*")):
        sid = sd.name
        nv = len(list(sd.glob("rgb/*.jpg")))
        items = None
        for idx in frames_for(nv, view):
            rgb = sd / "rgb" / f"{idx:06d}.jpg"
            seg = R.load_seg(batch, sid, idx)
            if not rgb.exists() or seg is None:
                continue
            if items is None:
                items = load_spec(batch, sid)
            im = Image.open(rgb).convert("RGB")
            food = (seg > 0) & (seg <= FOOD_MAX)
            bb = (R.bbox(seg == DISH_ID) or R.bbox(food)) if view == "top" else R.bbox(food)
            if bb is None:
                continue
            s = im.size[0] / seg.shape[1]
            k = R.clip_scale(ref_px / (max(bb[2] - bb[0], bb[3] - bb[1]) * s))
            sz = (round(im.size[0] * k), round(im.size[1] * k))
            im = im.resize(sz, Image.LANCZOS)
            segs = np.asarray(Image.fromarray(seg.astype(np.uint8)).resize(sz, Image.NEAREST))
            kinds = {"food": (segs > 0) & (segs <= FOOD_MAX), "table": np.isin(segs, TABLE_IDS)}
            if real == "n5k":
                kinds["dish"] = segs == DISH_ID
            for kind, mask in kinds.items():
                px = PX[(real, kind)]
                rng = random.Random(zlib.crc32(f"{batch.name}/{sid}/{idx}/{kind}".encode()))
                for c, f, x, y in sample(im, mask, NPER[kind], px, COV[kind], rng):
                    rec = dict(kind=kind, group=f"{batch.name}/{sid}", frame=idx, cov=f)
                    if kind == "food":
                        sub = segs[y:y + px, x:x + px]
                        cnt = np.bincount(sub[(sub > 0) & (sub <= FOOD_MAX)].ravel(), minlength=256)
                        top = int(cnt.argmax()); share = cnt[top] / max(cnt.sum(), 1)
                        it = items.get(top)
                        rec["label"] = item_label(it) if (share >= 0.6 and it) else "mixed"
                    else:
                        rec["label"] = kind
                    recs.append(rec); imgs.append(R.finish(c, 224))
    return recs, imgs


def real_records(real: str, n_img: int, ref_px: float, items):
    recs, imgs = [], []
    done = {"food": 0, "dish": 0, "table": 0}
    from scipy import ndimage as ndi
    for p, g in items:
        im = Image.open(p).convert("RGB")
        if real == "n5k":
            food, plate, tray = n5k_regions(im); masks = {"food": food, "dish": plate, "table": tray}
        else:
            mp = p.with_name(p.stem + "_m.png")
            if not mp.exists():
                continue
            food = np.asarray(Image.open(mp)) > 0
            bb = R.bbox(food)
            if bb is None:
                continue
            k = R.clip_scale(ref_px / max(bb[2] - bb[0], bb[3] - bb[1]))
            sz = (round(im.size[0] * k), round(im.size[1] * k))
            im = im.resize(sz, Image.LANCZOS)
            food = np.asarray(Image.fromarray((food * 255).astype(np.uint8)).resize(sz, Image.NEAREST)) > 0
            masks = {"food": food, "table": ~ndi.binary_dilation(food, np.ones((15, 15)))}
        for kind, mask in masks.items():
            if done[kind] >= n_img:
                continue
            rng = random.Random(zlib.crc32(f"{p}/{kind}".encode()))
            ps = sample(im, mask, NPER[kind], PX[(real, kind)], COV[kind], rng)
            done[kind] += bool(ps)
            for c, f, x, y in ps:
                recs.append(dict(kind=kind, group=g, label=kind, cov=f)); imgs.append(R.finish(c, 224))
        if all(done[k] >= n_img for k in masks):
            break
    return recs, imgs


# ------------------------------------------------------------------ features
def clip_features(imgs, dev, bs=32):
    import torch
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).eval().to(dev)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = np.stack([(np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32) / 255 - mean) / std
                          for im in imgs[i:i + bs]]).transpose(0, 3, 1, 2)
            x = torch.from_numpy(x).to(dev).to(m.dtype)
            e = m.visual_projection(m.vision_model(pixel_values=x).pooler_output)   # == get_image_features
            out.append(e.float().cpu().numpy())
    del m
    return np.concatenate(out)


def dino_features(imgs, dev):
    model, d = R.load_model()
    f = R.features(model, d, imgs, 224)[:, :768]       # CLS token only
    del model
    return f


def get_feats(name, recs, imgs, cache: Path, dev):
    p = cache / f"feat_{name}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["clip"], z["dino"], json.loads(str(z["recs"]))
    c, d = clip_features(imgs, dev), dino_features(imgs, dev)
    np.savez(p, clip=c, dino=d, recs=json.dumps(recs))
    return c, d, recs


# ------------------------------------------------------------------ metrics
def cmmd(x, y, sigma=10.0):
    """unbiased MMD^2 x 1000 of L2-normalised embeddings, RBF kernel."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True); y = y / np.linalg.norm(y, axis=1, keepdims=True)
    g = 1.0 / (2 * sigma ** 2)
    def k(a, b):
        return np.exp(-g * np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0))
    n, m = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    v = (kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.mean()
    return float(v * 1000)


def fd(x, y):
    mx, my = x.mean(0), y.mean(0)
    sx, sy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    w, v = np.linalg.eigh(sx); w = np.maximum(w, 0)
    r = (v * np.sqrt(w)) @ v.T
    ev = np.maximum(np.linalg.eigvalsh(r @ sy @ r), 0)
    return float(((mx - my) ** 2).sum() + np.trace(sx) + np.trace(sy) - 2 * np.sqrt(ev).sum())


class Pool:
    """features of one set, indexed by (kind,label) with scan/capture groups for resampling."""

    def __init__(self, clip, dino, recs):
        self.clip, self.dino, self.recs = clip, dino, recs
        self.group = np.array([r["group"] for r in recs])
        self.label = np.array([r["label"] for r in recs])
        self.kind = np.array([r["kind"] for r in recs])

    def select(self, kind=None, label=None, prefix=None):
        m = np.ones(len(self.recs), bool)
        if kind: m &= self.kind == kind
        if label: m &= self.label == label
        if prefix: m &= np.array([l.startswith(prefix) for l in self.label])
        return np.nonzero(m)[0]


def draw(idx, groups, rng, n):
    """bootstrap over groups, then subsample n patches (without replacement) from the drawn ones."""
    ug = np.unique(groups[idx])
    by = {g: idx[groups[idx] == g] for g in ug}
    ch = rng.choice(ug, len(ug))
    pool = np.concatenate([by[g] for g in ch])
    return rng.choice(pool, n, replace=len(pool) < n)


def class_table(pools, real, a, rng_seed=0):
    """-> {class: {'real_kind', 'n', batch: {cmmd/fd: {point, ci}} , 'delta': {...}}}."""
    from sklearn.decomposition import PCA
    names = list(pools)
    rp = real
    pca = PCA(n_components=a.fd_dim, random_state=0).fit(rp.dino) if a.fd_dim else None
    proj = (lambda z: pca.transform(z)) if pca else (lambda z: z)
    rd = proj(rp.dino)
    classes = {}
    for b in names:
        P = pools[b]
        for lab in sorted(set(P.label)):
            kind = P.kind[P.label == lab][0]
            classes.setdefault(lab, dict(kind=kind, counts={}))["counts"][b] = int((P.label == lab).sum())
    # aggregates
    aggs = {"food/*": ("food", None, None)}
    for pre in ("scan-mf3d/", "procedural/", "topping/"):
        aggs[f"food/{pre}*"] = ("food", None, pre)
    out = {}
    jobs = [(lab, c["kind"], dict(label=lab)) for lab, c in classes.items()]
    jobs += [(k, v[0], dict(kind=v[0], prefix=v[2])) for k, v in aggs.items()]
    rng = np.random.default_rng(rng_seed)
    # real halves (by capture) for the floor
    ug = np.unique(rp.group); half = {g: zlib.crc32(g.encode()) % 2 for g in ug}
    rh = np.array([half[g] for g in rp.group])
    for name, kind, sel in jobs:
        sels = {b: pools[b].select(**sel) for b in names}
        counts = {b: len(s) for b, s in sels.items() if len(s) >= a.min_n and len(np.unique(pools[b].group[s])) >= 3}
        if not counts:
            continue
        rkind = kind
        rsel = rp.select(kind=rkind)
        if len(rsel) < a.min_n:
            continue
        n = int(min(a.n_eval, min(counts.values()), len(rsel) // 2))
        if n < a.min_n:
            continue
        res = {b: {"cmmd": [], "fd": []} for b in counts}
        res["real/real"] = {"cmmd": [], "fd": []}
        for _ in range(a.boot):
            ri = draw(rsel, rp.group, rng, n)
            for b in counts:
                si = draw(sels[b], pools[b].group, rng, n)
                res[b]["cmmd"].append(cmmd(pools[b].clip[si], rp.clip[ri]))
                res[b]["fd"].append(fd(proj(pools[b].dino[si]), rd[ri]))
            # floor: halves of real
            h0, h1 = rsel[rh[rsel] == 0], rsel[rh[rsel] == 1]
            if len(h0) >= n and len(h1) >= n and len(np.unique(rp.group[h0])) > 2:
                i0, i1 = draw(h0, rp.group, rng, n), draw(h1, rp.group, rng, n)
                res["real/real"]["cmmd"].append(cmmd(rp.clip[i0], rp.clip[i1]))
                res["real/real"]["fd"].append(fd(rd[i0], rd[i1]))
        ent = dict(real_kind=rkind, n_eval=n, n_patches=counts)
        for b, v in res.items():
            if not v["cmmd"]:
                continue
            ent[b] = {m: dict(mean=float(np.mean(v[m])), ci95=[float(np.percentile(v[m], 2.5)), float(np.percentile(v[m], 97.5))])
                      for m in ("cmmd", "fd")}
        if a.baseline:
            ent["_raw"] = {b: v for b, v in res.items() if v["cmmd"]}
        out[name] = ent
    return out


def add_deltas(table, baseline):
    """delta vs baseline from the per-draw samples (paired on the real resample by draw index)."""
    for cname, ent in table.items():
        raw = ent.pop("_raw", None)
        if not raw or baseline not in raw:
            continue
        ent["delta_vs_" + baseline] = {}
        for b, v in raw.items():
            if b in (baseline, "real/real"):
                continue
            d = {}
            for m in ("cmmd", "fd"):
                dd = np.array(v[m]) - np.array(raw[baseline][m])
                lo, hi = np.percentile(dd, [2.5, 97.5])
                d[m] = dict(mean=float(dd.mean()), ci95=[float(lo), float(hi)],
                            verdict="improves" if hi < 0 else ("regresses" if lo > 0 else "n.s."))
            ent["delta_vs_" + baseline][b] = d


def markdown(report, baseline):
    lines = []
    for real, tab in report["tables"].items():
        batches = report["batches"]
        lines += [f"\n### vs real `{real}`  (CMMD x1000 | FD_DINOv2[PCA{report['args']['fd_dim']}]; mean [95% CI]; lower = closer)\n",
                  "| class | n | " + " | ".join(batches) + " | real/real floor |", "|---|---|" + "---|" * (len(batches) + 1)]
        for c, ent in tab.items():
            row = [c, str(ent["n_eval"])]
            for b in batches + ["real/real"]:
                if b in ent:
                    e = ent[b]
                    row.append(f"{e['cmmd']['mean']:.1f} [{e['cmmd']['ci95'][0]:.1f},{e['cmmd']['ci95'][1]:.1f}] \\| "
                               f"{e['fd']['mean']:.0f} [{e['fd']['ci95'][0]:.0f},{e['fd']['ci95'][1]:.0f}]")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        if baseline:
            lines += [f"\nPaired delta vs `{baseline}` (negative = better), verdict on CMMD / FD:\n",
                      "| class | " + " | ".join(b for b in batches if b != baseline) + " |", "|---|" + "---|" * (len(batches) - 1)]
            for c, ent in tab.items():
                d = ent.get("delta_vs_" + baseline)
                if not d:
                    continue
                lines.append(f"| {c} | " + " | ".join(
                    (f"{d[b]['cmmd']['mean']:+.1f} {d[b]['cmmd']['verdict']} / {d[b]['fd']['mean']:+.0f} {d[b]['fd']['verdict']}"
                     if b in d else "-") for b in batches if b != baseline) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--real", nargs="+", default=["mf3d", "n5k"], choices=["mf3d", "n5k"])
    ap.add_argument("--out", default="../results/synth/realism_metrics")
    ap.add_argument("--baseline", default=None, help="batch name for the paired accept/reject deltas")
    ap.add_argument("--real-images", type=int, default=250, help="max real images per region kind")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=250)
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--fd-dim", type=int, default=128, help="PCA dim for FD (0 = full 768)")
    ap.add_argument("--markdown", default=None)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    batches = [Path(s) for s in a.synth]
    names = [b.name for b in batches]
    report = dict(args=vars(a), batches=names, tables={})
    for real in a.real:
        rng = random.Random(0)
        ritems = R.real_items(real, a.real_images, rng, 1 if real == "n5k" else 3)
        ref_px = R.real_ref_px(real, ritems)
        cache = out;
        rf = out / f"feat_real_{real}.npz"
        if rf.exists():
            rrecs = None
        rrecs, rimgs = (None, None) if rf.exists() else real_records(real, a.real_images, ref_px, ritems)
        rc, rd_, rrecs = get_feats(f"real_{real}", rrecs, rimgs, out, dev)
        print(real, "real patches", {k: sum(r["kind"] == k for r in rrecs) for k in ("food", "dish", "table")}, flush=True)
        pools = {}
        for b in batches:
            f = out / f"feat_{b.name}_{real}.npz"
            recs, imgs = (None, None) if f.exists() else synth_records(b, real, ref_px)
            c, d, recs = get_feats(f"{b.name}_{real}", recs, imgs, out, dev)
            pools[b.name] = Pool(c, d, recs)
            print(real, b.name, len(recs), "patches", flush=True)
        tab = class_table(pools, Pool(rc, rd_, rrecs), a)
        if a.baseline:
            add_deltas(tab, a.baseline)
        report["tables"][real] = tab
    (out / "report.json").write_text(json.dumps(report, indent=1))
    md = markdown(report, a.baseline)
    (Path(a.markdown) if a.markdown else out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
