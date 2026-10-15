"""Train on synthetic, test on real (TSTR): does a segmenter trained only on our renders segment
REAL food photos?  This is the downstream test of synthetic realism (a DINOv2 real-vs-synthetic
probe is not the target: only what a model can *use* matters).

    cd pipeline
    flock <gpu.lock> python -m synth.tstr --synth ../data/synth/hard_v12 --steps 1500 \
        --seeds 0 1 2 --out ../results/synth/tstr                      # synthetic only
    ... --synth ../data/synth/hard_v12 --ft-real 32 --ft-steps 300      # + small real fine-tune
    ... --synth none --out ...                                          # real-train reference (upper)
    ... --synth none --real-n 32 --out ...                              # real-only with the same few images
    python -m synth.tstr --summarize ../results/synth/tstr              # markdown table of all reports

Task (the only one the available real masks support): binary food-vs-background.
  synthetic labels : food = raw seg 1..251 (food items); everything else (table 0/252, clutter 253,
                     dish 254) is background.  (The dish is NOT food; real frames have no dish.)
  real data        : MetaFood3D phone frames with *_m.png object masks (data/metafood3d/frames:
                     single foods on a black cloth, 336x448).  Split: the repo-wide capture-level
                     split.json (train 401 / test 95 captures); real VAL = a fixed 12 % of the train
                     captures (crc32 of the capture id), removed from real training.  Test is only
                     touched at the end; "best_val" checkpoint selection uses the real val split.
  optional (--eval-n5k N): N Nutrition5k overhead dishes with a PSEUDO mask (colour heuristic
                     realism.detect_food_n5k; pale foods missed) -> secondary, noisy, cafeteria-like.
Fixed budget: every arm sees --steps x --bs crops from the same number of frames (--frames-per-scan
evenly-spaced views per scan, default 8 => 160 frames for a 20-scan batch whatever the batch's
view count) with the same augmentation, model, optimiser and seed list.

Model: torchvision DeepLabV3-MobileNetV3-Large, ImageNet backbone (NO segmentation pre-training),
fp16 autocast, 320x320 crops; ~3 GB at bs 12.  Eval: whole real frame at 384x288.
Metrics: food IoU, mIoU (food+bg, dataset-level), boundary F (2-sided, --bf-tol px at eval res,
mean over images); 95 % CIs = bootstrap over real *captures* (seed-averaged per-image values).
Leakage flag: synthetic scenes use MetaFood3D meshes; a test capture whose own mesh appears in a
batch's specs is "seen".  Reported on all test images and on the "unseen" subset.
Licence: MetaFood3D is distributed for research use (Chen et al., "MetaFood3D"; see
data/datasets/MetaFood3D/_MetaFood3D_Readme.txt, which states no further terms); Nutrition5k is
CC BY 4.0 (Google Research).  Both are used here for evaluation only, no redistribution.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(os.environ.get("FOODSEG_DATA", "/home/kyle/Desktop/FoodSeg/data"))
MF3D = DATA / "metafood3d/frames"
N5K = DATA / "datasets/Nutrition5k/imagery/realsense_overhead"
FOOD_MIN, FOOD_MAX = 1, 251            # raw seg ids that are food items
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EVAL_HW = (384, 288)                   # portrait real frames; landscape ones are transposed
STORE_SHORT = 480                      # synthetic frames are stored at 640x480


# ------------------------------------------------------------------ real data
def _cap_dir(cap: str) -> Path:
    return MF3D / cap.replace("/", "__")


def real_split(per_cap=None, seed=0):
    """{'train','val','test'}: lists of (jpg path, mask path, capture id). Fixed and deterministic."""
    per_cap = per_cap or {"train": 8, "val": 4, "test": 6}
    sp = json.loads((MF3D / "split.json").read_text())
    out = {"train": [], "val": [], "test": []}
    for part, caps in (("train", sp["train"]), ("test", sp["test"])):
        for cap in caps:
            d = _cap_dir(cap)
            fr = sorted((p for p in d.glob("*.jpg") if (d / (p.stem + "_m.png")).exists()),
                        key=lambda p: p.name)
            if not fr:
                continue
            tgt = part
            if part == "train" and zlib.crc32(("val-" + cap).encode()) % 100 < 12:
                tgt = "val"
            n = min(per_cap[tgt], len(fr))
            idx = np.unique(np.linspace(0, len(fr) - 1, n).round().astype(int))
            out[tgt] += [(fr[i], d / (fr[i].stem + "_m.png"), cap) for i in idx]
    return out


def load_real(items):
    """-> list of dict(img uint8 HxWx3, m uint8 HxW {0,1}, group)."""
    out = []
    for p, mp, cap in items:
        im = np.asarray(Image.open(p).convert("RGB"))
        m = (np.asarray(Image.open(mp)) > 127).astype(np.uint8)
        if m.shape != im.shape[:2]:
            m = np.asarray(Image.fromarray(m * 255).resize((im.shape[1], im.shape[0]), Image.NEAREST)) > 127
            m = m.astype(np.uint8)
        out.append(dict(img=im, m=m, group=cap))
    return out


def load_n5k(n, seed=0):
    from . import realism as R
    dishes = sorted(N5K.glob("dish_*/rgb.png"))
    random.Random(seed).shuffle(dishes)
    out = []
    for p in dishes[:n]:
        im = Image.open(p).convert("RGB")
        m = R.detect_food_n5k(im)
        if m.mean() < 0.02 or m.mean() > 0.8:      # heuristic failed / degenerate
            continue
        out.append(dict(img=np.asarray(im), m=m.astype(np.uint8), group=p.parent.name))
    return out


# ------------------------------------------------------------------ synthetic data
def synth_frames(root: Path, fps: int, short=STORE_SHORT):
    """Evenly-spaced `fps` frames per scan with rgb + raw seg -> list of dict(img, m, group, sid)."""
    out, hist = [], np.zeros(256, np.int64)
    for sd in sorted(glob.glob(str(root / "scans/*"))):
        sid = Path(sd).name
        have = sorted(int(Path(p).stem) for p in glob.glob(f"{sd}/rgb/*.jpg")
                      if (root / "raw" / sid / f"frame_{int(Path(p).stem):06d}.npz").exists())
        if not have:
            continue
        pick = [have[i] for i in np.unique(np.linspace(0, len(have) - 1, min(fps, len(have))).round().astype(int))]
        for k in pick:
            im = Image.open(f"{sd}/rgb/{k:06d}.jpg")
            seg = np.load(root / "raw" / sid / f"frame_{k:06d}.npz")["seg"]
            w, h = im.size
            s = short / min(w, h)
            sz = (round(w * s), round(h * s))
            im = im.convert("RGB").resize(sz, Image.LANCZOS)
            seg = np.asarray(Image.fromarray(seg.astype(np.uint8)).resize(sz, Image.NEAREST))
            hist += np.bincount(seg.ravel(), minlength=256)
            m = ((seg >= FOOD_MIN) & (seg <= FOOD_MAX)).astype(np.uint8)
            out.append(dict(img=np.asarray(im), m=m, group=sid, sid=sid))
    return out, hist


def synth_assets(root: Path):
    """capture ids (split.json style 'Cat/capture') whose MetaFood3D mesh appears in the specs."""
    ids = set()
    for p in glob.glob(str(root / "specs/*/spec.json")):
        for it in json.loads(Path(p).read_text()).get("items", []):
            a = it.get("asset_id") or ""
            if a.startswith("metafood3d_"):
                ids.add(a)
    return ids


def capture_seen(cap: str, ids: set) -> bool:
    cat, name = cap.split("/", 1)
    return f"metafood3d_{cat}_{name}" in ids


# ------------------------------------------------------------------ training data
class CropDS:
    """Random food-anchored square crops (box -> 320x320), h-flip, light photometric jitter."""

    def __init__(self, recs, size, smin, smax, jitter, steps_len):
        self.r, self.size, self.smin, self.smax, self.jit, self.n = recs, size, smin, smax, jitter, steps_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch
        rng = np.random.default_rng((zlib.crc32(f"{torch.initial_seed()}-{i}".encode())))
        r = self.r[rng.integers(len(self.r))]
        img, m = r["img"], r["m"]
        h, w = m.shape
        side = float(rng.uniform(self.smin, self.smax)) * min(h, w)
        if rng.random() < 0.8 and m.any():
            ys, xs = np.nonzero(m[::4, ::4])
            j = rng.integers(len(ys)); cy, cx = ys[j] * 4 + 2, xs[j] * 4 + 2
            cy += rng.uniform(-side / 4, side / 4); cx += rng.uniform(-side / 4, side / 4)
        else:
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        x0 = min(max(cx - side / 2, 0), w - side); y0 = min(max(cy - side / 2, 0), h - side)
        box = (x0, y0, x0 + side, y0 + side)
        a = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR, box=box)
        b = Image.fromarray(m).resize((self.size, self.size), Image.NEAREST, box=box)
        a = np.asarray(a, np.float32) / 255
        b = np.asarray(b, np.int64)
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        if self.jit:
            a = a * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
            g = a.mean(2, keepdims=True)
            a = g + (a - g) * rng.uniform(0.8, 1.2)
            a = np.clip(a, 0, 1)
        a = (a - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(b))


def make_model(dev):
    import torch
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    m = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                                     num_classes=2, aux_loss=False)
    return m.to(dev)


def train(model, recs, steps, lr, a, dev, seed, evalfn=None, tag=""):
    """Returns (model, history, best_state_by_val_mIoU)."""
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(seed)
    ds = CropDS(recs, a.crop, a.smin, a.smax, not a.no_jitter, steps * a.bs)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True,
                    persistent_workers=False, worker_init_fn=lambda w: None,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    hist, best, best_v = [], None, -1
    model.train()
    t0 = time.time()
    for it, (x, y) in enumerate(dl, 1):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = torch.nn.functional.cross_entropy(model(x)["out"].float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if evalfn and (it % a.eval_every == 0 or it == steps):
            v = evalfn(model)
            hist.append(dict(step=it, loss=float(loss), val_miou=v["miou"], val_food_iou=v["food_iou"]))
            if v["miou"] > best_v:
                best_v, best = v["miou"], copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
            model.train()
            print(f"  {tag} step {it}/{steps} loss {loss.item():.3f} val mIoU {v['miou']:.3f} "
                  f"food {v['food_iou']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return model, hist, best


# ------------------------------------------------------------------ evaluation
def _bnd(m):
    from scipy import ndimage as ndi
    return m & ~ndi.binary_erosion(m, np.ones((3, 3)), border_value=1)


def boundary_f(p, g, tol):
    from scipy import ndimage as ndi
    pb, gb = _bnd(p), _bnd(g)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    prec = (ndi.distance_transform_edt(~gb)[pb] <= tol).mean()
    rec = (ndi.distance_transform_edt(~pb)[gb] <= tol).mean()
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def prep_eval(recs):
    """Resize to the fixed eval size once -> list of (tensor CHW float, mask bool HxW, group)."""
    import torch
    out = []
    for r in recs:
        h, w = r["m"].shape
        H, W = EVAL_HW if h >= w else EVAL_HW[::-1]
        a = np.asarray(Image.fromarray(r["img"]).resize((W, H), Image.BILINEAR), np.float32) / 255
        m = np.asarray(Image.fromarray(r["m"] * 255).resize((W, H), Image.NEAREST)) > 127
        out.append((torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy()), m, r["group"]))
    return out


def predict(model, ev, dev, bs=8):
    import torch
    model.eval()
    preds = [None] * len(ev)
    order = sorted(range(len(ev)), key=lambda i: tuple(ev[i][0].shape))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and j - i < bs and ev[order[j]][0].shape == ev[order[i]][0].shape:
                j += 1
            x = torch.stack([ev[k][0] for k in order[i:j]]).to(dev)
            p = model(x)["out"].argmax(1).cpu().numpy().astype(bool)
            for k, pk in zip(order[i:j], p):
                preds[k] = pk
            i = j
    return preds


def metrics(preds, ev, tol):
    """dataset-level IoUs + per-image arrays (I/U per class, boundary F) for bootstrapping."""
    rows = []
    for p, (_, g, grp) in zip(preds, ev):
        If, Uf = (p & g).sum(), (p | g).sum()
        Ib, Ub = (~p & ~g).sum(), (~p | ~g).sum()
        rows.append((If, Uf, Ib, Ub, boundary_f(p, g, tol)))
    r = np.array(rows, np.float64)
    return dict(food_iou=float(r[:, 0].sum() / max(r[:, 1].sum(), 1)),
                bg_iou=float(r[:, 2].sum() / max(r[:, 3].sum(), 1)),
                miou=float((r[:, 0].sum() / max(r[:, 1].sum(), 1) + r[:, 2].sum() / max(r[:, 3].sum(), 1)) / 2),
                bf=float(r[:, 4].mean())), r


def agg_rows(r):
    return dict(food_iou=r[:, 0].sum() / max(r[:, 1].sum(), 1),
                miou=(r[:, 0].sum() / max(r[:, 1].sum(), 1) + r[:, 2].sum() / max(r[:, 3].sum(), 1)) / 2,
                bf=r[:, 4].mean())


def boot_ci(r, groups, n=1000, seed=0):
    """95 % CI of dataset-level metrics, resampling captures."""
    groups = np.asarray(groups)
    ug = np.unique(groups)
    if len(ug) < 2:
        return {k: [float("nan")] * 2 for k in ("food_iou", "miou", "bf")}
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("food_iou", "miou", "bf")}
    for _ in range(n):
        ch = rng.choice(ug, len(ug))
        a = agg_rows(r[np.concatenate([idx[g] for g in ch])])
        for k in acc:
            acc[k].append(a[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def eval_set(model, ev, dev, tol):
    pr = predict(model, ev, dev)
    return metrics(pr, ev, tol)


# ------------------------------------------------------------------ orchestration
def summarise(per_seed_rows, groups, seen=None):
    """per_seed_rows: list over seeds of (N,5) arrays -> mean/sd over seeds + CI of the seed-mean."""
    per = [agg_rows(r) for r in per_seed_rows]
    mean_rows = np.mean(per_seed_rows, 0)
    out = {"seeds": len(per), "n_images": int(len(groups)), "n_captures": int(len(set(groups)))}
    ci = boot_ci(mean_rows, groups)
    for k in ("food_iou", "miou", "bf"):
        v = [p[k] for p in per]
        out[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), ci95=ci[k], per_seed=[float(x) for x in v])
    return out


def run(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp = real_split()
    if a.debug:                                   # tiny subsets for a smoke test
        sp = {k: v[::max(len(v) // 24, 1)] for k, v in sp.items()}
    train_real = load_real(sp["train"]); val_real = load_real(sp["val"]); test_real = load_real(sp["test"])
    ev_val, ev_test = prep_eval(val_real), prep_eval(test_real)
    ev_n5k = prep_eval(load_n5k(a.eval_n5k)) if a.eval_n5k else None
    print(f"real train {len(train_real)} val {len(val_real)} test {len(test_real)} images "
          f"({len({r['group'] for r in test_real})} test captures)", flush=True)
    tol = a.bf_tol
    ids, label = set(), "real"
    if a.synth.lower() != "none":
        root = Path(a.synth)
        label = root.name
        synth, hist = synth_frames(root, a.frames_per_scan)
        ids = synth_assets(root)
        fg = hist[FOOD_MIN:FOOD_MAX + 1].sum() / max(hist.sum(), 1)
        print(f"synthetic {label}: {len(synth)} frames, food fraction {fg:.3f}, "
              f"label ids present {np.nonzero(hist)[0].tolist()[:12]}...", flush=True)
        train_recs = synth
    else:
        rng = random.Random(0)
        train_recs = train_real
        if a.real_n:
            # n images spread over captures (deterministic)
            caps = sorted({r["group"] for r in train_real}); rng.shuffle(caps)
            by = {c: [r for r in train_real if r["group"] == c] for c in caps}
            train_recs, k = [], 0
            while len(train_recs) < a.real_n:
                c = caps[k % len(caps)]; j = k // len(caps)
                if j < len(by[c]):
                    train_recs.append(by[c][j])
                k += 1
                if k > len(caps) * 20: break
    seen = np.array([capture_seen(g, ids) for _, _, g in ev_test]) if ids else np.zeros(len(ev_test), bool)
    groups = [g for _, _, g in ev_test]
    name = a.name or (f"{label}" + (f"_ft{a.ft_real}" if a.ft_real else "") + (f"_n{a.real_n}" if (a.real_n and label == "real") else ""))
    rep = dict(name=name, args=vars(a), real=dict(train=len(train_real), val=len(val_real), test=len(test_real)),
               n_train_images=len(train_recs), test_seen_frac=float(seen.mean()), arms={})
    arms = {}

    def collect(arm, ckpt, model, rows_by_set):
        for k, v in rows_by_set.items():
            arms.setdefault(arm, {}).setdefault(ckpt, {}).setdefault(k, []).append(v)

    for seed in a.seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        t0 = time.time()
        model = make_model(dev)
        evfn = lambda m: eval_set(m, ev_val, dev, tol)[0]
        model, h, best = train(model, train_recs, a.steps, a.lr, a, dev, seed, evfn, tag=f"[{name} s{seed}]")
        stage = "synth_only" if a.synth.lower() != "none" else "real_only"
        for ck, state in (("final", None), ("best_val", best)):
            if state is not None:
                m2 = make_model(dev); m2.load_state_dict(state)
            else:
                m2 = model
            res = {"test": eval_set(m2, ev_test, dev, tol)[1]}
            if ev_n5k:
                res["n5k_pseudo"] = eval_set(m2, ev_n5k, dev, tol)[1]
            collect(stage, ck, m2, res)
            if state is not None:
                del m2
        arms[stage].setdefault("history", []).append(h)
        if a.synth.lower() != "none" and a.ft_real:
            rng = random.Random(100 + seed)
            caps = sorted({r["group"] for r in train_real}); rng.shuffle(caps)
            by = {c: [r for r in train_real if r["group"] == c] for c in caps}
            ft, k = [], 0
            while len(ft) < a.ft_real and k < len(caps) * 20:
                c = caps[k % len(caps)]; j = k // len(caps)
                if j < len(by[c]): ft.append(by[c][j])
                k += 1
            model, h2, best2 = train(model, ft, a.ft_steps, a.lr * 0.3, a, dev, seed + 1000, evfn, tag=f"[{name}+ft s{seed}]")
            for ck, state in (("final", None), ("best_val", best2)):
                m2 = model
                if state is not None:
                    m2 = make_model(dev); m2.load_state_dict(state)
                res = {"test": eval_set(m2, ev_test, dev, tol)[1]}
                if ev_n5k:
                    res["n5k_pseudo"] = eval_set(m2, ev_n5k, dev, tol)[1]
                collect("synth_plus_real_ft", ck, m2, res)
            arms["synth_plus_real_ft"].setdefault("history", []).append(h2)
        del model
        torch.cuda.empty_cache()
        print(f"[{name}] seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    for arm, d in arms.items():
        rep["arms"][arm] = {"history": d.pop("history", [])}
        for ck, sets in d.items():
            ent = {}
            for sname, rows in sets.items():
                if sname == "test":
                    ent["test_all"] = summarise(rows, groups)
                    if ids and (~seen).sum() >= 2:
                        ent["test_unseen"] = summarise([r[~seen] for r in rows], [g for g, s in zip(groups, seen) if not s])
                else:
                    ent[sname] = summarise(rows, [g for _, _, g in ev_n5k])
            rep["arms"][arm][ck] = ent
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {out / (name + '.json')}", flush=True)
    for arm, d in rep["arms"].items():
        for ck in ("final", "best_val"):
            t = d[ck]["test_all"]
            print(f"  {name} {arm:20s} {ck:9s} test food IoU {t['food_iou']['mean']:.3f} "
                  f"mIoU {t['miou']['mean']:.3f} (CI {t['miou']['ci95'][0]:.3f}-{t['miou']['ci95'][1]:.3f}) "
                  f"bF {t['bf']['mean']:.3f}")
    return rep


def summarize_dir(d):
    reps = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(d) / "*.json")))
            if "arms" in json.loads(Path(p).read_text())]
    lines = ["| run | arm | ckpt | seeds | food IoU | mIoU [95% CI] | boundary F | mIoU unseen-asset test |",
             "|---|---|---|---|---|---|---|---|"]
    for r in reps:
        for arm, d_ in r["arms"].items():
            for ck in ("final", "best_val"):
                t = d_[ck]["test_all"]; u = d_[ck].get("test_unseen")
                lines.append(f"| {r['name']} | {arm} | {ck} | {t['seeds']} | {t['food_iou']['mean']:.3f} +- {t['food_iou']['sd']:.3f} | "
                             f"{t['miou']['mean']:.3f} [{t['miou']['ci95'][0]:.3f}, {t['miou']['ci95'][1]:.3f}] | "
                             f"{t['bf']['mean']:.3f} | {('%.3f' % u['miou']['mean']) if u else '-'} |")
    print("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", default="none", help="synthetic batch root, or 'none' for the real-train reference")
    ap.add_argument("--out", default="../results/synth/tstr")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--smin", type=float, default=0.25, help="crop side / short image side, min")
    ap.add_argument("--smax", type=float, default=1.0)
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--frames-per-scan", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bf-tol", type=float, default=3.0)
    ap.add_argument("--real-n", type=int, default=0, help="with --synth none: train on only N real images")
    ap.add_argument("--ft-real", type=int, default=0, help="after synthetic training, fine-tune on N real train images")
    ap.add_argument("--ft-steps", type=int, default=300)
    ap.add_argument("--eval-n5k", type=int, default=0, help="also eval on N Nutrition5k pseudo-labelled dishes")
    ap.add_argument("--debug", action="store_true", help="tiny real subsets (smoke test)")
    ap.add_argument("--summarize", default=None, help="print a table of the JSON reports in this dir and exit")
    a = ap.parse_args()
    if a.summarize:
        summarize_dir(a.summarize)
        return
    run(a)


if __name__ == "__main__":
    main()
