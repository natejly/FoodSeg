"""MetaFood3D single-image volume / energy benchmark (MFP3D protocol), fine-tuned ConvNeXt.

    python -m eval.mf3d_train ../data/metafood3d --out ../results/metafood3d/img/s_rgb

Split, images and targets: MFP3D's released h5 files (github.com/jingema99/MFP3D):
401 train / 95 test objects, one 224x224 RGB image each (MetaFood3D still photos, not
video frames), targets volume, energy, weight, fat, carb, protein. Test = exactly those
95 h5 images (4-flip TTA), so numbers are comparable with MFP3D / PortionNet.
Training = the 401 h5 train images, plus (--frames > 0) video frames 0, 10, ... of the same
training objects (eval.mf3d_frames), squashed to 224x224 the same way, as extra views.
10 % of training *objects* (their h5 image) are held out to pick the epoch.
RGB only (the h5 images have no masks).

Metrics as in MFP3D / PortionNet: MAE and MAPE = mean over objects of |pred - gt| / gt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.n5k_train import MEAN, STD, augment, build_model, flips

TARGETS = ("volume", "energy", "weight", "fat", "carb", "protein")


_ARUCO = None


def aruco_features(img: np.ndarray) -> list[float]:
    """Scale cue from the printed ArUco board in every MetaFood3D capture (same board throughout):
    [found, log mean marker side (px, at 224), foreshortening (min/max side), log marker count]."""
    global _ARUCO
    if _ARUCO is None:
        _ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12),
                                         cv2.aruco.DetectorParameters())
    g = cv2.cvtColor(cv2.resize(img, (672, 672), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _ARUCO.detectMarkers(g)
    if ids is None or len(ids) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    sides = []
    for c in corners:
        q = c.reshape(4, 2) / 3.0                          # back to 224-px units
        sides.append(np.linalg.norm(q - np.roll(q, 1, 0), axis=1))
    sides = np.concatenate(sides)
    return [1.0, float(np.log(sides.mean())), float(sides.min() / sides.max()), float(np.log(len(ids)))]


class WithScale(nn.Module):
    """Backbone features + ArUco scale features -> head."""

    def __init__(self, arch: str, n_out: int, n_feat: int = 4):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, drop_path_rate=0.1)
        f = self.backbone.num_features
        self.feat = nn.Sequential(nn.Linear(n_feat, 64), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(f + 64), nn.Linear(f + 64, n_out))

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.set_grad_checkpointing(enable)

    def forward(self, x, z):
        return self.head(torch.cat([self.backbone(x), self.feat(z)], 1))


def load_h5(h5dir: Path, split: str) -> tuple[list[str], np.ndarray, dict]:
    with h5py.File(h5dir / f"gt_{split}.h5", "r") as f:
        objs = ["__".join(p.decode().split("/")[-4:-2]) for p in f["image_path"][:]]
        imgs = (f["image"][:].transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)   # (N,224,224,3) RGB
        tg = {o: {t: float(f[t][i]) for t in TARGETS} for i, o in enumerate(objs)}
    return objs, imgs, tg


def load_frames(fdir: Path, obj: str, frames: list[int]) -> np.ndarray:
    out = []
    for k in frames:
        p = fdir / obj / f"{k}.jpg"
        if p.exists():
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA))
    return np.stack(out) if out else np.zeros((0, 224, 224, 3), np.uint8)


def to_input(rgb: np.ndarray, dev: str) -> torch.Tensor:
    x = torch.from_numpy(rgb).to(dev).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN.to(dev)) / STD.to(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="data/metafood3d (with mfp3d_h5/ and frames/)")
    ap.add_argument("--arch", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=20, help="extra video frames per training object (0 = h5 image only)")
    ap.add_argument("--aruco", action="store_true", help="feed ArUco-board scale features to the head")
    ap.add_argument("--full", action="store_true", help="no val objects; keep the last epoch")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"

    fdir, h5dir = args.root / "frames", args.root / "mfp3d_h5"
    tr_objs, tr_imgs, tg_tr = load_h5(h5dir, "train")
    te_objs, te_imgs, tg_te = load_h5(h5dir, "test")
    tgt = {**tg_tr, **tg_te}
    perm = np.random.default_rng(0).permutation(len(tr_objs))
    n_val = 0 if args.full else len(tr_objs) // 10
    val_set = {tr_objs[i] for i in perm[:n_val]}
    ok = lambda o: all(np.isfinite(tgt[o][t]) and tgt[o][t] >= 0 for t in TARGETS)  # noqa: E731
    fit_idx = [i for i in range(len(tr_objs)) if tr_objs[i] not in val_set and ok(tr_objs[i])]
    print(f"dropped {sum(not ok(o) for o in tr_objs)} training objects with negative / missing labels", flush=True)
    val_objs = [o for o in tr_objs if o in val_set]

    frames = list(range(0, 200, 10))[:args.frames]
    R, Y = [tr_imgs[fit_idx]], [[tgt[tr_objs[i]] for i in fit_idx]]
    n_with = 0
    for i in fit_idx:
        if frames:
            fr = load_frames(fdir, tr_objs[i], frames)
            if len(fr):
                n_with += 1
                R.append(fr)
                Y.append([tgt[tr_objs[i]]] * len(fr))
    R = np.concatenate(R)
    Y = torch.tensor([[math.log1p(d[t]) for t in TARGETS] for ys in Y for d in ys])
    mu, sd = Y.mean(0), Y.std(0)
    Yn = (Y - mu) / sd
    print(f"train objects {len(fit_idx)} ({n_with} with video frames), val {len(val_objs)}, test {len(te_objs)}; "
          f"{len(R)} training images", flush=True)
    eval_imgs = {**{tr_objs[i]: tr_imgs[i:i + 1] for i in range(len(tr_objs))},
                 **{te_objs[i]: te_imgs[i:i + 1] for i in range(len(te_objs))}}
    nz = 4
    Z = torch.tensor([aruco_features(im) for im in R]) if args.aruco else torch.zeros(len(R), nz)
    eval_z = {o: (torch.tensor([aruco_features(im[0])]) if args.aruco else torch.zeros(1, nz))
              for o, im in eval_imgs.items()}
    if args.aruco:
        print(f"ArUco found in {int(Z[:, 0].sum())}/{len(Z)} training images, "
              f"{int(sum(z[0, 0] for z in eval_z.values()))}/{len(eval_z)} eval images", flush=True)

    model = (WithScale(args.arch, len(TARGETS)) if args.aruco
             else build_model(args.arch, 0, 0.1, R.shape[1:3], len(TARGETS))).to(dev).to(memory_format=torch.channels_last)
    fwd = (lambda x, z: model(x, z)) if args.aruco else (lambda x, z: model(x))  # noqa: E731
    model.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    spe = len(R) // args.bs
    total, warm = args.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler()

    @torch.no_grad()
    def predict(objs):
        model.eval()
        out = []
        for o in objs:
            x = to_input(eval_imgs[o], dev).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                z = eval_z[o].to(dev)
                p = torch.stack([fwd(v, z).float() for v in flips(x)]).mean(0)[0]
            out.append(p.cpu() * sd + mu)
        return torch.expm1(torch.stack(out)).clamp(min=0).numpy()

    def score(objs, pred):
        true = np.array([[tgt[o][t] for t in TARGETS] for o in objs])
        res = {}
        for i, t in enumerate(TARGETS):
            ok = true[:, i] > 0
            res[t] = {"mae": float(np.abs(pred[:, i] - true[:, i]).mean()),
                      "mape_pct": float(100 * np.mean(np.abs(pred[ok, i] - true[ok, i]) / true[ok, i])),
                      "pct_of_mean": float(100 * np.abs(pred[:, i] - true[:, i]).mean() / true[:, i].mean())}
        return res

    log, best, best_key = [], None, float("inf")
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        order = rng.permutation(len(R))
        for st in range(spe):
            bi = order[st * args.bs:(st + 1) * args.bs]
            x = augment(to_input(R[bi], dev)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                out = fwd(x, Z[bi].to(dev))
            loss = F.l1_loss(out.float(), Yn[bi].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        rec = {"epoch": ep, "loss": run / spe, "sec": round(time.time() - t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if val_objs:
            v = score(val_objs, predict(val_objs))
            k = (v["volume"]["mape_pct"] + v["energy"]["mape_pct"]) / 2
            rec.update(val_vol_mape=round(v["volume"]["mape_pct"], 2), val_eng_mape=round(v["energy"]["mape_pct"], 2))
        else:
            k = -ep
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if k < best_key:
            best_key, best = k, {n: p.detach().clone() for n, p in model.state_dict().items()}
    model.load_state_dict(best)
    pred = predict(te_objs)
    res = score(te_objs, pred)
    json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "best_val": best_key, "test": res, "log": log},
              open(args.out / "summary.json", "w"), indent=2)
    with open(args.out / "test_predictions.csv", "w") as fh:
        fh.write("object," + ",".join(f"true_{t},pred_{t}" for t in TARGETS) + "\n")
        for o, p in zip(te_objs, pred):
            fh.write(o + "," + ",".join(f"{tgt[o][t]:.3f},{p[i]:.3f}" for i, t in enumerate(TARGETS)) + "\n")
    print("TEST " + "  ".join(f"{t} MAE {r['mae']:.1f} MAPE {r['mape_pct']:.1f}%" for t, r in res.items()), flush=True)


if __name__ == "__main__":
    main()
