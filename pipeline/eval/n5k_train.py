"""Fine-tune a CNN on Nutrition5k overhead RGB(-D) to predict mass, kcal and macros.

    python -m eval.n5k_train ../data/datasets/Nutrition5k --channels rgbd_thick \
        --out ../results/n5k/train/rgbd_thick

Inputs come from eval.n5k_cache (448x336). --channels:
    rgb          RGB only
    rgbd         + height above the counter plane (the depth, minus the rig)
    rgbd_thick   + food height above its container (bowl surface / plate plane) + food mask
Extra channels enter through zero-initialised stem weights, so training starts
from the ImageNet RGB model exactly.

Targets: log1p of mass, kcal, fat, carb, protein, standardised on train; L1
loss on that scale (--loss log), on the raw values over their train mean
(--loss raw, the MAE metric itself; log-scale L1 shrinks big dishes), or both. Augmentation is metric-safe:
flips, any rotation and small shifts, brightness/contrast, never scaling (pixel
size carries portion size). 10 % of depth_train_ids is held out to pick the
epoch; depth_test_ids is scored once at the end, with 4-flip test-time
averaging. fp16 autocast (Turing has no bf16).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.nutrition5k import read_metadata, read_split

TARGETS = ("mass", "kcal", "fat", "carb", "protein")
EXTRA = {"rgb": 0, "rgbd": 1, "rgbd_thick": 3}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(arch: str, n_extra: int, drop_path: float, img_size: tuple[int, int],
                n_out: int = len(TARGETS)) -> nn.Module:
    vit = "vit" in arch
    kw = {"img_size": img_size, "dynamic_img_size": True} if vit else {}
    model = timm.create_model(arch, pretrained=True, num_classes=n_out, drop_path_rate=drop_path, **kw)
    if n_extra:
        old = model.patch_embed.proj if vit else model.stem[0]   # ViT / ConvNeXt patchify conv
        new = nn.Conv2d(3 + n_extra, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        if vit:
            model.patch_embed.proj = new
        else:
            model.stem[0] = new
    return model


class DualBranch(nn.Module):
    """RGB backbone + separate depth backbone (depth / container height / mask channels),
    pooled features concatenated into one head (NuNet / IGSMNet-style late fusion)."""

    def __init__(self, arch: str, arch_depth: str, n_extra: int, drop_path: float, n_out: int):
        super().__init__()
        self.rgb = timm.create_model(arch, pretrained=True, num_classes=0, drop_path_rate=drop_path)
        self.dep = timm.create_model(arch_depth, pretrained=True, num_classes=0, drop_path_rate=drop_path,
                                     in_chans=n_extra)      # timm adapts the stem's pretrained weights
        f = self.rgb.num_features + self.dep.num_features
        self.head = nn.Sequential(nn.LayerNorm(f), nn.Dropout(0.1), nn.Linear(f, n_out))

    def set_grad_checkpointing(self, enable: bool = True):
        self.rgb.set_grad_checkpointing(enable)
        self.dep.set_grad_checkpointing(enable)

    def forward(self, x):
        return self.head(torch.cat([self.rgb(x[:, :3]), self.dep(x[:, 3:])], 1))


class Data:
    """Whole cache in RAM as uint8/fp16 tensors; batches assembled and augmented on the GPU."""

    def __init__(self, cache: Path, channels: str):
        self.ids = (cache / "ids.txt").read_text().split()
        self.index = {d: i for i, d in enumerate(self.ids)}
        self.rgb = torch.from_numpy(np.load(cache / "rgb.npy"))                 # (N,H,W,3) u8
        self.extra = []
        if channels in ("rgbd", "rgbd_thick"):
            self.extra.append((torch.from_numpy(np.load(cache / "h_counter.npy")), 10.0))   # /0.1 m
        if channels == "rgbd_thick":
            self.extra.append((torch.from_numpy(np.load(cache / "thick.npy")), 20.0))       # /0.05 m
            self.extra.append((torch.from_numpy(np.load(cache / "mask.npy")), 1.0))

    def batch(self, rows: list[int], dev: str) -> torch.Tensor:
        idx = torch.tensor(rows)
        x = self.rgb[idx].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255
        x = (x - MEAN.to(dev)) / STD.to(dev)
        ex = [a[idx].to(dev).float()[:, None] * s for a, s in self.extra]
        return torch.cat([x] + ex, 1)


def augment(x: torch.Tensor, n_rgb: int = 3) -> torch.Tensor:
    B, _, H, W = x.shape
    dev = x.device
    th = torch.rand(B, device=dev) * 2 * math.pi
    sx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    sy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    t = (torch.rand(B, 2, device=dev) - 0.5) * 0.12
    c, s = torch.cos(th), torch.sin(th)
    ar = W / H                                   # rotate in pixel space, not normalised space
    A = torch.stack([torch.stack([c * sx, -s * sy / ar, t[:, 0]], 1),
                     torch.stack([s * sx * ar, c * sy, t[:, 1]], 1)], 1)
    grid = F.affine_grid(A, x.shape, align_corners=False)
    rgb = F.grid_sample(x[:, :n_rgb], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = [rgb * (1 + 0.3 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5))
           + 0.3 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)]
    if x.shape[1] > n_rgb:
        out.append(F.grid_sample(x[:, n_rgb:], grid, mode="nearest", padding_mode="zeros", align_corners=False))
    return torch.cat(out, 1)


def flips(x: torch.Tensor):
    return [x, x.flip(3), x.flip(2), x.flip(2, 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("../data/cache/n5k_448"))
    ap.add_argument("--channels", choices=list(EXTRA), default="rgbd_thick")
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=32, help="inference batch for the val/test forward (default %(default)s); lower for big backbones")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0, help="init / batch order / augmentation")
    ap.add_argument("--split-seed", type=int, default=0, help="which 10 %% of train is val (keep fixed to ensemble)")
    ap.add_argument("--loss", choices=["log", "raw", "mix"], default="log",
                    help="L1 on standardised log1p targets, on raw values / train mean (the metric), or their mean")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (big backbones)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights, e.g. 0.999; 0 = off")
    ap.add_argument("--full", action="store_true",
                    help="train on all of depth_train_ids, no val; keep the last epoch")
    ap.add_argument("--arch-depth", default="",
                    help="separate backbone for the non-RGB channels (late fusion), e.g. convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--comp", action="store_true",
                    help="predict log1p(mass) and log per-gram kcal/fat/carb/protein; totals = density x mass")
    ap.add_argument("--ingr-aux", type=float, default=0.0,
                    help="weight of an auxiliary head predicting log1p grams of each common ingredient "
                         "(training labels only; test input is still just the image); 0 = off")
    ap.add_argument("--ingr-min", type=int, default=10, help="ingredients seen in >= this many train dishes")
    ap.add_argument("--select", choices=["masskcal", "all5"], default="masskcal",
                    help="val score that picks the epoch: mean %% of mass and kcal, or of all five targets")
    ap.add_argument("--target-weights", default="",
                    help="comma list of 5 loss weights for mass,kcal,fat,carb,protein (normalised to mean 1); empty = equal")
    ap.add_argument("--uncertainty", action="store_true",
                    help="learned per-target log-variance weighting (Kendall-style, Laplace form: exp(-s)*L1 + s)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (smoke test)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda"

    meta = read_metadata(args.root)
    data = Data(args.cache, args.channels)
    tr_all = [d for d in read_split(args.root, "depth_train_ids") if d in data.index and d in meta]
    test = [d for d in read_split(args.root, "depth_test_ids") if d in data.index and d in meta]
    perm = np.random.default_rng(args.split_seed).permutation(len(tr_all))
    rng = np.random.default_rng(args.seed)
    n_val = 0 if args.full else int(len(tr_all) * args.val_frac)
    val = [tr_all[i] for i in perm[:n_val]]
    train = [tr_all[i] for i in perm[n_val:]]
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  channels {args.channels}", flush=True)

    assert not (args.comp and args.loss != "log"), "--comp works on the log scale only"

    def encode(d):
        m = meta[d]
        if not args.comp:
            return [math.log1p(m[t]) for t in TARGETS]
        return [math.log1p(m["mass"])] + [math.log((m[t] + 0.1) / (m["mass"] + 1.0)) for t in TARGETS[1:]]

    def decode(z: torch.Tensor) -> torch.Tensor:          # (N,5) log space -> raw values
        if not args.comp:
            return torch.expm1(z)
        mass = torch.expm1(z[:, :1])
        return torch.cat([mass, torch.exp(z[:, 1:]) * (mass + 1.0) - 0.1], 1)

    Y = lambda ids: torch.tensor([encode(d) for d in ids])  # noqa: E731
    y_tr = Y(train)
    mu, sd = y_tr.mean(0), y_tr.std(0)

    # auxiliary ingredient targets: log1p grams per common ingredient, from train dishes only
    counts: dict[str, int] = {}
    for d in train:
        for n, _ in meta[d]["ingredients"]:
            counts[n] = counts.get(n, 0) + 1
    vocab = sorted(n for n, c in counts.items() if c >= args.ingr_min) if args.ingr_aux else []
    vidx = {n: i for i, n in enumerate(vocab)}
    A_tr = torch.zeros(len(train), len(vocab))
    for r, d in enumerate(train):
        for n, g in meta[d]["ingredients"]:
            if n in vidx:
                A_tr[r, vidx[n]] += g
    A_tr = torch.log1p(A_tr)
    a_sd = float(A_tr[A_tr > 0].std()) if len(vocab) else 1.0
    A_tr /= a_sd
    if vocab:
        print(f"ingredient aux head: {len(vocab)} ingredients (>= {args.ingr_min} train dishes), weight {args.ingr_aux}",
              flush=True)
    NT = len(TARGETS)

    img_size = tuple(data.rgb.shape[1:3])
    if args.arch_depth:
        assert EXTRA[args.channels], "--arch-depth needs depth channels"
        model = DualBranch(args.arch, args.arch_depth, EXTRA[args.channels], args.drop_path,
                           NT + len(vocab)).to(dev).to(memory_format=torch.channels_last)
    else:
        model = build_model(args.arch, EXTRA[args.channels], args.drop_path, img_size, NT + len(vocab)).to(dev).to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)
    ema = None
    if args.ema:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train) // (args.bs * args.accum)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total, warm = args.epochs * steps_per_epoch, 2 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler()

    @torch.no_grad()
    def predict(ids: list[str], tta: bool) -> np.ndarray:
        if ema is not None:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(live[k].dtype) for k, v in ema.items()})
        model.eval()
        out = []
        for b in range(0, len(ids), args.eval_bs):
            x = data.batch([data.index[d] for d in ids[b:b + args.eval_bs]], dev).contiguous(memory_format=torch.channels_last)
            views = flips(x) if tta else [x]
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.stack([model(v).float()[:, :NT] for v in views]).mean(0)
            out.append((p.cpu() * sd + mu))
        if ema is not None:
            model.load_state_dict(live)
        return decode(torch.cat(out)).clamp(min=0).numpy()

    def score(ids, pred) -> dict:
        true = np.array([[meta[d][t] for t in TARGETS] for d in ids])
        mae = np.abs(pred - true).mean(0)
        return {t: (float(mae[i]), float(100 * mae[i] / true[:, i].mean())) for i, t in enumerate(TARGETS)}

    log, best, best_key = [], None, float("inf")
    tr_rows = np.array([data.index[d] for d in train])
    y_tr_n = (y_tr - mu) / sd
    raw_mean = decode(y_tr).mean(0).to(dev)
    mu_d, sd_d = mu.to(dev), sd.to(dev)

    tw = None
    if args.target_weights:
        tw = torch.tensor([float(v) for v in args.target_weights.split(",")], device=dev)
        assert len(tw) == NT
        tw = tw / tw.mean()
    log_var = nn.Parameter(torch.zeros(NT, device=dev)) if args.uncertainty else None
    if log_var is not None:
        opt.add_param_group({"params": [log_var], "lr": 1e-2, "weight_decay": 0.0})
        sched.base_lrs.append(1e-2); sched.lr_lambdas.append(sched.lr_lambdas[0])

    def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l_log = (out - y).abs().mean(0)                 # (NT,) per-target
        if args.loss == "log":
            lt = l_log
        else:
            raw_p = torch.expm1((out * sd_d + mu_d).clamp(max=12))
            raw_y = torch.expm1(y * sd_d + mu_d)
            l_raw = ((raw_p - raw_y).abs() / raw_mean).mean(0)
            lt = l_raw if args.loss == "raw" else 0.5 * (l_log + l_raw)
        if log_var is not None:
            return (torch.exp(-log_var) * lt + log_var).mean()
        return (lt * tw).mean() if tw is not None else lt.mean()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        order = rng.permutation(len(train))
        run = 0.0
        for st in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            for a in range(args.accum):
                b0 = (st * args.accum + a) * args.bs
                bi = order[b0:b0 + args.bs]
                x = augment(data.batch(tr_rows[bi].tolist(), dev)).contiguous(memory_format=torch.channels_last)
                y = y_tr_n[bi].to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                loss = loss_fn(out[:, :NT].float(), y)
                if vocab:
                    loss = loss + args.ingr_aux * F.smooth_l1_loss(out[:, NT:].float(), A_tr[bi].to(dev), beta=0.5)
                loss = loss / args.accum
                scaler.scale(loss).backward()
                run += loss.item()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(args.ema).add_(v.float(), alpha=1 - args.ema)
                        else:
                            ema[k].copy_(v)
        dt = time.time() - t0
        rec = {"epoch": ep, "loss": run / steps_per_epoch, "sec": round(dt, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if val:
            v = score(val, predict(val, tta=False))
            key = ((v["mass"][1] + v["kcal"][1]) / 2 if args.select == "masskcal"
                   else float(np.mean([v[t][1] for t in TARGETS])))
            rec.update(val_mass_pct=round(v["mass"][1], 2), val_kcal_pct=round(v["kcal"][1], 2),
                       val_mean5_pct=round(float(np.mean([v[t][1] for t in TARGETS])), 2))
        else:
            key = -ep                               # --full: keep the last epoch
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if key < best_key:
            best_key = key
            src = ema if ema is not None else model.state_dict()
            best = {k: t.detach().clone() for k, t in src.items()}

    torch.save(best, args.out / "best.pt")
    sd_live = model.state_dict()
    model.load_state_dict({k: v.to(sd_live[k].dtype) for k, v in best.items()})
    ema = None
    if val:
        vpred = predict(val, tta=True)
        with open(args.out / "val_predictions.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dish_id"] + [f"pred_{t}" for t in TARGETS])
            for d, p in zip(val, vpred):
                w.writerow([d] + [f"{x:.2f}" for x in p])
    pred = predict(test, tta=True)
    res = score(test, pred)
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "best_val": best_key,
               "test": res, "log": log}
    if val:
        summary["val"] = score(val, vpred)               # per-target, best epoch, 4-flip TTA
    if log_var is not None:
        summary["log_var"] = dict(zip(TARGETS, log_var.detach().cpu().tolist()))
    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    with open(args.out / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dish_id"] + [f"true_{t}" for t in TARGETS] + [f"pred_{t}" for t in TARGETS])
        for d, p in zip(test, pred):
            w.writerow([d] + [meta[d][t] for t in TARGETS] + [f"{x:.2f}" for x in p])
    print("TEST " + "  ".join(f"{t} {m:.1f} ({p:.1f}%)" for t, (m, p) in res.items()), flush=True)


if __name__ == "__main__":
    main()
