"""Fine-tune PTv3 (Sonata init) on synthetic fused scans: semantic (4 cls) + centre-offset heads.

  cd /home/kyle/Desktop/FoodSeg/pipeline
  python -m seg3d.train --smoke                       # 20 iterations end to end
  python -m seg3d.train --run ptv3_a --epochs 12      # see README for the full launch command
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import augment as A
from .data import DataConfig, FoodSegDataset, collate, list_scans, split_scans

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    run: str = "dev"
    model: str = "ptv3"                    # ptv3 | dgcnn
    pretrained: bool = True                # Sonata init (False = from scratch)
    patch_size: int = 1024
    drop_path: float = 0.1
    mass_head: bool = False
    mass_w: float = 0.0
    epochs: int = 12
    freeze_epochs: int = 3                 # Sonata encoder frozen (stem+decoder+heads train) for N epochs
    unfreeze: str = "top"                  # then: none | top (enc stages 3,4) | all
    batch_size: int = 1
    grad_accum: int = 2
    lr: float = 1e-3                       # heads / decoder / stem
    lr_backbone: float = 5e-5              # unfrozen encoder
    weight_decay: float = 0.02
    warmup_iters: int = 30
    amp: bool = True
    cls_weights: list = field(default_factory=lambda: [0.5, 1.0, 1.0, 2.0])
    off_w: float = 1.0
    dir_w: float = 0.1
    num_workers: int = 2
    val_every: int = 1
    max_minutes: float = 10.0
    seed: int = 0
    train_regime: str = "r1"             # r1: default lift corruption | r2: augment.HARD_LIFT corruption on every sample
    eval_regime: str = "r1"              # r1: stored lift as-is | r2: hard lift applied deterministically at eval
    max_mem_gb: float = 4.0                # abort if torch peak alloc exceeds (shared GPU)
    data: DataConfig = field(default_factory=DataConfig)
    aug: A.AugConfig = field(default_factory=A.AugConfig)
    decoder: str = "plain"                 # plain (offset+DBSCAN) | koc (keep-or-change head replaces it)
    koc_w: float = 1.0
    koc_eps: float = 0.05                  # label smoothing toward KEEP
    koc_change_w: float = 2.0              # loss weight of change points (target != own lift label)
    koc_gamma: float = 0.0                 # focal exponent
    koc_bias: float = 2.0                  # initial (learnable) logit bonus for KEEP
    koc_iters: int = 1                     # eval: 2 = re-run the head on the updated labels
    cluster: dict = field(default_factory=lambda: dict(eps=0.006, min_samples=8, min_pts=20, method="dbscan"))

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if k == "data":
                c.data = DataConfig(**v)
            elif k == "aug":
                c.aug = A.AugConfig(**{kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})
            elif hasattr(c, k):
                setattr(c, k, v)
        c.finalize()
        return c

    def finalize(self):
        if self.decoder == "koc":
            self.data.koc = True

    def apply_regime(self):
        if self.train_regime == "r2":
            for k, v in A.HARD_LIFT.items():
                setattr(self.aug, k, v)

    def apply_overrides(self, sets):
        sets = sorted(sets, key=lambda s: not s.startswith("train_regime="))     # regime first, explicit aug.* wins
        for s in sets:
            k, v = s.split("=", 1)
            tgt, parts = self, k.split(".")
            for p in parts[:-1]:
                tgt = tgt[p] if isinstance(tgt, dict) else getattr(tgt, p)
            val = json.loads(v) if v[:1] in "[{0123456789-tfn\"" else v
            if isinstance(tgt, dict):
                tgt[parts[-1]] = val
            else:
                cur = getattr(tgt, parts[-1])
                setattr(tgt, parts[-1], tuple(val) if isinstance(cur, tuple) else val)
            if k == "train_regime":
                self.apply_regime()
        self.finalize()

    def to_dict(self):
        return dataclasses.asdict(self)


def load_config(path, sets):
    cfg = Config()
    if path:
        import yaml
        cfg = Config.from_dict(yaml.safe_load(open(path)))
    cfg.apply_overrides(sets)
    return cfg


def build_model(cfg, pretrained=None):
    from .model import build
    return build(cfg.model, pretrained=cfg.pretrained if pretrained is None else pretrained, drop_path=cfg.drop_path,
                 patch_size=cfg.patch_size, mass_head=cfg.mass_head, koc=cfg.decoder == "koc",
                 koc_bias=cfg.koc_bias)


def make_optimizer(model, cfg):
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    ids = {id(p) for p in bb}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([dict(params=rest, lr=cfg.lr), dict(params=bb, lr=cfg.lr_backbone)],
                             weight_decay=cfg.weight_decay)


def set_phase(model, cfg, epoch):
    mode = "none" if epoch < cfg.freeze_epochs else cfg.unfreeze
    model.set_backbone_trainable(mode)
    return mode


def main(argv=None):
    from .eval import _to, evaluate
    from .loss import seg_loss
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--run")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. aug.crop_p=0.3 data.voxel_mm=2")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.run:
        cfg.run = a.run
    if a.epochs:
        cfg.epochs = a.epochs
    smoke_iters = 20
    if a.smoke:
        cfg.run, cfg.num_workers = "smoke", 0
    out = REPO / "results" / "seg3d" / cfg.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=list))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = "cuda"

    scans = list_scans(cfg.data.roots)
    tr, va = split_scans(scans, cfg.data)
    print(f"scans: {len(scans)} train {len(tr)} val {len(va)}")
    ds_tr = FoodSegDataset(tr, cfg.data, cfg.aug, train=True, seed=cfg.seed)
    ds_va = FoodSegDataset(va, cfg.data, A.eval_aug(cfg.eval_regime), train=False)
    model = build_model(cfg).to(dev)
    print("model:", getattr(model, "report", cfg.model))
    set_phase(model, cfg, 0)
    opt = make_optimizer(model, cfg)
    n_iter_ep = len(ds_tr) // (cfg.batch_size * cfg.grad_accum)
    total = smoke_iters if a.smoke else cfg.epochs * n_iter_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / cfg.warmup_iters) *
                                              0.5 * (1 + np.cos(np.pi * min(i / max(total, 1), 1))))
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    t0, it, hist = time.time(), 0, []
    torch.cuda.reset_peak_memory_stats()
    stop = False
    for ep in range(cfg.epochs if not a.smoke else 1):
        phase = set_phase(model, cfg, ep)
        if ep == cfg.freeze_epochs and not a.smoke:
            opt = make_optimizer(model, cfg)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i, s=it: 0.5 * (1 + np.cos(np.pi * min((s + i) / max(total, 1), 1))))
        ds_tr.set_epoch(ep)
        dl = torch.utils.data.DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                                         num_workers=cfg.num_workers, persistent_workers=False, drop_last=True)
        model.train()
        losses, te, n_oom = [], time.time(), 0
        for bi, batch in enumerate(dl):
            b = _to(batch, dev)
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                    o = model(b)
                loss, parts = seg_loss(o, b, cfg.cls_weights, cfg.off_w, cfg.dir_w, cfg.mass_w, model,
                                     koc=dict(w=cfg.koc_w, eps=cfg.koc_eps, change_w=cfg.koc_change_w, gamma=cfg.koc_gamma) if cfg.decoder == "koc" else None)
                scaler.scale(loss / cfg.grad_accum).backward()
            except torch.OutOfMemoryError:                    # shared GPU: skip the batch, keep going
                n_oom += 1
                print(f"OOM at batch {bi} (n={len(b['coord'])}); skipping [{n_oom}]")
                o = loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if n_oom >= 5 and model.mode != "none":
                    print("repeated OOM: falling back to frozen backbone")
                    cfg.unfreeze = "none"
                    model.set_backbone_trainable("none")
                    n_oom = 0
                continue
            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(float(loss.detach()))
            it += 1
            if a.smoke:
                print(f"it {it:2d} loss {float(loss.detach()):.4f} {parts} n={len(b['coord'])} "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB {(time.time() - te) / (bi + 1):.2f} s/it")
            peak = torch.cuda.max_memory_allocated() / 2**30
            if peak > cfg.max_mem_gb:
                print(f"ABORT: peak {peak:.2f} GiB > {cfg.max_mem_gb}")
                stop = True
            if (a.smoke and it >= smoke_iters) or (time.time() - t0) / 60 > cfg.max_minutes:
                stop = stop or not a.smoke or True
            if stop:
                break
        rec = dict(epoch=ep, phase=phase, iters=it, train_loss=float(np.mean(losses)), peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                   minutes=(time.time() - t0) / 60, train_s=time.time() - te, ep_iters=len(losses),
                   s_per_iter=(time.time() - te) / max(len(losses), 1))
        if a.smoke:
            rec["loss_first5"], rec["loss_last5"] = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
        if (ep + 1) % cfg.val_every == 0 or stop:
            torch.cuda.empty_cache()
            rec["val"] = evaluate(model, ds_va, cfg, baseline=(ep == 0))
            model.train()
        hist.append(rec)
        (out / "metrics.json").write_text(json.dumps(hist, indent=1))
        print(json.dumps({k: v for k, v in rec.items() if k != "val"}))
        if "val" in rec and rec["val"]["model"]:
            m = rec["val"]["model"]
            print(f"  val sem_miou {m['sem_miou']:.3f} inst_miou {m['inst_miou']:.3f} pq {m['pq']:.3f} count {m['exact_count']:.2f}")
            if ep == 0 and rec["val"]["lift_baseline"]:
                (out / "baseline_lift.json").write_text(json.dumps(rec["val"]["lift_baseline"], indent=1))
            best = max([h["val"]["model"]["inst_miou"] for h in hist if "val" in h and h["val"]["model"]
                        and h["val"]["model"]["inst_miou"] == h["val"]["model"]["inst_miou"]] or [-1])
            if m["inst_miou"] >= best:                   # selection on val inst mIoU only
                torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), epoch=ep), out / "best.pt")
        torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), epoch=ep), out / "last.pt")
        if stop:
            break
    print(f"done: {it} iters, {(time.time() - t0):.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
