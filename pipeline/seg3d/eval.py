"""Evaluate a checkpoint (or the lift-label baseline) on the val split and dump per-scan predictions.

  python -m seg3d.eval --ckpt results/seg3d/<run>/best.pt [--split val|train|all] [--dump DIR]
  python -m seg3d.eval --baseline
(run from /home/kyle/Desktop/FoodSeg/pipeline)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import augment as A
from . import metrics as M
from .cluster import anchored_instances, cluster_instances
from .data import FoodSegDataset, collate, list_scans, split_scans


def _to(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


DECODERS = {"plain": None, "anchored": dict(), "anchored_split": dict(split_merged=True),
            "anchored_nob": dict(use_boundary=False)}


def decode(name, s, sem, off, cfg):
    """One decoding variant -> (sem, inst).  s = dataset sample (needs xyz, lift, lift_share, boundary)."""
    if name == "plain":
        return sem, cluster_instances(s["xyz"], sem, off, **cfg.cluster)
    return anchored_instances(s["xyz"], sem, off, s["lift"], s["lift_share"], s["boundary"], **DECODERS[name], **cfg.cluster)


def make_sets(cfg, split="val", regime="r1"):
    scans = list_scans(cfg.data.roots)
    tr, va = split_scans(scans, cfg.data)
    pick = {"val": va, "train": tr, "all": scans}[split]
    return FoodSegDataset(pick, cfg.data, A.eval_aug(regime), train=False)


@torch.no_grad()
def koc_decode(model, out, b, s, cfg, iters=1):
    """Keep-or-change decode -> (N,) normalised labels in lift-id space.  iters=2 rebuilds the candidates from the
    updated labels and re-runs only the head (encoder features are reused)."""
    from .koc import build_candidates, decode_labels
    lab, _ = decode_labels(out["koc"], b["cand_id"].to(out["koc"].device))
    for _ in range(iters - 1):
        kc = build_candidates(s["xyz"], lab.cpu().numpy(), s["label"], cfg.data.koc_k, cfg.data.koc_r_mm * 1e-3, cfg.data.koc_m)
        b2 = _to(collate([dict(s, **kc)]), out["koc"].device)
        lg = model.koc(out["feat"], out["sem"], b2["feat"], b2)
        lab, _ = decode_labels(lg, b2["cand_id"])
    return lab.cpu().numpy()


@torch.no_grad()
def evaluate(model, ds, cfg, dump: Path | None = None, baseline=True, amp=False, decoders=None, volumes=True, scan_cb=None):   # spconv fp16 inference tuner fails on sm_75 in eval mode -> fp32 eval
    """Returns {"model": plain-decoding result, "model_<variant>": ..., "lift_baseline": ..., "lift_baseline_anchored": ...}."""
    dev = next(model.parameters()).device if model is not None else "cpu"
    primary = "koc" if getattr(cfg, "decoder", "plain") == "koc" else "plain"
    decoders = tuple(decoders) if decoders else (primary,)
    if model is not None and primary not in decoders:
        decoders = (primary,) + decoders
    if getattr(model, "koc", None) is None:
        decoders = tuple(d for d in decoders if not d.startswith("koc")) or ("plain",)
        primary = "plain" if primary == "koc" else primary
    accs = {d: M.Accumulator(volumes=volumes) for d in decoders} if model is not None else {}
    base = M.Accumulator(volumes=volumes) if baseline else None
    base_anch = M.Accumulator(volumes=volumes) if baseline else None
    S = cfg.data.coord_scale
    if model is not None:
        model.eval()
    for i in range(len(ds)):
        s = ds[i]
        b = collate([s])
        gt_sem, gt_inst, gt_label = s["sem"], s["inst"], s["label"]
        xyz = s["xyz"]
        sel = ~s["ignore"]                       # points of ignored (tiny/invisible) items leave the metrics entirely
        Fl = (lambda a: a[sel]) if not sel.all() else (lambda a: a)
        gs, gi, gl, xs, lf = Fl(gt_sem), Fl(gt_inst), Fl(gt_label), Fl(xyz), Fl(s["lift"])
        if base is not None:
            ps, pi = M.lift_baseline_pred(s["lift"])
            base.add(Fl(ps), Fl(pi), xs, gs, gi, gl, s["items"], lift=lf)
            # trivially lift-anchored: predicted "model" == lift (class and clusters), so nothing is reassigned
            asem, ainst = anchored_instances(xyz, ps, np.zeros_like(xyz), s["lift"], s["lift_share"], s["boundary"], clusters=pi)
            base_anch.add(Fl(asem), Fl(ainst), xs, gs, gi, gl, s["items"], lift=lf)
            if scan_cb is not None:
                scan_cb("lift", s, sel, ps, pi, None)
        if model is None:
            continue
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev != "cpu"):
            out = model(_to(b, dev))
        sem = out["sem"].argmax(1).cpu().numpy()
        off = out["offset"].float().cpu().numpy() / S
        for d in decoders:
            if d.startswith("koc"):
                lab = koc_decode(model, out, b, s, cfg, iters=2 if d == "koc2" else getattr(cfg, "koc_iters", 1))
                psem, inst = M.lift_baseline_pred(lab)
                accs[d].add(Fl(psem), Fl(inst), xs, gs, gi, gl, s["items"], lift=lf)
                accs[d].add_koc(Fl(lab), lf, gl, Fl((b["cand_tgt"] == b["cand_id"].shape[1] - 1).numpy()))
                if scan_cb is not None:
                    scan_cb(d, s, sel, psem, inst, (lab, (b["cand_tgt"] == b["cand_id"].shape[1] - 1).numpy()))
                continue
            psem, inst = decode(d, s, sem, off, cfg)
            accs[d].add(Fl(psem), Fl(inst), xs, gs, gi, gl, s["items"], lift=lf)
            if scan_cb is not None:
                scan_cb(d, s, sel, psem, inst, None)
            if dump is not None and d == "plain":
                dump.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dump / (s["name"].replace("/", "__") + ".npz"), xyz=xyz, label=gt_label, gt_sem=gt_sem,
                                    pred_sem=psem, pred_inst=inst, pred_offset=off.astype(np.float32), lift=s["lift"])
    res = {"model": accs[primary].result() if primary in accs else None,
           "lift_baseline": base.result() if base else None, "lift_baseline_anchored": base_anch.result() if base else None}
    for d in decoders:
        if d != primary:
            res["model_" + d] = accs[d].result()
    return res


def main():
    from .train import Config, load_config, build_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--regime", default="r1")
    ap.add_argument("--decoders", nargs="*", default=None, help="plain anchored ... koc koc2")
    ap.add_argument("--dump")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg, model = None, None
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ck["config"])
        cfg.apply_overrides(a.set)
        model = build_model(cfg, pretrained=False).cuda()
        model.load_state_dict(ck["model"])
    else:
        cfg = load_config(None, a.set)
    res = evaluate(model, make_sets(cfg, a.split, a.regime), cfg, Path(a.dump) if a.dump else None, decoders=tuple(a.decoders))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
