"""Round 4: train / test on REAL SAM 3 lifted votes (11 hard_v12 scans, data/synth/hard_v12_sam3_v2_image), leave-scans-out.

  python -m seg3d.ablate_v4 --setup | --baseline | --run [--budget-min N] | --final RUN | --report     (cwd .../FoodSeg/pipeline)
Data: results/seg3d/r4_data/real_v12 = symlinks to the real-vote scans + hard_v12's manifest (same item filter as round 3).
Folds: 3 (val 4/4/3 real scans).  train = real scans not in the fold (x REAL_REP) + split_v3 pseudo train scans minus pseudo copies of ALL 11
real scans.  Arm b trains once on pseudo only and is evaluated on all 11.  Every GPU process is wrapped in `flock -w 900 -E 99`; a lock
timeout (rc 99) stops the driver (the other session's render batch owns the GPU then).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ablate as A
from .data import item_status

REPO, OUT = A.REPO, A.OUT
REAL_SRC = REPO / "data/synth/hard_v12_sam3_v2_image/scans"
PSEUDO_ROOT = "data/synth/hard_v12"
R4 = OUT / "r4_data"
REAL_REL = "results/seg3d/r4_data/real_v12/scans"
LOCK = "/home/kyle/.claude/jobs/e47ee4f9/tmp/gpu.lock"
FLOCK = ["flock", "-w", "900", "-E", "99", LOCK]
REAL_REP = 3
FILTER = "manifest"
DEC = ("plain", "anchored", "koc", "koc2")
KOC = ["decoder=koc"]
MIX = "mix"
# (arm, extra sets, split kind)   priority order from the coordinator: c, d, e, b
ARMS = [("c_mix", KOC, "folds"), ("d_frag", KOC + ["aug.frag_p=0.3", "aug.merge_p=0.1"], "folds"),
        ("e_wide", KOC + ["data.koc_k=8", "data.koc_r_mm=50"], "folds"), ("b_pseudo", KOC, "pseudo")]


def setup():
    root = R4 / "real_v12"
    (root / "scans").mkdir(parents=True, exist_ok=True)
    sids = sorted(p.name for p in REAL_SRC.iterdir() if (p / "gt/seg_points.npz").exists())
    for sid in sids:
        ln = root / "scans" / sid
        if not ln.exists():
            ln.symlink_to(REAL_SRC / sid)
    (root / "manifest.jsonl").write_text((REPO / PSEUDO_ROOT / "manifest.jsonl").read_text())
    sp3 = json.loads((OUT / "split_v3.json").read_text())
    pseudo_copies = {f"{PSEUDO_ROOT}/scans/{s}" for s in sids}
    pseudo_train = [p for p in sp3["train"] if p not in pseudo_copies]
    valid = {}
    for sid in sids:
        z = np.load(REAL_SRC / sid / "gt/seg_points.npz")
        valid[sid] = len(item_status(root / "scans" / sid, z["label"], FILTER)[0])
    usable = [s for s in sids if valid[s] >= 1]
    perm = np.random.default_rng(0).permutation(usable).tolist()
    folds = [sorted(f) for f in np.array_split(perm, 3)]
    rel = lambda s: f"{REAL_REL}/{s}"                                           # noqa: E731
    info = dict(real_scans=sids, valid_items=valid, folds=folds, n_pseudo_train=len(pseudo_train), real_rep=REAL_REP)
    for k, val in enumerate(folds):
        tr_real = [s for s in usable if s not in val]
        (OUT / f"r4_split_f{k}.json").write_text(json.dumps(dict(train=pseudo_train + [rel(s) for s in tr_real] * REAL_REP, val=[rel(s) for s in val])))
    (OUT / "r4_split_pseudo.json").write_text(json.dumps(dict(train=pseudo_train, val=[rel(s) for s in usable])))
    (OUT / "r4_setup.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def sets(extra, seed, split):
    s = ["train_regime=r1", "eval_regime=r1", f"seed={seed}", "data.roots=" + json.dumps([PSEUDO_ROOT]),
         f"data.split_file=results/seg3d/{split}.json", f"data.item_filter={FILTER}"]
    sched = dict(A.SCHED, patch_size=512, val_every=8)
    return s + [f"{k}={v}" for k, v in sched.items()] + list(extra)


def run_job(arm, extra, split, tag, seed=0):
    run = f"r4_{arm}_{tag}_s{seed}"
    d = OUT / run
    if (d / "final.json").exists():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    t0 = time.time()
    if not (d / "last.pt").exists():
        cmd = FLOCK + [sys.executable, "-m", "seg3d.train", "--run", run, "--set"] + sets(extra, seed, split)
        with open(d / "train.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=REPO / "pipeline", env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=60 * 40).returncode
        (d / "run_info.json").write_text(json.dumps(dict(name=arm, tag=tag, seed=seed, rc=rc, wall_s=time.time() - t0, split=split, sets=extra)))
        print(f"{run}: rc={rc} wall {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            return rc
    rc = subprocess.run(FLOCK + [sys.executable, "-m", "seg3d.ablate_v4", "--final", run], cwd=REPO / "pipeline", env=env).returncode
    (d / "best.pt").unlink(missing_ok=True)
    return rc


def run(budget):
    t0 = time.time()
    for arm, extra, kind in ARMS:
        for tag, split in ([(f"f{k}", f"r4_split_f{k}") for k in range(3)] if kind == "folds" else [("all", "r4_split_pseudo")]):
            if (time.time() - t0) / 60 > budget - 8:
                print("budget exhausted, skipping", arm, tag, flush=True)
                continue
            rc = run_job(arm, extra, split, tag)
            if rc == 99:
                print("LOCK WAIT TIMEOUT: stop launching GPU runs", flush=True)
                return


# ----------------------------------------------------------------------------- per-scan raw statistics
def scan_stats(d, s, sel, psem, inst, extra):
    from . import metrics as M
    Fl = (lambda a: a[sel]) if not sel.all() else (lambda a: a)
    acc = M.Accumulator(volumes=False)
    gs, gi, gl, xs, lf = Fl(s["sem"]), Fl(s["inst"]), Fl(s["label"]), Fl(s["xyz"]), Fl(s["lift"])
    acc.add(Fl(psem), Fl(inst), xs, gs, gi, gl, s["items"], lift=lf)
    if extra is not None:
        acc.add_koc(Fl(extra[0]), lf, gl, Fl(extra[1]))
    pi, li = Fl(inst), M.lift_baseline_pred(lf)[1]
    g_ids, _, mp = M.match(gi, pi)
    _, _, ml = M.match(gi, li)
    items = []
    for g in g_ids:
        m = gi == g
        n = m.sum()
        ids, cnt = np.unique(li[m & (li > 0)], return_counts=True)
        frag = int((cnt >= 0.15 * n).sum()) >= 2
        pids, pc = np.unique(pi[m & (pi > 0)], return_counts=True)
        top_share = float(pc.max() / n) if len(pc) else 0.0
        liou, piou = float(ml[g][1]) if g in ml else 0.0, float(mp[g][1]) if g in mp else 0.0
        items.append(dict(g=int(g), lift_iou=liou, pred_iou=piou, frag_lift=bool(frag), whole_after=bool(piou > 0.5 and top_share >= 0.85)))
    return dict(conf=acc.conf.tolist(), rf={k: v.tolist() for k, v in acc.rf.items()}, tp=acc.tp, tp_iou=acc.tp_iou, fp=acc.fp, fn=acc.fn,
                count_ok=acc.count_ok, inst_iou=acc.inst_iou, koc=acc.koc.tolist(), items=items)


def final_eval(run_name):
    import torch
    from . import eval as E
    from .train import Config, build_model
    d = OUT / run_name
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    res = {}

    def cb(dec, s, sel, psem, inst, extra):
        res.setdefault(dec, {})[s["name"]] = scan_stats(dec, s, sel, psem, inst, extra)

    ds = E.make_sets(cfg, "val", "r1")
    E.evaluate(model, ds, cfg, baseline=True, decoders=DEC, volumes=False, scan_cb=cb)
    (d / "final.json").write_text(json.dumps(res))
    print("final done", run_name, {k: len(v) for k, v in res.items()})


def baseline():
    """Lift baseline on the 11 real scans (no GPU): reuse the evaluator with model=None."""
    from . import eval as E
    from .train import Config
    cfg = Config()
    cfg.data.roots, cfg.data.split_file, cfg.data.item_filter = [PSEUDO_ROOT], "results/seg3d/r4_split_pseudo.json", FILTER
    res = {}

    def cb(dec, s, sel, psem, inst, extra):
        res.setdefault(dec, {})[s["name"]] = scan_stats(dec, s, sel, psem, inst, extra)

    E.evaluate(None, E.make_sets(cfg, "val", "r1"), cfg, baseline=True, volumes=False, scan_cb=cb)
    (OUT / "r4_baseline.json").write_text(json.dumps(res))
    print({k: len(v) for k, v in res.items()})


# ----------------------------------------------------------------------------- report
def pooled(per):
    """per: {scan: stats} -> dict of pooled metrics + per-scan mean IoU dict."""
    conf = sum(np.array(v["conf"]) for v in per.values())
    iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
    all_iou = [x for v in per.values() for x in v["inst_iou"]]
    tp, tpi = sum(v["tp"] for v in per.values()), sum(v["tp_iou"] for v in per.values())
    fp, fn = sum(v["fp"] for v in per.values()), sum(v["fn"] for v in per.values())
    rf = {k: sum(np.array(v["rf"][k]) for v in per.values()) for k in ("food",)}
    ko = sum(np.array(v["koc"]) for v in per.values())
    its = [i for v in per.values() for i in v["items"]]
    out = dict(sem=float(iou[conf.sum(1) > 0].mean()), inst=float(np.mean(all_iou)), pq=tpi / max(tp + .5 * fp + .5 * fn, 1e-9),
               count=sum(v["count_ok"] for v in per.values()) / len(per), corrected=float(rf["food"][0] / max(rf["food"][1], 1)),
               broken=float(rf["food"][2] / max(rf["food"][3], 1)), keep=float(ko[1] / ko[0]) if ko[0] else None, n_items=len(all_iou),
               n_frag=sum(i["frag_lift"] for i in its), frag_merged=sum(i["frag_lift"] and i["whole_after"] for i in its),
               n_ok_lift=sum(i["lift_iou"] > 0.5 for i in its), broken_items=sum(i["lift_iou"] > 0.5 and i["pred_iou"] <= 0.5 for i in its),
               fixed_items=sum(i["lift_iou"] <= 0.5 and i["pred_iou"] > 0.5 for i in its))
    return out, {n: float(np.mean(v["inst_iou"])) for n, v in per.items() if v["inst_iou"]}


def report():
    rng = np.random.default_rng(0)
    base = json.loads((OUT / "r4_baseline.json").read_text())["lift"]
    arms = {}
    for arm, _, kind in ARMS:
        fs = sorted(OUT.glob(f"r4_{arm}_*_s0/final.json"))
        if fs:
            arms[arm] = [json.loads(f.read_text()) for f in fs]
    pb, sb = pooled(base)
    fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"                          # noqa: E731
    rows = ["| arm | decode | folds | scans | inst mIoU | PQ | count | d vs lift per scan (95% CI) | sem | corrected | broken | keep | frag. items merged | correct items broken | items fixed |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            f"| a lift (real votes) | - | - | {len(base)} | {pb['inst']:.3f} | {pb['pq']:.3f} | {pb['count']:.2f} | - | {pb['sem']:.3f} | 0 | 0 | 1 | 0 / {pb['n_frag']} | 0 / {pb['n_ok_lift']} | 0 |"]
    out = dict(lift=pb)
    seen = set()
    for arm, fl in arms.items():
        for dec in ("plain", "anchored", "koc", "koc2"):
            per = {}
            for f in fl:
                per.update(f.get(dec, {}))
            if not per:
                continue
            p, sm = pooled(per)
            names = sorted(set(sm) & set(sb))
            diff = np.array([sm[n] - sb[n] for n in names])
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            ci = (diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
            pl, _ = pooled({n: base[n] for n in per if n in base})
            out[f"{arm}/{dec}"] = dict(p, d=ci, scans=len(names), folds=len(fl), lift_same_scans=pl)
            if (arm, len(names)) not in seen:
                seen.add((arm, len(names)))
                rows.append(f"| a lift, same {len(names)} scans | - | - | {len(names)} | {pl['inst']:.3f} | {pl['pq']:.3f} | {pl['count']:.2f} | - | {pl['sem']:.3f} | 0 | 0 | 1 | 0 / {pl['n_frag']} | 0 / {pl['n_ok_lift']} | 0 |")
            rows.append(f"| {arm} | {dec} | {len(fl)} | {len(names)} | {p['inst']:.3f} | {p['pq']:.3f} | {p['count']:.2f} | {fmt(ci)} | {p['sem']:.3f} | "
                        f"{p['corrected']:.3f} | {p['broken']:.3f} | {'-' if p['keep'] is None else format(p['keep'], '.3f')} | {p['frag_merged']} / {p['n_frag']} | "
                        f"{p['broken_items']} / {p['n_ok_lift']} | {p['fixed_items']} |")
    (OUT / "round4.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true"); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--final"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--budget-min", type=float, default=75)
    a = ap.parse_args()
    if a.setup:
        setup()
    if a.baseline:
        baseline()
    if a.final:
        final_eval(a.final)
    if a.run:
        run(a.budget_min)
    if a.report:
        report()


if __name__ == "__main__":
    main()
