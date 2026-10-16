"""PTv3 as a *semantic* cleaner on real SAM 3 lifted votes (4 classes: table/background, container, clutter, food).

  python -m seg3d.semantic --setup | --run [--budget-min N] | --final RUN | --report      (cwd .../FoodSeg/pipeline)
Data: 40 real-vote scans (hard_v12/hard_v13 *_sam3_v2_image) symlinked under results/seg3d/semantic/data/real_v1{2,3}
(+ the batch's synthetic manifest for the item filter).  3 folds by scan, stratified by batch.  train = real scans not in the
fold (x REAL_REP) + split_v3 pseudo-vote train scans minus pseudo copies of the fold's val scans.  Arms: full (all inputs),
nolift (lift channel zeroed).  Baseline = the lift's own semantic reading.  Instance heads are trained with weight 0 (off_w=dir_w=0)
and their outputs ignored.  Every GPU process is wrapped in the shared flock.  Checkpoints are deleted after evaluation.
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

from .data import item_status

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "seg3d" / "semantic"
DATA = OUT / "data"
SRC = {"v12": REPO / "data/synth/hard_v12_sam3_v2_image/scans", "v13": REPO / "data/synth/hard_v13_sam3_v2_image/scans"}
MAN = {"v12": REPO / "data/synth/hard_v12/manifest.jsonl", "v13": REPO / "data/synth/hard_v13/manifest.jsonl"}
PSEUDO_ROOT = "data/synth/hard_v12"
LOCK = "/home/kyle/.claude/jobs/e47ee4f9/tmp/gpu.lock"
FLOCK = ["flock", "-w", "900", "-E", "99", LOCK]
REAL_REP = 3
ARMS = {"full": [], "nolift": ['data.zero_groups=["lift"]']}
SCHED = dict(epochs=4, freeze_epochs=4, **{"data.epoch_repeat": 4}, patch_size=512, max_minutes=6, grad_accum=2, num_workers=2, skip_val=True, off_w=0, dir_w=0)
CLS = ["table", "container", "clutter", "food"]


def rel(b, s):
    return f"results/seg3d/semantic/data/real_{b}/scans/{s}"


def setup():
    info = dict(valid_items={}, folds=[[], [], []])
    rng = np.random.default_rng(0)
    pseudo_train_all = json.loads((REPO / "results/seg3d/split_v3.json").read_text())["train"]
    usable = {}
    for b in SRC:
        root = DATA / f"real_{b}"
        (root / "scans").mkdir(parents=True, exist_ok=True)
        (root / "manifest.jsonl").write_text(MAN[b].read_text())
        for sid in sorted(p.name for p in SRC[b].iterdir() if (p / "gt/seg_points.npz").exists()):
            ln = root / "scans" / sid
            if not ln.exists():
                ln.symlink_to(SRC[b] / sid)
            z = np.load(SRC[b] / sid / "gt/seg_points.npz")
            n = len(item_status(root / "scans" / sid, z["label"], "manifest")[0])
            info["valid_items"][f"{b}/{sid}"] = n
            if n >= 1:
                usable.setdefault(b, []).append(sid)
    for b, sids in usable.items():
        perm = rng.permutation(sids).tolist()
        for k, f in enumerate(np.array_split(perm, 3)):
            info["folds"][k] += [(b, s) for s in sorted(f)]
    allr = [(b, s) for b, v in usable.items() for s in v]
    for k, val in enumerate(info["folds"]):
        vs = set(val)
        tr_real = [rel(b, s) for b, s in allr if (b, s) not in vs]
        excl = {f"{PSEUDO_ROOT}/scans/{s}" for b, s in val if b == "v12"}
        ptr = [p for p in pseudo_train_all if p not in excl]
        (OUT / f"split_f{k}.json").write_text(json.dumps(dict(train=ptr + tr_real * REAL_REP, val=[rel(b, s) for b, s in val])))
        info[f"n_pseudo_train_f{k}"] = len(ptr)
    info["folds"] = [[f"{b}/{s}" for b, s in f] for f in info["folds"]]
    (OUT / "setup.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: (v if k != "valid_items" else len(v)) for k, v in info.items()}, indent=1))


def sets(extra, seed, split):
    s = ["train_regime=r1", "eval_regime=r1", f"seed={seed}", "data.roots=" + json.dumps([PSEUDO_ROOT]),
         f"data.split_file=results/seg3d/semantic/{split}.json", "data.item_filter=manifest"]
    return s + [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in SCHED.items()] + list(extra)


def run_job(arm, fold, seed=0):
    run = f"semantic/{arm}_f{fold}"
    d = OUT / f"{arm}_f{fold}"
    if (d / "final.json").exists():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    t0 = time.time()
    if not (d / "last.pt").exists():
        cmd = FLOCK + [sys.executable, "-m", "seg3d.train", "--run", run, "--set"] + sets(ARMS[arm], seed, f"split_f{fold}")
        with open(d / "train.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=REPO / "pipeline", env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=60 * 20).returncode
        (d / "run_info.json").write_text(json.dumps(dict(arm=arm, fold=fold, rc=rc, wall_s=time.time() - t0)))
        print(f"{run}: rc={rc} wall {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            return rc
    rc = subprocess.run(FLOCK + [sys.executable, "-m", "seg3d.semantic", "--final", f"{arm}_f{fold}"], cwd=REPO / "pipeline", env=env).returncode
    if rc == 0:
        for f in ("last.pt", "best.pt"):
            (d / f).unlink(missing_ok=True)                       # checkpoints deleted after evaluation
    return rc


def run(budget):
    t0 = time.time()
    for fold in range(3):
        for arm in ARMS:
            if (time.time() - t0) / 60 > budget - 7:
                print("budget exhausted, skipping", arm, fold, flush=True)
                continue
            if run_job(arm, fold) == 99:
                print("LOCK WAIT TIMEOUT", flush=True)
                return


def sem_of(label):
    s = np.zeros(len(label), np.int64)
    s[label == 254], s[label == 253], s[(label > 0) & (label < 252)] = 1, 2, 3
    return s


def final_eval(name):
    import torch
    from . import augment as A
    from .data import FoodSegDataset, collate, split_scans
    from .eval import _to
    from .train import Config, build_model
    d = OUT / name
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    _, va = split_scans([], cfg.data)
    ds = FoodSegDataset(va, cfg.data, A.eval_aug("r1"), train=False)
    pdir = d / "preds"
    pdir.mkdir(exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    res = {}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            b = _to(collate([s]), "cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(b)
            pred = out["sem"].argmax(1)
            torch.cuda.synchronize()
            dt = time.time() - t0
            pred = pred.cpu().numpy()
            keep = ~s["ignore"]
            gt = s["sem"][keep]
            lift = sem_of(s["lift"])[keep]                       # lift_label -> class
            pr = pred[keep]
            conf = lambda p: np.bincount(gt * 4 + p, minlength=16).reshape(4, 4)        # noqa: E731 rows = GT
            res[s["name"]] = dict(net=conf(pr).tolist(), lift=conf(lift).tolist(), time_s=dt, n_vox=int(len(pred)),
                                  # food recovered / lost / leaked counts
                                  food_gt=int((gt == 3).sum()), food_lift_missed=int(((gt == 3) & (lift != 3)).sum()),
                                  food_net_rec=int(((gt == 3) & (lift != 3) & (pr == 3)).sum()),
                                  food_lift_hit=int(((gt == 3) & (lift == 3)).sum()), food_net_lost=int(((gt == 3) & (lift == 3) & (pr != 3)).sum()),
                                  nonfood_gt=int((gt != 3).sum()), nonfood_lift_leak=int(((gt != 3) & (lift == 3)).sum()),
                                  nonfood_net_leak=int(((gt != 3) & (pr == 3)).sum()))
            np.savez_compressed(pdir / (s["name"].replace("/", "__") + ".npz"), pred=pred.astype(np.int8), gt=s["sem"].astype(np.int8),
                                lift=sem_of(s["lift"]).astype(np.int8), ignore=s["ignore"], xyz=s["xyz"].astype(np.float16))
    res["_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (d / "final.json").write_text(json.dumps(res))
    print("final done", name, len(res) - 1, "scans, peak GiB", res["_peak_gib"])


# ----------------------------------------------------------------------------- report
def iou_of(conf):
    conf = np.asarray(conf, float)
    tp = np.diag(conf)
    return tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1), tp / np.maximum(conf.sum(1), 1), tp / np.maximum(conf.sum(0), 1)   # iou, recall, precision


def boot(x, rng, n=4000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def report():
    rng = np.random.default_rng(0)
    info = json.loads((OUT / "setup.json").read_text())
    arms = {}
    for arm in ARMS:
        r = {}
        peak = []
        for f in sorted(OUT.glob(f"{arm}_f*/final.json")):
            j = json.loads(f.read_text())
            peak.append(j.pop("_peak_gib"))
            r.update(j)
        if r:
            arms[arm] = (r, peak)
    names = sorted(set.intersection(*[set(r) for r, _ in arms.values()]))
    batch = lambda n: n.split("/")[0].replace("real_", "")                                # noqa: E731
    fmt = lambda t, p=3: f"{t[0]:.{p}f} [{t[1]:.{p}f}, {t[2]:.{p}f}]"                       # noqa: E731
    out = ["# PTv3 as a semantic cleaner on real SAM 3 votes", "",
           f"{len(names)} real-vote scans, 3 folds stratified by batch (v12/v13), voxel-level, item-filtered. Arms: lift = the lift's own reading; "
           "full = frozen Sonata PTv3, all inputs; nolift = same without the lift channel. Per-scan means with bootstrap 95% CI over scans; "
           "class tables are pooled over all val points (union over folds).", ""]

    def tables(sub, title):
        L = [f"## {title} ({len(sub)} scans)", "", "| arm | " + " | ".join(f"IoU {c}" for c in CLS) + " | mIoU (pooled) | mIoU per-scan mean [CI] | container R / P | clutter R / P |", "|---|" + "---|" * 8]
        for arm, key in (("lift", "lift"), ("full", "net"), ("nolift", "net")):
            if arm != "lift" and arm not in arms:
                continue
            src = arms.get("full" if arm == "lift" else arm)[0]
            conf = sum(np.array(src[n][key]) for n in sub)
            iou, rec, pre = iou_of(conf)
            per = [iou_of(src[n][key])[0][np.array(src[n][key]).sum(1) > 0].mean() for n in sub]
            L.append(f"| {arm} | " + " | ".join(f"{x:.3f}" for x in iou) + f" | {iou.mean():.3f} | {fmt(boot(per, rng))} | "
                     f"{rec[1]:.3f} / {pre[1]:.3f} | {rec[2]:.3f} / {pre[2]:.3f} |")
        return L + [""]

    out += tables(names, "All scans")
    for b in ("v12", "v13"):
        out += tables([n for n in names if batch(n) == b], f"Batch {b}")

    def pf(src, n, num, den):
        return src[n][num] / src[n][den] if src[n][den] else float("nan")

    out += ["## Food recovered / leaked / lost (per-scan means, CI over scans)", "",
            "recovered = GT-food points the lift did NOT call food that the net calls food (lift recovers 0 by definition). "
            "lost = GT-food points the lift called food that the net turned non-food. leaked = non-food points called food. "
            "d leak = net leak minus lift leak, paired per scan.", "",
            "| arm | batch | scans | food recovered | food lost | leaked (net) | leaked (lift) | d leak vs lift (paired) | d food recall vs lift (paired) |", "|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm in arms:
        for bsel in ("all", "v12", "v13"):
            sub = [n for n in names if bsel == "all" or batch(n) == bsel]
            src = arms[arm][0]
            rec = [pf(src, n, "food_net_rec", "food_lift_missed") for n in sub]
            lost = [pf(src, n, "food_net_lost", "food_lift_hit") for n in sub]
            ln = [pf(src, n, "nonfood_net_leak", "nonfood_gt") for n in sub]
            ll = [pf(src, n, "nonfood_lift_leak", "nonfood_gt") for n in sub]
            dl = [a - b for a, b in zip(ln, ll)]
            dr = [(src[n]["food_net_rec"] + src[n]["food_lift_hit"] - src[n]["food_net_lost"]) / max(src[n]["food_gt"], 1)
                  - src[n]["food_lift_hit"] / max(src[n]["food_gt"], 1) for n in sub]
            pooled_rec = sum(src[n]["food_net_rec"] for n in sub) / max(sum(src[n]["food_lift_missed"] for n in sub), 1)
            out.append(f"| {arm} | {bsel} | {sum(~np.isnan(rec))} | {fmt(boot(rec, rng))} (pooled {pooled_rec:.3f}) | {fmt(boot(lost, rng))} | {fmt(boot(ln, rng), 4)} | "
                       f"{fmt(boot(ll, rng), 4)} | {fmt(boot(dl, rng), 4)} | {fmt(boot(dr, rng))} |")
            summ[f"{arm}/{bsel}"] = dict(rec=boot(rec, rng), lost=boot(lost, rng), leak=boot(ln, rng), lift_leak=boot(ll, rng), d_leak=boot(dl, rng), d_recall=boot(dr, rng))
    pool_miss = sum(arms["full"][0][n]["food_lift_missed"] for n in names)
    out += ["", f"Lift food-recall (pooled) = {1 - pool_miss / max(sum(arms['full'][0][n]['food_gt'] for n in names), 1):.3f}; "
            f"{pool_miss} GT-food voxels are missed by the lift, {sum(arms['full'][0][n]['food_gt'] for n in names)} GT-food voxels in total.", "",
            "## Cost", "", "| arm | inference s/scan (mean, median) | voxels/scan | peak GPU GiB (eval, per fold) |", "|---|---|---|---|"]
    for arm, (r, peak) in arms.items():
        t = [r[n]["time_s"] for n in names]
        out.append(f"| {arm} | {np.mean(t):.2f}, {np.median(t):.2f} | {np.mean([r[n]['n_vox'] for n in names]):.0f} | {', '.join(f'{p:.2f}' for p in peak)} |")
    out.append("")
    (OUT / "report_body.md").write_text("\n".join(out))
    (OUT / "summary_stats.json").write_text(json.dumps(summ, indent=1, default=float))
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--final"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--budget-min", type=float, default=42)
    a = ap.parse_args()
    if a.setup:
        setup()
    if a.final:
        final_eval(a.final)
    if a.run:
        run(a.budget_min)
    if a.report:
        report()


if __name__ == "__main__":
    main()
