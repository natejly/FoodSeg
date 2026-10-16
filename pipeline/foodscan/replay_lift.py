"""CPU replay of stored raw SAM 3 per-frame maps through segment.py's linker + run_meal's lift, scored vs GT.

    python -m foodscan.replay_lift [--linker covote|repair] [--weighting cos+qual] [--procs 11] [--out DIR] [--only s000000 ...]

Per scan (hard_v12): segment_scan(from_raw=hard_v12/scans/<sid>/masks_sam3_v2i) -> masks/ + instances.json
in OUT/<sid>, then labels.lift_for_meal -> OUT/<sid>/lift.npz, on the scan's gt/seg_points.npz cloud (the
cloud results/lift_sweep used), scored with seg3d.metrics on the 2 mm canonical scene exactly like lift_sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline"))
SCANS = REPO / "data/synth/hard_v12_sam3_v2_image/scans"
RAWROOT = REPO / "data/synth/hard_v12/scans"
SIDS = ["s000000", "s000001", "s000002", "s000007", "s000009", "s000011", "s000012", "s000013", "s000014",
        "s000016", "s000017"]


def replay_scan(sid: str, out: Path, linker: str = "covote", weighting: str | None = None,
                cluster_thresh: float = 0.8, raw_name: str = "masks_sam3_v2i", post: dict | None = None,
                post_judge=None, return_lift: bool = False):
    from foodscan import labels as L
    from foodscan.loader import Scan
    from foodscan.segment import PostOpts, load_masks, make_linker, segment_scan
    sd, od = SCANS / sid, Path(out) / sid
    z = np.load(sd / "gt/seg_points.npz")
    xyz, nrm = z["xyz"].astype(np.float64), z["normal"].astype(np.float64)
    meta = segment_scan(sd, [], from_raw=RAWROOT / sid / raw_name, out_dir=od / "masks",
                        linker=make_linker(sd, xyz, nrm, method=linker, cluster_thresh=cluster_thresh),
                        postproc=PostOpts(**post) if isinstance(post, dict) else post, post_judge=post_judge)
    masks, meta = load_masks(od, od / "masks")
    w = weighting or (L.DEFAULT_LINKED_WEIGHTING if linker == "covote" else "cos")
    lift = L.lift_for_meal(Scan(sd), masks, xyz, nrm, meta, weighting=w, save_to=od / "lift.npz")
    res = evaluate_lift(sid, lift.lift_label)
    return (res, lift.lift_label) if return_lift else res


def evaluate_lift(sid: str, lift_full) -> dict:
    from eval.lift_sweep import evaluate
    from seg3d.data import load_scan, semantic_of, voxelize
    d = load_scan(SCANS / sid, "manifest")
    vi, _ = voxelize(d["xyz"], 2e-3)
    lab = d["label"][vi]
    sem = semantic_of(lab)
    sc = dict(xyz=d["xyz"][vi].astype(np.float64), label=lab, sem=sem, inst=np.where(sem == 3, lab, 0).astype(np.int64),
              ignore=np.isin(lab, d["ignore_ids"]) if d["ignore_ids"] else np.zeros(len(vi), bool))
    return evaluate(SimpleNamespace(sc=sc, vi=vi), lift_full)


def _run(a):
    sid, out, kw = a
    return sid, replay_scan(sid, out, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linker", default="covote", choices=["covote", "repair"])
    ap.add_argument("--weighting", default=None)
    ap.add_argument("--cluster-thresh", type=float, default=0.8)
    ap.add_argument("--post", default=None, help="JSON dict of segment.PostOpts fields (2D mask post-processing)")
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--out", default=str(REPO / "results/lift_replay"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    from eval.lift_sweep import pooled
    sids = a.only or SIDS
    kw = dict(linker=a.linker, weighting=a.weighting, cluster_thresh=a.cluster_thresh,
              post=json.loads(a.post) if a.post else None)
    with Pool(min(a.procs, len(sids))) as p:
        res = dict(p.map(_run, [(s, a.out, kw) for s in sids]))
    pool = pooled([res[s] for s in sids])
    for s in sids:
        print(s, f"inst mIoU {np.mean(res[s]['iou']):.3f}", f"ids {res[s]['n_ids']}", f"frag {res[s]['frag']}")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in pool.items()}, default=str))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"summary_{a.linker}.json").write_text(json.dumps(
        {"args": kw, "pooled": pool, "per_scan": res}, default=str))


if __name__ == "__main__":
    main()
