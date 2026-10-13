"""Scan-quality sim: per-item volume error of run_meal on rendered food scans, with fuse's
depth filters on/off, under mild vs wide LiDAR edge smear and a full vs half orbit.

    PYTHONPATH=.:<dir containing synth/> python tests/sim_scan_quality.py SCANS_DIR WORK_DIR

SCANS_DIR holds synth.render output (gt.json, gt/depth_clean, gt/masks, gt/container.npz);
`synth` (synth.evaluate) is on branch worktree-synth-scaffold. Conditions:
  mild_full  the scans as rendered (synth.lidar: 1 px bleed, edges marked low confidence)
  wide_full  depth averaged over 7x7 within 3 px of every jump, marked high confidence:
             the worst case, closer to the 1-4 cm skirts on real scan 20261002_210342
  wide_half  wide_full, keeping only a 200 deg arc of the orbit (that scan covered 203 deg)
Prints container-support MAPE per filter configuration.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import foodscan.fuse as F  # noqa: E402
import foodscan.run_meal as RM  # noqa: E402
from foodscan.loader import Scan  # noqa: E402

W, H = 256, 192
CONDS = {"mild_full": (False, False), "wide_full": (True, False), "wide_half": (True, True)}
CFGS = {"none": dict(edge_rel=0, max_incidence=0, min_sharpness=0),
        "edge": dict(max_incidence=0, min_sharpness=0),
        "incidence75": dict(edge_rel=0, max_incidence=75, min_sharpness=0),
        "sharpness": dict(edge_rel=0, max_incidence=0),
        "defaults": {},
        "defaults+incidence75": dict(max_incidence=75)}


def variant(scan: Path, out: Path, smear: bool, half: bool) -> Path:
    if out.exists():
        return out
    out.mkdir(parents=True)
    for sub in ("rgb", "gt"):
        (out / sub).symlink_to(scan.resolve() / sub)
    for f in ("gt.json", "meta.json"):
        shutil.copy(scan / f, out / f)
    recs = [json.loads(line) for line in open(scan / "frames.jsonl") if line.strip()]
    if half:
        c = RM.look_at(Scan(scan))
        C = np.array([np.array(r["transform"])[:3, 3] for r in recs])
        az = np.degrees(np.arctan2(C[:, 2] - c[2], C[:, 0] - c[0])) % 360   # ARKit: +Y up
        recs = [r for r, a in zip(recs, az) if (a - az[0]) % 360 <= 200]
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for sub in ("depth", "smooth_depth", "conf"):
        (out / sub).mkdir()
    for r in recs:
        n = f"{r['index']:06d}"
        conf = np.fromfile(scan / "conf" / f"{n}.u8", np.uint8).reshape(H, W)
        if smear:
            clean = np.fromfile(scan / "gt" / "depth_clean" / f"{n}.f32", "<f4").reshape(H, W)
            jump = ndimage.maximum_filter(clean, 3) - ndimage.minimum_filter(clean, 3) > 0.01
            region = ndimage.binary_dilation(jump, iterations=3)
            conf = conf.copy()
            conf[region] = 2
        for sub in ("depth", "smooth_depth"):
            d = np.fromfile(scan / sub / f"{n}.f32", "<f4").reshape(H, W)
            if smear:
                d = np.where(region, ndimage.uniform_filter(d, 7), d).astype("<f4")
            d.tofile(out / sub / f"{n}.f32")
        conf.tofile(out / "conf" / f"{n}.u8")
    return out


def main() -> None:
    from synth.evaluate import evaluate
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans")
    ap.add_argument("work")
    args = ap.parse_args()
    scans = sorted(p for p in Path(args.scans).iterdir() if (p / "gt.json").exists())
    print(f"{'condition':10s} " + " ".join(f"{k:>21s}" for k in CFGS))
    for cond, (smear, half) in CONDS.items():
        err = {k: [] for k in CFGS}
        for scan in scans:
            v = variant(scan, Path(args.work) / cond / scan.name, smear, half)
            for name, kw in CFGS.items():
                RM.fuse = partial(F.fuse, **kw)
                err[name] += [abs(r["err_pct"]) for r in evaluate(v, "container") if not r["excluded"]]
        print(f"{cond:10s} " + " ".join(f"{np.mean(e):20.1f}%" for e in err.values()), flush=True)


if __name__ == "__main__":
    main()
