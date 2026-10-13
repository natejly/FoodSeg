"""Sample, render and finalize scenes end to end (one Blender process per scene).

    python -m synth.batch --assets ../data/synth/assets.jsonl --root ../data/synth/v0 \
        --n 50 [--start 0] [--res-scale 0.5] [--spp 64] [--blender ~/blender/blender-4.2.1-linux-x64] \
        [--preset hard]

Layout under ROOT: specs/<sid>/ (spec + meshes), raw/<sid>/ (npz, deleted unless
--keep-raw), scans/<sid>/ (FoodScan format + gt). Scenes whose scans/<sid>/gt.json
exists are skipped, so a Slurm array can rerun the same command after a crash
with --start set per task.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .assets import load_index
from .finalize import finalize
from .spec import sample_scene
from .spec_hard import sample_scene_hard

RENDER = Path(__file__).with_name("render.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--res-scale", type=float, default=1.0)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--blender", default=str(Path.home() / "blender" / "blender-4.2.1-linux-x64"))
    ap.add_argument("--hdri-dir", default=None)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--keep-hr", action="store_true")
    ap.add_argument("--preset", choices=["v0", "hard"], default="v0", help="hard: see synth/spec_hard.py")
    args = ap.parse_args()

    root = Path(args.root)
    assets = load_index(args.assets)
    hdris = sorted(str(p.resolve()) for p in Path(args.hdri_dir).rglob("*.hdr")) if args.hdri_dir else None
    bproc = Path(sys.executable).with_name("blenderproc")
    for i in range(args.start, args.start + args.n):
        sid = f"s{i:06d}"
        sdir, rdir, odir = root / "specs" / sid, root / "raw" / sid, root / "scans" / sid
        if (odir / "gt.json").exists():
            continue
        t0 = time.time()
        sampler = sample_scene_hard if args.preset == "hard" else sample_scene
        sampler(np.random.default_rng([args.seed, i]), assets, sdir, sid, n_views=args.views,
                res_scale=args.res_scale, hdris=hdris)
        cmd = [str(bproc), "run", "--custom-blender-path", args.blender, str(RENDER),
               str(sdir / "spec.json"), str(rdir), "--spp", str(args.spp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not (rdir / "render_meta.json").exists():
            # usually the GPU ran out of memory (it may be shared); retry once on the CPU
            print(f"{sid}: GPU render failed, retrying on CPU", flush=True)
            r = subprocess.run(cmd + ["--cpu"], capture_output=True, text=True)
        if r.returncode != 0 or not (rdir / "render_meta.json").exists():
            (root / "failed").mkdir(parents=True, exist_ok=True)
            (root / "failed" / f"{sid}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
            print(f"{sid}: render FAILED (log in {root / 'failed'})")
            continue
        gt = finalize(sdir, rdir, odir, seed=args.seed, keep_hr=args.keep_hr)
        if not args.keep_raw:
            shutil.rmtree(rdir)
        off = [it["name"] for it in gt["items"] if it["off_container"]]
        print(f"{sid}: {gt['container']['kind']}, {len(gt['items'])} items, "
              f"{gt['total_volume_ml']:.0f} mL, {time.time() - t0:.0f}s"
              + (f"  WARNING off the dish: {off}" if off else ""), flush=True)


if __name__ == "__main__":
    main()
