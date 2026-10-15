"""Sample, render and finalize scenes end to end (one Blender process per scene).

    python -m synth.batch --assets ../data/synth/assets.jsonl --root ../data/synth/v0 \
        --n 50 [--start 0] [--res-scale 0.5] [--spp 64] [--blender ~/blender/blender-4.2.1-linux-x64] \
        [--preset hard]

Layout under ROOT: specs/<sid>/ (spec + meshes), raw/<sid>/ (npz, deleted unless
--keep-raw), scans/<sid>/ (FoodScan format + gt). Scenes whose scans/<sid>/gt.json
exists are skipped (scans are written to <sid>.tmp and renamed, gt.json last, so a crash never leaves a
half scan under the final name), so a Slurm array can rerun the same command after a crash
with --start set per task (or --shard i/n). A scene that raises or times out is logged to
ROOT/failed/<sid>.log (traceback + subprocess output) and the batch continues; ROOT/batch_report.json
lists done / skipped / failed scenes at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .assets import load_index
from .collision import ensure as ensure_collision
from .fetch_assets import HDRIS, SURFACES
from .finalize import finalize
from .spec import sample_scene
from .spec_hard import sample_scene_hard

RENDER = Path(__file__).with_name("render.py")
SURFACE_IDS = {i for ids in SURFACES.values() for i in ids}     # the curated list (some fetched sets were dropped)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--res-scale", type=float, default=1.0)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--blender", default=str(Path.home() / "blender" / "blender-4.2.1-linux-x64"))
    ap.add_argument("--hdri-dir", default=None, help="default (hard): <online>/polyhaven_hdri")
    ap.add_argument("--n5k-dir", default=None, help="Nutrition5k (meal compositions); default <online>/../Nutrition5k")
    ap.add_argument("--online-dir", default=None,
                    help="synth.fetch_assets output (HDRIs, tabletop textures); default <assets>/../../datasets/online")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--keep-hr", action="store_true")
    ap.add_argument("--preset", choices=["v0", "hard"], default="v0", help="hard: see synth/spec_hard.py")
    ap.add_argument("--collision-cache", default=None, help="CoACD parts cache (default: <assets dir>/collision)")
    ap.add_argument("--hull-only", action="store_true", help="single convex hull per scan (old physics)")
    ap.add_argument("--physics", choices=["mpm", "bullet"], default="mpm",
                    help="hard preset: mpm settles soft food and dropped scans together, rigid-only scenes too "
                         "(synth.mpm); bullet: Blender rigid bodies only (fallback, time-limited)")
    ap.add_argument("--mpm-time", type=float, default=1.0)
    ap.add_argument("--shard", default=None, metavar="I/N", help="process only scenes with index %% N == I")
    ap.add_argument("--render-timeout", type=float, default=1800.0, help="seconds per Blender subprocess")
    args = ap.parse_args()

    root = Path(args.root)
    assets = load_index(args.assets)
    cache = Path(args.collision_cache) if args.collision_cache else Path(args.assets).parent / "collision"
    online = Path(args.online_dir) if args.online_dir else Path(args.assets).resolve().parent.parent / "datasets" / "online"
    hdri_dir = Path(args.hdri_dir) if args.hdri_dir else (online / "polyhaven_hdri" if args.preset == "hard" else None)
    hdris = sorted(str(p.resolve()) for p in hdri_dir.rglob("*.hdr")
                   if args.hdri_dir or p.parent.name in HDRIS) if hdri_dir and hdri_dir.exists() else None   # curated rooms
    surfaces = []                                  # PBR tabletops (hard preset)
    if (online / "sources.jsonl").exists():
        for line in (online / "sources.jsonl").read_text().splitlines():
            r = json.loads(line)
            d = online / r["source"] / r["id"]
            if r.get("role") == "surface" and r["id"] in SURFACE_IDS and (d / "diffuse.jpg").exists():
                dims = r.get("dimensions_mm") or [1000, 1000]
                surfaces.append({"dir": str(d.resolve()), "kind": r["kind"], "tile_m": max(dims[:2]) / 1000.0})
    n5k = None                                     # real dish compositions for synth.meals
    n5k_dir = Path(args.n5k_dir) if args.n5k_dir else online.parent / "Nutrition5k"
    if args.preset == "hard" and (n5k_dir / "metadata").exists():
        from .composition import load_dishes, norm, resolve
        cats = {norm(a["category"]) for a in assets if a.get("role", "food") == "food"
                and a.get("volume_ok", a["watertight"]) and not a.get("container_shaped")}
        n5k = {"dishes": load_dishes(n5k_dir), "mapping": resolve(cats)}
    print(f"{len(hdris or [])} HDRIs, {len(surfaces)} tabletop textures, "
          f"{len(n5k['dishes']) if n5k else 0} Nutrition5k dishes", flush=True)
    bproc = Path(sys.executable).with_name("blenderproc")
    shard_i, shard_n = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    report = {"args": vars(args), "done": [], "skipped": [], "failed": {}}
    for i in range(args.start, args.start + args.n):
        sid = f"s{i:06d}"
        if i % shard_n != shard_i:
            continue
        sdir, rdir, odir = root / "specs" / sid, root / "raw" / sid, root / "scans" / sid
        if (odir / "gt.json").exists():
            report["skipped"].append(sid)
            continue
        out_log: list[str] = []         # subprocess stdout/stderr for the failure log
        try:
            gt = run_scene(i, sid, args, assets, cache, hdris, surfaces, n5k, bproc, sdir, rdir, odir, out_log)
            report["done"].append(sid)
        except Exception as e:  # noqa: BLE001  one bad scene must not kill the batch
            (root / "failed").mkdir(parents=True, exist_ok=True)
            (root / "failed" / f"{sid}.log").write_text(
                traceback.format_exc() + "\n--- subprocess output ---\n" + "\n".join(out_log))
            report["failed"][sid] = f"{type(e).__name__}: {e}"
            shutil.rmtree(odir.with_name(odir.name + ".tmp"), ignore_errors=True)
            print(f"{sid}: FAILED ({type(e).__name__}: {e}); log in {root / 'failed' / (sid + '.log')}", flush=True)
            continue
        finally:
            if not args.keep_raw and rdir.exists() and sid not in report["failed"]:
                shutil.rmtree(rdir, ignore_errors=True)
        off = [it["name"] for it in gt["items"] if it["off_container"]]
        print(f"{sid}: {gt['container']['kind']}, {len(gt['items'])} items, "
              f"{gt['total_volume_ml']:.0f} mL, {gt['_wall_s']:.0f}s"
              + (f"  WARNING off the dish: {off}" if off else ""), flush=True)
    name = "batch_report.json" if shard_n == 1 else f"batch_report_{shard_i}of{shard_n}.json"
    (root / name).write_text(json.dumps(report, indent=1, default=str))
    print(f"done {len(report['done'])}, skipped {len(report['skipped'])}, failed {len(report['failed'])}")


def _run_render_cmd(cmd: list[str], timeout: float, grace: float = 20.0) -> tuple[int, str, bool]:
    """Run cmd in its own process group; on timeout SIGTERM then SIGKILL the whole group so the
    Blender child of the blenderproc wrapper cannot be orphaned. Returns (returncode, output, timed_out)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    pgid = proc.pid                       # session leader: pgid == pid
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
            try:
                out, err = proc.communicate(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            out = err = ""
        try:
            os.killpg(pgid, signal.SIGKILL)   # reap any stragglers left after the leader exited
        except ProcessLookupError:
            pass
    return proc.returncode, (out or "")[-20000:] + "\n" + (err or "")[-20000:], timed_out


def _run(cmd: list[str], timeout: float, log: list[str]) -> subprocess.CompletedProcess:
    rc, output, timed_out = _run_render_cmd(cmd, timeout)
    if timed_out:
        log.append(f"TIMEOUT after {timeout:.0f}s\n{output}")
        raise subprocess.TimeoutExpired(cmd, timeout)
    log.append(output)
    return subprocess.CompletedProcess(cmd, rc)


def run_scene(i, sid, args, assets, cache, hdris, surfaces, n5k, bproc, sdir, rdir, odir, out_log) -> dict:
    t0 = time.time()
    sampler = sample_scene_hard if args.preset == "hard" else sample_scene
    extra = {"surfaces": surfaces, "n5k": n5k} if args.preset == "hard" else {}
    spec = sampler(np.random.default_rng([args.seed, i]), assets, sdir, sid, n_views=args.views,
                   res_scale=args.res_scale, hdris=hdris, **extra)
    if args.preset == "hard" and not args.hull_only:   # true-shape compound colliders for dropped scans
        by_id = {a["id"]: a for a in assets}
        for it in spec["items"]:
            if it["dynamic"] and it["fmt"] != "npz" and it["asset_id"] in by_id:
                it["collision"] = str(ensure_collision(by_id[it["asset_id"]], cache).resolve())
        (sdir / "spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))
    physics = "none"
    if args.preset == "hard" and args.physics == "mpm":
        physics = "mpm"
        from .mpm import settle               # taichi: only when used
        try:
            spec = settle(spec, sdir, sim_time=args.mpm_time, log=lambda m: print(f"{sid} {m}", flush=True))
        except Exception as e:                # fall back to Blender's Bullet for this scene
            print(f"{sid}: MPM failed ({type(e).__name__}: {e}), falling back to Bullet", flush=True)
            out_log.append(f"MPM failed:\n{traceback.format_exc()}")
            physics = "bullet (mpm failed)"
            spec = json.loads((sdir / "spec.json").read_text())
        (sdir / "spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))
    elif args.preset == "hard":
        physics = "bullet"
    shutil.rmtree(rdir, ignore_errors=True)    # never finalize a previous attempt's frames
    shutil.rmtree(odir.with_name(odir.name + ".tmp"), ignore_errors=True)
    cmd = [str(bproc), "run", "--custom-blender-path", args.blender, str(RENDER),
           str(sdir / "spec.json"), str(rdir), "--spp", str(args.spp)]
    ok = lambda r: r.returncode == 0 and (rdir / "render_meta.json").exists()    # noqa: E731
    try:
        r = _run(cmd, args.render_timeout, out_log)
        if not ok(r):
            raise RuntimeError(f"render exit {r.returncode}")
    except (RuntimeError, subprocess.TimeoutExpired):
        # usually the GPU ran out of memory (it may be shared); retry once on the CPU
        print(f"{sid}: GPU render failed, retrying on CPU", flush=True)
        shutil.rmtree(rdir, ignore_errors=True)
        r = _run(cmd + ["--cpu"], args.render_timeout, out_log)
        if not ok(r):
            raise RuntimeError(f"render failed (exit {r.returncode}) on GPU and CPU")
    prov = {"seed": args.seed, "scene_index": i, "views": args.views, "res_scale": args.res_scale,
            "spp": args.spp, "physics": physics, "preset_args": {"preset": args.preset, "hull_only": args.hull_only,
            "mpm_time": args.mpm_time, "n5k": bool(n5k), "n_hdris": len(hdris or []), "n_surfaces": len(surfaces)}}
    gt = finalize(sdir, rdir, odir, seed=args.seed, keep_hr=args.keep_hr, provenance=prov)
    gt["_wall_s"] = time.time() - t0
    return gt


if __name__ == "__main__":
    main()
