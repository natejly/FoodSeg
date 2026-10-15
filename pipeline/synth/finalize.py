"""Raw renders -> a FoodScan capture directory that every pipeline tool reads unchanged.

    python -m synth.finalize SCENE_DIR RAW_DIR SCAN_OUT [--seed 0] [--keep-hr]

Writes the same layout as the iPad app (meta.json, frames.jsonl, rgb/*.jpg,
depth/*.f32, smooth_depth/*.f32, conf/*.u8; see foodscan/loader.py) plus:

    gt.json              per-item true volume (mL), mass if known, names, container
                         parameters and capacity, LiDAR noise parameters
    gt/seg_ids/*.png     full instance ids per frame at RGB resolution: items 1..K, 253 clutter, 254 container
    gt/masks/*.png       the same with 253/254 zeroed (0 = not food; what run_meal --masks reads) and
    gt/masks/instances.json   -> `run_meal --masks SCAN/gt/masks` = oracle segmentation
    gt/container/*.png   container mask (for rim-exclusion training)
    gt/container.npz     exact ContainerTemplate (`run_meal --container SCAN/gt/container.npz`)
    gt/points.npz        surface samples of every object in its final pose, ARKit frame
                         (xyz, label: item id, 254 container, 253 clutter, 252 table) -> synth.seg_export
    gt/depth_clean/*.f32 noise-free 256x192 depth
    gt/depth_hr/*.f32    noise-free depth at RGB resolution (with --keep-hr; for depth_upsample evals)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import container_from_json
from .camera import PhoneCamera, develop, save_jpeg, scene_key
from .density import mass_from_volume
from .itemstats import item_gt_stats, with_table
from .lidar import D_H, D_W, LidarParams, PoseNoiseParams, perturb_poses, simulate, translucent_ids

GENERATOR_VERSION = "synth-2.1"     # 2.1: atomic scans, full gt/seg_ids, table label 252, mass/volume stats

CONTAINER_ID = 254
DISTRACTOR_ID = 253     # non-food clutter (hard preset)


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def finalize(scene_dir: Path, raw_dir: Path, out: Path, seed: int = 0, keep_hr: bool = False,
             params: LidarParams | None = None, provenance: dict | None = None,
             pose_noise: bool = True) -> dict:
    """Write the scan to `<out>.tmp`, gt.json last, then rename to `out` (gt.json = complete marker)."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        gt = _finalize(scene_dir, raw_dir, tmp, seed, keep_hr, params, provenance, pose_noise)
        if out.exists():
            shutil.rmtree(out)
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return gt


def _finalize(scene_dir: Path, raw_dir: Path, out: Path, seed: int, keep_hr: bool,
              params: LidarParams | None, provenance: dict | None, pose_noise: bool) -> dict:
    spec = json.loads((scene_dir / "spec.json").read_text())
    rmeta = json.loads((raw_dir / "render_meta.json").read_text())
    W2A = np.array(spec["world_to_arkit"])
    cam = spec["camera"]
    K = np.array(cam["K"])
    w, h = cam["width"], cam["height"]
    rng = np.random.default_rng([seed, int(spec["scene_id"].lstrip("s") or 0)])

    if params is None:                  # per-scan sensor: sigma0, bias field, dropout, translucency
        params = LidarParams.sample(rng)
    true_c2w = [np.array(c) for c in cam["c2w"]]
    pn = PoseNoiseParams.sample(rng) if pose_noise else None
    # exported (ARKit-like drifting) poses; GT stays in the true frame
    c2w_exp = perturb_poses(true_c2w, np.random.default_rng([seed, 7]), pn) if pn else true_c2w
    transl = translucent_ids(spec["items"])
    subs = ["rgb", "depth", "smooth_depth", "conf", "gt/masks", "gt/seg_ids", "gt/container", "gt/depth_clean"]
    if keep_hr:
        subs.append("gt/depth_hr")
    for s in subs:
        (out / s).mkdir(parents=True, exist_ok=True)

    frames = sorted(raw_dir.glob("frame_*.npz"))
    if rmeta.get("n_frames") is not None and rmeta["n_frames"] != len(frames):
        raise RuntimeError(f"{raw_dir}: render_meta says {rmeta['n_frames']} frames, found {len(frames)}")
    visible: dict[int, int] = {}
    phone = PhoneCamera.sample(rng) if spec.get("preset") == "hard" else None
    key = None
    if frames and "rgb_linear" in np.load(frames[0]).files:     # HDR render: meter the whole scan once
        key = scene_key([np.load(p)["rgb_linear"] for p in frames[::4]])
    with open(out / "frames.jsonl", "w") as f:
        for i, p in enumerate(frames):
            z = np.load(p)
            depth, seg = z["depth"], z["seg"]
            rgb = develop(z["rgb_linear"], key, phone.ev_bias if phone else 0.0) if key is not None else z["rgb"]
            name = f"{i:06d}"
            if phone and key is not None:      # full sensor + ISP chain on the linear render
                save_jpeg(phone.process(z["rgb_linear"], key, rng, i), out / "rgb" / f"{name}.jpg",
                          phone.jpeg_quality)
            elif phone:
                save_jpeg(phone.apply(rgb, rng), out / "rgb" / f"{name}.jpg", phone.jpeg_quality)
            else:
                Image.fromarray(rgb).save(out / "rgb" / f"{name}.jpg", quality=95)
            sim = simulate(depth, rgb, K, rng, params, c2w=true_c2w[i], seg=seg, translucent=transl)
            sim["depth"].astype("<f4").tofile(out / "depth" / f"{name}.f32")
            sim["smooth_depth"].astype("<f4").tofile(out / "smooth_depth" / f"{name}.f32")
            sim["conf"].tofile(out / "conf" / f"{name}.u8")
            sim["clean"].astype("<f4").tofile(out / "gt" / "depth_clean" / f"{name}.f32")
            if keep_hr:
                depth.astype("<f4").tofile(out / "gt" / "depth_hr" / f"{name}.f32")
            seg = np.asarray(seg).astype(np.uint8)
            Image.fromarray(seg).save(out / "gt" / "seg_ids" / f"{name}.png")     # food + 253 clutter + 254 container
            food = np.where((seg == CONTAINER_ID) | (seg == DISTRACTOR_ID), 0, seg).astype(np.uint8)   # run_meal --masks
            Image.fromarray(food).save(out / "gt" / "masks" / f"{name}.png")
            Image.fromarray(((seg == CONTAINER_ID) * 255).astype(np.uint8)).save(
                out / "gt" / "container" / f"{name}.png")
            for lid in np.unique(food):
                if lid:
                    visible[int(lid)] = visible.get(int(lid), 0) + 1
            T = W2A @ c2w_exp[i]   # camera axes already ARKit-style
            rec = {"index": i, "timestamp": cam["timestamps"][i], "trackingState": "normal",
                   "intrinsics": K.tolist(), "transform": T.tolist(),
                   "imageWidth": w, "imageHeight": h, "depthWidth": D_W, "depthHeight": D_H,
                   "exposureDuration": 0.01, "exposureOffset": 0.0}
            f.write(json.dumps(rec) + "\n")

    inst = {it["id"]: {"name": it["name"], "concept_index": k, "mean_score": 1.0, "gt": True}
            for k, it in enumerate(spec["items"])}
    (out / "gt" / "masks" / "instances.json").write_text(json.dumps(inst, indent=1))

    c = container_from_json(spec["container"])
    if spec["container"].get("asset") and (raw_dir / "gt_points.npz").exists():
        # real scanned dish: its own surface samples (dish frame: Z up, table at z = 0), not the
        # analytic stand-in, plus a ring of table so the template registers like the procedural one
        z = np.load(raw_dir / "gt_points.npz")
        dish = z["xyz"][z["label"] == CONTAINER_ID].astype(np.float64)
        trng = np.random.default_rng(0)
        r, th = 0.15 * np.sqrt(trng.random(6000)), trng.random(6000) * 2 * np.pi
        table = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(r)], 1)
        table = table[np.hypot(table[:, 0], table[:, 1]) > np.hypot(dish[:, 0], dish[:, 1]).max()]
        np.savez_compressed(out / "gt" / "container.npz", points=np.concatenate([dish, table]),
                            name=spec["container"]["asset"]["name"])
    else:
        np.savez_compressed(out / "gt" / "container.npz", points=c.surface_points(rng=0),
                            name=f"synthetic_{c.kind}")

    by_id = {r["id"]: r for r in rmeta["items"]}
    items = []
    for it in spec["items"]:
        r = by_id[it["id"]]
        off = r["radius_from_axis_m"] > c.R + c.rim_w
        items.append({k: it[k] for k in ("id", "name", "category", "source", "asset_id", "volume_ml",
                                         "mass_g", "procedural", "generated", "density_g_ml", "mass_source")
                         if k in it}
                     | {"scene_volume_ml": r["scene_volume_ml"], "frames_visible": visible.get(it["id"], 0),
                        "off_container": bool(off), "pose_arkit": (W2A @ np.array(r["local2world"])).tolist(),
                        "fill": it.get("fill"), "shape": it.get("shape")})
        if it.get("cloth"):        # leaves: the simulated, thickened sheets' own solid volume
            items[-1]["volume_ml"] = r["scene_volume_ml"]
        if it.get("density_g_ml"):     # spec mass came from the pre-settle volume: redo it from the final one
            items[-1]["mass_g"] = items[-1]["volume_ml"] * it["density_g_ml"]
            items[-1]["mass_source"] = "density"
    if (raw_dir / "gt_points.npz").exists():     # per-point 3D labels, ARKit frame
        z = np.load(raw_dir / "gt_points.npz")
        xyz = z["xyz"].astype(np.float64) @ W2A[:3, :3].T + W2A[:3, 3]
        xyz, lab = with_table(xyz.astype(np.float32), z["label"], W2A)      # + table plane samples (252)
        np.savez_compressed(out / "gt" / "points.npz", xyz=xyz, label=lab)
        item_gt_stats({"items": items, "world_to_arkit": W2A.tolist()}, xyz, lab)
    else:
        for it in items:
            if it.get("mass_g") is None:
                it["mass_g"], it["mass_source"] = mass_from_volume(it)
    gt = {"scene_id": spec["scene_id"], "synthetic": True, "preset": spec.get("preset", "v0"),
          "distractors": [d["name"] for d in spec.get("distractors", [])],
          "phone_camera": phone.to_json() if phone else None,
          "container": {k: v for k, v in spec["container"].items() if k != "mesh"},
          "table_plane_arkit": {"normal": (W2A[:3, :3] @ [0, 0, 1]).tolist(),
                                "point": W2A[:3, 3].tolist()},
          "world_to_arkit": W2A.tolist(), "items": items,
          "total_volume_ml": float(sum(it["volume_ml"] for it in items)),
          "lidar_params": asdict(params), "pose_noise": asdict(pn) if pn else None, "render": {"spp": rmeta["spp"], "timing_s": rmeta["timing_s"]},
          "provenance": {"seed": seed, "n_views": len(frames), "views": len(cam["c2w"]),
                         "preset": spec.get("preset", "v0"), "git_commit": git_commit(),
                         "generator_version": GENERATOR_VERSION, **(provenance or {})}}
    meta = {"formatVersion": 1, "app": "synth/blenderproc", "device": "synthetic", "synthetic": True,
            "savedFrames": len(frames), "scene": spec["scene_id"], "name": spec["scene_id"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "gt.json").write_text(json.dumps(gt, indent=1))      # LAST: its presence marks a complete scan
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("raw_dir")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-hr", action="store_true")
    ap.add_argument("--no-pose-noise", action="store_true", help="export the true camera poses (debugging)")
    args = ap.parse_args()
    gt = finalize(Path(args.scene_dir), Path(args.raw_dir), Path(args.out), args.seed, args.keep_hr,
                  pose_noise=not args.no_pose_noise)
    print(f"{gt['scene_id']}: {len(gt['items'])} items, {gt['total_volume_ml']:.1f} mL ->", args.out)
    for it in gt["items"]:
        flag = " OFF-CONTAINER" if it["off_container"] else ""
        print(f"  {it['id']} {it['name']:<16} spec {it['volume_ml']:7.1f} mL  scene {it['scene_volume_ml']:7.1f} mL"
              f"  visible in {it['frames_visible']} frames{flag}")


if __name__ == "__main__":
    main()
