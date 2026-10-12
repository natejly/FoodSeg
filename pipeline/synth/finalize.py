"""Raw renders -> a FoodScan capture directory that every pipeline tool reads unchanged.

    python -m synth.finalize SCENE_DIR RAW_DIR SCAN_OUT [--seed 0] [--keep-hr]

Writes the same layout as the iPad app (meta.json, frames.jsonl, rgb/*.jpg,
depth/*.f32, smooth_depth/*.f32, conf/*.u8; see foodscan/loader.py) plus:

    gt.json              per-item true volume (mL), mass if known, names, container
                         parameters and capacity, LiDAR noise parameters
    gt/masks/*.png       instance ids per frame at RGB resolution (0 = not food) and
    gt/masks/instances.json   -> `run_meal --masks SCAN/gt/masks` = oracle segmentation
    gt/container/*.png   container mask (for rim-exclusion training)
    gt/container.npz     exact ContainerTemplate (`run_meal --container SCAN/gt/container.npz`)
    gt/depth_clean/*.f32 noise-free 256x192 depth
    gt/depth_hr/*.f32    noise-free depth at RGB resolution (with --keep-hr; for depth_upsample evals)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import container_from_json
from .lidar import D_H, D_W, LidarParams, simulate

CONTAINER_ID = 254


def finalize(scene_dir: Path, raw_dir: Path, out: Path, seed: int = 0, keep_hr: bool = False,
             params: LidarParams = LidarParams()) -> dict:
    spec = json.loads((scene_dir / "spec.json").read_text())
    rmeta = json.loads((raw_dir / "render_meta.json").read_text())
    W2A = np.array(spec["world_to_arkit"])
    cam = spec["camera"]
    K = np.array(cam["K"])
    w, h = cam["width"], cam["height"]
    rng = np.random.default_rng([seed, int(spec["scene_id"].lstrip("s") or 0)])

    subs = ["rgb", "depth", "smooth_depth", "conf", "gt/masks", "gt/container", "gt/depth_clean"]
    if keep_hr:
        subs.append("gt/depth_hr")
    for s in subs:
        (out / s).mkdir(parents=True, exist_ok=True)

    frames = sorted(raw_dir.glob("frame_*.npz"))
    visible: dict[int, int] = {}
    with open(out / "frames.jsonl", "w") as f:
        for i, p in enumerate(frames):
            z = np.load(p)
            rgb, depth, seg = z["rgb"], z["depth"], z["seg"]
            name = f"{i:06d}"
            Image.fromarray(rgb).save(out / "rgb" / f"{name}.jpg", quality=95)
            sim = simulate(depth, rgb, K, rng, params)
            sim["depth"].astype("<f4").tofile(out / "depth" / f"{name}.f32")
            sim["smooth_depth"].astype("<f4").tofile(out / "smooth_depth" / f"{name}.f32")
            sim["conf"].tofile(out / "conf" / f"{name}.u8")
            sim["clean"].astype("<f4").tofile(out / "gt" / "depth_clean" / f"{name}.f32")
            if keep_hr:
                depth.astype("<f4").tofile(out / "gt" / "depth_hr" / f"{name}.f32")
            food = np.where(seg == CONTAINER_ID, 0, seg).astype(np.uint8)
            Image.fromarray(food).save(out / "gt" / "masks" / f"{name}.png")
            Image.fromarray(((seg == CONTAINER_ID) * 255).astype(np.uint8)).save(
                out / "gt" / "container" / f"{name}.png")
            for lid in np.unique(food):
                if lid:
                    visible[int(lid)] = visible.get(int(lid), 0) + 1
            T = W2A @ np.array(cam["c2w"][i])   # camera axes already ARKit-style
            rec = {"index": i, "timestamp": cam["timestamps"][i], "trackingState": "normal",
                   "intrinsics": K.tolist(), "transform": T.tolist(),
                   "imageWidth": w, "imageHeight": h, "depthWidth": D_W, "depthHeight": D_H,
                   "exposureDuration": 0.01, "exposureOffset": 0.0}
            f.write(json.dumps(rec) + "\n")

    inst = {it["id"]: {"name": it["name"], "concept_index": k, "mean_score": 1.0, "gt": True}
            for k, it in enumerate(spec["items"])}
    (out / "gt" / "masks" / "instances.json").write_text(json.dumps(inst, indent=1))

    c = container_from_json(spec["container"])
    np.savez_compressed(out / "gt" / "container.npz", points=c.surface_points(rng=0),
                        name=f"synthetic_{c.kind}")

    by_id = {r["id"]: r for r in rmeta["items"]}
    items = []
    for it in spec["items"]:
        r = by_id[it["id"]]
        off = r["radius_from_axis_m"] > c.R + c.rim_w
        items.append({k: it[k] for k in ("id", "name", "category", "source", "asset_id", "volume_ml",
                                         "mass_g", "procedural", "generated")}
                     | {"scene_volume_ml": r["scene_volume_ml"], "frames_visible": visible.get(it["id"], 0),
                        "off_container": bool(off), "pose_arkit": (W2A @ np.array(r["local2world"])).tolist(),
                        "fill": it.get("fill")})
    gt = {"scene_id": spec["scene_id"], "synthetic": True,
          "container": {k: v for k, v in spec["container"].items() if k != "mesh"},
          "table_plane_arkit": {"normal": (W2A[:3, :3] @ [0, 0, 1]).tolist(),
                                "point": W2A[:3, 3].tolist()},
          "world_to_arkit": W2A.tolist(), "items": items,
          "total_volume_ml": float(sum(it["volume_ml"] for it in items)),
          "lidar_params": asdict(params), "render": {"spp": rmeta["spp"], "timing_s": rmeta["timing_s"]}}
    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    meta = {"formatVersion": 1, "app": "synth/blenderproc", "device": "synthetic", "synthetic": True,
            "savedFrames": len(frames), "scene": spec["scene_id"], "name": spec["scene_id"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("raw_dir")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-hr", action="store_true")
    args = ap.parse_args()
    gt = finalize(Path(args.scene_dir), Path(args.raw_dir), Path(args.out), args.seed, args.keep_hr)
    print(f"{gt['scene_id']}: {len(gt['items'])} items, {gt['total_volume_ml']:.1f} mL ->", args.out)
    for it in gt["items"]:
        flag = " OFF-CONTAINER" if it["off_container"] else ""
        print(f"  {it['id']} {it['name']:<16} spec {it['volume_ml']:7.1f} mL  scene {it['scene_volume_ml']:7.1f} mL"
              f"  visible in {it['frames_visible']} frames{flag}")


if __name__ == "__main__":
    main()
