"""Refine ARKit poses with COLMAP: SIFT + sequential matching, triangulate with poses
fixed, bundle-adjust poses only (intrinsics fixed), re-align to the ARKit frame.

    python -m foodscan.refine_poses SCAN_DIR [--stride 3] [--work SCAN_DIR/colmap]

Writes SCAN_DIR/poses_refined.json: {frame_index: 4x4 camera-to-world in ARKit convention}.
`Scan(dir, poses="refined")` (see loader) then uses them for fusion.

Written against pycolmap 3.12.x (frames/rigs API). Facts that shaped it (from the
COLMAP source): triangulate_points never moves poses; bundle_adjustment fixes gauge on
two cameras, so the result drifts and must be re-aligned; alignment via projection
centres should return scale ~1 because the input was metric already.
Untested until the first real scan (the synthetic scan has no texture for SIFT).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .loader import _ARKIT_TO_CV, Scan


def refine(scan: Scan, work: Path, stride: int = 3, max_features: int = 8192,
           overlap: int = 10, max_center_err: float = 0.05) -> dict[int, np.ndarray]:
    import pycolmap

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    imgs = work / "images"
    out = work / "sparse"
    if db.exists():
        db.unlink()
    imgs.mkdir(exist_ok=True)

    frames = scan.good_frames(stride)
    poses_c2w_arkit = {}
    for f in frames:
        name = f"{f.index:06d}.jpg"
        dst = imgs / name
        if not dst.exists():
            shutil.copy(scan.dir / "rgb" / name, dst)
        poses_c2w_arkit[name] = f.T_cw @ np.linalg.inv(_ARKIT_TO_CV)  # back to ARKit c2w

    K = frames[0].K_rgb
    w, h = frames[0].rgb_size
    ro = pycolmap.ImageReaderOptions(camera_model="PINHOLE",
                                     camera_params=f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}")
    so = pycolmap.SiftExtractionOptions(max_num_features=max_features)
    pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model="PINHOLE", reader_options=ro, sift_options=so)
    mo = pycolmap.SequentialMatchingOptions(overlap=overlap, quadratic_overlap=True, loop_detection=False)
    pycolmap.match_sequential(db, matching_options=mo)

    rec = pycolmap.Reconstruction()
    with pycolmap.Database(str(db)) as database:
        cache = pycolmap.DatabaseCache.create(database, 0, False, set())
    rec.load(cache)
    for image_id, image in rec.images.items():
        T_cw = np.linalg.inv(poses_c2w_arkit[image.name] @ _ARKIT_TO_CV)   # OpenCV world->cam
        rec.frames[image.frame_id].rig_from_world = pycolmap.Rigid3d(T_cw[:3, :4])
        rec.register_frame(image.frame_id)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.triangulation.min_angle = 1.0
    rec = pycolmap.triangulate_points(rec, db, imgs, out / "tri", clear_points=True,
                                      options=opts, refine_intrinsics=False)
    ref = pycolmap.Reconstruction(rec)
    print(f"triangulated {rec.num_points3D()} points from {rec.num_reg_images()} images")

    ba = pycolmap.BundleAdjustmentOptions(refine_focal_length=False, refine_principal_point=False,
                                          refine_extra_params=False, refine_rig_from_world=True,
                                          refine_sensor_from_rig=False)
    pycolmap.bundle_adjustment(rec, ba)

    sim = pycolmap.align_reconstructions_via_proj_centers(rec, ref, max_proj_center_error=max_center_err)
    if sim is None:
        raise RuntimeError("alignment to ARKit frame failed")
    print(f"re-alignment scale {sim.scale:.4f} (expect ~1.0)")
    if abs(sim.scale - 1) > 0.02:
        print("WARNING: scale drifted >2%; forcing metric scale from ARKit")
        sim = pycolmap.Sim3d(1.0, sim.rotation, sim.translation)
    rec.transform(sim)
    rec.write(out / "ba")

    refined = {}
    for image_id, image in rec.images.items():
        T_cw = np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]])
        refined[int(Path(image.name).stem)] = np.linalg.inv(T_cw) @ _ARKIT_TO_CV  # ARKit c2w

    # report the correction size
    deltas = [np.linalg.norm(refined[int(Path(n).stem)][:3, 3] - p[:3, 3]) for n, p in poses_c2w_arkit.items()
              if int(Path(n).stem) in refined]
    print(f"pose correction: median {np.median(deltas)*1000:.1f} mm, max {np.max(deltas)*1000:.1f} mm")
    return refined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    scan = Scan(args.scan_dir)
    work = Path(args.work) if args.work else Path(args.scan_dir) / "colmap"
    refined = refine(scan, work, stride=args.stride)
    (Path(args.scan_dir) / "poses_refined.json").write_text(
        json.dumps({k: v.tolist() for k, v in refined.items()}, indent=0))
    print(f"wrote {len(refined)} refined poses")


if __name__ == "__main__":
    main()
