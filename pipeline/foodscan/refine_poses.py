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
           overlap: int = 10, max_center_err: float = 0.05, reuse_db: bool = False,
           seed_frames: int | None = None) -> dict[int, np.ndarray]:
    """seed_frames=None: seed every frame from ARKit. seed_frames=N: pure SfM (incremental mapper), then
    Sim3-align to the ARKit poses of the first N subsampled frames (for ARKit runaway scans)."""
    import pycolmap

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    imgs = work / "images"
    out = work / "sparse"
    out.mkdir(parents=True, exist_ok=True)
    for d in ("tri", "ba", "sfm"):
        (out / d).mkdir(exist_ok=True)
    if db.exists() and not reuse_db:
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
    run_match = not (reuse_db and db.exists())
    ro = pycolmap.ImageReaderOptions(camera_model="PINHOLE",
                                     camera_params=f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}")
    so = pycolmap.SiftExtractionOptions(max_num_features=max_features)
    if run_match:
        pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE,
                                  camera_model="PINHOLE", reader_options=ro, sift_options=so)
        mo = pycolmap.SequentialMatchingOptions(overlap=overlap, quadratic_overlap=True, loop_detection=False)
        pycolmap.match_sequential(db, matching_options=mo)

    if seed_frames is not None:
        return _sfm_aligned(db, imgs, out, poses_c2w_arkit, seed_frames)

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
        cen = lambda r: np.mean([np.linalg.inv(np.vstack([i.cam_from_world().matrix(), [0, 0, 0, 1]]))[:3, 3]
                                 for i in r.images.values()], 0)
        R = sim.rotation.matrix()
        sim = pycolmap.Sim3d(1.0, sim.rotation, cen(ref) - R @ cen(rec))   # keep centroids matched at scale 1
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


def _sfm_aligned(db, imgs, out, poses_c2w_arkit, n_seed):
    """Unseeded SfM (intrinsics fixed), then Sim3 (Umeyama on camera centres) to ARKit poses of the first n_seed frames."""
    import pycolmap
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, imgs, out / "sfm", opts)
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"SfM: {len(recs)} models, largest has {rec.num_reg_images()} images, {rec.num_points3D()} points")
    names = sorted(poses_c2w_arkit)[:n_seed]
    C = {}
    for im in rec.images.values():
        C[im.name] = np.linalg.inv(np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]]))[:3, 3]
    common = [n for n in names if n in C]
    if len(common) < 6:
        raise RuntimeError(f"only {len(common)} seed frames registered")
    X = np.array([C[n] for n in common])
    Y = np.array([poses_c2w_arkit[n][:3, 3] for n in common])
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((X - mx) ** 2).sum()
    res = np.linalg.norm((s * (X - mx) @ R.T + my) - Y, axis=1)
    print(f"seed alignment: {len(common)} frames, scale {s:.4f}, residual median {np.median(res)*1000:.1f} mm")
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), my - s * R @ mx))
    refined = {}
    for im in rec.images.values():
        T_cw = np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]])
        refined[int(Path(im.name).stem)] = np.linalg.inv(T_cw) @ _ARKIT_TO_CV
    return refined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--work", default=None)
    ap.add_argument("--reuse-db", action="store_true")
    ap.add_argument("--seed-frames", type=int, default=None, help="pure SfM aligned to first N ARKit frames")
    args = ap.parse_args()
    scan = Scan(args.scan_dir, poses="arkit")
    work = Path(args.work) if args.work else Path(args.scan_dir) / "colmap"
    refined = refine(scan, work, stride=args.stride, reuse_db=args.reuse_db, seed_frames=args.seed_frames)
    (Path(args.scan_dir) / "poses_refined.json").write_text(
        json.dumps({k: v.tolist() for k, v in refined.items()}, indent=0))
    print(f"wrote {len(refined)} refined poses")


if __name__ == "__main__":
    main()
