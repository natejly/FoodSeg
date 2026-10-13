"""Fuse a FoodScan capture into a metric mesh / point cloud with TSDF integration.

Usage:
    python -m foodscan.fuse SCAN_DIR [--voxel 0.003] [--stride 2] [--max-depth 0.8]
                            [--min-conf 2] [--smoothed] [--out OUT_PREFIX]
                            [--edge-rel 0.02] [--max-incidence 0] [--min-sharpness 0.5]

Writes OUT_PREFIX.ply (mesh) and OUT_PREFIX_points.ply (point cloud).

Also exposes `single_frame_cloud(frame)` so a single RGB-D frame can be
compared with the fused scan (research question 1).

Uses the tensor-based VoxelBlockGrid. The legacy ScalableTSDFVolume returns
an empty mesh with Open3D 0.20 on Apple Silicon, so avoid it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from .loader import CONF_HIGH, Frame, Scan

# Depth cleaning defaults. ARKit sceneDepth is a 256x192 map densified from a sparse LiDAR
# dot grid, so it interpolates across object silhouettes ("flying pixels": a ramp between
# foreground and background) and is least reliable where the surface is seen edge-on.
# On scan 20261002_210342 those two effects put ~3.6k points 1-4 cm outside the box and
# 4-6 mm of error on its sides, while the head-on top face was within 1.5 mm.
EDGE_REL = 0.02         # drop a pixel if its 3x3 depth range exceeds 2 % of its depth
MAX_INCIDENCE = 0.0     # off: dropping grazing pixels (e.g. 75 deg) eroded small items in the
                        # food sim (cheese cube -21 -> -43 %) for ~2 points of MAPE gain
MIN_SHARPNESS = 0.5     # drop frames sharper than this fraction of the scan median


def clean_depth(depth: np.ndarray, K: np.ndarray, edge_rel: float = EDGE_REL,
                max_incidence: float = MAX_INCIDENCE) -> np.ndarray:
    """Zero depth at silhouettes (edge_rel) and grazing-angle surfaces (max_incidence, deg).
    Either test is disabled by passing 0. Invalid pixels must already be 0."""
    from scipy.ndimage import maximum_filter, median_filter, minimum_filter, uniform_filter
    d = depth.copy()
    valid = d > 0
    if edge_rel > 0:
        # range of a median-denoised map, so per-pixel noise on oblique flat surfaces
        # (the table, a plate) does not read as an edge
        dm = np.where(valid, median_filter(depth, 3), 0.0)
        near = minimum_filter(np.where(dm > 0, dm, np.inf), 3)
        rng = maximum_filter(dm, 3) - near
        # keep the foreground side of the jump: the object's own rim is real, the smeared
        # ramp and the background behind it are not (removing the rim too erased small items)
        behind = dm - near > 0.5 * edge_rel * depth
        d[valid & (rng > edge_rel * depth) & behind] = 0.0
    if max_incidence > 0:
        # normals from a lightly smoothed depth over a +-2 px baseline: the raw map's ~1 mm
        # noise against a ~1.7 mm pixel footprint would otherwise flag flat surfaces.
        w = uniform_filter(valid.astype(np.float32), 3)
        ds = np.where(w > 0, uniform_filter(np.where(valid, depth, 0.0), 3) / np.maximum(w, 1e-6), 0.0)
        h, wd = d.shape
        u, v = np.meshgrid(np.arange(wd), np.arange(h))
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(ds)], -1)
        P = ray * ds[..., None]
        dx = np.zeros_like(P); dy = np.zeros_like(P)
        dx[:, 2:-2] = P[:, 4:] - P[:, :-4]
        dy[2:-2] = P[4:] - P[:-4]
        n = np.cross(dx, dy)
        cos = np.abs((n * ray).sum(-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(ray, axis=-1) + 1e-12)
        known = np.linalg.norm(n, axis=-1) > 0
        d[known & (cos < np.cos(np.radians(max_incidence)))] = 0.0
    return d


def _depth_image(frame: Frame, min_conf: int, smoothed: bool, max_depth: float,
                 edge_rel: float, max_incidence: float) -> o3d.t.geometry.Image:
    depth = frame.masked_depth(min_conf=min_conf, smoothed=smoothed, max_depth=max_depth)
    depth = clean_depth(depth, frame.K_depth, edge_rel, max_incidence)
    return o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth.astype(np.float32))))


def _color_image(frame: Frame) -> o3d.t.geometry.Image:
    # Color is only for visualisation; downsample it to the depth grid so both
    # share K_depth. Volume comes from depth alone. VoxelBlockGrid wants float
    # color when depth is float.
    color = np.asarray(Image.fromarray(frame.rgb()).resize(frame.depth_size, Image.BILINEAR))
    return o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(color.astype(np.float32) / 255.0)))


def _K(frame: Frame) -> o3d.core.Tensor:
    return o3d.core.Tensor(frame.K_depth, o3d.core.float64)


def _extrinsic(frame: Frame) -> o3d.core.Tensor:
    return o3d.core.Tensor(frame.T_wc, o3d.core.float64)


def single_frame_cloud(frame: Frame, min_conf: int = CONF_HIGH, smoothed: bool = False,
                       max_depth: float = 0.8, edge_rel: float = 0.0,
                       max_incidence: float = 0.0) -> o3d.geometry.PointCloud:
    """Point cloud from one RGB-D frame, in world coordinates (the RGB-D baseline).

    Depth cleaning is off by default: with a single view nothing refills the silhouette
    pixels it removes, so it erodes objects (-23 % volume on the synthetic box)."""
    w, h = frame.depth_size
    K = frame.K_depth
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    depth = frame.masked_depth(min_conf=min_conf, smoothed=smoothed, max_depth=max_depth)
    depth = clean_depth(depth, K, edge_rel, max_incidence)
    color = np.asarray(Image.fromarray(frame.rgb()).resize((w, h), Image.BILINEAR))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
        depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, frame.T_wc)


def fuse(scan: Scan, voxel: float = 0.0015, stride: int = 2, min_conf: int = CONF_HIGH,
         smoothed: bool = False, max_depth: float = 0.8, device: str = "CPU:0",
         edge_rel: float = EDGE_REL, max_incidence: float = MAX_INCIDENCE,
         min_sharpness: float = MIN_SHARPNESS):
    """TSDF-integrate all good, sharp frames. Returns (legacy mesh, legacy point cloud)."""
    frames = scan.sharp_frames(stride, min_sharpness)
    if not frames:
        raise RuntimeError("no frames with normal tracking")
    dev = o3d.core.Device(device)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel,
        block_resolution=8,
        block_count=50000,
        device=dev,
    )
    for f in tqdm(frames, desc="integrate"):
        depth = _depth_image(f, min_conf, smoothed, max_depth, edge_rel, max_incidence).to(dev)
        color = _color_image(f).to(dev)
        K, E = _K(f), _extrinsic(f)
        blocks = vbg.compute_unique_block_coordinates(depth, K, E, 1.0, max_depth)
        vbg.integrate(blocks, depth, color, K, K, E, 1.0, max_depth)
    mesh = vbg.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()
    pcd = vbg.extract_point_cloud().to_legacy()
    return mesh, pcd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--voxel", type=float, default=0.0015,
                    help="voxel size in meters (1-1.5 mm for volume work; coarser fattens walls)")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth good frame")
    ap.add_argument("--max-depth", type=float, default=0.8, help="ignore depth beyond this (m)")
    ap.add_argument("--min-conf", type=int, default=CONF_HIGH, choices=[0, 1, 2])
    ap.add_argument("--smoothed", action="store_true", help="use ARKit smoothedSceneDepth")
    ap.add_argument("--device", default="CPU:0", help="CPU:0 or CUDA:0 on the cluster")
    ap.add_argument("--edge-rel", type=float, default=EDGE_REL,
                    help="drop depth at silhouettes: 3x3 range > this x depth (0 = off)")
    ap.add_argument("--max-incidence", type=float, default=MAX_INCIDENCE,
                    help="drop depth seen more edge-on than this many degrees (0 = off)")
    ap.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS,
                    help="drop frames blurrier than this fraction of the median (0 = off)")
    ap.add_argument("--out", default=None, help="output prefix (default: SCAN_DIR/fused)")
    args = ap.parse_args()

    scan = Scan(args.scan_dir)
    print(f"{len(scan)} frames, {len(scan.good_frames())} with normal tracking, "
          f"device {scan.meta.get('device')}")
    mesh, pcd = fuse(scan, voxel=args.voxel, stride=args.stride, min_conf=args.min_conf,
                     smoothed=args.smoothed, max_depth=args.max_depth, device=args.device,
                     edge_rel=args.edge_rel, max_incidence=args.max_incidence,
                     min_sharpness=args.min_sharpness)
    out = Path(args.out) if args.out else Path(args.scan_dir) / "fused"
    mesh_path = out.with_suffix(".ply")
    pts_path = out.parent / (out.name + "_points.ply")
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(pts_path), pcd)
    bb = mesh.get_axis_aligned_bounding_box()
    print(f"mesh: {len(mesh.vertices)} vertices -> {mesh_path}")
    print(f"points: {len(pcd.points)} -> {pts_path}")
    print(f"bbox min {np.round(bb.min_bound, 3)} max {np.round(bb.max_bound, 3)} (m)")


if __name__ == "__main__":
    main()
