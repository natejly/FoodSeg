"""Fuse a FoodScan capture into a metric mesh / point cloud with TSDF integration.

Usage:
    python -m foodscan.fuse SCAN_DIR [--voxel 0.003] [--stride 2] [--max-depth 0.8]
                            [--min-conf 2] [--smoothed] [--out OUT_PREFIX]

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


def _depth_image(frame: Frame, min_conf: int, smoothed: bool, max_depth: float) -> o3d.t.geometry.Image:
    depth = frame.masked_depth(min_conf=min_conf, smoothed=smoothed, max_depth=max_depth)
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
                       max_depth: float = 0.8) -> o3d.geometry.PointCloud:
    """Point cloud from one RGB-D frame, in world coordinates (the RGB-D baseline)."""
    w, h = frame.depth_size
    K = frame.K_depth
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    depth = frame.masked_depth(min_conf=min_conf, smoothed=smoothed, max_depth=max_depth)
    color = np.asarray(Image.fromarray(frame.rgb()).resize((w, h), Image.BILINEAR))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
        depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, frame.T_wc)


def fuse(scan: Scan, voxel: float = 0.0015, stride: int = 2, min_conf: int = CONF_HIGH,
         smoothed: bool = False, max_depth: float = 0.8, device: str = "CPU:0"):
    """TSDF-integrate all good frames. Returns (legacy mesh, legacy point cloud)."""
    frames = scan.good_frames(stride)
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
        depth = _depth_image(f, min_conf, smoothed, max_depth).to(dev)
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
    ap.add_argument("--out", default=None, help="output prefix (default: SCAN_DIR/fused)")
    args = ap.parse_args()

    scan = Scan(args.scan_dir)
    print(f"{len(scan)} frames, {len(scan.good_frames())} with normal tracking, "
          f"device {scan.meta.get('device')}")
    mesh, pcd = fuse(scan, voxel=args.voxel, stride=args.stride, min_conf=args.min_conf,
                     smoothed=args.smoothed, max_depth=args.max_depth, device=args.device)
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
