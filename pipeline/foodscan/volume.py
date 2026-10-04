"""Volume of objects sitting on a plane, from a fused point cloud or mesh.

Estimators, so they can be compared on calibration objects:

    mesh_prism_volume   integrate the fused TSDF mesh's top surface down to the
                        support plane, triangle by triangle       <- primary
                        (synthetic 6x4x3 cm box: +1.3 % at 1 mm voxels,
                        +12 % at 3 mm; error is voxel fattening of walls)
    height_map_volume   integrate mean height above the plane over a 2 mm grid
                        (Nutrition5k / Lu et al. style); works on bare points,
                        so it is the estimator for the single-RGB-D baseline
    poisson_volume      Poisson-reconstruct the object surface, close it to the
                        plane, take the signed volume of the watertight mesh
    convex_hull_volume  convex hull of the points above the plane (the
                        LiDARCalorieCam approach; over-estimates concave items)

Always run `cluster_objects` on the above-plane points first; stray table
points above the threshold otherwise inflate the hull and Poisson results.

All functions take an (N, 3) numpy array of points in metres in world
coordinates, plus a Plane. Volumes are returned in cm^3 (= mL).

Typical use:
    plane = fit_support_plane(all_points)          # table / plate surface
    obj = points_above_plane(all_points, plane, min_height=0.003)
    v = height_map_volume(obj, plane)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

M3_TO_CM3 = 1e6


@dataclass
class Plane:
    """n . x + d = 0, with n unit length and oriented so 'up' is +n."""
    normal: np.ndarray
    d: float

    def height(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance of points above the plane (metres)."""
        return pts @ self.normal + self.d

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Two orthonormal in-plane axes (u, v)."""
        n = self.normal
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return u, v

    def project_2d(self, pts: np.ndarray) -> np.ndarray:
        u, v = self.basis()
        return np.stack([pts @ u, pts @ v], axis=-1)


def fit_support_plane(pts: np.ndarray, dist_thresh: float = 0.002, n_iter: int = 2000,
                      up_hint: np.ndarray | None = None) -> Plane:
    """RANSAC plane through the dominant flat surface (table or plate).

    `up_hint` (world direction the camera considers up, e.g. gravity from ARKit's
    .gravity alignment = +Y in ARKit coordinates, converted the same way as the
    poses) orients the normal so objects are at positive height. Without it, the
    side with more points off-plane is taken as 'up'.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3,
                                       num_iterations=n_iter)
    a, b, c, d = model
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    plane = Plane(n, d)

    if up_hint is not None:
        if np.dot(n, up_hint) < 0:
            plane = Plane(-n, -d)
    else:
        h = plane.height(pts)
        if np.sum(h < -dist_thresh) > np.sum(h > dist_thresh):
            plane = Plane(-n, -d)
    return plane


def points_above_plane(pts: np.ndarray, plane: Plane, min_height: float = 0.003,
                       max_height: float = 0.25) -> np.ndarray:
    h = plane.height(pts)
    return pts[(h > min_height) & (h < max_height)]


def cluster_objects(obj_pts: np.ndarray, eps: float = 0.006, min_points: int = 10,
                    min_cluster: int = 50) -> list[np.ndarray]:
    """Split above-plane points into objects with DBSCAN; drops noise and tiny blobs.

    Returned largest first. Use before any volume estimator so stray table
    points above the height threshold don't inflate hulls.
    """
    if len(obj_pts) == 0:
        return []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = [obj_pts[labels == k] for k in range(labels.max() + 1)]
    clusters = [c for c in clusters if len(c) >= min_cluster]
    return sorted(clusters, key=len, reverse=True)


def mesh_prism_volume(mesh: o3d.geometry.TriangleMesh, plane: Plane,
                      min_height: float = 0.001) -> float:
    """Volume between the mesh's upward-facing triangles and the plane.

    For each triangle whose normal points away from the plane, the volume of the
    prism down to the plane is (projected area) x (mean vertex height). Summing
    over the visible top surface gives the volume with no grid discretisation.
    Pass a mesh already cropped to one object (see `crop_mesh_to_points`).
    """
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    if len(t) == 0:
        return 0.0
    h = plane.height(v)
    tri_h = h[t]                                  # (T, 3)
    keep = tri_h.mean(1) > min_height
    a, b, c = v[t[keep, 0]], v[t[keep, 1]], v[t[keep, 2]]
    n = np.cross(b - a, c - a)                    # 2 x area x normal
    signed_proj_area = 0.5 * (n @ plane.normal)   # + for upward-facing
    vol = np.sum(signed_proj_area * tri_h[keep].mean(1))
    return float(vol * M3_TO_CM3)


def crop_mesh_to_points(mesh: o3d.geometry.TriangleMesh, pts: np.ndarray,
                        pad: float = 0.004) -> o3d.geometry.TriangleMesh:
    """Sub-mesh inside the padded bounding box of `pts` (one clustered object)."""
    bbox = o3d.geometry.AxisAlignedBoundingBox(pts.min(0) - pad, pts.max(0) + pad)
    return mesh.crop(bbox)


def height_map_volume(obj_pts: np.ndarray, plane: Plane, cell: float = 0.002,
                      fill_holes: bool = True) -> float:
    """Sum over grid cells of (max height in cell) * cell area.

    Using the max per cell approximates the visible top surface; cells with no
    points inside the object's footprint are filled from neighbours when
    `fill_holes` is set, which matters for LiDAR dropouts on glossy patches.
    """
    if len(obj_pts) == 0:
        return 0.0
    uv = plane.project_2d(obj_pts)
    h = plane.height(obj_pts)
    ij = np.floor((uv - uv.min(0)) / cell).astype(int)
    shape = ij.max(0) + 1
    # mean height per cell. Max-per-cell latches onto noise peaks once the
    # cloud is dense (fine TSDF voxels), so use the mean; on a flat-topped
    # object it is unbiased except for the perimeter cells.
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    sums = np.bincount(flat, weights=h, minlength=shape[0] * shape[1])
    counts = np.bincount(flat, minlength=shape[0] * shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = (sums / counts).reshape(shape)

    if fill_holes:
        grid = _fill_nan_inside_footprint(grid)
    return float(np.nansum(grid) * cell * cell * M3_TO_CM3)


def _fill_nan_inside_footprint(grid: np.ndarray, iters: int = 20) -> np.ndarray:
    """Iteratively fill NaN cells that have >= 3 valid 4-neighbours with their mean.

    Interior dropouts get filled; the exterior (no neighbours) stays NaN.
    """
    g = grid.copy()
    for _ in range(iters):
        nan = np.isnan(g)
        if not nan.any():
            break
        padded = np.pad(g, 1, constant_values=np.nan)
        nbrs = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
        count = np.sum(~np.isnan(nbrs), axis=0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(nbrs, axis=0)
        fill = nan & (count >= 3)
        if not fill.any():
            break
        g[fill] = mean[fill]
    return g


def convex_hull_volume(obj_pts: np.ndarray, plane: Plane) -> float:
    """Convex hull of the object points plus their projections onto the plane."""
    if len(obj_pts) < 4:
        return 0.0
    base = obj_pts - np.outer(plane.height(obj_pts), plane.normal)
    pts = np.vstack([obj_pts, base])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    hull, _ = pcd.compute_convex_hull()
    return float(hull.get_volume() * M3_TO_CM3)


def poisson_volume(obj_pts: np.ndarray, plane: Plane, depth: int = 8,
                   base_spacing: float = 0.002) -> float:
    """Close the scanned surface to the plane and Poisson-reconstruct a watertight mesh.

    The underside is synthesised as a flat disc of points on the plane under the
    object's footprint, with normals pointing down, so Poisson has a closed shell.
    """
    if len(obj_pts) < 50:
        return 0.0
    top = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    top.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # orient normals away from the plane (roughly 'up' and outward)
    centroid = obj_pts.mean(0)
    top.orient_normals_towards_camera_location(centroid + plane.normal * 0.5)

    # base disc: grid over footprint, kept where a top point exists nearby
    uv = plane.project_2d(obj_pts)
    u, v = plane.basis()
    origin = -plane.d * plane.normal
    us = np.arange(uv[:, 0].min(), uv[:, 0].max(), base_spacing)
    vs = np.arange(uv[:, 1].min(), uv[:, 1].max(), base_spacing)
    gu, gv = np.meshgrid(us, vs)
    grid_uv = np.stack([gu.ravel(), gv.ravel()], -1)
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.c_[uv, np.zeros(len(uv))])))
    keep = np.array([tree.search_radius_vector_3d([p[0], p[1], 0.0], base_spacing * 1.5)[0] > 0
                     for p in grid_uv])
    base_uv = grid_uv[keep]
    # plane origin has uv (origin.u, origin.v); express base points in world
    base_pts = origin + np.outer(base_uv[:, 0] - origin @ u, u) + np.outer(base_uv[:, 1] - origin @ v, v)
    base = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_pts))
    base.normals = o3d.utility.Vector3dVector(np.tile(-plane.normal, (len(base_pts), 1)))

    closed = top + base
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(closed, depth=depth)
    # trim Poisson's bubble to the object's bounding box (padded)
    bbox = closed.get_axis_aligned_bounding_box()
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox.min_bound - 0.003, bbox.max_bound + 0.003)
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight():
        # fall back to voxel fill of the cropped mesh
        return _voxel_fill_volume(mesh, plane)
    return float(abs(mesh.get_volume()) * M3_TO_CM3)


def _voxel_fill_volume(mesh: o3d.geometry.TriangleMesh, plane: Plane, voxel: float = 0.001) -> float:
    """Volume between the mesh's top surface and the plane, via height-map on mesh vertices."""
    pts = np.asarray(mesh.vertices)
    pts = pts[plane.height(pts) > 0]
    return height_map_volume(pts, plane, cell=voxel * 2)


def estimate_all(obj_pts: np.ndarray, plane: Plane,
                 mesh: o3d.geometry.TriangleMesh | None = None) -> dict[str, float]:
    out = {
        "height_map_cm3": height_map_volume(obj_pts, plane),
        "poisson_cm3": poisson_volume(obj_pts, plane),
        "convex_hull_cm3": convex_hull_volume(obj_pts, plane),
        "n_points": float(len(obj_pts)),
        "max_height_mm": float(plane.height(obj_pts).max() * 1000) if len(obj_pts) else 0.0,
    }
    if mesh is not None:
        out["mesh_prism_cm3"] = mesh_prism_volume(crop_mesh_to_points(mesh, obj_pts), plane)
    return out
