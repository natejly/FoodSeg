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


def _lsq_plane(p: np.ndarray) -> tuple[np.ndarray, float]:
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    return n, float(-n @ c)


def fit_support_plane_info(pts: np.ndarray, dist_thresh: float = 0.002, n_iter: int = 2000,
                           up_hint: np.ndarray | None = None, robust: bool = True,
                           band: float = 0.02, max_tilt_deg: float = 2.0, seed: int = 0) -> tuple[Plane, dict]:
    """Robust support-plane fit. Returns (plane, info).

    1. coarse plane from Open3D RANSAC (as before);
    2. lowest band: only points at most `band` metres above the coarse plane (anything below the
       container rim / food), so food piles and rims cannot pull the plane up;
    3. RANSAC on the band with a `dist_thresh` inlier threshold, then least-squares refinement on
       the inliers (two passes);
    4. with `up_hint` (gravity, ARKit +Y): if the fitted normal is more than `max_tilt_deg` away from
       it, use the gravity-aligned plane through the robust lowest band instead and flag it
       (`info["gravity_fallback"]`). `info["tilt_deg"]` is the disagreement of the *fit* with gravity.
    `robust=False` keeps the old single-pass Open3D RANSAC.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3,
                                       num_iterations=n_iter)
    n = np.array(model[:3])
    norm = np.linalg.norm(n)
    n, d = n / norm, model[3] / norm
    info: dict = {"robust": bool(robust), "gravity_fallback": False, "tilt_deg": None, "n_inliers": len(inliers)}
    if robust and up_hint is not None:
        # The dominant plane of a room scan can be a wall. With gravity known, only accept coarse
        # planes within 15 deg of horizontal (RANSAC over a subsample of the cloud).
        u0 = np.asarray(up_hint, float) / np.linalg.norm(up_hint)
        if n @ u0 < 0:
            n, d = -n, -d
        if n @ u0 < np.cos(np.radians(15)):
            rng = np.random.default_rng(seed)
            sub = pts[rng.choice(len(pts), min(len(pts), 30000), replace=False)]
            best_cnt = -1
            for _ in range(n_iter):
                tri = sub[rng.integers(0, len(sub), 3)]
                nn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                ln = np.linalg.norm(nn)
                if ln < 1e-12:
                    continue
                nn = nn / ln
                if nn @ u0 < 0:
                    nn = -nn
                if nn @ u0 < np.cos(np.radians(15)):
                    continue
                dd = -nn @ tri[0]
                cnt = int(np.sum(np.abs(sub @ nn + dd) < dist_thresh))
                if cnt > best_cnt:
                    best_cnt, n, d = cnt, nn, dd
            info["coarse_replaced"] = True
    if robust:
        h = pts @ n + d
        # orient the coarse normal first so "lowest band" means below the objects
        if up_hint is not None:
            if n @ up_hint < 0:
                n, d, h = -n, -d, -h
        elif np.sum(h < -dist_thresh) > np.sum(h > dist_thresh):
            n, d, h = -n, -d, -h
        low = pts[h < band]
        if len(low) >= 3:
            rng = np.random.default_rng(seed)
            best, best_cnt = (n, d), -1
            for _ in range(n_iter):
                tri = low[rng.integers(0, len(low), 3)]
                nn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                ln = np.linalg.norm(nn)
                if ln < 1e-12:
                    continue
                nn = nn / ln
                if nn @ n < 0:
                    nn = -nn
                if nn @ n < 0.9:                  # stay within ~25 deg of the coarse plane
                    continue
                dd = -nn @ tri[0]
                cnt = int(np.sum(np.abs(low @ nn + dd) < dist_thresh))
                if cnt > best_cnt:
                    best, best_cnt = (nn, dd), cnt
            n, d = best
            for _ in range(2):                    # least-squares refinement on the inliers
                inl = low[np.abs(low @ n + d) < dist_thresh]
                if len(inl) < 3:
                    break
                nn, dd = _lsq_plane(inl)
                if nn @ n < 0:
                    nn, dd = -nn, -dd
                n, d = nn, dd
            info["n_inliers"] = int(np.sum(np.abs(low @ n + d) < dist_thresh))
        if up_hint is not None:
            u = np.asarray(up_hint, float) / np.linalg.norm(up_hint)
            tilt = float(np.degrees(np.arccos(np.clip(n @ u, -1, 1))))
            info["tilt_deg"] = tilt
            if tilt > max_tilt_deg:
                # gravity-aligned plane through the robust lowest band (inliers of the fit)
                ref = low[np.abs(low @ n + d) < dist_thresh] if len(low) else pts
                if len(ref) == 0:
                    ref = pts
                n, d = u, float(-np.median(ref @ u))
                info["gravity_fallback"] = True
    plane = Plane(n, d)

    if up_hint is not None:
        if np.dot(plane.normal, up_hint) < 0:
            plane = Plane(-plane.normal, -plane.d)
    else:
        h = plane.height(pts)
        if np.sum(h < -dist_thresh) > np.sum(h > dist_thresh):
            plane = Plane(-plane.normal, -plane.d)
    return plane, info


def fit_support_plane(pts: np.ndarray, dist_thresh: float = 0.002, n_iter: int = 2000,
                      up_hint: np.ndarray | None = None, robust: bool = True) -> Plane:
    """RANSAC plane through the dominant flat surface (table or plate); see `fit_support_plane_info`.

    `up_hint` (world direction the camera considers up, e.g. gravity from ARKit's
    .gravity alignment = +Y in ARKit coordinates, converted the same way as the
    poses) orients the normal so objects are at positive height. Without it, the
    side with more points off-plane is taken as 'up'.
    """
    return fit_support_plane_info(pts, dist_thresh, n_iter, up_hint, robust)[0]


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

    Each triangle contributes (signed projected area) x (mean vertex height):
    upward faces add, downward faces (overhang undersides) subtract. Summing
    gives the volume with no grid discretisation; a negative total means the
    crop is mostly undersides, i.e. not an object resting on the plane.
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


def label_mesh_vertices(mesh: o3d.geometry.TriangleMesh, pts: np.ndarray, labels: np.ndarray,
                        radius: float = 0.003) -> np.ndarray:
    """Item id per mesh vertex: the label of the nearest labelled point (label > 0) within `radius`
    metres, else 0. `pts`/`labels` are the full cloud, so neighbouring items compete for vertices."""
    from scipy.spatial import cKDTree
    v = np.asarray(mesh.vertices)
    out = np.zeros(len(v), np.int64)
    m = labels > 0
    if not m.any() or len(v) == 0:
        return out
    d, i = cKDTree(pts[m]).query(v, distance_upper_bound=radius)
    ok = np.isfinite(d)
    out[ok] = labels[m][i[ok]]
    return out


def crop_mesh_by_label(mesh: o3d.geometry.TriangleMesh, vertex_labels: np.ndarray, item_id: int,
                       item_pts: np.ndarray | None = None, min_faces: int = 50,
                       pad: float = 0.004) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Sub-mesh of the faces with at least two of their three vertices labelled `item_id` (see
    `label_mesh_vertices`); the strict all-three rule drops a one-face seam between touching items (-12 % on a
    3 cm block with 4 mm faces).

    Falls back to the padded bbox crop of `item_pts` when the label crop has fewer than `min_faces`
    faces (or no points are given to fall back on only if nothing at all was found). The info dict
    reports `crop` ("label" | "bbox"), `n_faces` and `label_faces`.
    """
    tri = np.asarray(mesh.triangles)
    mine = (vertex_labels[tri] == item_id) if len(tri) else np.zeros((0, 3), bool)
    keep = mine.sum(1) >= 2          # majority rule: a seam strip between touching items is split, not lost
    sub = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(tri[keep]))
    n_label = len(sub.triangles)
    if n_label >= min_faces or item_pts is None:
        return sub, {"crop": "label", "n_faces": n_label, "label_faces": n_label}
    sub = crop_mesh_to_points(mesh, item_pts, pad)
    return sub, {"crop": "bbox", "n_faces": len(sub.triangles), "label_faces": n_label}


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


# ---- per-item result: envelope + solid estimate + geometry for the mass step --------------

def item_geometry(item_pts: np.ndarray, ref, cell: float = 0.002) -> dict:
    """Footprint area (cm^2) and height above the support (mm, p95 and max) of one item's points.

    `ref` is a Plane or a container.Support (anything with `.relative_height`). The footprint is the
    2 mm-cell occupancy of the points' plan view, closed by one cell and hole-filled (so LiDAR
    dropouts inside the item count), which is what the thin-layer mass rule multiplies by thickness."""
    if len(item_pts) == 0:
        return {"footprint_cm2": 0.0, "height_mm": 0.0, "height_max_mm": 0.0}
    from scipy import ndimage
    plane = getattr(ref, "plane", ref)
    h = ref.relative_height(item_pts) if hasattr(ref, "relative_height") else ref.height(item_pts)
    uv = plane.project_2d(item_pts)
    ij = np.floor((uv - uv.min(0)) / cell).astype(int) + 2
    occ = np.zeros(ij.max(0) + 3, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3), bool))
    occ = ndimage.binary_fill_holes(occ)
    return {"footprint_cm2": float(occ.sum() * cell * cell * 1e4),
            "height_mm": float(np.percentile(h, 95) * 1e3), "height_max_mm": float(h.max() * 1e3)}


def item_volume_result(volume_envelope_ml: float, item_pts: np.ndarray, ref, name: str,
                       kind: str | None = None) -> dict:
    """The volume fields every item carries.

    volume_envelope_ml  mesh-prism / height-map integral above the support (visible top surface closed
                        to the container, voids included); `volume_ml` is kept as an alias of it
    volume_solid_est_ml envelope / ENVELOPE_RATIO[kind] (density_table.py, medians from results/rq1/envelope.md)
    kind                `kind` if valid, else inferred from the name via density_table.infer_kind
    """
    from .density_table import ENVELOPE_RATIO, infer_kind, volume_solid_est
    k, how = infer_kind(name, kind)
    return {"kind": k, "kind_source": how, "volume_envelope_ml": float(volume_envelope_ml),
            "volume_ml": float(volume_envelope_ml), "envelope_ratio": ENVELOPE_RATIO[k],
            "volume_solid_est_ml": float(volume_solid_est(volume_envelope_ml, k)),
            **item_geometry(item_pts, ref)}
