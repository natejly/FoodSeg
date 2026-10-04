"""Container-aware volume: food height measured against a registered empty-container scan.

Workflow
    # once per plate / bowl
    tmpl = ContainerTemplate.from_scan(empty_scan_dir)            # fuse + table plane + surface
    tmpl.save("data/containers/bowl_A.npz")

    # per meal
    tmpl = ContainerTemplate.load("data/containers/bowl_A.npz")
    T = tmpl.register(scene_points, scene_plane)                  # ICP, returns 4x4 template->scene
    support = tmpl.support(T, scene_plane)                        # height of the container surface
    food = support.points_above(scene_points, min_height=0.002)   # or your SAM 3 labelled points
    v = support.mesh_prism_volume(crop_mesh_to_points(mesh, food))

The template is stored as points expressed relative to its own table plane, so a
template scanned on one table registers onto another. Heights are always measured
along the *scene's* table normal, which is what Lu et al. 2020 and Nutrition5k do.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .volume import M3_TO_CM3, Plane, fit_support_plane


def _plane_frame(plane: Plane) -> np.ndarray:
    """4x4 world->plane transform: plane normal becomes +Z, origin on the plane."""
    u, v = plane.basis()
    n = plane.normal
    R = np.stack([u, v, n])                       # rows: new axes in world coords
    origin = -plane.d * n
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return T


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


@dataclass
class ContainerTemplate:
    """Empty container as points in a plane-aligned frame (Z up, table at z = 0)."""
    points: np.ndarray          # (N, 3) in template frame
    name: str = "container"

    @classmethod
    def from_points(cls, pts: np.ndarray, plane: Plane, name: str = "container",
                    band: float = 0.15, min_dev: float = 0.0015) -> "ContainerTemplate":
        """Keep points within `band` metres of the container's centre that deviate from the
        table plane by more than `min_dev` (rim, walls, interior), plus a ring of table
        points around them so ICP has the plane constraint too."""
        T = _plane_frame(plane)
        p = _apply(T, pts)
        dev = np.abs(p[:, 2]) > min_dev
        if dev.sum() < 100:
            raise ValueError("no container geometry found above/below the plane")
        centre = p[dev, :2].mean(0)
        r = np.linalg.norm(p[:, :2] - centre, axis=1)
        keep = (r < band) & (dev | (r < band))   # container + surrounding table disc
        p = p[keep]
        p[:, :2] -= centre                        # centre the template in XY
        return cls(points=p, name=name)

    @classmethod
    def from_scan(cls, scan_dir: str | Path, name: str | None = None, voxel: float = 0.0015,
                  up_hint: np.ndarray | None = None) -> "ContainerTemplate":
        from .fuse import fuse
        from .loader import Scan
        scan = Scan(scan_dir)
        _, pcd = fuse(scan, voxel=voxel, stride=2)
        pts = np.asarray(pcd.points)
        plane = fit_support_plane(pts, up_hint=up_hint)
        return cls.from_points(pts, plane, name=name or Path(scan_dir).name)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, points=self.points, name=self.name)

    @classmethod
    def load(cls, path: str | Path) -> "ContainerTemplate":
        z = np.load(path, allow_pickle=False)
        return cls(points=z["points"], name=str(z["name"]))

    # ---- registration --------------------------------------------------------

    def register(self, scene_pts: np.ndarray, scene_plane: Plane, init_xy: np.ndarray | None = None,
                 max_dist: float = 0.01, voxel: float = 0.002, food_min_height: float = 0.0015) -> np.ndarray:
        """ICP the template into the scene. Returns 4x4 template-frame -> world.

        Coarse init: scene table plane gives the normal, the centroid of off-plane
        scene points (or `init_xy`) gives the XY offset. Then point-to-plane ICP
        against scene points near the container (deviating from the table plane).
        Rotation about the plane normal is left to ICP; for round bowls it is
        irrelevant, for rimmed plates the rim geometry constrains it.
        """
        T_ws = np.linalg.inv(_plane_frame(scene_plane))         # scene-plane frame -> world
        s = _apply(_plane_frame(scene_plane), scene_pts)         # scene in plane frame
        dev = np.abs(s[:, 2]) > 0.0015
        centre = np.asarray(init_xy) if init_xy is not None else s[dev, :2].mean(0)
        # candidate scene points: within the template's radius of the centre
        rad = np.linalg.norm(self.points[:, :2], axis=1).max() + 0.02
        near = np.linalg.norm(s[:, :2] - centre, axis=1) < rad
        target = s[near]

        T0 = np.eye(4)
        T0[:2, 3] = centre                                       # template centre -> scene centre
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target)).voxel_down_sample(voxel)
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        tgt_tree = o3d.geometry.KDTreeFlann(tgt)
        tgt_np = np.asarray(tgt.points)

        def icp(src_pts: np.ndarray, init: np.ndarray, dist: float):
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts)).voxel_down_sample(voxel)
            return o3d.pipelines.registration.registration_icp(
                src, tgt, dist, init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

        # Stage 1: coarse, all template points.
        res = icp(self.points, T0, max_dist)
        # Stage 2: drop template points the scene covers (food sitting on the
        # container): where the nearest scene point is well above the template,
        # there is no true correspondence and ICP would drag the template upward.
        moved = _apply(res.transformation, self.points)
        covered = np.zeros(len(moved), bool)
        for i, p in enumerate(moved):
            k, idx, _ = tgt_tree.search_knn_vector_3d(p, 1)
            if k:
                covered[i] = (tgt_np[idx[0], 2] - p[2]) > food_min_height
        free = self.points[~covered]
        self.covered_fraction = float(covered.mean())
        if len(free) > 200:
            res = icp(free, res.transformation, min(max_dist, 0.004))
        self.last_fitness, self.last_rmse = res.fitness, res.inlier_rmse
        return T_ws @ res.transformation                         # template -> world

    def support(self, T_template_to_world: np.ndarray, scene_plane: Plane, cell: float = 0.002) -> "Support":
        pts_w = _apply(T_template_to_world, self.points)
        return Support.from_points(pts_w, scene_plane, cell=cell)


def _fill_nan_nearest(grid: np.ndarray, max_dist_cells: int = 3) -> np.ndarray:
    """Fill NaN cells from the nearest valid cell, up to `max_dist_cells` away.

    Small gaps inside the template footprint (sparser points than the grid) are
    filled; cells far from any template point stay NaN and fall back to the plane.
    """
    from scipy import ndimage
    nan = np.isnan(grid)
    if not nan.any():
        return grid
    dist, (iy, ix) = ndimage.distance_transform_edt(nan, return_distances=True, return_indices=True)
    out = grid.copy()
    fill = nan & (dist <= max_dist_cells)
    out[fill] = grid[iy[fill], ix[fill]]
    return out


@dataclass
class Support:
    """Height field of the support surface (container + table) over the scene plane."""
    plane: Plane
    origin_uv: np.ndarray       # (2,) uv of grid cell (0, 0)
    cell: float
    grid: np.ndarray            # (nu, nv) height in metres; NaN = no template -> use plane (0)

    @classmethod
    def from_points(cls, pts: np.ndarray, plane: Plane, cell: float = 0.002) -> "Support":
        uv = plane.project_2d(pts)
        h = plane.height(pts)
        origin = uv.min(0)
        ij = np.floor((uv - origin) / cell).astype(int)
        shape = ij.max(0) + 1
        flat = ij[:, 0] * shape[1] + ij[:, 1]
        # Mean height per cell. The container surface is single-valued from above,
        # and max-per-cell biases sloped walls upward by ~slope x cell/2.
        sums = np.bincount(flat, weights=h, minlength=shape[0] * shape[1])
        counts = np.bincount(flat, minlength=shape[0] * shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = sums / counts
        grid[counts == 0] = np.nan
        grid = _fill_nan_nearest(grid.reshape(shape), max_dist_cells=3)
        return cls(plane=plane, origin_uv=origin, cell=cell, grid=grid)

    def height_at(self, pts: np.ndarray) -> np.ndarray:
        """Support height under each point (metres), 0 (plane) where no template cell."""
        uv = self.plane.project_2d(pts)
        ij = np.floor((uv - self.origin_uv) / self.cell).astype(int)
        out = np.zeros(len(pts))
        inb = (ij[:, 0] >= 0) & (ij[:, 0] < self.grid.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.grid.shape[1])
        g = self.grid[ij[inb, 0], ij[inb, 1]]
        out[inb] = np.where(np.isnan(g), 0.0, g)
        return out

    def relative_height(self, pts: np.ndarray) -> np.ndarray:
        return self.plane.height(pts) - self.height_at(pts)

    def points_above(self, pts: np.ndarray, min_height: float = 0.002, max_height: float = 0.25) -> np.ndarray:
        h = self.relative_height(pts)
        return pts[(h > min_height) & (h < max_height)]

    def mesh_prism_volume(self, mesh: o3d.geometry.TriangleMesh, min_height: float = 0.001) -> float:
        """Like volume.mesh_prism_volume but each vertex's height is relative to the support."""
        v = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)
        if len(t) == 0:
            return 0.0
        h = self.relative_height(v)
        tri_h = h[t]
        keep = tri_h.mean(1) > min_height
        a, b, c = v[t[keep, 0]], v[t[keep, 1]], v[t[keep, 2]]
        n = np.cross(b - a, c - a)
        signed_proj_area = 0.5 * (n @ self.plane.normal)
        return float(np.sum(signed_proj_area * tri_h[keep].mean(1)) * M3_TO_CM3)

    def height_map_volume(self, pts: np.ndarray, cell: float | None = None) -> float:
        """Grid integration of mean relative height (for point-only inputs)."""
        cell = cell or self.cell
        if len(pts) == 0:
            return 0.0
        uv = self.plane.project_2d(pts)
        h = self.relative_height(pts)
        ij = np.floor((uv - uv.min(0)) / cell).astype(int)
        shape = ij.max(0) + 1
        flat = ij[:, 0] * shape[1] + ij[:, 1]
        sums = np.bincount(flat, weights=h, minlength=shape[0] * shape[1])
        counts = np.bincount(flat, minlength=shape[0] * shape[1])
        mean = sums[counts > 0] / counts[counts > 0]
        return float(np.sum(mean) * cell * cell * M3_TO_CM3)
