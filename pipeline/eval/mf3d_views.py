"""Volume from simulated depth views of real MetaFood3D meshes: 1 frame vs N-view orbits.

    python -m eval.mf3d_views ~/Downloads/_MetaFood3D_new_3D_Mesh.tar.gz ../data/metafood3d/nutrition.xlsx \
        --out ../results/metafood3d/views

Each scanned mesh (streamed from the archive, never extracted) is set on a table
plane (z = 0) and ray-cast from virtual depth cameras 35 cm from its centre
(320x240, 60 deg FoV, depth noise sigma 1 mm along the ray, table hits dropped
as a perfect food mask would):
    top1     one camera straight down
    oblique1 one camera at 45 deg elevation
    orbitN   N cameras evenly around a 45 deg ring plus one overhead (N = 2, 4, 8, 16)
Two volumes per view set:
    <views>         top-surface height map of the visible food points above the table
                    (max height per 2 mm cell, holes filled): plane closing, every
                    column under the visible top counted solid
    carve_<views>   space carving on a 2 mm voxel grid: a voxel survives unless a
                    camera sees past it; low views carve undercuts plane closing keeps The mesh's own volume
(watertight meshes only) is reported to check the sheet's ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import openpyxl

from foodscan.volume import _fill_nan_inside_footprint

W, H, FOV = 320, 240, 60.0
DIST, CELL, NOISE = 0.35, 0.002, 0.001
MIN_VOL = 20.0      # mL; single almonds / spears (< 20 mL) are below what 2 mm voxels resolve
VIEWS = {"top1": [(90, 0)], "oblique1": [(45, 0)]}
for n in (2, 4, 8, 16):
    VIEWS[f"orbit{n}"] = [(90, 0)] + [(45, 360 * k / n) for k in range(n)]


def cameras(elev_az: list[tuple[float, float]], centre: np.ndarray) -> list[np.ndarray]:
    eyes = []
    for el, az in elev_az:
        el, az = np.radians(el), np.radians(az)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        eyes.append(centre + DIST * d)
    return eyes


def camera(eye: np.ndarray, centre: np.ndarray):
    """(K, R, eye): OpenCV-style camera (x right, y down, z forward) looking at centre."""
    f = (W / 2) / np.tan(np.radians(FOV) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    z = centre - eye
    z /= np.linalg.norm(z)
    up = np.array([0, 1.0, 0]) if abs(z[2]) > 0.99 else np.array([0, 0, 1.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return K, np.stack([x, y, z]), eye


def render(scene, cam, rng) -> tuple[np.ndarray, np.ndarray]:
    """Depth image (z along the optical axis, inf = no hit, noisy) and geometry ids."""
    K, R, eye = cam
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    d_w = d_cam.reshape(-1, 3) @ R                       # R^T applied to row vectors
    rays = np.c_[np.tile(eye, (H * W, 1)), d_w].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    t = hit["t_hit"].numpy()                              # = z depth since d_cam.z = 1
    noisy = t * (1 + rng.normal(0, NOISE, t.shape) / np.maximum(t * np.linalg.norm(d_w, axis=1), 1e-6))
    return noisy.reshape(H, W), hit["geometry_ids"].numpy().reshape(H, W)


def visible_points(depth: np.ndarray, gid: np.ndarray, food_id: int, cam) -> np.ndarray:
    K, R, eye = cam
    v, u = np.nonzero((gid == food_id) & np.isfinite(depth))
    z = depth[v, u]
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
    return pc @ R + eye


def carve_volume(views: list, bounds: tuple[np.ndarray, np.ndarray], vox: float = 0.002,
                 margin: float = 0.002) -> float:
    """Space carving: a voxel above the table survives unless some camera sees past it
    (its depth < observed depth - margin, or its pixel saw nothing)."""
    lo, hi = bounds
    axes = [np.arange(lo[k] + vox / 2, hi[k], vox) for k in range(3)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    X = X[X[:, 2] > 0]
    keep = np.ones(len(X), bool)
    for (K, R, eye), depth in views:
        c = (X[keep] - eye) @ R.T
        z = c[:, 2]
        u = np.floor(K[0, 0] * c[:, 0] / z + K[0, 2]).astype(int)
        v = np.floor(K[1, 1] * c[:, 1] / z + K[1, 2]).astype(int)
        inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        free = np.zeros(len(z), bool)
        d = depth[v[inside], u[inside]]
        free[inside] = z[inside] < d - margin           # inf (no hit) carves too
        idx = np.nonzero(keep)[0]
        keep[idx[free]] = False
    return float(keep.sum() * vox ** 3 * 1e6)


def top_surface_volume(pts: np.ndarray) -> float:
    if len(pts) < 20:
        return 0.0
    pts = pts[pts[:, 2] > 0.001]
    ij = np.floor(pts[:, :2] / CELL).astype(np.int64)
    ij -= ij.min(0)
    shape = ij.max(0) + 1
    grid = np.full(shape, np.nan)
    np.fmax.at(grid, (ij[:, 0], ij[:, 1]), pts[:, 2])
    grid = _fill_nan_inside_footprint(grid)
    return float(np.nansum(grid) * CELL * CELL * 1e6)


def process_mesh(path: str, rng) -> dict:
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=False)
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return {}
    ext = v.max(0) - v.min(0)
    if ext.max() > 2.0:                                  # stored in mm
        mesh.scale(1e-3, center=np.zeros(3))
        v = np.asarray(mesh.vertices)
    mesh.translate([-v[:, 0].mean(), -v[:, 1].mean(), -v[:, 2].min()])
    out = {"extent_cm": (100 * (np.asarray(mesh.vertices).max(0) - np.asarray(mesh.vertices).min(0))).round(1).tolist(),
           "watertight": bool(mesh.is_watertight())}
    out["mesh_vol"] = float(abs(mesh.get_volume()) * 1e6) if out["watertight"] else float("nan")
    scene = o3d.t.geometry.RaycastingScene()
    food_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    table = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 0.01).translate([-0.5, -0.5, -0.01])
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(table))
    vv = np.asarray(mesh.vertices)
    centre = np.array([0.0, 0.0, vv[:, 2].max() / 2])
    lo = np.r_[vv[:, :2].min(0) - 0.01, 0.0]
    hi = np.r_[vv[:, :2].max(0) + 0.01, vv[:, 2].max() + 0.01]
    for name, ea in VIEWS.items():
        views, pts = [], []
        for e in cameras(ea, centre):
            cam = camera(e, centre)
            depth, gid = render(scene, cam, rng)
            views.append((cam, depth))
            pts.append(visible_points(depth, gid, food_id, cam))
        out[name] = top_surface_volume(np.concatenate(pts))
        out[f"carve_{name}"] = carve_volume(views, (lo, hi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", type=Path)
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../results/metafood3d/views"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = list(openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    sheet = {}
    for r in rows[1:]:
        if r[0] and all(isinstance(r[k], (int, float)) for k in (3, 4, 8)) and r[8] > 0:
            sheet[(str(r[0]).strip(), str(r[1]).strip().lower())] = {"mass": float(r[3]), "kcal": float(r[4]),
                                                                     "vol_gt": float(r[8])}
    cache = args.out / "per_object.jsonl"
    done = {(d["cat"], d["inst"]) for d in map(json.loads, open(cache))} if cache.exists() else set()
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    n = 0
    with tarfile.open(args.tar, "r|gz") as t, open(cache, "a") as fh:
        for m in t:
            if not (m.isfile() and m.name.lower().endswith(".obj")):
                continue
            parts = m.name.split("/")
            cat, inst = parts[1], parts[2]
            key = (cat, inst.lower())
            if key not in sheet or (cat, inst) in done:
                continue
            p = os.path.join(tmp, "m.obj")
            with open(p, "wb") as f:
                f.write(t.extractfile(m).read())
            try:
                rec = {"cat": cat, "inst": inst, **sheet[key], **process_mesh(p, rng)}
            except Exception as e:                       # noqa: BLE001
                rec = {"cat": cat, "inst": inst, **sheet[key], "error": repr(e)}
            os.remove(p)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n} meshes", flush=True)
    summarize(cache, args.out)
    summarize(cache, args.out, min_vol=0.0, name="summary_all_sizes.json")


def summarize(cache: Path, out: Path, min_vol: float = MIN_VOL, name: str = "summary.json") -> None:
    R = [r for r in map(json.loads, open(cache)) if "error" not in r and "top1" in r and r["vol_gt"] >= min_vol]
    ests = ["mesh_vol"] + list(VIEWS) + [f"carve_{v}" for v in VIEWS]
    gt = np.array([r["vol_gt"] for r in R])
    rep = {"n_objects": len(R), "n_watertight": int(sum(r["watertight"] for r in R)), "volume": {}, "mass_loo_category_density": {},
           "kcal_loo_category": {}}
    cats = np.array([r["cat"] for r in R])
    mass, kcal = np.array([r["mass"] for r in R]), np.array([r["kcal"] for r in R])
    for e in ["vol_gt"] + ests:
        v = np.array([r.get(e, np.nan) if e != "vol_gt" else r["vol_gt"] for r in R], float)
        ok = np.isfinite(v) & (v > 0)
        if e != "vol_gt":
            ape = 100 * np.abs(v[ok] - gt[ok]) / gt[ok]
            rep["volume"][e] = {"n": int(ok.sum()), "mape_pct": float(ape.mean()), "median_ape_pct": float(np.median(ape)),
                                "median_bias_pct": float(np.median(100 * (v[ok] - gt[ok]) / gt[ok]))}
        mp, kp = np.full(len(R), np.nan), np.full(len(R), np.nan)
        for i in np.nonzero(ok)[0]:
            others = (np.arange(len(R)) != i) & ok
            same = others & (cats == cats[i])
            pool = same if same.any() else others
            dens = np.median(mass[pool] / v[pool])
            mp[i] = dens * v[i]
            kp[i] = mp[i] * np.median(kcal[pool] / mass[pool])
        k = np.isfinite(mp)
        rep["mass_loo_category_density"][e] = float(100 * np.mean(np.abs(mp[k] - mass[k]) / mass[k]))
        rep["kcal_loo_category"][e] = float(100 * np.mean(np.abs(kp[k] - kcal[k]) / kcal[k]))
    rep["min_vol_ml"] = min_vol
    json.dump(rep, open(out / name, "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
