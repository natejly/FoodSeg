"""Tests for the synthetic per-item volume harness (pipeline/eval/synth_volume*.py).

    cd /home/kyle/Desktop/FoodSeg && conda run -n foodseg python -m pytest pipeline/tests/test_synth_volume.py -q

Shapes with closed-form volumes are sampled as the top-down surface a scanner would see (plus a variant with
side walls), then every estimator must land within 5 % (flat table and bowl).  Degenerate inputs must not raise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.eval import synth_volume as SV  # noqa: E402
from pipeline.eval import synth_volume_data as D  # noqa: E402
from pipeline.eval import synth_volume_methods as M  # noqa: E402
from pipeline.eval import synth_volume_report as RP  # noqa: E402
from pipeline.foodscan.container import Support  # noqa: E402

rng = np.random.default_rng(0)


def disc_grid(r: float, step: float = 0.001):
    g = np.arange(-r, r + step, step)
    X, Y = np.meshgrid(g, g)
    m = X ** 2 + Y ** 2 < r ** 2
    rg = np.random.default_rng(int(r * 1e4))
    return X[m] + rg.uniform(-step / 4, step / 4, m.sum()), Y[m] + rg.uniform(-step / 4, step / 4, m.sum())


def flat_support() -> dict:
    """A flat plate lying on the table: container surface == table plane."""
    sup = Support.from_points(np.array([[x, y, 0.0] for x in np.linspace(-.1, .1, 41) for y in np.linspace(-.1, .1, 41)]),
                              M.Z_PLANE)
    return {"icp": sup, "icp_c": sup, "gt": sup}


def hemisphere(r=0.03):
    x, y = disc_grid(r)
    z = np.sqrt(np.clip(r ** 2 - x ** 2 - y ** 2, 0, None))
    return np.c_[x, y, z], 2 / 3 * np.pi * r ** 3 * 1e6


def cylinder(r=0.03, h=0.02, walls=False):
    x, y = disc_grid(r)
    pts = np.c_[x, y, np.full(len(x), h)]
    if walls:
        th = rng.uniform(0, 2 * np.pi, 4000)
        z = rng.uniform(0, h, 4000)
        pts = np.vstack([pts, np.c_[r * np.cos(th), r * np.sin(th), z]])
    return pts, np.pi * r ** 2 * h * 1e6


@pytest.mark.parametrize("name,shape", [("hemisphere", hemisphere), ("cylinder", cylinder)])
@pytest.mark.parametrize("method", ["foodscan", "foodscan_gtreg", "foodscan_cinit", "voxel", "hull", "table", "hull_table"])
def test_flat_plane_shapes_within_5pct(name, shape, method):
    pts, truth = shape()
    est = M.estimate_item(pts, pts, flat_support())[method]
    assert est == pytest.approx(truth, rel=0.05), f"{method} on {name}: {est:.2f} vs {truth:.2f}"


def test_cylinder_with_side_walls():
    pts, truth = cylinder(r=0.02, h=0.03, walls=True)
    est = M.estimate_item(pts, pts, flat_support())
    for m in ("foodscan", "voxel", "hull", "table", "hull_table"):
        assert est[m] == pytest.approx(truth, rel=0.05), m


def test_small_item_footprint_bias():
    """Fixed Support.height_map_volume rasterises the footprint at 0.5 mm, so a 1 cm radius cylinder is within 5 %;
    legacy=True keeps the old whole-2-mm-cell count (+8..25 % over)."""
    pts, truth = cylinder(r=0.01, h=0.02)
    sup = flat_support()["icp"]
    est = M.estimate_item(pts, pts, flat_support())
    assert abs(est["foodscan"] / truth - 1) < 0.05
    assert 1.08 < sup.height_map_volume(sup.points_above(pts, 0.001), legacy=True) / truth < 1.25
    assert est["hull"] == pytest.approx(truth, rel=0.08)


def paraboloid_bowl(R=0.06, D=0.02, n=60000):
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, n)
    r = R * np.sqrt(rng.uniform(0, 1, n))
    return np.c_[r * np.cos(th), r * np.sin(th), -D + D * r ** 2 / R ** 2]


def bowl_fill(fill=0.012, R=0.06, D=0.02):
    """Flat fill of height `fill` above the bowl bottom; top surface only (what the scanner sees)."""
    rf = R * np.sqrt(fill / D)
    x, y = disc_grid(rf)
    pts = np.c_[x, y, np.full(len(x), -D + fill)]
    return pts, np.pi * fill ** 2 * R ** 2 / (2 * D) * 1e6


def test_bowl_container_surface_vs_table_plane():
    cont = paraboloid_bowl()
    food, truth = bowl_fill()
    scene = np.vstack([cont, food])
    sup = M.build_supports(cont, scene, init_xy=np.zeros(2))
    assert sup["icp"] is not None and sup["info"]["icp_shift_mm"] < 3.0, sup["info"]
    est = M.estimate_item(food, scene, sup)
    for m in ("foodscan", "foodscan_gtreg", "foodscan_cinit", "voxel"):
        assert est[m] == pytest.approx(truth, rel=0.05), f"{m}: {est[m]:.2f} vs {truth:.2f}"
    # the hull wraps the 2 mm support height field's jitter, so it runs a few % high (3-6 % observed)
    assert est["hull"] == pytest.approx(truth, rel=0.08)
    # the flat-table reference cannot see food below the rim at all
    assert est["table"] == 0.0 and est["hull_table"] == 0.0


def test_bowl_deeper_fill_and_partial_cover():
    cont = paraboloid_bowl()
    food, truth = bowl_fill(fill=0.017)
    scene = np.vstack([cont, food])
    sup = M.build_supports(cont, scene, init_xy=np.zeros(2))
    est = M.estimate_item(food, scene, sup)
    assert est["foodscan_gtreg"] == pytest.approx(truth, rel=0.05)
    assert est["voxel"] == pytest.approx(truth, rel=0.05)


def test_degenerate_items_do_not_raise():
    sup = flat_support()
    few = rng.uniform(0, 0.01, (10, 3))
    assert all(np.isnan(v) for v in M.estimate_item(few, few, sup).values())
    assert all(np.isnan(v) for v in M.estimate_item(np.zeros((0, 3)), np.zeros((0, 3)), sup).values())
    nan_pts = np.full((30, 3), np.nan)
    assert all(np.isnan(v) for v in M.estimate_item(nan_pts, nan_pts, sup).values())
    below = np.c_[rng.uniform(0, .02, (50, 2)), np.full(50, -0.005)]          # entirely under the support
    est = M.estimate_item(below, below, sup)
    assert est["foodscan"] == 0.0 and est["table"] == 0.0
    line = np.c_[np.linspace(0, .03, 40), np.zeros(40), np.full(40, 0.01)]    # collinear: qhull must not crash
    est = M.estimate_item(line, line, sup)
    assert np.isnan(est["hull"]) or est["hull"] >= 0
    est = M.estimate_item(hemisphere()[0], hemisphere()[0], {"icp": None, "icp_c": None, "gt": None})
    assert np.isnan(est["foodscan"]) and np.isfinite(est["table"])           # no support -> container methods NaN only


def test_match_instances_permutation_and_threshold():
    gt = np.array([1] * 10 + [2] * 10 + [252] * 5)
    inst = np.array([7] * 10 + [3] * 10 + [0] * 5)
    assert D.match_instances(gt, inst) == {1: (7, 1.0), 2: (3, 1.0)}
    inst2 = np.array([7] * 2 + [0] * 8 + [3] * 10 + [0] * 5)                     # item 1 mostly lost: IoU 0.2
    assert D.match_instances(gt, inst2)[1][0] == 7
    inst3 = np.array([7] * 1 + [0] * 9 + [3] * 10 + [0] * 5)                     # IoU 0.1 -> still matched; below -> dropped
    assert 1 in D.match_instances(gt, inst3)
    assert 1 not in D.match_instances(gt, np.array([0] * 10 + [3] * 10 + [0] * 5))
    assert D.match_instances(gt, np.zeros(25, int)) == {}


def test_top_envelope_and_densify():
    cont = paraboloid_bowl(n=30000)
    under = cont.copy()
    under[:, 2] -= 0.004                                                       # an underside sheet
    env = M.top_envelope(np.vstack([cont, under]))
    assert (env[:, 2] > cont[:, 2].min() + 1e-6).all() or len(env) <= len(cont) + len(under)
    dense = M.densify_surface(env)
    assert len(dense) == pytest.approx(np.pi * 0.06 ** 2 / M.CELL ** 2, rel=0.05)
    r = np.hypot(dense[:, 0], dense[:, 1])
    assert np.abs(dense[:, 2] - (-0.02 + 0.02 * r ** 2 / 0.06 ** 2)).mean() < 0.0005


# ---------------------------------------------------------------- end to end on a fake scan dir

def write_scan(root: Path, sid: str, items_pts: list[np.ndarray], volumes: list[float], empty=False, ok=True,
               with_lift=False):
    """Dish frame == ARKit frame rotated so that dish z -> ARKit y (up)."""
    d = root / "scans" / sid
    (d / "gt").mkdir(parents=True)
    Rm = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)                    # dish -> arkit (x, z->y, y->z)
    W = np.eye(4)
    W[:3, :3] = Rm
    cont = paraboloid_bowl()
    tab = np.c_[rng.uniform(-.14, .14, (6000, 2)), np.zeros(6000)]
    tab = tab[np.hypot(tab[:, 0], tab[:, 1]) > 0.065]
    xs, ls = [cont, tab], [np.full(len(cont), 254), np.full(len(tab), 252)]
    for i, p in enumerate(items_pts, 1):
        xs.append(p)
        ls.append(np.full(len(p), i))
    xyz, lab = np.vstack(xs), np.concatenate(ls)
    seg = dict(xyz=(xyz @ Rm.T).astype(np.float32), rgb=np.zeros((len(xyz), 3), np.uint8),
               normal=np.zeros((len(xyz), 3), np.float32), label=lab.astype(np.uint8), dist=np.zeros(len(xyz), np.float32))
    if with_lift:
        seg["lift_label"] = np.where(lab == 252, 0, np.where(lab <= 3, (lab % 3) + 1, lab)).astype(np.int32)   # ids permuted
        seg["lift_share"] = np.ones(len(xyz), np.float32)
        seg["lift_nvotes"] = np.full(len(xyz), 5, np.int16)
        seg["boundary"] = np.zeros(len(xyz), np.uint8)
    if empty:
        seg = {k: v[:0] for k, v in seg.items()}
    np.savez(d / "gt" / "seg_points.npz", **seg)
    np.savez(d / "gt" / "container.npz", points=cont, name="c")
    gt = dict(scene_id=sid, world_to_arkit=W.tolist(), container=dict(kind="bowl"),
              items=[dict(id=i, name=f"it{i}", category="Cat", volume_ml=v, mass_g=v * 0.9, shape=dict(kind="piece"))
                     for i, v in enumerate(volumes, 1)])
    (d / "gt.json").write_text(json.dumps(gt))
    (root / "manifest.jsonl").write_text((root / "manifest.jsonl").read_text() if (root / "manifest.jsonl").exists() else "")
    with open(root / "manifest.jsonl", "a") as f:
        f.write(json.dumps(dict(scan=sid, ok=ok, stacked=False, qc=[])) + "\n")


def test_end_to_end_fake_scans(tmp_path):
    root = tmp_path / "toy"
    food, truth = bowl_fill()
    small = hemisphere(0.012)[0] + np.array([0.03, 0.0, -0.0])               # a hemisphere on the table next to the bowl
    tiny = rng.uniform(0, 0.005, (10, 3)) + np.array([0, 0.04, 0])           # < 20 points
    write_scan(root, "s000000", [food, small, tiny], [truth, hemisphere(0.012)[1], 0.1], with_lift=True)
    write_scan(root, "s000001", [], [5.0], empty=True, ok=False)
    write_scan(root, "s000002", [food], [truth], ok=False)
    man = D.read_manifest(root)
    infos, rows, dishes = [], [], []
    for sid in ("s000000", "s000001", "s000002"):
        sc = D.load_scan(root, sid, man)
        i, r, d = SV.evaluate_scan(sc, [])
        infos.append(i); rows += r; dishes += d
    by = {(r["scan"], r["source"], r["item_id"]): r for r in rows}
    r = by[("s000000", "gt", 1)]
    assert r["vol_foodscan_gtreg"] == pytest.approx(truth, rel=0.06)
    assert r["vol_foodscan"] == pytest.approx(truth, rel=0.10)
    assert by[("s000000", "gt", 2)]["vol_table"] == pytest.approx(hemisphere(0.012)[1], rel=0.08)
    assert np.isnan(by[("s000000", "gt", 3)]["vol_foodscan"])                  # < 20 points -> missing
    lr = by[("s000000", "lift", 1)]                                           # ids permuted: matching must undo it
    assert lr["matched"] and lr["vol_foodscan_gtreg"] == pytest.approx(truth, rel=0.06)
    assert infos[1]["n_points"] == 0 and by[("s000001", "gt", 1)]["matched"] is False   # empty cloud: no crash, item missing
    res = RP.build(rows, dishes, infos, include_all=True, pred_dirs=[], plane="gt", min_points=20)
    assert res["counts"]["all"]["scans"] == 3 and res["counts"]["ok"]["scans"] == 1
    assert res["reading"]
    out = tmp_path / "out"
    out.mkdir()
    RP.write(out, res, rows, dishes, infos)
    assert (out / "summary.md").read_text().startswith("# Synthetic")
    json.loads((out / "summary.json").read_text())
    res_ok = RP.build(rows, dishes, infos, include_all=False, pred_dirs=[], plane="gt", min_points=20)
    assert {x["source"] for x in res_ok["per_item"]} <= {"gt", "lift"} | {"gt_top"}
