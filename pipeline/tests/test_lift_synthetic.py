"""Lift oracle masks onto GT points of hard_v10 scans: lift_label must match GT on > 90% of visible food.

    python tests/test_lift_synthetic.py        (also collected by pytest)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _lift_fixture import load, scene_dirs  # noqa: E402

from foodscan.labels import lift_scan, load_lift  # noqa: E402


def _food_agreement(r, gt):
    vis = (gt >= 1) & (gt < 253) & (r.lift_nvotes >= 2)
    return float((r.lift_label[vis] == gt[vis]).mean()), int(vis.sum())


def test_lift_matches_gt():
    for scene in scene_dirs():
        scan, xyz, gt, masks, nrm = load(scene)
        r = lift_scan(scan, masks, xyz, normals=nrm)
        acc, n = _food_agreement(r, gt)
        print(f"{scene.name}: food agreement {acc:.3f} on {n} visible pts, "
              f"repair merges {r.stats['repair']['merges']}, agree frac {r.lift_agree[r.lift_nvotes > 0].mean():.3f}")
        assert acc > 0.90, (scene.name, acc)
        assert r.stats["repair"]["merges"] == 0           # consistent oracle ids must not merge
        cont = (gt == 254) & (r.lift_nvotes >= 2)
        assert (r.lift_label[cont] == 254).mean() > 0.5   # containers survive as class 254


def test_dtypes_and_roundtrip():
    scene = scene_dirs(1)[0]
    scan, xyz, gt, masks, nrm = load(scene)
    r = lift_scan(scan, masks, xyz, normals=nrm)
    assert r.lift_label.dtype == np.int32 and r.lift_share.dtype == np.float32
    assert r.lift_nvotes.dtype == np.int16 and r.lift_agree.dtype == np.uint8
    assert ((r.lift_share >= 0) & (r.lift_share <= 1)).all()
    with tempfile.TemporaryDirectory() as d:
        r.save(Path(d) / "lift.npz", points=xyz, normals=nrm, meta={1: {"name": "x"}})
        z = load_lift(Path(d) / "lift.npz")
    for k, dt in [("lift_label", np.int32), ("lift_share", np.float32), ("lift_nvotes", np.int16),
                  ("lift_agree", np.uint8), ("xyz", np.float32)]:
        assert z[k].dtype == dt and len(z[k]) == len(xyz), k
    assert z["meta"]["container_label"] == 254 and z["meta"]["instances"]["1"]["name"] == "x"


def test_smoothing_flag_default_off_and_runs():
    scene = scene_dirs(1)[0]
    scan, xyz, gt, masks, nrm = load(scene)
    sub = np.arange(0, len(xyz), 4)
    a = lift_scan(scan, masks, xyz[sub], normals=nrm[sub])
    b = lift_scan(scan, masks, xyz[sub], normals=nrm[sub], smooth_radius=0.004)
    acc_a, _ = _food_agreement(a, gt[sub])
    acc_b, _ = _food_agreement(b, gt[sub])
    print(f"smoothing: off {acc_a:.3f} on {acc_b:.3f}")
    assert acc_b > 0.85


def test_id_break_is_repaired():
    """Tracker break: id 2 becomes id 7 in odd frames. Repair must merge it back (1 merge)."""
    scene = scene_dirs(1)[0]
    scan, xyz, gt, masks, nrm = load(scene)
    broken = {}
    for i, (fi, m) in enumerate(sorted(masks.items())):
        m = m.copy()
        if i % 2:
            m[m == 2] = 7
        broken[fi] = m
    meta = {1: {"concept_index": 0}, 2: {"concept_index": 0}, 3: {"concept_index": 1}, 7: {"concept_index": 0}}
    off = lift_scan(scan, broken, xyz, normals=nrm, repair=False)
    on = lift_scan(scan, broken, xyz, normals=nrm, meta=meta, repair=True)
    a_off, _ = _food_agreement(off, gt)
    a_on, _ = _food_agreement(on, gt)
    print(f"id break: no repair {a_off:.3f}, repair {a_on:.3f}, merges {on.stats['repair']['merges']} {on.merge_map}")
    assert on.stats["repair"]["merges"] >= 1 and on.merge_map.get(7) == 2
    assert not (on.lift_label == 7).any()
    assert a_on > 0.90


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
