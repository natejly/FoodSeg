import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import augment as A                                     # noqa: E402
from seg3d.data import FEAT_DIM, DataConfig, FoodSegDataset, list_scans, split_scans, voxelize  # noqa: E402


def toy_scene(rng, n=6000):
    xyz = rng.uniform(-0.1, 0.1, (n, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(0, 0.03, n)
    label = np.zeros(n, np.int32)
    label[xyz[:, 0] > 0.03] = 1
    label[(xyz[:, 0] < -0.03)] = 2
    label[(np.abs(xyz[:, 1]) > 0.08)] = 254
    label[(xyz[:, 1] < -0.09)] = 253
    return dict(xyz=xyz, rgb=rng.uniform(0, 1, (n, 3)).astype(np.float32),
                normal=np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1)), label=label)


def test_scale_bound_and_rigid():
    rng = np.random.default_rng(0)
    d = toy_scene(rng)
    c = A.AugConfig(crop_p=0, cap_p=0, jitter=False, flip=False)
    for _ in range(50):
        o = A.geometric(dict(d), rng, c)
        assert 0.95 - 1e-6 <= o["scale"] <= 1.05 + 1e-6
        # pairwise distance ratio equals scale (rotation/tilt are rigid)
        r = np.linalg.norm(o["xyz"][:50] - o["xyz"][1:51], axis=1) / np.linalg.norm(d["xyz"][:50] - d["xyz"][1:51], axis=1)
        assert np.allclose(r, o["scale"], atol=1e-3)
        assert np.allclose(np.linalg.norm(o["normal"], axis=1), 1, atol=1e-4)
        assert abs(o["normal"][:, 2].min()) > np.cos(np.radians(3.5)) - 1e-3       # tilt <= 3 deg


def test_jitter_clip_and_crop_cap():
    rng = np.random.default_rng(1)
    d = toy_scene(rng)
    o = A.geometric(dict(d), rng, A.AugConfig(rot_z=False, tilt_deg=0, scale=0, flip=False, crop_p=0, cap_p=0))
    assert np.abs(o["xyz"] - d["xyz"]).max() <= 3e-3 + 1e-6
    o = A.geometric(dict(d), rng, A.AugConfig(crop_p=1, cap_p=0, crop_radius_m=(0.05, 0.05), jitter=False, scale=0))
    assert len(o["xyz"]) < len(d["xyz"])
    o = A.geometric(dict(d), rng, A.AugConfig(crop_p=0, cap_p=1, cap_frac=(0.3, 0.3)))
    assert 0.6 < len(o["xyz"]) / len(d["xyz"]) < 0.8


def test_lift_synth_and_corrupt():
    rng = np.random.default_rng(2)
    d = toy_scene(rng)
    lift, share, nv = A.synth_lift(d["label"], rng)
    assert lift.shape == share.shape == nv.shape == d["label"].shape
    assert set(np.unique(lift)) <= {0, 1, 2, 253, 254}
    c = A.AugConfig(blob_p=1, merge_p=1, drop_inst_p=0, morph_p=1)
    l2, s2, n2 = A.corrupt_lift(d["xyz"], lift, share, nv, rng, c, 0.002)
    assert (l2 != lift).any() and s2.min() >= 0 and s2.max() <= 1
    assert ((l2 == 253) == (lift == 253)).all()            # clutter/container untouched


def test_voxelize_unique():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 0.05, (5000, 3))
    idx, g = voxelize(x, 0.005, rng)
    assert len(np.unique(g, axis=0)) == len(idx) <= 1000


def test_dataset_real_shapes():
    scans = list_scans(DataConfig().roots)
    if not scans:
        pytest.skip("no synthetic scans")
    cfg = DataConfig()
    tr, va = split_scans(scans, cfg)
    assert set(map(str, tr)).isdisjoint(map(str, va)) and va
    ds = FoodSegDataset(tr, cfg, A.AugConfig(), train=True)
    s = ds[0]
    n = len(s["coord"])
    assert s["feat"].shape == (n, FEAT_DIM) and s["grid_coord"].shape == (n, 3) and s["offset"].shape == (n, 3)
    assert s["sem"].max() <= 3 and (s["offset"][s["sem"] != 3] == 0).all()
    assert (s["grid_coord"] >= 0).all()
    dv = FoodSegDataset(va, cfg, A.AugConfig(enable=False, lift_corrupt=True), train=False)
    a, b = dv[0], dv[0]
    assert np.array_equal(a["lift"], b["lift"]) and np.allclose(a["coord"], b["coord"])      # deterministic val
