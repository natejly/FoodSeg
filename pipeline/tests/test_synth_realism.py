"""Blender-free checks of the realism / composition changes (synth.lidar, synth.spec_hard, synth.composition).

    cd pipeline && python -m pytest tests/test_synth_realism.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth.lidar import D_H, D_W, LidarParams, PoseNoiseParams, perturb_poses, simulate, translucent_ids, world_bias

ASSETS = Path("/home/kyle/Desktop/FoodSeg/data/synth/assets_v3.jsonl")


def _flat_scene(tx: float):
    z = 0.35
    depth = np.full((D_H, D_W), z, np.float32)
    rgb = np.full((D_H, D_W, 3), 128, np.uint8)
    K = np.array([[200.0, 0, D_W / 2], [0, 200.0, D_H / 2], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, 3] = [tx, 0.0, z]               # looks straight down (-Z) at the table
    return depth, rgb, K, c2w


def _quiet(**kw) -> LidarParams:
    base = dict(sigma0_m=1e-7, sigma_z2=0.0, lateral_px=0.0, noise_blur_px=0.0, bias_amp_m=0.003,
                bias_innov_frac=0.0, p_drop_dark=0.0, p_drop_grazing=0.0, p_drop_spec=0.0, bleed_px=0.0)
    base.update(kw)
    return LidarParams(**base)


def test_bias_field_is_persistent_in_world_frame():
    p = _quiet(field_seed=5)
    dA, rA, K, cA = _flat_scene(0.0)
    dB, rB, _, cB = _flat_scene(0.007)           # 4 depth pixels at 0.35 m, fx = 200
    A = simulate(dA, rA, K, np.random.default_rng(1), p, c2w=cA)
    B = simulate(dB, rB, K, np.random.default_rng(2), p, c2w=cB)
    eA, eB = A["depth"] - A["clean"], B["depth"] - B["clean"]
    assert eA.std() > 0.5e-3                      # the field is there (mm scale)
    np.testing.assert_allclose(eA[:, 4:], eB[:, :-4], atol=1e-5)
    # legacy per-frame field is not shared between frames
    q = _quiet(persistent=False, bias_sigma_m=0.003)
    a = simulate(dA, rA, K, np.random.default_rng(1), q, c2w=cA)
    b = simulate(dB, rB, K, np.random.default_rng(2), q, c2w=cB)
    assert np.abs((a["depth"] - a["clean"])[:, 4:] - (b["depth"] - b["clean"])[:, :-4]).max() > 5e-4


def test_world_bias_has_two_scales():
    p = LidarParams(field_seed=3)
    x = np.linspace(-0.1, 0.1, 400)
    pts = np.stack([x, np.zeros_like(x), np.full_like(x, 0.05)], -1)
    b = world_bias(pts, p)
    assert 0.3 < b.std() < 2.0 and np.abs(np.diff(b)).max() < 3 * b.std()      # smooth, unit-ish rms


def test_translucent_bias_and_dropout_options():
    d, rgb, K, c = _flat_scene(0.0)
    seg = np.zeros((D_H, D_W), np.uint8)
    seg[:, :100] = 7
    p = _quiet(translucent_bias_m=0.003, bias_amp_m=0.0)
    s = simulate(d, rgb, K, np.random.default_rng(0), p, c2w=c, seg=seg, translucent=[7])
    e = s["depth"] - s["clean"]
    assert abs(e[:, :90].mean() - 0.003) < 2e-4 and abs(e[:, 110:].mean()) < 2e-4
    # specular highlights drop out and are smeared (no NaN, conf 0)
    hot = rgb.copy()
    hot[50:80, 50:80] = 255
    s = simulate(d, hot, K, np.random.default_rng(0), _quiet(p_drop_spec=1.0, bias_amp_m=0.0), c2w=c)
    assert (s["conf"][55:75, 55:75] == 0).all() and np.isfinite(s["depth"]).all()
    # sampled params stay in range and are recorded as a plain dict
    from dataclasses import asdict
    q = LidarParams.sample(np.random.default_rng(4))
    dd = asdict(q)
    assert 0.0008 <= dd["sigma0_m"] <= 0.003 and 1.0 <= dd["noise_blur_px"] <= 1.5 and 0.5 <= dd["drop_mult"] <= 2.0
    assert translucent_ids([{"id": 1, "category": "yogurt"}, {"id": 2, "category": "steak"}]) == [1]


def test_pose_drift():
    rng = np.random.default_rng(0)
    T = []
    for i in range(40):
        t = np.eye(4)
        t[:3, 3] = [0.3 * np.cos(i / 6), 0.3 * np.sin(i / 6), 0.25]
        T.append(t)
    out = perturb_poses(T, rng, PoseNoiseParams(jump_p=0.0))
    assert len(out) == len(T) and np.allclose(out[0], T[0])
    steps = [np.linalg.norm(out[i][:3, 3] - T[i][:3, 3] - (out[i - 1][:3, 3] - T[i - 1][:3, 3])) for i in range(1, 40)]
    assert 1e-4 < np.mean(steps) < 5e-3                          # mm-scale steps
    R = [out[i][:3, :3] @ T[i][:3, :3].T for i in range(40)]
    assert all(np.allclose(r @ r.T, np.eye(3), atol=1e-9) for r in R)           # rotations stay rotations
    jumpy = perturb_poses(T, np.random.default_rng(1), PoseNoiseParams(jump_p=0.5, trans_step_m=0.0, rot_step_deg=0.0))
    assert max(np.linalg.norm(jumpy[i][:3, 3] - T[i][:3, 3]) for i in range(40)) > 0.005
    off = perturb_poses(T, rng, PoseNoiseParams(enabled=False))
    assert all(np.allclose(a, b) for a, b in zip(off, T))
    s = PoseNoiseParams.sample(rng)
    assert 0.0005 <= s.trans_step_m <= 0.003 and 0.05 <= s.rot_step_deg <= 0.2


def test_composition_mappings_and_coverage():
    from synth.composition import INGREDIENTS, resolve, visible
    m = resolve({"cheese cube", "meatloaf", "pork chop"})
    assert INGREDIENTS["corn"][0] == "fill" and INGREDIENTS["noodles"] == ("fill", "spaghetti")
    assert INGREDIENTS["eggplant"][0] == "fill" and INGREDIENTS["eggplant"][1] != "roasted potato"
    assert INGREDIENTS["ham"][0] == "asset"
    pizza = visible([("cheese pizza", 100.0), ("mozzarella cheese", 20.0)], m)
    assert [x[2] for x in pizza if x[0] == "mozzarella cheese"] == [("topping", "melted cheese")]
    salad = visible([("lettuce", 50.0), ("mozzarella cheese", 20.0), ("arugula", 20.0)], m)
    assert [x[2][0] for x in salad] == ["leaf", "asset", "leaf"]


@pytest.mark.skipif(not ASSETS.exists(), reason="needs data/synth/assets_v3.jsonl")
def test_spec_hard_composition(tmp_path):
    from synth.assets import load_index
    from synth.spec_hard import sample_scene_hard
    assets = load_index(ASSETS)
    fills, unf = [], 0
    for i in range(12):
        sp = sample_scene_hard(np.random.default_rng([7, i]), assets, tmp_path / f"s{i}", f"s{i}", n_views=4)
        assert "meal_unfulfilled" in sp["meal"] and 0.0 <= sp["fill_fraction"] <= 1.0
        ids = [it["id"] for it in sp["items"]]
        assert ids == list(range(1, len(ids) + 1)) and max(ids) < 252
        assert all(it["mass_g"] for it in sp["items"] if it["source"] == "procedural_piece")
        fills.append(sp["fill_fraction"])
        unf += bool(sp["meal"]["meal_unfulfilled"])
    assert np.mean(fills) > 0.4
