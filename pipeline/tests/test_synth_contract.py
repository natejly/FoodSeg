"""Data-contract tests: pseudo-SAM corruption and atomic finalize."""
import json

import numpy as np
import pytest

from synth import votes as V


def _masks():
    m = np.zeros((192, 256), np.uint8)
    m[40:100, 40:100] = 1
    m[40:100, 100:160] = 2      # touches 1
    m[120:180, 60:200] = 254
    return {i: m.copy() for i in range(6)}


def test_pseudo_sam_is_seeded_and_keeps_ids():
    a, pa = V.pseudo_sam(_masks(), 5, [1, 2])
    b, pb = V.pseudo_sam(_masks(), 5, [1, 2])
    assert pa == pb and all((a[k] == b[k]).all() for k in a)
    for m in a.values():
        assert m.shape == (192, 256) and set(np.unique(m)) <= {0, 1, 2, 3, 254}
    assert any((a[k] != _masks()[k]).any() for k in a)       # boundary jitter always applies


def test_pseudo_sam_rates():
    n = {"merge": 0, "container_fp": 0}
    for s in range(200):
        _, p = V.pseudo_sam(_masks(), s, [1, 2])
        n["merge"] += p["merge"] is not None
        n["container_fp"] += p["container_fp"] is not None
    assert 0.1 < n["merge"] / 200 < 0.3 and 0.2 < n["container_fp"] / 200 < 0.4


def test_finalize_is_atomic(tmp_path, monkeypatch):
    from synth import finalize as F
    out = tmp_path / "scans" / "s000001"

    def boom(scene, raw, tmp, *a, **k):
        tmp.mkdir(parents=True)
        (tmp / "rgb.jpg").write_text("partial")
        raise RuntimeError("mid-write failure")

    monkeypatch.setattr(F, "_finalize", boom)
    with pytest.raises(RuntimeError):
        F.finalize(tmp_path, tmp_path, out)
    assert not out.exists() and not (tmp_path / "scans" / "s000001.tmp").exists()


def test_finalize_writes_gt_last(tmp_path):
    from test_synth import _fake_raw
    from synth.assets import build_primitives
    from synth.finalize import finalize
    from synth.spec import sample_scene
    rng = np.random.default_rng(3)
    spec = sample_scene(rng, build_primitives(tmp_path / "prims", n_per=1), tmp_path / "spec", "s000003",
                        n_views=4, res_scale=0.25)
    _fake_raw(spec, tmp_path / "raw", rng)
    gt = finalize(tmp_path / "spec", tmp_path / "raw", tmp_path / "scans" / "s000003", seed=3)
    d = tmp_path / "scans" / "s000003"
    assert (d / "gt.json").exists() and not (tmp_path / "scans" / "s000003.tmp").exists()
    assert (d / "gt" / "seg_ids" / "000000.png").exists()
    assert gt["provenance"]["seed"] == 3 and gt["provenance"]["n_views"] == 4
    assert all(it["mass_g"] and it["mass_source"] for it in gt["items"])
    assert gt["provenance"]["generator_version"] == "synth-2.2"
    on_disk = json.loads((d / "gt.json").read_text())
    for it in on_disk["items"]:
        for k in ("volume_envelope_ml", "hull_volume_ml", "envelope_ratio", "bulk_density_g_ml"):
            assert k in it


def test_finalize_pose_noise_and_sampled_lidar(tmp_path):
    from test_synth import _fake_raw
    from synth.assets import build_primitives
    from synth.finalize import finalize
    from synth.spec import sample_scene
    rng = np.random.default_rng(3)
    spec = sample_scene(rng, build_primitives(tmp_path / "prims", n_per=1), tmp_path / "spec", "s000003",
                        n_views=6, res_scale=0.25)
    _fake_raw(spec, tmp_path / "raw", rng)
    gt = finalize(tmp_path / "spec", tmp_path / "raw", tmp_path / "a", seed=3)
    gt2 = finalize(tmp_path / "spec", tmp_path / "raw", tmp_path / "b", seed=3, pose_noise=False)
    assert gt["pose_noise"]["trans_step_m"] > 0 and gt2["pose_noise"] is None
    assert 0.0008 <= gt["lidar_params"]["sigma0_m"] <= 0.003 and "field_seed" in gt["lidar_params"]
    W2A = np.array(spec["world_to_arkit"])
    err = []
    for line, c in zip((tmp_path / "a" / "frames.jsonl").read_text().splitlines(), spec["camera"]["c2w"]):
        T = np.array(json.loads(line)["transform"])
        err.append(np.linalg.norm(T[:3, 3] - (W2A @ np.array(c))[:3, 3]))
    assert max(err) > 1e-4 and max(err) < 0.05
    for line, c in zip((tmp_path / "b" / "frames.jsonl").read_text().splitlines(), spec["camera"]["c2w"]):
        assert np.allclose(np.array(json.loads(line)["transform"]), W2A @ np.array(c))


def test_run_render_cmd_kills_whole_process_group(tmp_path):
    import os
    import subprocess
    import time
    from synth.batch import _run_render_cmd
    marker = f"sleep {300 + os.getpid() % 1000}"
    sh = tmp_path / "fake_render.sh"
    sh.write_text(f"#!/bin/bash\n{marker} &\necho started\nsleep 300\n")
    t0 = time.time()
    rc, out, timed_out = _run_render_cmd(["bash", str(sh)], timeout=2, grace=5)
    assert timed_out and time.time() - t0 < 30 and "started" in out
    time.sleep(0.3)
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()
    assert not alive, f"orphaned child survived: {alive}"
    rc, out, timed_out = _run_render_cmd(["bash", "-c", "echo hi; echo err >&2; exit 3"], timeout=10)
    assert (rc, timed_out) == (3, False) and "hi" in out and "err" in out


def _surface_cloud(rng, n):
    """Sphere cap (label 1), box top + sides (label 2), container bowl (254): constrains all 6 DOF."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    sph = 0.04 * u + [0.0, 0.0, 0.04]
    sph = sph[sph[:, 2] > 0.035]
    box = rng.uniform(0, 1, (n, 3)) * [0.05, 0.03, 0.02]
    face = rng.integers(0, 3, n)
    box[np.arange(n), face] = rng.integers(0, 2, n) * np.array([0.05, 0.03, 0.02])[face]
    box += [0.06, -0.02, 0.0]
    bowl = 0.09 * u + [0.0, 0.0, 0.09]
    bowl = bowl[bowl[:, 2] < 0.05]
    xyz = np.vstack([sph, box, bowl])
    lab = np.concatenate([np.full(len(sph), 1), np.full(len(box), 2), np.full(len(bowl), 254)]).astype(np.uint8)
    return xyz, lab


def test_seg_export_icp_recovers_rigid_misalignment():
    pytest.importorskip("open3d")
    from synth.seg_export import align_and_label
    rng = np.random.default_rng(0)
    gxyz, glab = _surface_cloud(rng, 60000)
    fxyz, _ = _surface_cloud(np.random.default_rng(1), 60000)       # independent samples of the same surfaces
    th = np.deg2rad(0.3)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    c = gxyz.mean(0)
    t = np.array([0.003, -0.0015, 0.0015])                           # |t| = 3.8 mm
    fused = (fxyz - c) @ R.T + c + t
    r = align_and_label(fused, np.zeros_like(fused), gxyz, glab)
    resid = (fused @ r["T"][:3, :3].T + r["T"][:3, 3]) - fxyz         # corrected vs the unperturbed truth
    assert np.linalg.norm(np.median(resid, axis=0)) < 0.0005
    assert np.abs(resid).max() < 0.0015
    truth, tl = _surface_cloud(np.random.default_rng(1), 60000)
    agree = (r["label"] == tl).mean()
    assert agree > 0.98, agree
    assert 0.004 <= r["tol"] <= 0.008 and r["icp"]["post"] < r["icp"]["pre"]
    r0 = align_and_label(fxyz, np.zeros_like(fxyz), gxyz, glab)       # clean cloud: ~identity
    assert np.linalg.norm(r0["T"][:3, 3]) < 0.0005


def _fake_scan(root, name, pn, pre_mm, post_mm, n_views, n_pts=40):
    d = root / name
    (d / "gt").mkdir(parents=True)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 0.05, (200, 3))
    np.savez(d / "gt" / "points.npz", xyz=xyz, label=np.ones(200, np.uint8))
    lab = np.concatenate([np.ones(n_pts, np.uint8), np.zeros(10, np.uint8)])
    np.savez(d / "gt" / "seg_points.npz", label=lab, dist=np.full(len(lab), post_mm / 1000, np.float32))
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "rgb.jpg") if False else None
    gt = {"items": [{"id": 1, "category": "a", "off_container": False, "frames_visible": 5, "volume_ml": 10,
                     "scene_volume_ml": 10, "shape": {"kind": "single"}}],
          "world_to_arkit": np.eye(4).tolist(), "container": {"R": 0.1, "kind": "plate"},
          "total_volume_ml": 10, "provenance": {"n_views": n_views}, "pose_noise": pn,
          "lidar_params": {"sigma0_m": 0.0012},
          "seg_export": {"median_pre_mm": pre_mm, "median_post_mm": post_mm, "tol": 0.004}}
    (d / "gt.json").write_text(json.dumps(gt))
    return d


def test_manifest_pose_aware_flags(tmp_path, monkeypatch):
    from synth import manifest as M
    monkeypatch.setattr(M, "_item_colours", lambda scan, n: {})
    pn = {"trans_step_m": 0.0015, "momentum": 0.6, "jump_p": 0.05, "jump_m": [0.005, 0.01], "rot_step_deg": 0.1}
    b = M.drift_bound(pn, 8)
    assert 0.004 < b < 0.05 and M.drift_bound(pn, 32) > b
    # drifting scan: big pre-ICP, small post-ICP -> ok, not misaligned
    r = M.describe(_fake_scan(tmp_path, "a", pn, 3.0, 0.8, 8))
    assert r["ok"] and "fusion_misaligned" not in r["qc"] and "pose_drift_large" not in r["qc"]
    assert r["n_views"] == 8 and abs(r["lidar_sigma0_mm"] - 1.2) < 1e-9
    # pre-ICP above the bound -> informational pose_drift_large; ICP failing -> dropped
    r = M.describe(_fake_scan(tmp_path, "b", pn, b * 1000 + 5, 0.8, 8))
    assert "pose_drift_large" in r["qc"] and r["ok"]
    r = M.describe(_fake_scan(tmp_path, "c", pn, 6.0, 3.0, 8))
    assert "fusion_misaligned" in r["qc"] and not r["ok"]
    # no pose_noise in gt.json (legacy / clean): no drift flag, post threshold still applies
    r = M.describe(_fake_scan(tmp_path, "d", None, 30.0, 0.8, 8))
    assert "pose_drift_large" not in r["qc"] and r["drift_bound_mm"] is None and r["ok"]
    r = M.describe(_fake_scan(tmp_path, "e", None, 3.0, 2.5, 8))
    assert "fusion_misaligned" in r["qc"]


def test_manifest_tiny_item_scales_with_views(tmp_path, monkeypatch):
    from synth import manifest as M
    monkeypatch.setattr(M, "_item_colours", lambda scan, n: {})
    assert M.min_points_threshold(24) == 50 and M.min_points_threshold(48) == 50 and abs(M.min_points_threshold(8) - 50 / 3) < 1e-9
    r = M.describe(_fake_scan(tmp_path, "f", None, 1, 1, 8, n_pts=20))     # 20 >= 16.7 at 8 views
    assert "tiny_item" not in r["qc"] and abs(r["min_item_points_threshold"] - 50 / 3) < 1e-9
    r = M.describe(_fake_scan(tmp_path, "g", None, 1, 1, 24, n_pts=20))
    assert "tiny_item" in r["qc"]


def _sphere_pts(c, r, n, rng):
    v = rng.normal(size=(n, 3))
    return np.asarray(c) + r * v / np.linalg.norm(v, axis=1, keepdims=True)


def test_envelope_hemisphere_matches_analytic():
    from synth.itemstats import envelope_volume
    rng = np.random.default_rng(0)
    r = 0.04
    p = _sphere_pts([0, 0, 0], r, 400000, rng)
    p = p[p[:, 2] >= 0]
    env, hull = envelope_volume(p, lambda xy, zmin: np.zeros(len(xy)))
    assert env == pytest.approx(2 / 3 * np.pi * r ** 3 * 1e6, rel=0.03)
    assert hull == pytest.approx(2 / 3 * np.pi * r ** 3 * 1e6, rel=0.03)


def test_envelope_sphere_pile_exceeds_solid():
    from synth.itemstats import envelope_volume
    rng = np.random.default_rng(1)
    r, d = 0.01, 0.022
    pts = np.concatenate([_sphere_pts([x * d, y * d, r + z * d], r, 20000, rng)
                          for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    solid = 8 * 4 / 3 * np.pi * r ** 3 * 1e6
    env, hull = envelope_volume(pts, lambda xy, zmin: np.zeros(len(xy)))
    assert env / solid > 1.3
    assert hull / solid > env / solid
