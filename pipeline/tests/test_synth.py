"""synth package without Blender: exact volumes, LiDAR model, spec -> finalize -> loader round trip.

    cd pipeline && python -m pytest tests/test_synth.py -q
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from synth.geometry import box_mesh, ellipsoid_mesh, fill_mesh, mesh_volume, sample_container
from synth.lidar import LidarParams, simulate


def test_fill_matches_analytic_capacity():
    rng = np.random.default_rng(1)
    for kind in ("bowl", "plate"):
        for _ in range(4):
            c = sample_container(rng, kind)
            h = c.floor + 0.6 * (c.H - c.floor)
            V, F = fill_mesh(c, h, mound=0.0, lift=0.0)
            assert abs(mesh_volume(V, F) * 1e6 - c.capacity_ml(h)) / c.capacity_ml(h) < 0.01
            Vc, Fc = c.mesh()
            assert mesh_volume(Vc, Fc) > 0


def test_primitives():
    V, F = box_mesh(0.06, 0.04, 0.03)
    assert abs(mesh_volume(V, F) * 1e6 - 72.0) < 1e-6
    V, F = ellipsoid_mesh(0.03, 0.02, 0.01, n=64)
    assert abs(mesh_volume(V, F) / (4 / 3 * np.pi * 0.03 * 0.02 * 0.01) - 1) < 0.005


def test_lidar_shapes_and_noise_level():
    rng = np.random.default_rng(0)
    d = np.full((360, 480), 0.30, np.float32)
    d[150:210, 200:280] = 0.27                       # a 3 cm step
    rgb = np.full((360, 480, 3), 128, np.uint8)
    K = np.array([[360.0, 0, 240], [0, 360.0, 180], [0, 0, 1]])
    out = simulate(d, rgb, K, rng, LidarParams())
    assert out["depth"].shape == (192, 256) and out["conf"].dtype == np.uint8
    flat = out["conf"] == 2
    assert flat.mean() > 0.8
    err = (out["depth"] - out["clean"])[flat]
    assert 0.0003 < err.std() < 0.005


def _fake_raw(spec: dict, raw, rng):
    """Stand-in for render.py: flat table depth, one item visible in the centre."""
    cam = spec["camera"]
    w, h = cam["width"], cam["height"]
    raw.mkdir(parents=True)
    for i in range(len(cam["c2w"])):
        seg = np.zeros((h, w), np.uint8)
        seg[h // 3:2 * h // 3, w // 3:2 * w // 3] = 254
        seg[2 * h // 5:3 * h // 5, 2 * w // 5:3 * w // 5] = 1
        np.savez_compressed(raw / f"frame_{i:06d}.npz", rgb=rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
                            depth=np.full((h, w), 0.3, np.float32), seg=seg)
    items = [{"id": it["id"], "local2world": np.eye(4).tolist(), "scene_volume_ml": it["volume_ml"],
              "radius_from_axis_m": 0.0} for it in spec["items"]]
    (raw / "render_meta.json").write_text(json.dumps({"spp": 1, "timing_s": {}, "items": items}))


def test_spec_finalize_loader_roundtrip(tmp_path):
    from foodscan.loader import Scan
    from synth.assets import build_primitives
    from synth.finalize import finalize
    from synth.spec import sample_scene

    assets = build_primitives(tmp_path / "prims", n_per=1)
    rng = np.random.default_rng(3)
    spec = sample_scene(rng, assets, tmp_path / "spec", "s000003", n_views=6, res_scale=0.25)
    _fake_raw(spec, tmp_path / "raw", rng)
    gt = finalize(tmp_path / "spec", tmp_path / "raw", tmp_path / "scan")

    scan = Scan(tmp_path / "scan")
    assert len(scan.good_frames()) == 6
    f = scan.frames[0]
    assert f.depth().shape == (192, 256) and f.confidence().shape == (192, 256)
    assert f.rgb().shape == (360, 480, 3)
    # ARKit world is Y up: cameras orbit above the table, i.e. higher Y than the table point
    table_y = gt["table_plane_arkit"]["point"][1]
    assert all(fr.T_cw[1, 3] > table_y + 0.1 for fr in scan.frames)
    assert np.allclose(gt["table_plane_arkit"]["normal"], [0, 1, 0], atol=1e-9)
    # OpenCV optical axis (+Z) points down toward the table on the top-down frames
    assert scan.frames[-1].T_cw[1, 2] < -0.9
    m = np.asarray(Image.open(tmp_path / "scan" / "gt" / "masks" / "000000.png"))
    assert set(np.unique(m)) == {0, 1}
    assert np.load(tmp_path / "scan" / "gt" / "container.npz")["points"].shape[1] == 3
