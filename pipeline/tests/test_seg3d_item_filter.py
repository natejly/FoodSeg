import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import augment as A                                     # noqa: E402
from seg3d import eval as E                                        # noqa: E402
from seg3d.data import DataConfig, FoodSegDataset, item_status     # noqa: E402


def make_scan(root: Path, dropped=None):
    """Scan with food items 1 (400 pts), 2 (30 pts, tiny), 3 (300 pts) + table + container, lift = GT."""
    sd = root / "scans" / "s000000"
    (sd / "gt").mkdir(parents=True)
    rng = np.random.default_rng(0)
    counts = {1: 400, 2: 30, 3: 300, 252: 800, 254: 300}
    lab = np.concatenate([np.full(c, k) for k, c in counts.items()]).astype(np.int32)
    n = len(lab)
    xyz = rng.uniform(-0.05, 0.05, (n, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(0, 0.02, n)
    for k, cx in ((1, -0.03), (2, 0.0), (3, 0.03)):               # separate the items in x
        xyz[lab == k, 0] = cx + rng.normal(0, 0.003, (lab == k).sum())
    np.savez(sd / "gt" / "seg_points.npz", xyz=xyz, rgb=rng.uniform(0, 1, (n, 3)).astype(np.float32),
             normal=np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1)), label=lab, dist=np.zeros(n, np.float32),
             lift_label=lab, lift_share=np.ones(n, np.float32), lift_nvotes=np.full(n, 5, np.int16))
    (sd / "gt.json").write_text(json.dumps(dict(items=[dict(id=i, category=f"c{i}", volume_ml=10.0) for i in (1, 2, 3)],
                                                table_plane_arkit=dict(normal=[0, 0, 1], point=[0, 0, 0]))))
    rec = dict(scan="s000000", ok=False)
    if dropped is not None:
        rec["dropped_items"] = dropped
    (root / "manifest.jsonl").write_text(json.dumps(rec) + "\n")
    return sd, lab


def test_item_status_modes(tmp_path):
    sd, lab = make_scan(tmp_path, dropped=[3])
    assert item_status(sd, lab, "manifest") == ([1, 2], [3])          # manifest wins even if item 2 is tiny
    assert item_status(sd, lab, "points", 50) == ([1, 3], [2])
    assert item_status(sd, lab, "none") == ([1, 2, 3], [])
    sd2, lab2 = make_scan(tmp_path / "b", dropped=None)               # manifest without dropped_items -> points rule
    assert item_status(sd2, lab2, "manifest", 50) == ([1, 3], [2])


def test_loss_weight_and_metrics_ignore(tmp_path):
    sd, lab = make_scan(tmp_path, dropped=[2])
    cfg = DataConfig(roots=[], item_filter="manifest", voxel_mm=2.0)
    ds = FoodSegDataset([sd], cfg, A.eval_aug("r1"), train=False)
    s = ds[0]
    tiny = s["label"] == 2
    assert tiny.any() and s["ignore"][tiny].all() and not s["ignore"][~tiny].any()
    assert (s["weight"][tiny] == 0).all() and (s["weight"][~tiny & (s["boundary"] == 0)] == 1).all()
    # perfect prediction on the kept items: with the filter the tiny item is not a GT item, without it it is
    from seg3d import metrics as M
    for mode, n_items in (("manifest", 2), ("none", 3)):
        cfg2 = DataConfig(roots=[], item_filter=mode, voxel_mm=2.0)
        s2 = FoodSegDataset([sd], cfg2, A.eval_aug("r1"), train=False)[0]
        sel = ~s2["ignore"]
        acc = M.Accumulator(volumes=False)
        ps, pi = M.lift_baseline_pred(s2["lift"])
        acc.add(ps[sel], pi[sel], s2["xyz"][sel], s2["sem"][sel], s2["inst"][sel], s2["label"][sel], s2["items"], lift=s2["lift"][sel])
        r = acc.result()
        assert r["n_items"] == n_items and abs(r["inst_miou"] - 1.0) < 1e-9
