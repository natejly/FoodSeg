"""Headless tests for tools.label_items on hard_v10 synthetic scans (skipped if the data is absent)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import label_items as L                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "data" / "synth" / "hard_v10" / "scans"
SCANS = sorted(ROOT.glob("s*"))[:6] if ROOT.exists() else []
pytestmark = pytest.mark.skipif(not SCANS, reason="hard_v10 synthetic scans not available")


def gt_seeds(scan, rng):
    gt = scan["gt_label"]
    items = []
    for i in np.unique(gt[(gt > 0) & (gt < 252)]):
        pts = np.nonzero(gt == i)[0]
        items.append({"name": scan.get("gt_names", {}).get(int(i), f"item{i}"), "seeds": [int(rng.choice(pts))], "gt": int(i)})
    return items


def run_scan(path, grow, tmp):
    scan = L.load_scan(path)
    rng = np.random.default_rng(0)
    items = gt_seeds(scan, rng)
    sess = L.Session(scan, grow=grow)
    sess.apply_seeds_spec({"items": [{k: v for k, v in it.items() if k != "gt"} for it in items]})
    sess.save(tmp)
    return scan, sess


def test_lift_grow_matches_gt(tmp_path):
    best, nums = 0.0, []
    for p in SCANS:
        scan, sess = run_scan(p, "lift", tmp_path / p.name)
        label, names = sess.result()
        r = L.score_labels(scan["xyz"], scan["gt_label"], label)
        nums.append(round(r["inst_miou"], 3))
        best = max(best, r["inst_miou"])
        assert len(names) == r["n_ref"]
    print("lift-grow inst mIoU vs GT per scan:", nums)
    assert best > 0.9


def test_geodesic_grow_runs(tmp_path):
    nums = []
    for p in SCANS[:3]:
        scan, sess = run_scan(p, "geodesic", tmp_path / p.name)
        label, _ = sess.result()
        nums.append(L.score_labels(scan["xyz"], scan["gt_label"], label)["inst_miou"])
        assert (L._food(label) > 0).any()
    print("geodesic inst mIoU vs GT:", np.round(nums, 3))
    assert np.nanmax(nums) > 0.5


def test_outputs_and_score(tmp_path):
    p = SCANS[0]
    scan, sess = run_scan(p, "lift", tmp_path)
    z = np.load(tmp_path / "labels_manual.npz")
    assert z["label"].dtype == np.int32 and len(z["label"]) == len(scan["xyz"])
    for k in ("names", "seeds", "method", "timestamp", "annotator"):
        assert k in z.files
    assert json.loads(str(z["names"]))
    summ = json.loads((tmp_path / "labels_manual.json").read_text())
    assert summ["n_items"] == len(json.loads(str(z["names"])))
    # score helper (CLI path) against lift, plus a "refined" prediction file
    np.savez(tmp_path / "refined.npz", label=scan["gt_label"].clip(0, 254))
    out = L.cmd_score(type("A", (), dict(scan=str(p), ref=str(tmp_path / "labels_manual.npz"),
                                         pred=str(tmp_path / "refined.npz"), json=None))())
    assert "lift" in out and len(out) == 2
    assert out["refined.npz:label"]["inst_miou"] > 0.9


def test_undo_split_merge_marks(tmp_path):
    scan = L.load_scan(SCANS[0])
    prior = scan["prior"]
    ids = [i for i in np.unique(prior) if 0 < i < 252]
    big = max(ids, key=lambda i: (prior == i).sum())
    pts = np.nonzero(prior == big)[0]
    far = [int(pts[0]), int(pts[np.argmax(np.linalg.norm(scan["xyz"][pts] - scan["xyz"][pts[0]], axis=1))])]
    s = L.Session(scan)
    s.add_items([{"name": "a", "seeds": [far[0]]}, {"name": "b", "seeds": [far[1]]}])    # split
    lab, names = s.result()
    assert set(names.values()) == {"a", "b"}
    assert (lab == 1).sum() > 0 and (lab == 2).sum() > 0 and ((lab > 0) == (prior == big)).all()
    s.undo()
    assert not s.result()[1]
    if len(ids) > 1:
        other = [i for i in ids if i != big][0]
        s.add_items([{"name": "m", "seeds": [far[0], int(np.nonzero(prior == other)[0][0])]}])   # merge
        assert ((s.result()[0] == 1) == np.isin(prior, [big, other])).all()
    s.add_mark("container", [int(np.nonzero(prior == 254)[0][0])])
    assert (s.result()[0] == 254).sum() == (prior == 254).sum()


def test_export_frames(tmp_path):
    p = SCANS[0]
    scan = L.load_scan(p)
    label = np.where((scan["gt_label"] > 0) & (scan["gt_label"] < 252), scan["gt_label"], 0).astype(np.int32)
    paths = L.export_frames(p, label, tmp_path / "masks_manual", xyz=scan["xyz"], scale=0.25, fill_px=10)
    n_frames = len(list((p / "rgb").glob("*.jpg")))
    assert len(paths) == n_frames == len(list((tmp_path / "masks_manual").glob("*.png")))
    from PIL import Image
    from foodscan.loader import Scan
    ids_all, agree, tot = set(), 0, 0
    for fr, pth in zip(Scan(p).frames, paths):
        img = np.asarray(Image.open(pth))
        assert img.dtype == np.uint8 and img.shape == (fr.rgb_size[1] // 4, fr.rgb_size[0] // 4)
        ids_all |= set(np.unique(img).tolist())
        idx, u, v, z = L.project_frame_points(fr, scan["xyz"])
        keep = label[idx] > 0
        ui, vi = (u[keep] * 0.25).astype(int), (v[keep] * 0.25).astype(int)
        agree += (img[vi, ui] == label[idx][keep]).sum()
        tot += keep.sum()
    assert ids_all <= set(np.unique(label).tolist())
    assert tot > 50 and agree / tot > 0.9, (agree, tot)
