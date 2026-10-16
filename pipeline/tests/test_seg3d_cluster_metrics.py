import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import metrics as M                                      # noqa: E402
from seg3d.cluster import cluster_instances                         # noqa: E402


def blobs(rng):
    c = np.array([[0, 0, 0.02], [0.03, 0, 0.02], [0.0, 0.05, 0.02]])
    pts = np.concatenate([ci + rng.normal(0, 0.004, (300, 3)) for ci in c])
    gt = np.repeat([1, 2, 3], 300)
    return pts, gt, c


def test_cluster_offsets_split_touching():
    rng = np.random.default_rng(0)
    pts, gt, c = blobs(rng)
    off = c[gt - 1] - pts                                         # perfect centre vote
    sem = np.full(len(pts), 3)
    inst = cluster_instances(pts, sem, off, eps=0.004)
    assert len(np.unique(inst)) == 3
    ps, pi, = M.lift_baseline_pred(gt)
    acc = M.Accumulator(volumes=False)
    acc.add(sem, inst, pts, np.full(len(pts), 3), gt, gt, {})
    r = acc.result()
    assert r["inst_miou"] > 0.99 and r["exact_count"] == 1.0 and r["pq"] > 0.99


def test_cluster_background_and_small():
    rng = np.random.default_rng(1)
    pts, gt, c = blobs(rng)
    sem = np.where(gt == 3, 0, 3)
    inst = cluster_instances(pts, sem, np.zeros_like(pts), eps=0.01)
    assert (inst[gt == 3] == 0).all()
    assert len(np.unique(inst[inst > 0])) >= 1


def test_metrics_merge_error():
    rng = np.random.default_rng(2)
    pts, gt, c = blobs(rng)
    pred = np.where(gt == 3, 2, gt)                                # merge 2+3
    acc = M.Accumulator(volumes=False)
    acc.add(np.full(len(pts), 3), pred, pts, np.full(len(pts), 3), gt, gt, {})
    r = acc.result()
    assert r["exact_count"] == 0.0 and r["recall50"] < 1.0
