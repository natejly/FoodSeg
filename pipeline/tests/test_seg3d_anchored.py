import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import metrics as M                                      # noqa: E402
from seg3d.cluster import anchored_instances                        # noqa: E402


def toy(rng):
    """Three food blobs 1,2,3 (400 pts each, 3 cm apart) + a table slab."""
    c = np.array([[0, 0, 0.02], [0.03, 0, 0.02], [0.0, 0.06, 0.02]])
    pts = np.concatenate([ci + rng.normal(0, 0.004, (400, 3)) for ci in c])
    gt = np.repeat([1, 2, 3], 400)
    tab = np.c_[rng.uniform(-0.1, 0.1, (300, 2)), np.zeros(300)]
    xyz = np.concatenate([pts, tab])
    gt_inst = np.r_[gt, np.zeros(300, int)]
    gt_sem = np.r_[np.full(1200, 3), np.zeros(300, int)]
    return xyz, c, gt_inst, gt_sem


def test_anchored_identity_when_model_agrees():
    rng = np.random.default_rng(0)
    xyz, c, gi, gs = toy(rng)
    off = np.where(gs[:, None] == 3, c[np.maximum(gi - 1, 0)] - xyz, 0)
    sem, inst = anchored_instances(xyz, gs, off, gi, np.ones(len(xyz)), np.zeros(len(xyz), np.uint8), eps=0.004)
    assert (inst == gi).all() and (sem == gs).all()                    # nothing uncertain -> lift untouched


def test_anchored_fixes_blob_keeps_rest_and_spawns_id():
    rng = np.random.default_rng(1)
    xyz, c, gi, gs = toy(rng)
    off = np.where(gs[:, None] == 3, c[np.maximum(gi - 1, 0)] - xyz, 0)
    lift = gi.copy()
    flip = np.nonzero(gi == 2)[0][:100]
    lift[flip] = 1                                                      # blob of item 2 wrongly given id 1
    share = np.ones(len(xyz))
    share[flip] = 0.4                                                   # ... with low vote share
    drop = gi == 3
    lift[drop] = 0                                                      # item 3 missed entirely by the lift
    share[drop] = 0.4
    sem, inst = anchored_instances(xyz, gs, off, lift, share, np.zeros(len(xyz), np.uint8), eps=0.004)
    assert (inst[gi == 2] == inst[gi == 2][0]).all()                   # blob moved back to its own cluster's id
    assert inst[gi == 2][0] == 2 and (inst[gi == 1] == 1).all()
    new = np.unique(inst[gi == 3])
    assert len(new) == 1 and new[0] not in (0, 1, 2)                  # spawned a fresh id (no lift overlap)
    # low-share points the model agrees with are not "broken"
    assert (sem == gs).all()


def test_anchored_confident_lift_points_are_never_reassigned():
    rng = np.random.default_rng(2)
    xyz, c, gi, gs = toy(rng)
    lift = gi.copy()
    lift[gi == 3] = 2                                                   # confident (share 1) wrong merge
    off = np.where(gs[:, None] == 3, c[np.maximum(gi - 1, 0)] - xyz, 0)
    _, inst = anchored_instances(xyz, gs, off, lift, np.ones(len(xyz)), np.zeros(len(xyz), np.uint8), eps=0.004)
    assert (inst[gi == 3] == 2).all()                                   # refine, don't replace
    _, inst = anchored_instances(xyz, gs, off, lift, np.ones(len(xyz)), np.zeros(len(xyz), np.uint8), eps=0.004, split_merged=True)
    assert len(np.unique(inst[gi == 3])) == 1 and inst[gi == 3][0] != inst[gi == 2][0]    # opt-in split


def test_class_disagreement_triggers_reassignment():
    rng = np.random.default_rng(3)
    xyz, c, gi, gs = toy(rng)
    lift = gi.copy()
    lift[gi == 0] = np.where(np.arange((gi == 0).sum()) < 50, 1, 0)      # 50 table points lifted as food 1, share 1
    sem = gs.copy()                                                      # model says table
    off = np.where(gs[:, None] == 3, c[np.maximum(gi - 1, 0)] - xyz, 0)
    s2, i2 = anchored_instances(xyz, sem, off, lift, np.ones(len(xyz)), np.zeros(len(xyz), np.uint8), eps=0.004)
    assert (s2 == gs).all() and (i2[gi == 0] == 0).all()


def test_corrected_broken_metric():
    gt_inst = np.array([1] * 4 + [2] * 4 + [0] * 4)
    gt_sem = np.array([3] * 8 + [0] * 4)
    xyz = np.random.default_rng(0).normal(size=(12, 3))
    lift = np.array([1] * 4 + [1] * 2 + [2] * 2 + [0] * 4)             # two points of item 2 mislabelled as 1 -> wrong 2
    ps = gt_sem.copy()
    pi = gt_inst.copy()
    pi[0] = 0                                                          # model breaks one right point (becomes background but food)
    ps[0] = 0
    acc = M.Accumulator(volumes=False)
    acc.add(ps, pi, xyz, gt_sem, gt_inst, gt_inst, {}, lift=lift)
    r = acc.result()
    # food domain = 8 food points.  lift wrong: pts 4,5 (id 1 instead of 2) -> and point correctness is by matched instance:
    assert r["n_wrong_food"] == 2
    assert r["corrected_food"] == 1.0                                   # both wrong points fixed by the model
    assert abs(r["broken_food"] - 1 / 6) < 1e-9                         # 1 of the 6 lift-right food points broken
    acc2 = M.Accumulator(volumes=False)
    ls, li = M.lift_baseline_pred(lift)
    acc2.add(ls, li, xyz, gt_sem, gt_inst, gt_inst, {}, lift=lift)
    assert acc2.result()["corrected_food"] == 0 and acc2.result()["broken_food"] == 0
