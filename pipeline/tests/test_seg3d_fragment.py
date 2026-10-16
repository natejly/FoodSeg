import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import augment as A                                     # noqa: E402


def scene(seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.concatenate([rng.normal([0, 0, 0], 0.01, (400, 3)), rng.normal([0.1, 0, 0], 0.01, (300, 3))]).astype(np.float32)
    lift = np.r_[np.full(400, 3), np.full(300, 5)].astype(np.int32)
    return xyz, lift, np.ones(700, np.float32), np.full(700, 5, np.int16)


def test_fragment_splits_one_instance_with_fresh_ids():
    c = A.AugConfig(frag_p=1.0)
    n_new_tot = 0
    for seed in range(20):
        xyz, lift, share, nv = scene()
        l2, s2, nv2, n_new = A.fragment_lift(xyz, lift, share, nv, np.random.default_rng(seed), c)
        n_new_tot += n_new
        assert len(np.unique(l2[l2 > 0])) == 2 + n_new and n_new <= 2
        changed = l2 != lift
        assert changed.sum() < 0.85 * (lift == lift[changed][0]).sum() if n_new else not changed.any()
        # only one original instance is touched; the new ids are unused food ids; GT-free (lift only)
        if n_new:
            assert len(np.unique(lift[changed])) == 1
            new = np.unique(l2[changed])
            assert ((new > 0) & (new < 252)).all() and not np.isin(new, [3, 5]).any()
            assert (s2[changed] < 0.91).all() and (s2[~changed] == 1).all()
            assert (nv2 == nv).all()
    assert n_new_tot >= 20


def test_apply_lift_fragments_pseudo_not_real():
    xyz, lift, share, nv = scene()
    d = dict(xyz=xyz, label=lift, lift_label=lift, lift_share=share, lift_nvotes=nv)
    c = A.AugConfig(frag_p=1.0, lift_corrupt=False)
    out = A.apply_lift(d, np.random.default_rng(1), c, 0.002, True)
    assert len(np.unique(out["lift_label"])) > 2
    out = A.apply_lift(dict(d, real_votes=True), np.random.default_rng(1), c, 0.002, True)
    assert (out["lift_label"] == lift).all()
    off = A.apply_lift(d, np.random.default_rng(1), A.AugConfig(frag_p=0.0, lift_corrupt=False), 0.002, True)
    assert (off["lift_label"] == lift).all()
    ev = A.apply_lift(d, np.random.default_rng(1), A.eval_aug("r1"), 0.002, True, force_corrupt=True)
    assert (ev["lift_label"] == lift).all()
