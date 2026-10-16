"""Toy-data tests for the lift-quality knobs: vote weighting, margin / background abstention, superpoints.

    python tests/test_lift_sweep.py        (also collected by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodscan import labels as L  # noqa: E402


def _fv(lab, w=None, score=None, edge=None, qual=None):
    n = len(lab)
    return L._FrameVotes(0, np.arange(n, dtype=np.int32), np.asarray(lab, np.int32),
                         np.ones(n, np.float32) if w is None else np.asarray(w, np.float32),
                         None if score is None else np.asarray(score, np.float32),
                         None if edge is None else np.asarray(edge, np.float32),
                         None if qual is None else np.asarray(qual, np.float32))


def test_vote_weights_schemes():
    fv = _fv([1, 1, 1], w=[1.0, 0.5, 1.0], score=[0.9, 0.5, 0.2], edge=[10.0, 3.0, 0.0], qual=[1.0, 0.5, 1.0])
    assert np.allclose(L.vote_weights(fv, "cos"), [1.0, 0.5, 1.0])
    assert np.allclose(L.vote_weights(fv, "cos+score"), [0.9, 0.25, 0.2])
    e = L.vote_weights(fv, "cos+edge", edge_px=6.0, edge_floor=0.1)
    assert np.allclose(e, [1.0, 0.5 * (0.1 + 0.9 * 0.5), 0.1])          # far from the edge: full weight; on it: floor
    q = L.vote_weights(fv, "cos+score+edge+qual", edge_px=6.0, edge_floor=0.0)
    assert np.allclose(q, [0.9, 0.5 * 0.5 * 0.5 * 0.5, 0.0])
    try:
        L.vote_weights(fv, "cos+bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown token must raise")


def test_edge_weighting_flips_a_vote():
    # two frames vote 1 near a mask edge (weight ~floor), one frame votes 2 in the mask interior
    a = _fv([1], edge=[0.0], score=[1.0], qual=[1.0])
    b = _fv([1], edge=[0.0], score=[1.0], qual=[1.0])
    c = _fv([2], edge=[20.0], score=[1.0], qual=[1.0])
    fvs = [a, b, c]
    for i, f in enumerate(fvs):
        f.pos = i
    ids = np.array([0, 1, 2])
    v_cos, _ = L._accumulate(fvs, ids, {}, 1)
    v_edge, _ = L._accumulate(fvs, ids, {}, 1, weighting="cos+edge", weight_kwargs=dict(edge_px=6.0, edge_floor=0.1))
    assert L._decide_cols(v_cos, np.array([3]), ids, 1)[0][0] == 1
    assert L._decide_cols(v_edge, np.array([3]), ids, 1)[0][0] == 2


def test_frame_mask_features():
    m = np.zeros((40, 60), np.int32)
    m[5:25, 5:25] = 7                                   # compact 20x20 square
    m[30:32, 5:55] = 8                                  # thin strip (low area)
    edge, score, qual = L.frame_mask_features(m, {7: 0.9, 8: 0.4})
    assert edge[4, 15] == 0.0 and edge[15, 15] > 5
    assert score[7] == np.float32(0.9) and score[0] == 1.0
    assert 0.8 < qual[7] <= 1.0 and qual[8] < qual[7]


def test_margin_and_background_abstention():
    ids = np.array([0, 1, 2])
    votes = np.array([[3.0, 2.0, 0.0],        # bg wins narrowly over 1
                      [1.0, 1.1, 1.0],        # near three-way tie
                      [0.0, 0.0, 0.0]], np.float32)
    nv = np.array([5, 5, 0])
    assert L._decide_cols(votes, nv, ids, 1)[0].tolist() == [0, 1, 0]
    assert L._decide_cols(votes, nv, ids, 1, bg_abstain=True)[0].tolist() == [1, 1, 0]     # bg votes dropped
    assert L._decide_cols(votes, nv, ids, 1, margin=0.2)[0].tolist() == [0, 0, 0]          # near-tie abstains; bg row is 0 anyway
    assert L._decide_cols(votes, nv, ids, 6)[0].tolist() == [0, 0, 0]                      # min_nvotes gate


def _two_squares(n=30, rng=None):
    rng = rng or np.random.default_rng(0)
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) * 0.004
    pts = np.concatenate([np.c_[g, np.zeros(len(g))], np.c_[g + [0.2, 0], np.zeros(len(g))]])
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    rgb = np.concatenate([np.tile([200, 40, 40], (len(g), 1)), np.tile([40, 40, 200], (len(g), 1))]).astype(np.uint8)
    return pts, nrm, rgb, len(g)


def test_superpoints_respect_colour_and_connectivity():
    pts, nrm, rgb, h = _two_squares()
    sp = L.compute_superpoints(pts, nrm, rgb, n_target=8, w_color=1.0)
    assert len(sp) == len(pts) and sp.dtype == np.int32
    # no superpoint spans both (disconnected) squares
    assert not any((sp[:h] == s).any() and (sp[h:] == s).any() for s in np.unique(sp))
    assert 6 <= sp.max() + 1 <= 40


def test_superpoints_split_a_colour_edge():
    # one connected flat surface, left half red / right half blue: colour-aware superpoints align with the seam
    n = 40
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) * 0.004
    pts = np.c_[g, np.zeros(len(g))]
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    right = g[:, 0] > g[:, 0].mean() + 0.003
    rgb = np.where(right[:, None], [40, 40, 200], [200, 40, 40]).astype(np.uint8)
    sp = L.compute_superpoints(pts, nrm, rgb, n_target=10, w_color=3.0)
    mixed = sum(1 for s in np.unique(sp) if 0 < right[sp == s].mean() < 1 and min(right[sp == s].mean(), 1 - right[sp == s].mean()) > 0.2)
    assert mixed == 0


def test_superpoint_majority_vote_toy():
    sp = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    lab = np.array([1, 1, 2, 0, 3, 3, 3, 4])
    out = L.superpoint_majority(lab, sp)
    assert out.tolist() == [1, 1, 1, 1, 3, 3, 3, 4]                       # sp0: 1 wins 2 vs 1 vs 1
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 0], bool)                       # sp2 has no valid point -> background
    assert L.superpoint_majority(lab, sp, valid=valid).tolist() == [1, 1, 1, 1, 3, 3, 3, 0]
    w = np.array([1, 1, 5, 1, 1, 1, 1, 1.0])                               # weights can overturn the count
    assert L.superpoint_majority(lab, sp, weights=w)[:4].tolist() == [2, 2, 2, 2]


def test_superpoint_soft_votes_and_lift_from_votes():
    sp = np.array([0, 0, 0, 1, 1])
    votes = np.array([[0, 1.0, 0], [0, 0.4, 0.5], [0, 0.4, 0.5], [0, 0, 1.0], [0, 0, 0.0]], np.float32)
    s = L.superpoint_votes(votes, sp)
    assert np.allclose(s[0], s[2]) and np.allclose(s[0], [0, 1.8, 1.0]) and np.allclose(s[3], [0, 0, 1.0])
    # end to end through lift_from_votes: a 1-vote outlier inside a superpoint is overruled; hard and soft agree here
    n = 5
    fvs = [L._FrameVotes(p, np.arange(n, dtype=np.int32), np.array([1, 1, 2, 2, 2], np.int32), np.ones(n, np.float32))
           for p in range(3)]
    fvs[0].lab = np.array([1, 1, 1, 2, 2], np.int32)
    plain = L.lift_from_votes(fvs, n, repair=False, agree=False, min_nvotes=1)
    assert plain.lift_label.tolist() == [1, 1, 2, 2, 2]
    sp = np.array([0, 0, 0, 1, 1])
    for mode in ("hard", "soft"):
        r = L.lift_from_votes(fvs, n, repair=False, agree=False, min_nvotes=1, superpoints=sp, sp_mode=mode)
        assert r.lift_label.tolist() == [1, 1, 1, 2, 2], (mode, r.lift_label)


def test_mutual_and_cluster_linking_toy():
    # two views of one object (ids 10, 11) and a second object (20, 21), disjoint point sets; mask ids differ per frame
    n = 200
    f0 = L._FrameVotes(0, np.arange(n, dtype=np.int32), np.where(np.arange(n) < 100, 10, 20).astype(np.int32), np.ones(n, np.float32))
    f1 = L._FrameVotes(1, np.arange(n, dtype=np.int32), np.where(np.arange(n) < 100, 11, 21).astype(np.int32), np.ones(n, np.float32))
    for fn in (lambda: L.repair_ids([f0, f1], n, aux_ids={999}, min_points=5, min_support=1, mutual=True),
               lambda: L.cluster_ids([f0, f1], n, thresh=0.6, aux_ids={999}, min_points=5)):
        remap, _ = fn()
        assert remap == {11: 10, 21: 20}, remap


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
