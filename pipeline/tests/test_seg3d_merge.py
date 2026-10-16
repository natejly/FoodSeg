import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import merge as G                                       # noqa: E402

REAL = Path(__file__).resolve().parents[2] / "results/seg3d/r4_data/real_v12/scans/s000002"


def blob(c, n, rng, rgb, r=0.02):
    p = rng.normal(0, 1, (n, 3))
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    p[:, 2] = np.abs(p[:, 2])                                      # upper half of a flat ellipsoid (axes 3.5 x 3.5 x 1 cm)
    ax = np.array([1.75, 1.75, 0.5]) * r
    nrm = p / ax
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return p * ax + c, np.tile(rgb, (n, 1)), nrm


def toy(seed=0):
    """One red half-sphere cut by a plane into fragments 1 and 2 (ids), plus a touching but blue/high neighbour (id 3)."""
    rng = np.random.default_rng(seed)
    p, c, n = blob(np.zeros(3), 6000, rng, [0.8, 0.2, 0.2])
    left = p[:, 0] < 0
    q, cq, nq = blob(np.array([0.068, 0, 0.0]), 3000, rng, [0.1, 0.2, 0.9])
    xyz = np.r_[p, q]
    lift = np.r_[np.where(left, 1, 2), np.full(len(q), 3)].astype(np.int32)
    # voxel thin
    keep = np.unique(np.floor(xyz / 0.002).astype(int), axis=0, return_index=True)[1]
    xyz, lift = xyz[keep], lift[keep]
    rgb = np.r_[c, cq][keep]
    nrm = np.r_[n, nq][keep]
    return dict(xyz=xyz, rgb=rgb.astype(np.float32), normal=nrm, lift=lift, share=np.full(len(xyz), 0.9, np.float32),
                nv=np.full(len(xyz), 8, np.int16), label=lift.copy(), vox=0.002)


def test_toy_merges_fragments_not_neighbour():
    sc = toy()
    merged, geo = G.merge_scene(sc, G.heuristic_decider(0.1, 12.0, 40.0, 0.006))
    assert len(np.unique(merged[sc["lift"] == 1])) == 1
    assert merged[sc["lift"] == 1][0] == merged[sc["lift"] == 2][0]
    assert merged[sc["lift"] == 3][0] != merged[sc["lift"] == 1][0]


def test_toy_features_distinguish_seam():
    sc = toy()
    geo = G.SceneGeo(sc)
    es = {(e["a"], e["b"]): e["f"] for e in geo.edges(np.arange(512), {i: frozenset([i]) for i in range(512)})}
    assert (1, 2) in es
    f12 = es[(1, 2)]
    assert f12["dE"] < 3 and f12["ang"] < 20
    f13 = es.get((2, 3), es.get((1, 3)))
    assert f13 is not None and f13["dE"] > 30


def test_safety_blocks_low_convexity_and_container_seam():
    f = dict(convexity=0.4, cont_frac=0.0, share_small=0.9, share_large=0.9, n_small=10, n_large=2000)
    assert not G.safe(f, c_min=0.6)
    assert not G.safe(dict(f, convexity=1.0, cont_frac=0.9))
    assert not G.safe(dict(f, convexity=1.0, share_small=0.1))        # low-share small side next to a large one
    assert G.safe(dict(f, convexity=1.0))


@pytest.mark.skipif(not REAL.exists(), reason="hard_v12 real-vote scan not available")
def test_features_finite_on_real_scan():
    sc = G.load_scene(REAL)
    geo = G.SceneGeo(sc)
    es = geo.edges(np.arange(512), {i: frozenset([i]) for i in range(512)})
    assert es
    for e in es:
        v = G.fvec(e["f"])
        assert v.shape == (len(G.FEATS),) and np.isfinite(v).all()
        assert 0 <= e["f"]["contact_ratio"] <= 1.0 + 1e-9
