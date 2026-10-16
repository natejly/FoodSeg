"""Tests for the learned hidden-volume head (pipeline/eval/volume_head*.py).

    cd /home/kyle/Desktop/FoodSeg && conda run -n foodseg python -m pytest pipeline/tests/test_volume_head.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.eval import synth_volume_data as D  # noqa: E402
from pipeline.eval import volume_head as VH  # noqa: E402
from pipeline.eval import volume_head_features as F  # noqa: E402

V12 = ROOT / "data" / "synth" / "hard_v12"


def test_toy_hidden_fraction_is_learned():
    """hidden fraction = deterministic function of one feature -> < 5 % median APE out of sample."""
    rng = np.random.default_rng(0)
    n = 500
    vis = np.exp(rng.normal(3.0, 0.8, n))
    f = rng.uniform(0, 1, n)
    noise = rng.normal(size=(n, 3))
    true = vis * (1.0 + 4.0 * f ** 2)            # hidden volume up to 4x visible
    X = np.c_[f, noise]
    tr, te = np.arange(350), np.arange(350, n)
    m = VH.HeadModel("hgb", 0).fit(X[tr], np.log(vis[tr]), np.log(true[tr]), None)
    pred = np.exp(m.predict(X[te], np.log(vis[te]), None))
    ape = np.abs(pred - true[te]) / true[te]
    assert np.median(ape) < 0.05
    base = np.abs(vis[te] - true[te]) / true[te]
    assert np.median(base) > 0.3                  # the toy really needs the correction


def test_baselines_degenerate():
    X = np.zeros((5, 2))
    lv = np.log(np.arange(1, 6.0))
    for name in ("none", "global", "kind", "ridge", "hgb", "hgb_direct"):
        m = VH.HeadModel(name).fit(X, lv, lv + 0.5, np.array(["a"] * 5))
        p = m.predict(X, lv, np.array(["a", "b", "a", "a", "a"]))
        assert np.isfinite(p).all()


def test_boot_diff_sign():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 1.0, 120)
    model = base - 0.2
    r = VH.boot_diff(model, base, np.repeat(np.arange(40), 3), B=200)
    assert r["d_med_hi"] < 0 and r["p_better"] == 1.0


@pytest.mark.skipif(not V12.exists(), reason="hard_v12 not present")
def test_features_finite_on_v12_scan():
    man = D.read_manifest(V12)
    sc = D.load_scan(V12, "s000002", man)
    rows = F.scan_features(sc)
    assert len(rows) >= 5
    for r in rows:
        for k in F.FEATURES:
            if k in ("lift_share", "lift_nvotes"):
                continue
            assert np.isfinite(r[k]), (r["item_id"], k)
        assert r["vol_true"] > 0
    # no GT-only quantity is an input
    banned = {"occl_frac", "n_gt_points", "vol_true", "mass_g", "rho", "vol_above_container"}
    assert not banned & set(F.FEATURES)


FEAT = ROOT / "results" / "volume_head" / "features.npz"
ENV = ROOT / "results" / "rq1" / "envelope_items.csv"


@pytest.mark.skipif(not (FEAT.exists() and ENV.exists()), reason="cached features / envelope table missing")
def test_envelope_target_v12_category_disjoint_beats_baseline():
    """--target envelope: target = log(V_envelope_gt / V_vis); on hard_v12 with category-disjoint CV the
    corrected median item APE is below the no-correction baseline."""
    De = VH.Data(FEAT, "valid", "envelope")
    Ds = VH.Data(FEAT, "valid", "solid")
    assert De.n == Ds.n and De.n > 200                       # same population for both targets
    assert np.allclose(De.true, De.d["vol_env"]) and np.allclose(Ds.true, Ds.d["vol_solid"])
    Dv = De.subset(De.batch == "hard_v12")
    assert Dv.n >= 100
    res = VH.run_protocol(Dv, "catcv", ["none", "hgb_l1"], repeats=2)
    base = np.median(VH.mean_ape(res, "none", Dv))
    corr = np.median(VH.mean_ape(res, "hgb_l1", Dv))
    assert np.isfinite(corr) and corr < base, (corr, base)
    # the baseline is the raw visible volume vs the envelope: under-counts (negative bias)
    assert np.median(Dv.lv_true - Dv.lv_vis) > 0
