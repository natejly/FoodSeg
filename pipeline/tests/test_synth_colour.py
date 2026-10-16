"""Pure-function tests for synth.colour_audit and synth.tex_grade (no Blender, no data)."""
import numpy as np

from synth import colour_audit as ca
from synth import tex_grade as tg


def test_lab_roundtrip_and_hue():
    rgb = np.array([[200, 100, 50], [10, 200, 30], [128, 128, 128]], np.uint8)
    lab = ca.srgb_to_lab(rgb)
    assert np.allclose(ca.lab_to_srgb(lab) * 255, rgb, atol=0.05)
    assert ca.chroma(lab[2]) < 0.1
    assert ca.hue_diff(350, 10) == 20 and ca.hue_diff(10, 350) == 20


def test_classify_flags():
    tgt = dict(L=31, L_p50=31, C_p50=25, C_p95=33, hue=112)
    neon = dict(L_p50=55, C_p50=48, C_p90=70, hue=118)
    assert "neon" in ca.classify(neon, tgt)
    black = dict(L_p50=8, C_p50=3, C_p90=6, hue=112)
    assert "black" in ca.classify(black, dict(tgt, L_p50=50))
    grey = dict(L_p50=40, C_p50=2, C_p90=5, hue=112)
    assert "grey" in ca.classify(grey, tgt)
    shifted = dict(L_p50=31, C_p50=25, C_p90=30, hue=40)
    assert "hue" in ca.classify(shifted, tgt)
    ok = dict(L_p50=33, C_p50=24, C_p90=30, hue=110)
    assert ca.classify(ok, tgt) == []


def test_n5k_key():
    keys = ["broccoli", "white rice", "waffles", "cherry tomatoes", "potatoes"]
    assert ca.n5k_key("Broccoli", keys) == "broccoli"
    assert ca.n5k_key("waffle", keys) == "waffles"
    assert ca.n5k_key("potato", keys) == "potatoes"
    assert ca.n5k_key("boiled egg", keys) is None
    assert ca.n5k_key("rice", keys) is None


def _atlas():
    """64x64 atlas: fill colour (20,30,20) on the left half, green food right, a neutral grey patch inside."""
    a = np.empty((64, 64, 3), np.uint8)
    a[:] = (20, 30, 20)
    rng = np.random.default_rng(0)
    a[:, 32:] = np.clip(np.array([90, 200, 40]) + rng.integers(-12, 12, (64, 32, 3)), 0, 255)
    a[10:30, 40:60] = (110, 112, 112)
    return a


def test_detect_and_fill():
    a = _atlas()
    m = tg.detect_fill(a)
    assert m is not None and m[:, :30].all() and not m[:, 34:].any()
    f = tg.fill_background(a, tg.dilate(m, 2))
    assert np.abs(f[:, :20].astype(int) - np.median(a[:, 36:].reshape(-1, 3), 0)).max() < 40


def test_grey_fill_makes_patch_coloured():
    a = _atlas()
    lab = ca.srgb_to_lab(a).reshape(-1, 3).astype(np.float32)
    valid = np.zeros((64, 64), bool)
    valid[:, 32:] = True
    out, share = tg.fill_grey(lab, valid.reshape(-1))
    assert share > 0.05
    patch = out.reshape(64, 64, 3)[15:25, 45:55]
    assert ca.chroma(patch).mean() > 25


def test_grade_reduces_neon_and_lifts_grey():
    tgt = ca.n5k_target("broccoli")
    stats = dict(L50=80.0, C50=62.0, C90=98.0, hue=119.0)
    p = tg.grade_params(stats, tgt)
    assert p["L_to"] < 70 and p["C_to"] < 55
    lab = np.array([[80, -30, 54], [60, -60, 85]], np.float32)
    g = tg.grade_lab(lab, p, stats)
    assert ca.chroma(g)[1] < ca.chroma(lab)[1] * 0.8
    sal = ca.n5k_target("salmon grill")
    st = dict(L50=55.0, C50=8.0, C90=15.0, hue=40.0)
    p = tg.grade_params(st, sal)
    assert p["C_to"] > 20 and p["lift_hue"] is not None
    g = tg.grade_lab(np.array([[55, 1, 1]], np.float32), p, st)
    assert ca.chroma(g)[0] > 15


def test_generic_caps_only_reduce():
    st = dict(L50=50.0, C50=30.0, C90=40.0, hue=90.0)
    p = tg.grade_params(st, None)
    assert p["L_to"] == 50.0 and p["C_to"] == 30.0 and p["dh"] == 0.0


def test_streak_ratio():
    rng = np.random.default_rng(1)
    healthy = rng.random((64, 64)) * 255
    streaks = np.repeat(rng.random((64, 1)) * 255, 64, axis=1)
    assert ca.streak_ratio(healthy) < ca.STREAK_RATIO < ca.streak_ratio(streaks)
