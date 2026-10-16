"""Toy-frame tests for the 2D mask post-processing rules in foodscan.segment (CPU only)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
from foodscan import segment as S

H, W = 48, 64


def _nrm(z):
    return np.tile(np.array([0, 0, 1.0], np.float32), (*z.shape, 1))


def test_depth_split_two_levels():
    z = np.full((H, W), 0.5, np.float32)
    z[:, 32:] = 0.6                                   # 100 mm step in the middle of one mask
    m = np.zeros((H, W), bool)
    m[10:40, 10:54] = True
    part = S.split_by_depth(m, z, _nrm(z))
    assert part is not None and set(np.unique(part[m])) == {1, 2}
    assert part[20, 15] != part[20, 50]


def test_depth_no_split_on_flat_or_tiny_part():
    z = np.full((H, W), 0.5, np.float32)
    m = np.zeros((H, W), bool)
    m[10:40, 10:54] = True
    assert S.split_by_depth(m, z, _nrm(z)) is None
    z[10:40, 52:54] = 0.7                             # sliver < 20 % of the mask
    assert S.split_by_depth(m, z, _nrm(z)) is None


def test_colour_split_and_reject():
    lab = np.zeros((H, W, 3), np.float32)
    lab[..., 0] = 50
    lab[:, 32:, 1] = 40                               # two clearly different, spatially compact halves
    m = np.zeros((H, W), bool)
    m[10:40, 10:54] = True
    part = S.split_by_colour(m, lab)
    assert part is not None and part[20, 15] != part[20, 50]
    lab2 = lab.copy()
    lab2[:, 32:, 1] = 5                               # centroids < 12 apart
    assert S.split_by_colour(m, lab2) is None
    lab3 = lab.copy()                                 # checkerboard: clusters not spatially separable
    lab3[..., 1] = 0
    lab3[::2, ::2, 1] = 40
    lab3[1::2, 1::2, 1] = 40
    assert S.split_by_colour(m, lab3) is None


def test_multi_name_split():
    f = np.zeros((H, W), bool)
    f[10:40, 10:54] = True
    a = np.zeros_like(f); a[10:40, 10:42] = True      # ~73 % of f, IoU >= 0.7 -> baseline does not carve it
    b = np.zeros_like(f); b[10:40, 42:54] = True      # ~27 %
    food = {1: (0.9, f)}
    dets = [(2, "rice", 0.8, a), (3, "egg", 0.7, b)]
    base = S.refine_frame(food, dets)
    multi = S.refine_frame(food, dets, multi_name_cover=0.25)
    assert sorted(r[2] for r in multi.values()) == ["egg", "rice"]
    assert len(multi) == 2 and len(base) == 2          # a is named, b carved: same outcome here
    # only a triggers (b under the threshold): flag changes nothing
    b2 = np.zeros_like(f); b2[10:12, 42:54] = True
    assert S.refine_frame(food, [(2, "rice", .8, a), (3, "egg", .7, b2)], multi_name_cover=0.25).keys() == \
        S.refine_frame(food, [(2, "rice", .8, a), (3, "egg", .7, b2)]).keys()


def test_hull_ratio():
    m = np.zeros((H, W), bool)
    m[10:30, 10:30] = True
    assert S._hull_ratio(m) > 0.9
    m2 = np.zeros((H, W), bool)
    m2[10:12, 5:60] = True
    m2[10:40, 5:7] = True                              # L shape
    assert S._hull_ratio(m2) < 0.6


def _toy_scan(monkeypatch, lab, z):
    monkeypatch.setattr(S, "_lowres_frames", lambda *a, **k: {0: dict(lab=lab, z=z, nrm=_nrm(z))})


def test_merge_touching_same_name(monkeypatch):
    lab = np.zeros((H, W, 3), np.float32); lab[..., 0] = 50
    _toy_scan(monkeypatch, lab, np.full((H, W), 0.5, np.float32))
    r = np.zeros((H * 4, W * 4), np.uint16)            # full-res map is 4x the low-res one
    r[40:160, 40:128] = 1000
    r[40:160, 130:216] = 1001                          # 2 low-res px gap -> touches under a 2 px dilation
    npx = {1000: {"rice": 100}, 1001: {"rice": 90}}
    sc = {0: {1000: 0.9, 1001: 0.8}}
    o = S.PostOpts(merge_same=True, lowres=(W, H))
    out, nn, fs = S.postprocess_raw("x", {0: r}, npx, sc, o)
    assert len(np.unique(out[0][out[0] > 0])) == 1 and o.log[0]["kind"] == "merge"
    assert nn[1000]["rice"] == 190
    npx2 = {1000: {"rice": 100}, 1001: {"egg": 90}}    # different names never merge
    o2 = S.PostOpts(merge_same=True, lowres=(W, H))
    out2, _, _ = S.postprocess_raw("x", {0: r}, npx2, sc, o2)
    assert len(np.unique(out2[0][out2[0] > 0])) == 2 and not o2.log


def test_filter_low_score_and_tiny(monkeypatch):
    r = np.zeros((H * 4, W * 4), np.uint16)
    r[40:160, 40:200] = 1000                           # big, good
    r[20:30, 220:230] = 1001                           # tiny, low score
    npx = {1000: {"rice": 1}, 1001: {"rice": 1}}
    sc = {0: {1000: 0.9, 1001: 0.25}}
    o = S.PostOpts(min_score=0.3)
    out, _, _ = S.postprocess_raw("x", {0: r}, npx, sc, o)
    assert set(np.unique(out[0])) == {0, 1000} and o.log[0]["rule"] == "score"
    o = S.PostOpts(min_area_frac=0.003)
    out, _, _ = S.postprocess_raw("x", {0: r}, npx, {0: {1000: 0.9, 1001: 0.9}}, o)
    assert set(np.unique(out[0])) == {0, 1000} and o.log[0]["rule"] == "area"


def test_depth_split_in_postprocess_and_defaults_off(monkeypatch):
    lab = np.zeros((H, W, 3), np.float32)
    z = np.full((H, W), 0.5, np.float32); z[:, 32:] = 0.6
    _toy_scan(monkeypatch, lab, z)
    r = np.zeros((H * 4, W * 4), np.uint16)
    r[40:160, 40:216] = 1000
    npx, sc = {1000: {"rice": 100}}, {0: {1000: 0.8}}
    assert not S.PostOpts().active
    o = S.PostOpts(depth_split=True, lowres=(W, H))
    out, nn, fs = S.postprocess_raw("x", {0: r}, npx, sc, o)
    ids = np.unique(out[0][out[0] > 0])
    assert len(ids) == 2 and all(fs[0][int(i)] == 0.8 for i in ids)
    assert o.log[0]["kind"] == "split" and o.log[0]["rule"] == "depth"
