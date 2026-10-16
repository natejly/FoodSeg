import numpy as np

from eval import sam3_variants as V


def test_pack_roundtrip():
    m = np.random.default_rng(0).random((3, *V.SIZE)) > 0.5
    assert (V.unpack(V.pack(m)) == m).all()


def test_to_store_resizes_nearest():
    m = np.zeros((2, 756, 1008), bool)
    m[:, :378] = True
    s = V.to_store(m)
    assert s.shape == (2, 720, 960) and s[0, 100].all() and not s[0, 600].any()
    assert V.to_store(np.zeros((0, 756, 1008), bool)).shape == (0, 720, 960)


def test_instance_prompt_box_points_and_negatives():
    rng = np.random.default_rng(1)
    px = rng.uniform([100, 100], [200, 180], (400, 2))
    near = rng.uniform([190, 120], [260, 170], (200, 2))     # neighbour overlapping the box's right edge
    far = rng.uniform([600, 500], [650, 550], (50, 2))       # neighbour outside the box
    d = V.instance_prompt(px, [near, far])
    x0, y0, x1, y1 = d["box"]
    assert x0 < 105 and x1 > 195 and y0 < 105 and y1 > 175
    assert 1 <= len(d["pos"]) <= 3 and all(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in d["pos"])
    assert len(d["neg"]) == 1 and d["neg"][0][0] > 190          # only the neighbour with pixels inside the box


def test_compose_instance_maps_score_wins_and_overwrites_aux():
    base = np.zeros(V.SIZE, np.uint8)
    base[:, :10] = 254
    a = np.zeros(V.SIZE, bool); a[:, :20] = True
    b = np.zeros(V.SIZE, bool); b[:, 15:30] = True
    out = V.compose_instance_maps([(1, 0.6, a), (2, 0.9, b)], base)
    assert (out[:, :15] == 1).all() and (out[:, 15:30] == 2).all() and (out[:, 30:] == 0).all()


def test_count_errors_fragmented_and_merged():
    gi = np.array([1] * 10 + [2] * 10 + [3] * 10)
    pi = np.array([5] * 5 + [6] * 5 + [7] * 10 + [7] * 5 + [8] * 5)   # item 1 split over 5/6; item 2 and half of 3 share 7
    frag, merged, n = V.count_errors(gi, pi)
    assert n == 3 and frag == 2 and merged == 2        # item 1 (5+6 split) and 3 (7+8 split) fragmented; 2 and 3 merged via id 7
