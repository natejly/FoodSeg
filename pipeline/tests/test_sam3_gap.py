import numpy as np

from eval.sam3_gap import kind_of, mask_stats, match_ingredient, simple_name


def test_mask_stats_threshold_filters():
    ref = np.zeros((10, 10), bool); ref[:5] = True
    m = np.zeros((2, 10, 10), bool); m[0, :5] = True; m[1, 5:] = True
    s = np.array([0.4, 0.9], np.float32)
    hi = mask_stats(m, s, ref, 0.5)
    lo = mask_stats(m, s, ref, 0.3)
    assert hi["any"] and hi["iou"] == 0.0 and lo["iou"] == 1.0 and lo["n"] == 2
    assert not mask_stats(m, s, ref, 0.95)["any"]


def test_names():
    assert simple_name("yeast bread") == "bread" and simple_name("broccoli") == "broccoli"
    assert match_ingredient("Cherry Tomatoes") == "cherry tomato"
    assert match_ingredient("mashed potatoes") == "mashed potato"
    assert kind_of("metafood3d") == "scan" and kind_of("primitive") == "procedural"
