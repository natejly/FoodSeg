import json
import numpy as np
from PIL import Image

from eval.sam3_synth_votes import concepts_for, lift_stats, match_scan


def test_concepts_dedupe(tmp_path):
    items = [{"name": "tuna salad", "category": "Tuna_salad"}, {"name": "cucumber", "category": "Cucumber"},
             {"name": "cucumber", "category": "cucumber"}]
    (tmp_path / "gt.json").write_text(json.dumps({"items": items}))
    assert concepts_for(tmp_path, "names") == ["tuna salad", "cucumber"]
    assert concepts_for(tmp_path, "generic") == ["food"]


def test_match_remaps_and_keeps_aux(tmp_path):
    (tmp_path / "gt" / "seg_ids").mkdir(parents=True)
    (tmp_path / "masks_sam3").mkdir()
    g = np.zeros((20, 20), np.uint8)
    g[:, :8] = 1
    g[:, 8:14] = 2
    g[:, 14:] = 254
    s = np.zeros((40, 40), np.uint8)       # SAM ids swapped (5 -> item 2, 3 -> item 1), plus a stray instance
    s[:, :16] = 3
    s[:, 16:28] = 5
    s[:4, 28:34] = 7
    s[:, 28:] = np.where(s[:, 28:] == 7, 7, 254)
    Image.fromarray(g).save(tmp_path / "gt" / "seg_ids" / "000000.png")
    Image.fromarray(s).save(tmp_path / "masks_sam3" / "000000.png")
    (tmp_path / "masks_sam3" / "000000.json").write_text("{}")
    for f in ("concepts.json", "instances.json"):
        (tmp_path / "masks_sam3" / f).write_text("{}")
    info = match_scan(tmp_path, "names")
    out = np.asarray(Image.open(tmp_path / "masks_sam3_gt" / "000000.png"))
    assert info["sam_to_gt"]["3"] == 1 and info["sam_to_gt"]["5"] == 2
    assert info["sam_to_gt"]["7"] == 3 and info["unmatched_sam"] == [7]
    assert out[0, 0] == 1 and out[0, 20] == 2 and out[39, 39] == 254


def test_lift_stats(tmp_path):
    lab = np.repeat([1, 2], 500).astype(np.uint8)
    ll = lab.astype(np.int32).copy()
    ll[500:] = 1                                   # item 2 merged into 1
    np.savez(tmp_path / "p.npz", label=lab, lift_label=ll, lift_share=np.ones(1000, np.float32))
    s = lift_stats(tmp_path / "p.npz", min_pts=100)
    assert s["merges"] == 1 and s["splits"] == 0 and abs(s["wrong"] - 0.5) < 1e-9


def test_item_recalls(tmp_path):
    from eval.sam3_synth_votes import item_recalls
    lab = np.repeat([1, 2], 100).astype(np.uint8)
    ll = lab.astype(np.int32).copy()
    ll[100:150] = 0
    np.savez(tmp_path / "p.npz", label=lab, lift_label=ll)
    (tmp_path / "gt.json").write_text(json.dumps({"items": [{"id": 1, "name": "a", "source": "metafood3d"},
                                                             {"id": 2, "name": "b", "source": "primitive"}]}))
    r = item_recalls(tmp_path / "p.npz", tmp_path / "gt.json", min_pts=50)
    assert [(x["kind"], x["recall"]) for x in r] == [("scan", 1.0), ("procedural", 0.5)]
