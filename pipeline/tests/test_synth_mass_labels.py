"""Tests for the USDA mass side labels (pipeline/eval/synth_mass_labels.py) and the --mass-source option.

    cd /home/kyle/Desktop/FoodSeg && conda run -n foodseg python -m pytest pipeline/tests/test_synth_mass_labels.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.eval import synth_mass_labels as ML  # noqa: E402
from pipeline.eval import volume_head as VH  # noqa: E402
from pipeline.eval import volume_head_envelope as VHE  # noqa: E402
from pipeline.foodscan.density_table import DensityTable  # noqa: E402

V12 = ROOT / "data" / "synth" / "hard_v12" / "scans"
TABLE = DensityTable()


def _first_scan():
    for p in sorted(V12.glob("*/gt.json")):
        return p.parent
    pytest.skip("no hard_v12 scan")


def test_item_label_arithmetic():
    it = dict(id=3, category="white rice", shape={"kind": "pile"}, volume_ml=100.0, mass_g=80.0, density_g_ml=0.8,
              mass_source="density")
    r = ML.item_label(it, 200.0, TABLE)
    row, _, _ = TABLE.match("white rice", "pile")
    assert r["mass_usda_bulk_g"] == pytest.approx(200.0 * row.bulk)
    assert r["mass_usda_material_g"] == pytest.approx(100.0 * row.material)
    assert r["mass_asset_g"] is None and r["mass_source_synth"] == "density"
    r2 = ML.item_label(dict(it, mass_source="asset"), None, TABLE)
    assert r2["mass_asset_g"] == 80.0 and r2["mass_usda_bulk_g"] is None


def test_side_file_has_all_keys(tmp_path):
    src = _first_scan()
    dst = tmp_path / src.name
    dst.mkdir()
    (dst / "gt.json").write_text((src / "gt.json").read_text())
    items = ML.write_scan("hard_v12", dst, ML.read_envelopes(), TABLE)
    d = json.loads((dst / "gt" / "mass_labels.json").read_text())
    gt = json.loads((src / "gt.json").read_text())
    assert len(d["items"]) == len(gt["items"]) == len(items)
    for it in d["items"]:
        assert set(ML.KEYS) <= set(it)
    assert (src / "gt" / "mass_labels.json").exists()           # the real run wrote it next to the scan
    assert json.loads((src / "gt.json").read_text()) == gt      # gt.json untouched


def test_ratio_stats():
    s = ML.log_ratio_stats([2.0, 2.0, 0.5], [1.0, 1.0, 1.0])
    assert s["med_ratio"] == pytest.approx(2.0)
    assert s["med_abs_log"] == pytest.approx(np.log(2.0))
    s = ML.log_ratio_stats([1.0, 4.0], [1.0, 1.0])
    assert s["med_abs_log"] == pytest.approx(np.log(4.0) / 2)


def test_apply_mass_source_replaces_and_nans():
    items = [dict(batch="hard_v12", scan=_first_scan().name, item_id=1, mass_g=123.0),
             dict(batch="hard_v12", scan="nope", item_id=1, mass_g=5.0)]
    out = ML.apply_mass_source([dict(i) for i in items], "usda_bulk")
    lk = ML.load_mass_lookup("usda_bulk", batches=["hard_v12"])
    assert out[0]["mass_g"] == pytest.approx(lk[("hard_v12", items[0]["scan"], 1)])
    assert np.isnan(out[1]["mass_g"])
    assert ML.apply_mass_source(items, "synth") is items


def test_mass_source_option_runs_on_tiny_subset():
    feat = VH.OUT / "features.npz"
    if not feat.exists():
        pytest.skip("features.npz missing")
    D0 = VH.Data(feat, "valid", "envelope")
    D1 = VH.Data(feat, "valid", "envelope", "usda_bulk")
    assert D0.n == D1.n and not np.allclose(D0.mass, D1.mass, equal_nan=True)
    tiny = D1.subset(np.isin(D1.scan, np.unique(D1.scan)[:6]))
    res = VH.run_protocol(tiny, "catcv", ["none", "global"], repeats=1)
    assert res["global"]["pred"].shape[1] > 0
    VHE.MASS_SOURCE = "usda_bulk"
    try:
        items = VHE.E.read_items(VH.ENV_CSV)
        dens, ratio = VHE.densities(items)
        assert dens["_all"]["bulk"] > 0
    finally:
        VHE.MASS_SOURCE = "synth"
