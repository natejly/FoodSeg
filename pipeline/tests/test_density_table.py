import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodscan.density_table import DEFAULT_CSV, DensityTable, bulk_vs_material, lookup  # noqa: E402


def test_exact():
    d, src, conf = lookup("white rice cooked")
    assert conf == 1.0 and 0.6 < d < 0.75 and "USDA" in src


def test_exact_normalisation_plural_underscore():
    assert lookup("Blueberries")[2] == 1.0
    assert lookup("Fried_Rice")[0] == lookup("fried rice")[0]


def test_synonym():
    d, _, conf = lookup("jasmine rice")
    assert conf == 0.9 and d == lookup("white rice cooked")[0]
    assert lookup("penne")[2] == 0.9


def test_fuzzy_head_noun():
    row, method, _ = DensityTable().match("chocolate milk")
    assert method == "synonym" or row.food == "milk"
    row, method, _ = DensityTable().match("almond milk")
    assert row.food == "milk"


def test_kind_fallback():
    t = DensityTable()
    d, src, conf = t.lookup("zzzz unknown thing", kind="leaves")
    assert conf == 0.35 and d == t.kind_median["leaf"] < 0.3 and "leaf" in src
    d, _, conf = t.lookup("zzzz unknown thing", kind="pile")
    assert d == t.kind_median["pile"]


def test_global_fallback():
    t = DensityTable()
    d, _, conf = t.lookup("zzzz unknown thing")
    assert conf == 0.2 and d == t.global_median


def test_bulk_vs_material():
    bulk, mat, ratio = bulk_vs_material("spinach raw")
    assert mat is not None and bulk < 0.2 < mat and ratio < 0.2
    assert bulk_vs_material("zzzz")[1] is None


def _rows():
    with open(DEFAULT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def test_every_row_has_source_and_basis():
    for r in _rows():
        assert r["source"].strip() and r["basis"].strip() and r["uncertainty"].strip(), r["food"]
        assert r["kind"] in {"pile", "pieces", "leaf", "puree", "liquid", "solid", "scan"}


def test_density_range():
    # floor 0.03 not 0.05: air-popped popcorn is 8 g per cup = 0.034 g/mL (USDA SR Legacy)
    for r in _rows():
        assert 0.03 <= float(r["bulk_density_g_ml"]) <= 1.6, r["food"]
        if r["material_density_g_ml"]:
            assert 0.05 <= float(r["material_density_g_ml"]) <= 2.2, r["food"]


def test_bulk_not_above_material():
    for r in _rows():
        if r["material_density_g_ml"]:
            assert float(r["bulk_density_g_ml"]) <= float(r["material_density_g_ml"]) * 1.02, r["food"]
