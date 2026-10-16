"""Mass step: envelope x bulk density with per-kind fallbacks (density_table.estimate_mass, run_meal --mass-rule).

    cd /home/kyle/Desktop/FoodSeg && conda run -n foodseg python -m pytest pipeline/tests/test_mass_step.py -q
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from foodscan import density_table as DT  # noqa: E402
from foodscan.density_table import ENVELOPE_RATIO, estimate_mass, infer_kind, lookup, material_lookup  # noqa: E402
from foodscan.volume import Plane, item_geometry, item_volume_result  # noqa: E402

Z = Plane(np.array([0.0, 0.0, 1.0]), 0.0)


def test_infer_kind():
    assert infer_kind("white rice cooked")[0] == "pile"
    assert infer_kind("spinach raw")[0] == "leaf"
    assert infer_kind("tomato sauce")[0] == "topping"
    assert infer_kind("gravy")[0] == "topping"
    assert infer_kind("penne in tomato sauce")[0] == "topping"   # head noun is "sauce": the dish is not matched first
    assert infer_kind("chicken piece")[0] == "solid"
    assert infer_kind("zzzz unknown") == ("pile", "default")
    assert infer_kind("anything", hint="leaves") == ("leaf", "hint")


def test_ratio_table_has_every_kind():
    assert ENVELOPE_RATIO["pieces"] == 1.08 and ENVELOPE_RATIO["pile"] == 1.18
    assert ENVELOPE_RATIO["topping"] == 3.5 and ENVELOPE_RATIO["leaf"] == 15.0
    for k in DT.MASS_KINDS:
        assert k in DT.RULE_BY_KIND


@pytest.mark.parametrize("name,kind", [("white rice cooked", "pile"), ("chicken piece", "solid"),
                                       ("milk", "liquid"), ("cottage cheese", "puree")])
def test_bulk_rule_kinds(name, kind):
    m = estimate_mass(name, None, 100.0, 50.0, 30.0)
    d = lookup(name, kind)[0]
    assert m["kind"] == kind and m["mass_rule"] == "envelope_bulk" and m["density_kind"] == "bulk"
    assert m["mass_g"] == pytest.approx(100.0 * d)
    assert m["volume_solid_est_ml"] == pytest.approx(100.0 / ENVELOPE_RATIO[kind])
    assert m["mass_confidence"] == pytest.approx(lookup(name, kind)[2])
    assert m["density_source"] == lookup(name, kind)[1]


@pytest.mark.parametrize("name,kind", [("spinach raw", "leaf"), ("tomato sauce", "topping")])
def test_solid_material_rule(name, kind):
    m = estimate_mass(name, None, 60.0, 40.0, 25.0)       # tall: no thin layer
    mat = material_lookup(name, kind)[0]
    assert m["mass_rule"] == "solid_material" and m["density_kind"] == "material"
    assert m["mass_g"] == pytest.approx(60.0 / ENVELOPE_RATIO[kind] * mat)
    assert m["mass_confidence"] == pytest.approx(0.6 * material_lookup(name, kind)[2], abs=1e-3)


def test_thin_layer_rule():
    m = estimate_mass("tomato sauce", None, 5.0, footprint_cm2=20.0, height_mm=2.0)
    mat = material_lookup("tomato sauce", "topping")[0]
    assert m["mass_rule"] == "thin_layer" and m["thickness_mm"] == DT.THIN_THICKNESS_MM["topping"]
    assert m["mass_g"] == pytest.approx(20.0 * 0.15 * mat)
    # exactly at / above the threshold the thin rule does not fire
    assert estimate_mass("tomato sauce", None, 5.0, 20.0, DT.THIN_HEIGHT_MM)["mass_rule"] == "solid_material"
    # leaves too
    assert estimate_mass("spinach raw", None, 5.0, 20.0, 3.0)["mass_rule"] == "thin_layer"
    # a short pile keeps the envelope x bulk rule
    assert estimate_mass("white rice cooked", None, 5.0, 20.0, 1.0)["mass_rule"] == "envelope_bulk"
    # thin rule needs geometry
    assert estimate_mass("tomato sauce", None, 5.0)["mass_rule"] == "solid_material"


def test_oracle_and_confidence_ordering():
    assert estimate_mass("white rice", None, 10.0, bulk_oracle=0.5)["mass_g"] == pytest.approx(5.0)
    assert estimate_mass("spinach raw", None, 30.0, 10.0, 20.0, material_oracle=0.2)["mass_g"] == pytest.approx(30 / 15 * 0.2)
    exact = estimate_mass("white rice cooked", None, 10.0)["mass_confidence"]
    unknown = estimate_mass("zzzz unknown", None, 10.0)["mass_confidence"]
    thin = estimate_mass("spinach raw", None, 5.0, 10.0, 2.0)["mass_confidence"]
    assert exact > unknown and exact > thin


def test_item_geometry_and_result():
    g = np.arange(-0.02, 0.0201, 0.001)
    X, Y = np.meshgrid(g, g)
    pts = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 0.010)]       # 4x4 cm square top at 10 mm
    r = item_geometry(pts, Z)
    assert r["footprint_cm2"] == pytest.approx(16.0, rel=0.15) and r["height_mm"] == pytest.approx(10.0)
    res = item_volume_result(16.0, pts, Z, "white rice cooked")
    assert res["volume_ml"] == res["volume_envelope_ml"] == 16.0
    assert res["volume_solid_est_ml"] == pytest.approx(16.0 / 1.18) and res["kind"] == "pile"
    assert item_volume_result(16.0, pts, Z, "x", kind="topping")["volume_solid_est_ml"] == pytest.approx(16 / 3.5)


@pytest.fixture(scope="module")
def box_scan(tmp_path_factory):
    out = tmp_path_factory.mktemp("box_scan") / "scan"
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "tests" / "make_synthetic_scan.py"), str(out),
                    "--frames", "12"], check=True, capture_output=True)
    return out


def test_run_meal_envelope_vs_legacy(box_scan):
    from foodscan.run_meal import run
    new = run(box_scan, names=["white rice cooked"], nutrition=False, up_hint=np.array([0.0, 0.0, 1.0]))
    old = run(box_scan, names=["white rice cooked"], nutrition=False, up_hint=np.array([0.0, 0.0, 1.0]),
              mass_rule="legacy")
    n, o = new["items"][0], old["items"][0]
    assert new["mass_rule"] == "envelope" and old["mass_rule"] == "legacy"
    assert n["volume_ml"] == pytest.approx(o["volume_ml"], rel=1e-3) and n["volume_envelope_ml"] == n["volume_ml"]
    assert n["mass_g"] == pytest.approx(n["volume_envelope_ml"] * 0.668) and n["mass_rule"] == "envelope_bulk"
    assert "mass_g" not in o and "mass_rule" not in o       # legacy without nutrition: no mass, as before
    assert new["totals"]["mass_g"] == pytest.approx(n["mass_g"])
    with pytest.raises(ValueError):
        run(box_scan, mass_rule="nope")


def test_synthetic_scan_smoke():
    batch = ROOT / "data" / "synth" / "hard_v12"
    if not batch.exists():
        pytest.skip("hard_v12 not available")
    from pipeline.eval import synth_volume_data as D
    from pipeline.tools import mass_step_synth as MS
    man = D.read_manifest(batch)
    sid = next(s for s, m in sorted(man.items()) if m.get("ok"))
    rows = MS.scan_rows(D.load_scan(batch, sid, man), {})
    ok = [r for r in rows if r.get("ok")]
    assert ok and all(r["m_new"] >= 0 and r["rule"] in ("envelope_bulk", "solid_material", "thin_layer") for r in ok)
    txt = MS.tables(rows)
    assert "Dish level" in txt
