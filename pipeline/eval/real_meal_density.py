"""Hard-coded density post-step for the two real-meal example scans (NOT a general fix).

    python pipeline/eval/real_meal_density.py RESULT_DIR --assign "1=kiwi,2=sweet potato" --density "kiwi=1.05,sweet potato=1.0"

# ponytail: hard-coded for this example. SAM 3 prompts collapse kiwi -> "fruit" (segment.SIMPLE_NAMES) and the
# kind=pile lookup then uses 0.613 g/mL; the general fix (keep the specific name, whole-item kind) is still open.
# Item ids are assigned by hand (checked against volumes) because the instance names are all "fruit".
Keeps the original run in result_default_density.json; rewrites name, density and mass in result.json.
"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("result_dir")
ap.add_argument("--assign", required=True, help='"id=name,..."')
ap.add_argument("--density", required=True, help='"name=g/mL,..."')
a = ap.parse_args()
rd = Path(a.result_dir)
orig = rd / "result_default_density.json"
if not orig.exists():
    shutil.copy(rd / "result.json", orig)
res = json.loads(orig.read_text())
name_of = {int(i): n for i, n in (p.split("=") for p in a.assign.split(","))}
dens = {k: float(v) for k, v in (p.split("=") for p in a.density.split(","))}
for it in res["items"]:
    n = name_of[it["id"]]
    it.update(name=n, density_used_g_ml=dens[n], density_source="hard-coded example override", density_kind="whole",
              mass_rule="override_envelope", mass_g=it["volume_envelope_ml"] * dens[n])
res["totals"]["mass_g"] = sum(it["mass_g"] for it in res["items"])
(rd / "result.json").write_text(json.dumps(res, indent=1))
for it in res["items"]:
    print(f"{it['name']}: {it['volume_ml']:.0f} mL x {it['density_used_g_ml']} = {it['mass_g']:.0f} g")
