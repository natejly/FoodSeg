"""Food name -> density -> nutrients, via USDA FoodData Central (FNDDS) and a density table.

    from foodscan.nutrition import Nutrition
    n = Nutrition(api_key=os.environ["FDC_API_KEY"])     # DEMO_KEY works for testing (30 req/h)
    item = n.lookup("white rice cooked")
    est = n.estimate(item, volume_ml=180)
    est.mass_g, est.kcal, est.protein_g, est.carb_g, est.fat_g

Data sources
- FoodData Central search API, dataType "Survey (FNDDS)": foods as eaten, incl. cooked
  and mixed dishes; nutrients per 100 g. https://fdc.nal.usda.gov/api-guide/
- Density: a small hand-curated table in data/densities.csv (g/mL), seeded from
  FAO/INFOODS Density Database v2.0 and FNDDS "cup" portion weights. Add rows as your
  food set grows; every row should cite its source.
- Overrides: data/name_overrides.csv maps your free-text names to a fixed fdcId, so the
  VLM's wording never silently changes the lookup between runs.

All lookups are cached on disk (data/fdc_cache.json) to stay under the 1,000 req/h limit.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import urllib.parse
import urllib.request

FDC_SEARCH = "https://api.nal.usda.gov/fdc/v1/foods/search"
NUTRIENT_IDS = {"kcal": 1008, "protein_g": 1003, "carb_g": 1005, "fat_g": 1004}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class FoodItem:
    query: str
    fdc_id: int
    description: str
    data_type: str
    kcal_per_100g: float
    protein_per_100g: float
    carb_per_100g: float
    fat_per_100g: float
    density_g_per_ml: float | None
    density_source: str


@dataclass
class Estimate:
    name: str
    volume_ml: float
    density_g_per_ml: float
    mass_g: float
    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    fdc_id: int
    fdc_description: str


class Nutrition:
    def __init__(self, api_key: str = "DEMO_KEY", data_dir: Path | str = DEFAULT_DATA_DIR,
                 data_type: str = "Survey (FNDDS)"):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.cache_path = self.data_dir / "fdc_cache.json"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.densities = self._load_csv(self.data_dir / "densities.csv")
        self.overrides = self._load_csv(self.data_dir / "name_overrides.csv")

    # ---- public -----------------------------------------------------------

    def lookup(self, name: str) -> FoodItem:
        key = _norm(name)
        ov = self.overrides.get(key)
        if ov and ov.get("fdc_id"):
            hit = self._fdc_search(ov.get("fdc_query") or name, fdc_id=int(ov["fdc_id"]))
        else:
            hit = self._fdc_search(name)
        dens, src = self.density_for(name, hit["description"])
        nut = hit["nutrients"]
        return FoodItem(
            query=name, fdc_id=hit["fdcId"], description=hit["description"],
            data_type=hit["dataType"],
            kcal_per_100g=nut.get("kcal", 0.0), protein_per_100g=nut.get("protein_g", 0.0),
            carb_per_100g=nut.get("carb_g", 0.0), fat_per_100g=nut.get("fat_g", 0.0),
            density_g_per_ml=dens, density_source=src,
        )

    def estimate(self, item: FoodItem, volume_ml: float, density_override: float | None = None) -> Estimate:
        dens = density_override or item.density_g_per_ml
        if dens is None:
            raise ValueError(f"no density for '{item.query}' ({item.description}); add it to densities.csv")
        mass = volume_ml * dens
        f = mass / 100.0
        return Estimate(
            name=item.query, volume_ml=volume_ml, density_g_per_ml=dens, mass_g=mass,
            kcal=item.kcal_per_100g * f, protein_g=item.protein_per_100g * f,
            carb_g=item.carb_per_100g * f, fat_g=item.fat_per_100g * f,
            fdc_id=item.fdc_id, fdc_description=item.description,
        )

    def density_for(self, name: str, fdc_description: str = "") -> tuple[float | None, str]:
        """Best density row: exact override name, then longest keyword match."""
        key = _norm(name)
        row = self.densities.get(key)
        if row:
            return float(row["density_g_per_ml"]), row.get("source", "densities.csv")
        best, best_len = None, 0
        hay = f"{key} {_norm(fdc_description)}"
        for k, r in self.densities.items():
            if k in hay and len(k) > best_len:
                best, best_len = r, len(k)
        if best:
            return float(best["density_g_per_ml"]), best.get("source", "densities.csv") + " (keyword)"
        return None, "missing"

    # ---- FDC --------------------------------------------------------------

    def _fdc_search(self, query: str, fdc_id: int | None = None) -> dict:
        ck = f"{self.data_type}|{query}|{fdc_id or ''}"
        if ck in self.cache:
            return self.cache[ck]
        # POST with a JSON body: the GET form's URL-encoded "Survey (FNDDS)" was
        # rejected intermittently by the API's front end.
        url = FDC_SEARCH + "?" + urllib.parse.urlencode({"api_key": self.api_key})
        body = json.dumps({"query": query, "dataType": [self.data_type], "pageSize": 10}).encode()
        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise RuntimeError(
                        "FDC rate limit hit (DEMO_KEY: 30/h; real key: 1,000/h). "
                        "Get a free key at https://fdc.nal.usda.gov/api-key-signup") from e
                if attempt == 2:
                    raise RuntimeError(f"FDC request failed for '{query}': HTTP {e.code}") from e
                time.sleep(2 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise RuntimeError(f"FDC request failed for '{query}': {e}") from e
                time.sleep(2 * (attempt + 1))
        foods = data.get("foods", [])
        if not foods:
            raise LookupError(f"FDC: nothing found for '{query}' in {self.data_type}")
        if fdc_id is not None:
            chosen = next((f for f in foods if f["fdcId"] == fdc_id), None)
            if chosen is None:
                raise LookupError(f"FDC: override fdcId {fdc_id} not in results for '{query}'")
        else:
            chosen = _rank(query, foods)[0]
        hit = {
            "fdcId": chosen["fdcId"], "description": chosen["description"],
            "dataType": chosen.get("dataType", ""),
            "nutrients": _extract_nutrients(chosen.get("foodNutrients", [])),
            "alternatives": [(f["fdcId"], f["description"]) for f in foods[:5]],
        }
        self.cache[ck] = hit
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1))
        return hit

    # ---- files ------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        with open(path, newline="") as f:
            return {_norm(r["name"]): r for r in csv.DictReader(f) if r.get("name")}


_PENALTY_WORDS = ("with sauce", "with gravy", "as ingredient", "baby food", "fast food",
                  "restaurant", "frozen", "canned", "from fast food", "ns as to")


def _rank(query: str, foods: list[dict]) -> list[dict]:
    """Prefer the plainest description that covers the query words.

    FDC's own ranking often puts compound dishes first ("Chicken breast, grilled
    with sauce, skin eaten" for "grilled chicken breast"). Score = query-word
    coverage minus a penalty per comma-separated qualifier and per flagged phrase.
    """
    qwords = set(_norm(query).split())

    def score(f: dict) -> float:
        d = _norm(f["description"])
        dwords = set(d.replace(",", " ").split())
        coverage = len(qwords & dwords) / max(len(qwords), 1)
        extra = len(dwords - qwords)              # words the user didn't ask for
        qualifiers = d.count(",")
        penalty = sum(1 for w in _PENALTY_WORDS if w in d)
        return coverage * 10 - 0.5 * qualifiers - 0.3 * extra - 2 * penalty

    return sorted(foods, key=score, reverse=True)


def _extract_nutrients(food_nutrients: list[dict]) -> dict[str, float]:
    out = {}
    for fn in food_nutrients:
        nid = fn.get("nutrientId") or fn.get("nutrient", {}).get("id")
        val = fn.get("value", fn.get("amount"))
        for k, want in NUTRIENT_IDS.items():
            if nid == want and val is not None:
                out[k] = float(val)
    return out


def _norm(s: str) -> str:
    return " ".join(s.lower().replace(",", " ").split())


def to_dict(x) -> dict:
    return asdict(x)
