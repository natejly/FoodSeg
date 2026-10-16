# Bulk-density table for the real-meal mass step

Files: `data/density/bulk_density.csv` (163 rows), `data/density/sources.md` (provenance),
`pipeline/foodscan/density_table.py` (lookup + CLI), `pipeline/tests/test_density_table.py`.
Note `data/` is in `.gitignore`: the CSV must be force-added (`git add -f`) if it is to be committed.

## Why

The synthetic studies (`results/rq1/envelope.md`, `results/volume_head/summary.md`) say the
pipeline should estimate *envelope* volume (the visible top surface closed to the container, voids
included) and convert with a *bulk* density. With oracle volume, the per-kind density table is then
the dominant mass error (32 % item, 20 % dish). The solid-volume route needs the other (material)
density, which is 1.5-10x higher for piles and leaves, so the two must not be mixed:

    mass = envelope_volume_ml x bulk_density      (this table)
    mass = solid_volume_ml    x material_density  (material column, where known)

## How it was built

1. ~110 common plated foods and food kinds were chosen from the Nutrition5k ingredient list,
   the synthetic category lists and general plated-meal coverage; ~50 more were added to reach the
   synonym cover below.
2. For each food, a USDA FoodData Central SR Legacy household-measure record was picked
   (cooked rice "1 cup = 158 g", raw spinach "1 cup = 30 g", shredded cheddar "1 cup = 113 g", ...)
   and bulk density = gram weight / volume of the measure. 131 rows come this way; the exact fdc_id
   and modifier are in `basis`.
3. Rows with no volume measure (bakery, fries, bacon, sushi, composite dishes, whole muscle pieces)
   are author estimates from food-engineering tables and flagged `is_estimate=1` (32 rows).
4. `kind` is pile / pieces / leaf / puree / liquid / solid (no `scan` rows yet; MetaFood3D scans
   carry measured mass). Material density is filled for 144 rows from Rahman / Choi-Okos.
   Medians of bulk density: pile 0.61, pieces 0.63, puree 1.01, liquid 1.02, solid 1.05, leaf 0.20.

Both loose-cup and whole-piece variants exist where they differ a lot: `chicken diced` 0.59 vs
`chicken piece` 1.05, `cheese diced` 0.56 vs `cheese block` 1.07, `egg boiled chopped` 0.58 vs
`egg whole boiled` 1.03. Pick by what the scan shows (a pile of chunks vs one intact piece).

## Using it

    from foodscan.density_table import lookup
    d, source, conf = lookup(vlm_name, kind)       # conf 1.0 exact, 0.9 synonym, 0.5-0.8 fuzzy,
    mass_g = envelope_volume_ml * d                #      0.35 kind median, 0.2 global median

`Nutrition(density_lookup=lookup)` (optional hook in `foodscan/nutrition.py`) tries the table first
and uses it when confidence >= `density_min_conf` (0.5); otherwise the old `densities.csv`
logic runs, so default behaviour is unchanged. Carry `conf` and `source` into the per-item output and
flag items with conf < 0.5 as low-confidence mass. Matching normalises case/plural/underscores,
then synonyms, then head-noun containment (last word must agree: "almond milk" -> milk, not almonds),
then difflib >= 0.88.

CLI: `cd pipeline && python -m foodscan.density_table [--list-misses] [--compare] [--envelope]`.
Coverage (2026-10-16): Nutrition5k 551 ingredient names 99.8 % mapped to a named row (75 exact,
434 synonym, 41 fuzzy, 1 fallback: "plate only"); synthetic DENSITY categories 66/66; envelope_items
categories 101/101. Caveat: "mapped" includes broad synonyms (e.g. all hard cheeses -> `cheese block`,
all fish -> `fish fillet`); exact+synonym is the trustworthy part, fuzzy (41 names) less so.

## Limits

- USDA cup weights are measuring-cup packing, not plate packing. Expect +-25 % for piles,
  +-40 % for leaves, +-20 % for pieces; the table has no information on how a given plate was loaded.
- Cooked-state ambiguity: the VLM name rarely says raw/cooked, and the rows assume as-served (cooked).
- Sauces/dressings/toppings: bulk density of the sauce itself is ~1.0, but a thin topping layer's
  *envelope* (closed to the container floor) is much larger than its volume. This is a volume-step
  issue, not a table issue (see disagreements below).
- 32 rows are estimates; confidence is not propagated to an interval in the pipeline yet.
- No real-meal check has been made. Everything here is from reference tables.

## Sanity comparison with the synthetic densities (not ground truth)

`python -m foodscan.density_table --compare` (synthetic DENSITY dict vs this table, > 30 %):
broccoli 0.40 vs 0.66, cheese cube 1.10 vs 0.56, cherry tomato 0.90 vs 0.63, fried mushrooms 0.60
vs 0.30, hashbrown 0.60 vs 0.35, kale 0.20 vs 0.09, mac and cheese 0.95 vs 0.58, X
vs 0.48 (shredded, wrong state), mixed greens 0.18 vs 0.12, olive 0.95 vs 0.60, risotto 0.90 vs 0.66,
sausage 1.05 vs 0.80, tortilla chips 0.15 vs 0.11. Main pattern: the synthetic table uses
solid-ish densities for cheese, olives, tomatoes, mac and cheese, sausage (1.0 vs the table's
0.56-0.95), and is ~2x too dense for kale/mixed greens/hashbrown/mushrooms relative to USDA cups.
Of 66 synthetic categories, 12 differ by > 30 % (melted cheese no longer maps to shredded).

`--envelope` compares pooled mass / GT envelope volume per category from `results/rq1/envelope_items.csv`
(synthetic masses: asset or synthetic-table masses) with this table: 51 of 101 categories differ by
> 30 %. Largest: leaves (lettuce 0.01, spinach 0.01, arugula 0.00, kale 0.02, mixed greens 0.02 vs
0.09-0.20 in the table), toppings (tomato sauce 0.00, salsa 0.12, melted cheese 0.28, pesto 0.13,
sour cream 0.22, gravy 0.30 vs ~1.0), tofu, sushi, almonds (0.28 vs 0.60), broccoli 0.36 vs 0.66,
blueberries 0.42 vs 0.63, strawberries 0.35 vs 0.61. This is the envelope itself: a small mass of
leaves/toppings/pile in a big container has an envelope tens of times its solid volume, so
mass/envelope is far below any food's real bulk density. The bulk table is only right if the
envelope is a physical heap volume; for sparse items the pipeline needs a per-kind correction or
to use solid volume x material density there. Use the synthetic numbers only to see where the
volume model and the table disagree, not as evidence for either.

## Recalibrating with weighed real meals

Weigh single-component servings (rice, pasta, salad, fries, berries, nuts, diced fruit/veg, shredded
cheese, a steak) on a kitchen scale, run the scan, and record envelope volume per item.
Per kind (and per food where n >= 5): correction c_k = median(measured_mass / (envelope_volume x table_bulk)),
then apply `bulk' = bulk x c_k`. Report the IQR of the ratio per kind as the density uncertainty. Aim for
>= 10 servings per kind, with several plating styles (heaped vs flat, bowl vs plate); kinds likely
to move most are `leaf` (compaction) and `pile` (packing). Keep corrections separate from the CSV
(a `correction_by_kind.csv`) until there are enough weighings to justify editing rows.
