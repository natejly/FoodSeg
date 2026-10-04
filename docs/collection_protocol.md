# Data collection protocol

## Equipment
- Kitchen scale, 0.1 g or 1 g resolution.
- Calibration objects (measure and record true volume in `data/calibration.csv`):
  soda can (355 mL), rectangular box (ruler), measuring cup of water, LEGO/Rubik's cube,
  at least one glossy and one dark object.
- Plates: white matte, dark, glossy. Record which one in the CSV.
- iPad with FoodScan installed; charged; ≥ 5 GB free (a 30 s scan ≈ 250 MB).

## Scan technique
- Distance 25–35 cm. LiDAR bias explodes below 20 cm.
- Wait for tracking `normal` before pressing record.
- Orbit slowly for 20–40 s: one full circle at ~45° elevation, then one pass
  nearly top-down. Keep the whole plate in frame.
- Avoid strong backlight and direct sun (LiDAR washout). Indoor diffuse light is ideal.
- If tracking drops to `limited`, keep going; frames are flagged and skipped at fusion.

## Calibration session (do first, once)
For each object: 5 scans, at least two on different plates/backgrounds.
Scan name: `cal_<object>_<n>`, e.g. `cal_can_01`.
Also record 5 scans of an empty plate (`cal_emptyplate_01`) for plane-fit noise.

## Meal session
1. Empty plate on scale → tare. Record `plate_type`.
2. Add item 1 → record `name`, `grams`. Add item 2 → record. Repeat (incremental weighing,
   so items never need to be separated). Write names as you'd say them: "grilled chicken breast",
   "white rice", "steamed broccoli".
3. Scan A: orbit, name `m<meal_id>_orbit`.
4. Scan B: hold the iPad still ~35 cm directly above for 5 s, name `m<meal_id>_overhead`
   (the single-RGB-D baseline).
5. Optional photo of the scale display with the final weight as a backup.
6. Fill `data/meals.csv` immediately; memory is the biggest error source.

Exclusions (state in the report): soups, drinks, sauces poured over food, translucent items
(jelly), food overhanging the plate edge.

## Target
- Calibration: 5 objects × 5 scans + 5 empty plates = 30 scans, one afternoon.
- Meals: 40–60 meals, 3–5 items each, spread over ≥ 3 weeks and ≥ 2 venues; ~150–250 items.
  3 meals/day from mid-October gets there by Nov 15.

## After each session
```
~/Desktop/Thesis/scripts/pull_scans.sh          # iPad → data/scans/
rsync -av ~/Desktop/Thesis/data/ bouchet:/nfs/roberts/project/cpsc4900/cpsc4900_njl36/data/
```
