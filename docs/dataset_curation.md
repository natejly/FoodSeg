# Curating the synthetic full-capture dataset

Draft 2026-10-13. Companion to `docs/synthetic_data.md` (how scenes are made); this file is
about *which* scenes to make, how to check them, and how to split and version them.

## What the dataset is for

One record = one simulated iPad capture: a 24-view hand-held orbit (RGB, LiDAR depth,
confidence, poses) plus ground truth. Three consumers, in priority order:

1. **3D instance segmentation on the fused cloud** (DGCNN baseline, PTv3 / Sonata fine-tune,
   SAM 3 lift + 3D refine): `gt/seg_points.npz`.
2. **Per-item volume** from segmented points against the dish surface: `gt.json` volumes.
3. **Per-item mass** via volume x learned density: MetaFood3D weights where the asset has one.

Everything below is chosen so that numbers on the synthetic test split say something about
real meals, and so that the real weighed test set (section 7) is the final word.

## 1. Composition: sample meals like real ones

Real dish statistics, Nutrition5k (5,006 cafeteria dishes, `metadata/dish_metadata_cafe*.csv`,
ingredients with grams; computed 2026-10-13, to become `synth/composition.py`):

| | Nutrition5k |
|---|---|
| visible components per dish (ingredients >= 10 g) | mean 2.67; 1: 40 %, 2: 12 %, 3: 22 %, 4: 10 %, 5+: 15 % |
| dish mass | p10 41 g, median 145 g, p90 381 g |
| distinct visible ingredients | 218; top 40 = 67 % of components |
| most common | broccoli, chicken, mixed greens, cauliflower, scrambled eggs, carrot, pizza, berries, bacon, olives, cucumbers, arugula, fish, cherry tomatoes, white rice ... |
| strongest pairs | brown rice + salmon, egg whites + sweet potato, bacon + scrambled eggs, pork + white rice |

Plan:
- Map each Nutrition5k ingredient to an asset category, a procedural fill / topping, or
  "not rendered" (oil, salt, dressing). Sample a real dish's visible ingredient list and
  per-ingredient grams, then pick assets of those categories and scale fills to the grams
  (volume = grams / density). This gives real co-occurrence and portion sizes for free.
- **Gaps** (common in real meals, absent from our assets): leafy greens / salad, scrambled
  eggs and egg whites, cherry tomatoes and olives as piles, cubed / sliced vegetables
  (squash, cucumber slices). Candidates: OmniObject3D food scans (CC BY 4.0, metric, no
  weights), procedural leaf / cube / scramble generators. Leafy salad is the hardest case
  for every method (thin, interleaved, no clean surface) and worth its own stratum.
- **Coverage today** (`python -m synth.composition`, ~90 most common ingredients mapped):
  of 13,157 real visible components, 57.5 % map to a MetaFood3D scan category, 7.1 % to a
  procedural fill, 1.1 % to a topping, **34.3 % cannot be rendered**; 43.9 % of dishes can
  be rendered in full. Gaps by size: leafy greens 1,476 components, unmapped long tail
  1,121, small piles (olives, chickpeas, corn, brussels sprouts) 616, scrambled eggs /
  egg whites 531, cubed vegetables (roasted potato, squash, zucchini) 482, whole fruit we
  lack (orange, lemon, pear) 206. Small piles, scrambles and cubes are cheap procedural
  generators (~12 % of components); leafy greens are the hard, valuable one.
- **After adding primitives** (olive, chickpea, brussels sprout, roasted potato / squash
  cubes, zucchini slices, orange, lemon, pear) and fills (scrambled eggs, egg whites, brown
  rice, quinoa), index `data/synth/assets_v2.jsonl`: 66.6 % scan / primitive, 11.1 % fill,
  1.1 % topping, **21.2 % gap**; **62.6 % of dishes render in full** (was 43.9 %). Left:
  leafy greens 1,476 components, unmapped tail 1,121, corn kernels / radishes 111,
  sliced meat 75.
- Keep a share (~20 %) of uniform random compositions so models don't learn the cafeteria
  prior only.

## 2. Difficulty strata: tag every scene, balance on purpose

Tags computed after rendering from ground truth (`synth.manifest`):

| tag | definition |
|---|---|
| `n_items` | food items in the scene |
| `touching_pairs` | item pairs whose GT surfaces come within 3 mm (sample spacing ~1 mm + collision margins) |
| `same_category_pairs` | touching pairs of the same category (piles: the counting case) |
| `similar_colour_pairs` | touching pairs whose mean rendered colour differs by < 0.1 (linear RGB) |
| `stacked` | a topping or an item resting on another food |
| `container` | plate / bowl |
| `clutter` | non-food objects on the table |
| `min_item_points` | fused points on the smallest item (tiny items are a failure mode) |

Report every metric per stratum, not only overall: "PTv3 beats SAM 3 lifting on touching
same-colour pairs" is the thesis claim; an overall mIoU hides it. Sample to targets
(e.g. at least 30 % of scenes with a same-category touching pair) rather than hoping.

## 3. Quality control: what gets a scene dropped or flagged

Automatic (`synth.manifest`):
- render failed / CPU fallback failed (already logged by `synth.batch`);
- any item `off_container` or visible in < 3 frames (physics threw it away);
- an item with < 50 fused points (cannot be segmented or measured; flag, keep for stats);
- fused-to-GT median distance above 2 mm on above-table points (pose / fusion bug);
- `scene_volume_ml` differs from the spec volume by > 1 % (mesh import / scale bug);
- interpenetration after physics: GT points of one item inside another's hull.

Manual: a contact sheet of a random 2 % of each batch (`contact_sheet.jpg` style), looked
at before the batch is used. Every problem found becomes an automatic check.

## 4. Splits: disjoint by object, plus a held-out-category split

- **Object-disjoint** train / val / test over MetaFood3D *objects* (an object's mesh
  never appears in two splits), stratified by category: 80 / 10 / 10. Procedural fills
  share types across splits but draw disjoint parameter seeds.
- **Novel-category test**: ~10 categories never seen in training (choose across shape
  types: a pile food, a flat food, a round fruit) to measure generalisation.
- Scenes inherit the split of their assets; a scene mixing splits is not generated.
- The split file is generated once, committed, and never regenerated (`data/synth/splits.json`).

## 5. Size and budget

Local rate: 1.5-8.5 min per 24-view scene at 960x720 / 32 spp on the shared RTX 2060,
mostly physics (about 10 scenes / hour). Proposed v1: 2,000 train / 200 val / 300 test
(+ 200 novel-category) = 2,700 scenes, roughly 270 local GPU-hours or a day on cluster
GPUs. Start with 300 to train the DGCNN baseline and check the learning curve before
scaling; if 300 -> 1,000 scenes changes little, stop there.

## 6. Versioning and provenance

- Every `gt.json` records the preset, all spec parameters, LiDAR and phone-camera
  parameters, and (to add) the generator git commit.
- `manifest.jsonl` (one line per scan: split, tags, QC flags) is the dataset index; a
  dataset version = a manifest + the commit that made it (`data/synth/<name>/MANIFEST`).
- Dataset card with sources and licences: MetaFood3D is CC BY-NC 4.0, so the derived
  dataset is non-commercial; OmniObject3D CC BY 4.0 would need attribution.

## 7. The real test set (what makes the numbers mean something)

20-30 real iPad captures of plated meals, each item weighed separately on a kitchen scale
before plating, volume where measurable (water displacement for solids), photos of each
item. Instance masks: SAM 3 on the frames, corrected by hand on ~5 frames per scan, then
lifted to 3D. Match the strata above (touching, piles, bowl soups, a salad). Never used
for tuning; report every method once at the end.

## Open questions

- Fill scale from grams needs a density per Nutrition5k ingredient (FNDDS / INFOODS
  table in `foodscan/nutrition.py` already exists).
- How much leafy / thin food can a procedural generator make convincingly?
- Domain gap check: train on synthetic, evaluate on the real set; does fine-tuning on
  ~10 real scans close it (the literature says synthetic-pretrain + real-fine-tune wins)?
