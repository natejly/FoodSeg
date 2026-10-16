# Sources for bulk_density.csv

Built 2026-10-16. 163 rows: 131 derived from a USDA record, 32 flagged `is_estimate=1`
(author estimates from food-engineering tables, or a USDA record for a different but
structurally similar food, marked "PROXY" in `basis`).

## Primary: USDA FoodData Central, SR Legacy (April 2018 release)

`https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip`,
file `food_portion.csv` (household-measure gram weights). For every row marked
"SR Legacy fdc_id N ... portion '...'" in `basis`:

    bulk_density_g_ml = gram_weight / (amount * 236.588 mL)      # 1 US cup = 236.588 mL

The fdc_id, record description and modifier (e.g. "cup, shredded", "cup spaghetti not packed")
are in the `basis` column, so each value can be looked up at
`https://fdc.nal.usda.gov/fdc-app.html#/food-details/<fdc_id>/portions`.
The USDA cup weights are for foods measured *in a cup* (spooned, not tamped unless the modifier says
"packed"). That is a reasonable stand-in for a pile on a plate, not an exact one: a heaped plate
can be 10-25 % denser (packing) or looser (tossed salad) than a measuring cup.

## Secondary (material density, estimates)

- Rahman, M.S. (2009) *Food Properties Handbook*, 2nd ed., CRC Press: true (material) densities
  of muscle foods, fish, cheese, eggs, chocolate, starch and sugar solids.
- Choi, Y. and Okos, M.R. (1986) "Effects of temperature and composition on the thermal properties
  of foods", in *Food Engineering and Process Applications*: composition-based density model
  (butter 0.91, oils 0.915, water 1.0, starch 1.5, protein 1.33, fat 0.92, ash 2.4).
- Katz, E.E. and Labuza, T.P. (1981) J Food Sci 46:403: densities of meat/fish/dairy.
- FAO/INFOODS Density Database v2.0 (638 foods; `https://openknowledge.fao.org/handle/20.500.14283/ap805e`):
  used only to confirm beverage and liquid-food ranges (0.98-1.05 g/mL). Its Excel table was not
  downloaded here, so no row cites a specific INFOODS record.

## Known weak points

- Bakery items (bread, bagel, croissant, muffin, cake, pizza, pita, cornbread group), french fries,
  bacon, sushi, burrito, composite dishes: author estimates; no cup/volume record exists.
- Whole-piece rows (steak, chicken piece, fish fillet, cheese block, boiled egg) are material
  densities of the muscle/curd, 1.03-1.07: right when the piece is intact, too high for breaded
  or fried items (0.8-0.9).
- PROXY rows: `ground beef cooked` and `beef diced cooked` use diced cooked pork (cup 135 g);
  `sour cream` uses cream-cheese spread (240 g/cup); `soup broth based`, `cream soup`, `shrimp`
  (canned), `lasagna` (vegetable lasagna).
- SR portion amounts occasionally disagree with intuition (e.g. rich vanilla ice cream cup = 107 g);
  `ice cream` was therefore taken from the 1/2-cup 66 g records. `butter` SR cup (227 g) is the
  8-oz convention; true density 0.91 is used.
