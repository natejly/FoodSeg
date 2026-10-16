# USDA vs synthetic mass labels

473 items in 92 scans; 459 with a GT envelope volume (mass_usda_bulk defined). Reproduce: `python -m pipeline.eval.synth_mass_labels`.


## 1. Accuracy check: USDA-based mass vs real asset weights (MetaFood3D scanned assets)

Ratio = USDA mass / real weight (1.0 is perfect); the synthetic label equals the real weight here by construction. Material path uses volume_ml x USDA material density; bulk path uses GT envelope x USDA bulk density (the envelope includes air gaps and the column fill, so it is expected to overshoot solid mass for single/piece items).

| group | n | bulk: median ratio | bulk IQR | bulk median abs log | material: median ratio | material IQR | material median abs log |
|---|---|---|---|---|---|---|---|
| all asset items | 263 | 1.16 | 0.77-1.67 | 0.40 | 1.27 | 0.94-1.60 | 0.27 |
| kind pile | 135 | 1.03 | 0.74-1.40 | 0.32 | 1.19 | 0.93-1.34 | 0.22 |
| kind single | 128 | 1.36 | 0.79-1.82 | 0.49 | 1.43 | 0.97-1.75 | 0.42 |


Per category (n >= 2):


| group | n | bulk: median ratio | bulk IQR | bulk median abs log | material: median ratio | material IQR | material median abs log | USDA row |
|---|---|---|---|---|---|---|---|---|
| Almonds | 14 | 1.47 | 1.21-2.06 | 0.56 | 0.95 | 0.85-0.98 | 0.17 | almonds |
| Asparagus | 10 | 0.66 | 0.22-1.64 | 1.25 | 0.91 | 0.73-1.70 | 0.49 | green beans |
| Bacon | 8 | 1.10 | 0.28-1.82 | 0.93 | 1.14 | 0.94-1.20 | 0.18 | bacon |
| Bell Pepper | 2 | 1.82 | 1.82-1.82 | 0.60 | 1.63 | 1.63-1.63 | 0.49 | bell pepper chopped |
| Blueberries | 23 | 1.23 | 0.74-2.87 | 0.64 | 0.73 | 0.73-1.61 | 0.32 | blueberries |
| Broccoli | 25 | 1.67 | 1.44-1.89 | 0.51 | 1.72 | 1.70-1.94 | 0.54 | broccoli florets |
| Cauliflower | 6 | 0.85 | 0.76-0.91 | 0.20 | 1.30 | 1.30-1.30 | 0.26 | cauliflower |
| Chicken_Nugget | 2 | 4.15 | 3.34-5.15 | 1.42 | 1.81 | 1.76-1.87 | 0.59 | chicken piece |
| Chicken_tender | 3 | 1.99 | 1.83-2.40 | 0.69 | 1.19 | 1.14-1.31 | 0.18 | chicken piece |
| Chicken_thighs | 3 | 1.76 | 1.44-1.97 | 0.57 | 1.26 | 1.21-1.36 | 0.23 | chicken piece |
| Coleslaw | 2 | 0.83 | 0.81-0.85 | 0.18 | - | - | - | coleslaw |
| Cookies | 2 | 2.42 | 2.09-2.80 | 0.88 | 1.60 | 1.60-1.60 | 0.47 | chocolate |
| Cucumber | 36 | 0.72 | 0.34-1.07 | 0.44 | 0.93 | 0.93-1.34 | 0.07 | cucumber sliced |
| Falafel | 22 | 1.06 | 0.97-1.27 | 0.17 | - | - | - | burrito |
| French_toast | 2 | 1.24 | 1.16-1.32 | 0.21 | 2.62 | 2.59-2.66 | 0.96 | cornbread |
| Fried mushrooms | 8 | 0.69 | 0.54-0.79 | 0.37 | 1.31 | 1.26-1.31 | 0.27 | mushrooms sliced |
| Grapes | 3 | 0.67 | 0.53-0.79 | 0.40 | 0.94 | 0.86-0.94 | 0.06 | grapes |
| Green_beans | 2 | 5.39 | 4.96-5.87 | 1.69 | 2.31 | 2.31-2.31 | 0.84 | green beans |
| Ice_cream | 2 | 0.78 | 0.77-0.79 | 0.25 | 0.82 | 0.82-0.82 | 0.19 | ice cream |
| Kiwi | 2 | 0.67 | 0.64-0.71 | 0.40 | 0.94 | 0.92-0.96 | 0.06 | kiwi sliced |
| Melons | 12 | 0.87 | 0.77-1.26 | 0.28 | 0.92 | 0.91-1.20 | 0.09 | cantaloupe cubes |
| Omelet | 2 | 1.42 | 1.31-1.54 | 0.35 | 0.64 | 0.49-0.84 | 0.53 | egg scrambled |
| Quickbread | 2 | 1.67 | 1.63-1.72 | 0.52 | 2.73 | 2.73-2.73 | 1.01 | muffin |
| Rice | 3 | 1.49 | 1.29-1.49 | 0.40 | 1.58 | 1.57-1.75 | 0.46 | white rice cooked |
| Salmon_Grill | 2 | 1.21 | 1.19-1.23 | 0.19 | 0.95 | 0.95-0.96 | 0.05 | fish fillet |
| Sausages | 3 | 1.21 | 1.18-1.30 | 0.19 | 0.95 | 0.88-0.95 | 0.05 | bacon |
| Shrimp | 12 | 0.90 | 0.80-1.22 | 0.23 | 1.43 | 1.31-1.57 | 0.36 | shrimp |
| Strawberry | 20 | 1.12 | 0.78-1.39 | 0.29 | 1.19 | 1.19-1.23 | 0.17 | strawberries |
| Sushi | 5 | 1.66 | 1.33-1.84 | 0.51 | - | - | - | sushi roll |
| Tofu | 2 | 2.28 | 2.24-2.31 | 0.82 | 1.13 | 1.10-1.16 | 0.12 | tofu |
| Yeast_bread | 7 | 0.91 | 0.79-1.41 | 0.27 | 3.82 | 3.68-4.22 | 1.34 | bread slice |


## 2. Per-category density: generator vs USDA

`synth dens` = generator density (median over procedural items; for asset items the implied real weight/solid volume). `synth bulk` = pooled synthetic mass / GT envelope. `mass ratio` = median mass_usda_bulk / mass_synth. `dens ratio` = synth dens / USDA bulk (the table-to-table comparison of docs/density.md; the generator applies its density to the SOLID volume, so for leaves and toppings the pooled bulk column is far below it). Sorted by |log(dens ratio)|.

| category | synth source | n | synth dens | synth bulk | USDA bulk | dens ratio | pooled bulk ratio | mass ratio | USDA row |
|---|---|---|---|---|---|---|---|---|---|
| Tortilla_Chips | asset | 1 | 0.89 | 0.10 | 0.11 | 8.09 | 0.89 | 1.12 | tortilla chips |
| Hashbrown | asset | 1 | 1.70 | 0.55 | 0.35 | 4.85 | 1.57 | 0.64 | french fries |
| Fried mushrooms | asset | 8 | 0.76 | 0.42 | 0.30 | 2.57 | 1.40 | 0.70 | mushrooms sliced |
| Cucumber | asset | 36 | 1.07 | 0.48 | 0.44 | 2.43 | 1.09 | 0.72 | cucumber sliced |
| Brownies | asset | 1 | 1.17 | 0.97 | 0.50 | 2.34 | 1.94 | 0.51 | cake |
| Blueberries | asset | 23 | 1.44 | 0.42 | 0.63 | 2.30 | 0.67 | 1.23 | blueberries |
| kale | density | 1 | 0.20 | 0.02 | 0.09 | 2.25 | 0.20 | 5.04 | kale raw |
| Asparagus | asset | 10 | 1.15 | 0.56 | 0.53 | 2.18 | 1.05 | 0.66 | green beans |
| Green_Bean | asset | 1 | 1.06 | 0.40 | 0.53 | 2.00 | 0.76 | 1.31 | green beans |
| Omelet | asset | 2 | 1.83 | 0.64 | 0.93 | 1.97 | 0.69 | 1.44 | egg scrambled |
| cheese cube | density | 2 | 1.10 | 0.38 | 0.56 | 1.97 | 0.68 | 1.46 | cheese diced |
| Almonds | asset | 14 | 1.16 | 0.28 | 0.60 | 1.93 | 0.47 | 1.47 | almonds |
| Pie | asset | 1 | 0.91 | 0.23 | 0.50 | 1.81 | 0.45 | 2.22 | cornbread |
| Chicken_Nugget | asset | 2 | 0.58 | 0.27 | 1.05 | 0.55 | 0.26 | 4.54 | chicken piece |
| bread roll | density | 2 | 0.90 | 0.76 | 0.50 | 1.80 | 1.51 | 0.65 | cornbread |
| cake slice | density | 1 | 0.90 | 0.95 | 0.50 | 1.80 | 1.90 | 0.53 | cake |
| Grapes | asset | 3 | 1.11 | 0.95 | 0.64 | 1.74 | 1.49 | 0.67 | grapes |
| penne in tomato sauce | density | 1 | 0.90 | 0.82 | 0.52 | 1.72 | 1.57 | 0.64 | spaghetti cooked |
| Carrot | asset | 1 | 0.40 | 0.09 | 0.66 | 0.60 | 0.14 | 6.97 | carrots sliced cooked |
| Pasta_mixed_dishes | asset | 1 | 0.87 | 0.81 | 0.52 | 1.65 | 1.54 | 0.65 | spaghetti cooked |
| mac and cheese | density | 2 | 0.95 | 0.87 | 0.58 | 1.64 | 1.50 | 0.66 | macaroni and cheese |
| Pancake | asset | 1 | 0.80 | 0.31 | 0.50 | 1.61 | 0.62 | 1.62 | cornbread |
| Cookies | asset | 2 | 0.81 | 0.51 | 1.30 | 0.63 | 0.40 | 2.53 | chocolate |
| olive | density | 11 | 0.95 | 0.60 | 0.60 | 1.58 | 1.00 | 0.98 | olives |
| Cauliflower | asset | 6 | 0.81 | 0.59 | 0.52 | 1.54 | 1.13 | 0.85 | cauliflower |
| Melons | asset | 12 | 1.03 | 0.70 | 0.68 | 1.53 | 1.03 | 0.87 | cantaloupe cubes |
| mixed greens | density | 3 | 0.18 | 0.02 | 0.12 | 1.53 | 0.14 | 5.13 | mixed greens |
| Papaya | asset | 1 | 0.95 | 0.29 | 0.65 | 1.46 | 0.45 | 2.24 | peaches sliced |
| Strawberry | asset | 20 | 0.88 | 0.49 | 0.61 | 1.45 | 0.80 | 1.12 | strawberries |
| cookies | density | 3 | 0.90 | 0.22 | 1.30 | 0.69 | 0.17 | 7.37 | chocolate |
| radicchio | density | 1 | 0.17 | 0.01 | 0.12 | 1.44 | 0.04 | 22.77 | mixed greens |
| cherry tomato | density | 26 | 0.90 | 0.95 | 0.63 | 1.43 | 1.50 | 0.64 | cherry tomatoes |
| Kiwi | asset | 2 | 1.07 | 1.18 | 0.76 | 1.40 | 1.55 | 0.67 | kiwi sliced |
| Rice | asset | 3 | 0.92 | 0.48 | 0.67 | 1.37 | 0.72 | 1.49 | white rice cooked |
| risotto | density | 3 | 0.90 | 0.85 | 0.66 | 1.36 | 1.28 | 0.79 | barley cooked |
| Shrimp | asset | 12 | 0.73 | 0.52 | 0.54 | 1.35 | 0.96 | 0.90 | shrimp |
| lettuce | density | 5 | 0.15 | 0.01 | 0.20 | 0.75 | 0.03 | 35.59 | lettuce shredded |
| Sausages | asset | 3 | 1.05 | 0.63 | 0.80 | 1.31 | 0.79 | 1.21 | bacon |
| Tortilla | asset | 1 | 0.39 | 0.08 | 0.50 | 0.79 | 0.17 | 5.92 | pita bread |
| arugula | density | 3 | 0.15 | 0.00 | 0.12 | 1.27 | 0.00 | 2667.63 | mixed greens |
| Chicken_thighs | asset | 3 | 0.83 | 0.60 | 1.05 | 0.79 | 0.57 | 1.76 | chicken piece |
| black beans | density | 3 | 0.90 | 0.86 | 0.73 | 1.24 | 1.18 | 0.85 | black beans cooked |
| ramen | density | 1 | 0.55 | 0.54 | 0.68 | 0.81 | 0.79 | 1.26 | egg noodles cooked |
| white rice | density | 7 | 0.80 | 0.77 | 0.67 | 1.20 | 1.15 | 0.87 | white rice cooked |
| Chicken_tender | asset | 3 | 0.88 | 0.49 | 1.05 | 0.84 | 0.47 | 1.99 | chicken piece |
| Ice_cream | asset | 2 | 0.67 | 0.71 | 0.56 | 1.19 | 1.26 | 0.78 | ice cream |
| corn | density | 3 | 0.75 | 0.69 | 0.63 | 1.19 | 1.10 | 0.90 | corn kernels |
| squash | density | 6 | 0.90 | 0.73 | 0.76 | 1.18 | 0.97 | 0.98 | zucchini sliced |
| spinach (raw) | density | 4 | 0.15 | 0.01 | 0.13 | 1.18 | 0.08 | 12.74 | spinach raw |
| Burrito | asset | 1 | 0.94 | 0.70 | 0.80 | 1.17 | 0.88 | 1.14 | burrito |
| Green_beans | asset | 2 | 0.45 | 0.10 | 0.53 | 0.86 | 0.18 | 5.47 | green beans |
| sushi | density | 3 | 1.10 | 0.52 | 0.95 | 1.16 | 0.55 | 2.01 | sushi roll |
| Tuna_salad | asset | 1 | 1.00 | 1.53 | 0.87 | 1.16 | 1.77 | 0.56 | tuna salad |
| granola | density | 1 | 0.45 | 0.37 | 0.52 | 0.87 | 0.72 | 1.39 | granola |
| spaghetti | density | 1 | 0.60 | 0.58 | 0.52 | 1.15 | 1.11 | 0.90 | spaghetti cooked |
| boiled egg | density | 3 | 0.90 | 1.03 | 1.03 | 0.87 | 1.00 | 0.95 | egg whole boiled |
| Quickbread | asset | 2 | 0.46 | 0.24 | 0.40 | 1.14 | 0.60 | 1.68 | muffin |
| Mashed_Potato | asset | 1 | 1.01 | 1.00 | 0.89 | 1.14 | 1.12 | 0.89 | mashed potatoes |
| mashed potato | density | 8 | 1.00 | 0.92 | 0.89 | 1.13 | 1.04 | 0.95 | mashed potatoes |
| Yeast_bread | asset | 7 | 0.31 | 0.27 | 0.28 | 1.12 | 0.98 | 0.91 | bread slice |
| Tofu | asset | 2 | 0.93 | 0.46 | 1.05 | 0.89 | 0.43 | 2.28 | tofu |
| Coleslaw | asset | 2 | 0.90 | 0.97 | 0.81 | 1.12 | 1.20 | 0.83 | coleslaw |
| egg whites | density | 5 | 1.03 | 0.99 | 0.93 | 1.11 | 1.07 | 0.94 | egg scrambled |
| fried rice | density | 6 | 0.75 | 0.70 | 0.68 | 1.10 | 1.03 | 0.99 | fried rice |
| Falafel | asset | 22 | 0.88 | 0.72 | 0.80 | 1.10 | 0.90 | 1.06 | burrito |
| Sushi | asset | 5 | 0.86 | 0.56 | 0.95 | 0.91 | 0.59 | 1.66 | sushi roll |
| Bacon | asset | 8 | 0.88 | 0.51 | 0.80 | 1.10 | 0.64 | 1.17 | bacon |
| French_toast | asset | 2 | 0.46 | 0.40 | 0.50 | 0.91 | 0.80 | 1.25 | cornbread |
| salsa | density | 2 | 1.00 | 0.12 | 1.09 | 0.92 | 0.11 | 10.30 | meat sauce |
| tomato sauce | density | 1 | 1.00 | 0.00 | 1.09 | 0.92 | 0.00 | 333.45 | meat sauce |
| couscous | density | 2 | 0.72 | 0.67 | 0.66 | 1.08 | 1.01 | 0.97 | couscous cooked |
| Broccoli | asset | 25 | 0.61 | 0.36 | 0.66 | 0.93 | 0.54 | 1.67 | broccoli florets |
| lentil stew | density | 2 | 0.95 | 0.88 | 1.02 | 0.93 | 0.86 | 1.19 | soup broth based |
| Pork_Chop | asset | 1 | 1.14 | 1.22 | 1.06 | 1.07 | 1.15 | 0.87 | steak beef |
| brown rice | density | 7 | 0.80 | 0.78 | 0.85 | 0.94 | 0.91 | 1.10 | brown rice cooked |
| brussels sprout | density | 5 | 0.70 | 0.49 | 0.66 | 1.06 | 0.74 | 1.44 | brussels sprouts |
| Salmon_Grill | asset | 2 | 1.10 | 0.86 | 1.05 | 1.05 | 0.82 | 1.21 | fish fillet |
| curry sauce | density | 1 | 1.03 | 0.17 | 0.98 | 1.05 | 0.18 | 5.67 | gravy |
| refried beans | density | 2 | 1.05 | 0.93 | 1.01 | 1.04 | 0.93 | 1.07 | refried beans |
| chili | density | 1 | 0.98 | 1.01 | 1.02 | 0.96 | 0.99 | 1.01 | soup broth based |
| Peanuts | asset | 1 | 0.59 | 0.08 | 0.62 | 0.96 | 0.13 | 7.43 | peanuts |
| pesto | density | 5 | 0.95 | 0.13 | 0.98 | 0.96 | 0.13 | 10.64 | gravy |
| gravy | density | 3 | 1.02 | 0.30 | 0.98 | 1.04 | 0.31 | 3.85 | gravy |
| pumpkin soup | density | 1 | 1.00 | 0.98 | 1.03 | 0.97 | 0.95 | 1.05 | cream soup |
| tomato soup | density | 1 | 1.00 | 0.81 | 1.03 | 0.97 | 0.78 | 1.27 | cream soup |
| Bell Pepper | asset | 2 | 0.61 | 0.35 | 0.63 | 0.97 | 0.55 | 1.82 | bell pepper chopped |
| scrambled eggs | density | 2 | 0.95 | 0.87 | 0.93 | 1.02 | 0.94 | 1.07 | egg scrambled |
| polenta | density | 1 | 1.02 | 0.96 | 1.00 | 1.02 | 0.96 | 1.04 | polenta |
| melted cheese | density | 6 | 1.05 | 0.28 | 1.07 | 0.98 | 0.26 | 3.35 | cheese block |
| Steak | asset | 1 | 1.08 | 0.77 | 1.06 | 1.02 | 0.73 | 1.37 | steak beef |
| chicken curry | density | 1 | 1.00 | 0.82 | 0.98 | 1.02 | 0.84 | 1.20 | gravy |
| green curry | density | 5 | 1.00 | 0.60 | 0.98 | 1.02 | 0.61 | 1.24 | gravy |
| strawberry | density | 9 | 0.60 | 0.35 | 0.61 | 0.99 | 0.58 | 1.69 | strawberries |
| congee | density | 1 | 1.00 | 0.97 | 0.99 | 1.01 | 0.99 | 1.01 | oatmeal cooked |
| oatmeal | density | 4 | 1.00 | 0.95 | 0.99 | 1.01 | 0.96 | 1.05 | oatmeal cooked |
| hummus | density | 4 | 1.05 | 0.98 | 1.04 | 1.01 | 0.95 | 1.07 | hummus |
| yogurt | density | 3 | 1.03 | 0.94 | 1.04 | 0.99 | 0.91 | 1.11 | yogurt |
| sour cream | density | 4 | 1.01 | 0.22 | 1.01 | 1.00 | 0.21 | 4.31 | sour cream |
| quinoa | density | 5 | 0.78 | 0.73 | 0.78 | 1.00 | 0.94 | 1.05 | quinoa cooked |
| tofu | density | 2 | 1.05 | 0.73 | 1.05 | 1.00 | 0.70 | 1.41 | tofu |
| chicken breast | density | 1 | 1.05 | 0.77 | 1.05 | 1.00 | 0.74 | 1.36 | chicken piece |


## 3. Per-kind median |log ratio| of USDA mass to synthetic mass

| kind | n | bulk |log| | bulk median ratio | material |log| | material median ratio |
|---|---|---|---|---|---|
| leaves | 17 | 2.97 | 19.56 | 1.90 | 6.67 |
| piece | 83 | 0.08 | 1.01 | 0.35 | 1.42 |
| pile | 196 | 0.34 | 0.99 | 0.17 | 1.11 |
| single | 141 | 0.48 | 1.34 | 0.37 | 1.39 |
| topping | 22 | 1.78 | 5.94 | 0.02 | 1.02 |


## 4. Items whose mass changes by more than 30 %

Bulk path: 244 of 459 items (53 %); material path: 191 of 410. Full list: `changed_items.csv`. By category (bulk path):


| category | changed / total | median ratio |
|---|---|---|
| Cucumber | 23 / 36 | 0.47 |
| Broccoli | 22 / 25 | 1.70 |
| cherry tomato | 18 / 26 | 0.57 |
| Blueberries | 16 / 23 | 2.31 |
| Almonds | 10 / 14 | 1.96 |
| Strawberry | 9 / 20 | 1.42 |
| Asparagus | 9 / 10 | 0.60 |
| strawberry | 7 / 9 | 1.77 |
| Bacon | 7 / 8 | 1.56 |
| melted cheese | 6 / 6 | 3.35 |
| lettuce | 5 / 5 | 35.59 |
| Falafel | 5 / 22 | 1.46 |
| Fried mushrooms | 4 / 8 | 0.53 |
| Melons | 4 / 12 | 1.49 |
| pesto | 4 / 5 | 10.94 |
| Sushi | 4 / 5 | 1.75 |
| sour cream | 4 / 4 | 4.31 |
| spinach (raw) | 4 / 4 | 12.74 |
| Shrimp | 4 / 12 | 1.46 |
| arugula | 3 / 3 | 2667.63 |
| olive | 3 / 11 | 1.41 |
| Yeast_bread | 3 / 7 | 1.85 |
| gravy | 3 / 3 | 3.85 |
| mixed greens | 3 / 3 | 5.13 |
| Chicken_tender | 3 / 3 | 1.99 |
| brussels sprout | 3 / 5 | 1.44 |
| cookies | 3 / 3 | 7.37 |
| Quickbread | 2 / 2 | 1.68 |
| Bell Pepper | 2 / 2 | 1.82 |
| mac and cheese | 2 / 2 | 0.66 |
| cheese cube | 2 / 2 | 1.46 |
| Chicken_Nugget | 2 / 2 | 4.54 |
| salsa | 2 / 2 | 10.30 |
| green curry | 2 / 5 | 3.97 |
| Tofu | 2 / 2 | 2.28 |
| Rice | 2 / 3 | 1.49 |
| Green_beans | 2 / 2 | 5.47 |
| Chicken_thighs | 2 / 3 | 1.98 |
| Grapes | 2 / 3 | 0.54 |
| sushi | 2 / 3 | 2.33 |
| Cookies | 2 / 2 | 2.53 |
| Brownies | 1 / 1 | 0.51 |
| kale | 1 / 1 | 5.04 |
| Peanuts | 1 / 1 | 7.43 |
| granola | 1 / 1 | 1.39 |
| Green_Bean | 1 / 1 | 1.31 |
| curry sauce | 1 / 1 | 5.67 |
| Hashbrown | 1 / 1 | 0.64 |
| tofu | 1 / 2 | 1.81 |
| Tortilla | 1 / 1 | 5.92 |
| Omelet | 1 / 2 | 1.68 |
| Carrot | 1 / 1 | 6.97 |
| French_toast | 1 / 2 | 1.41 |
| tomato sauce | 1 / 1 | 333.45 |
| penne in tomato sauce | 1 / 1 | 0.64 |
| Pancake | 1 / 1 | 1.62 |
| Pie | 1 / 1 | 2.22 |
| Steak | 1 / 1 | 1.37 |
| Tuna_salad | 1 / 1 | 0.56 |
| bread roll | 1 / 2 | 0.57 |
| squash | 1 / 6 | 1.58 |
| Kiwi | 1 / 2 | 0.60 |
| Pasta_mixed_dishes | 1 / 1 | 0.65 |
| cake slice | 1 / 1 | 0.53 |
| Sausages | 1 / 3 | 1.39 |
| radicchio | 1 / 1 | 22.77 |
| Papaya | 1 / 1 | 2.24 |
| chicken breast | 1 / 1 | 1.36 |
