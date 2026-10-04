# Container geometry: bowls, rimmed plates, stacked food

Compiled 2026-09-28. Verified against primary sources unless marked **[U]**.

## What prior systems do
| System | Container handling |
|---|---|
| SNAQ (Herzig 2020) | RANSAC *table* plane; one standard plate/bowl type; no bowl treatment |
| **Lu et al. 2020** (arXiv 2003.08273) | CNN classifies 5 plate types; **prior 3D plate model** placed from segmentation + tray normal; volume = ∫(food − plate surface) along the tray normal. Works for plates and bowls |
| Nutrition5k | flat, fixed overhead camera |
| Hungry Networks (Naritomi & Yanai 2021/22) | reconstruct "dish" and "plate-only" meshes with a plate-consistency loss; volume = difference |
| Ogishi, Tanabe & Yanai 2025 (ACM MM MetaFood ws, doi 10.1145/3746264.3760487) | fine-tuned diffusion inpainter completes the plate under food, then volume difference. **[abstract only]** |
| MetaFood 2025 | plates/utensils as implicit scale only |
| PerBite 2026 | plate removed manually in Blender |
| **Jia/Sun bowl papers** (MADiMa 2023; J. Food Eng. 2024; Sensors 2024, PMC11435675) | pre-measured bowls; interior as surface of revolution z = H (r/R)^q; GT capacity by **water fill + weighing**; fit error 0.3–2.6 % of height; power law fails on deep/rimmed plates |
| MetaFood3D | "representing empty spaces beneath food objects remains a challenge"; stacking unsolved |

No "container-aware" method or reference-dish database exists as of 2026. Nobody quantifies the flat-plane error for bowls; a 12 cm × 2 cm bowl with 200 mL of rice gives −60 to −80 % with "volume above the rim plane" (agent's own calculation).

## Tools
- Efficient RANSAC (Schnabel 2007) in CGAL Shape Detection; PCL `SACMODEL_CYLINDER`; **pyRANSAC-3D** (plane, sphere, cylinder, cone, circle) for rim circle fits; Open3D has planes only.
- Axis-of-revolution from partial surfaces: Pottmann & Randrup 1998; Mara & Sablatnig 2006 (pottery sherds, closest analogue to a partly visible bowl).
- Learned completion of *food* (not container): Point2Volume 2020; VolE-Complete 2026 (github.com/GCVCG/VolE-Complete, symmetry-guided).

## Recommended implementation (ranked)
1. **Empty-container template + ICP** (all containers). Scan each plate/bowl once, store the mesh by ID. At meal time: table plane → container points (rim, exposed interior, outer wall) → ICP the template (4-DoF if the table normal is trusted). Per-item volume = ∫ over the item footprint of (food height − template height) along the table normal. Also gives bowl capacity for water-fill validation. → `pipeline/foodscan/container.py`
2. **Axisymmetric fallback** for unknown dishes: RANSAC circle on the rim, axis = rim normal, project visible interior/outer wall to (r, z), fit a monotone spline (not the power law), revolve.
3. **Flat plate**: fit the plane to the plate *well*, clip integration to the well, treat the rim as container.
4. **Stacked / overhanging**: support surface = max(container template, other items' visible surfaces under the footprint); volume outside the container footprint relative to the table; flag it. No published method validates this case.
5. **Validation**: water-fill each bowl; report flat-plane vs template error per container (unmeasured in the literature).

## Sources
PMC7142738 · 2003.08273 · 2103.03375 · 10.1145/3444685.3446275 · 10.1145/3552484.3555750 · 10.1145/3746264.3760487 · 2602.13041 · 2409.01966 · 2606.02021 · 10.1145/3607828.3617789 · 10.1016/j.jfoodeng.2024.111943 · PMC11435675 · 10.1111/j.1467-8659.2007.01016.x · doc.cgal.org Shape_detection · github.com/leomariga/pyRANSAC-3D · 10.1007/bf02684378 · 10.1109/3dpvt.2006.105 · 10.1109/tii.2019.2942831 · github.com/GCVCG/VolE-Complete
