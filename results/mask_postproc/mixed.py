import sys, json
sys.path.insert(0, "results/mask_postproc"); sys.path.insert(0, "pipeline")
import sweep
from foodscan.replay_lift import SIDS, RAWROOT
n = t = 0
for s in SIDS:
    a, b = sweep.mixed_masks(s, RAWROOT / s / "masks_sam3_v2i/raw"); n += a; t += b
print(json.dumps({"mixed_raw_masks": n, "raw_food_masks": t}))
