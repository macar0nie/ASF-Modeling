"""Write a standalone 3D replay (one HTML file that works offline).

    python make_3d.py                                   # baseline no-action outbreak
    python make_3d.py --scenario S2 --level 1 --out fast_detection.html
"""
import argparse

from asfsim.analysis import scenario_params
from asfsim.config import BASELINE, FARMS, REFERENCE, SCENARIO_BY_KEY
from asfsim.model import run_abm
from asfsim.viz3d import farm_html

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", choices=list(SCENARIO_BY_KEY), help="S1, S2 or S3 (default: no-action reference)")
ap.add_argument("--level", type=int, default=1, help="0, 1 or 2 (low, baseline, high)")
ap.add_argument("--farm", choices=["commercial", "backyard"], default="commercial")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="asf_farm_3d.html")
a = ap.parse_args()

farm = list(FARMS.values())[0 if a.farm == "commercial" else 1]
base = BASELINE.with_(**farm, seed=a.seed)
if a.scenario:
    sc = SCENARIO_BY_KEY[a.scenario]
    p = scenario_params(base, sc, sc.levels[a.level])
    sub = f"{sc.name.split(':')[0]}: {sc.level_names[a.level]}"
else:
    p, sub = base.with_(**REFERENCE), "No action (reference)"
res = run_abm(p)
with open(a.out, "w", encoding="utf-8") as f:
    f.write(farm_html(res, subtitle=sub))
print(f"wrote {a.out}: {res.metrics()['lost']} of {p.n_pigs} pigs lost over {len(res.history) - 1} days")
