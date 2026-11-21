"""Run the three experimental scenarios plus the no-action reference and save the results.

    python run_experiments.py              # 1,000 replications per level (about 2 minutes)
    python run_experiments.py --n 200 --farm backyard
"""
import argparse
from pathlib import Path

from asfsim import analysis as an
from asfsim.config import BASELINE, FARMS

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000, help="replications per level")
ap.add_argument("--farm", choices=["commercial", "backyard"], default="commercial")
ap.add_argument("--out", default="results")
a = ap.parse_args()

base = BASELINE.with_(**list(FARMS.values())[0 if a.farm == "commercial" else 1])
out = Path(a.out)
out.mkdir(exist_ok=True)
runs = an.run_all(base, a.n)
table = an.scenario_table(runs)
runs.to_csv(out / f"scenario_runs_{a.farm}.csv", index=False)
table.to_csv(out / f"scenario_summary_{a.farm}.csv", index=False)

cols = ["attack_rate", "lost", "deaths", "culled", "peak_day", "duration", "detection_day"]
lines = [f"# Scenario results ({a.farm} farm, {base.n_pigs} pigs, {a.n} replications per level)", "",
         "Means with ± 95% confidence-interval half-widths.", "",
         "| Scenario | Level | " + " | ".join(an.KPI_LABELS[c] for c in cols) + " |",
         "|---|---|" + "---|" * len(cols)]
for _, r in table.iterrows():
    cells = []
    for c in cols:
        v, ci = r.get(c), r.get(c + "_ci", 0)
        cells.append("–" if v != v else (f"{v:.1%} ± {ci:.1%}" if c == "attack_rate" else f"{v:.1f} ± {ci:.1f}"))
    lines.append(f"| {r.scenario} | {r.level} | " + " | ".join(cells) + " |")
(out / f"scenario_summary_{a.farm}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
