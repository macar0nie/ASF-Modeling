"""Blocky 3D farm view.

The 3D view does NOT run its own simulation: it replays the day-by-day state of every pig from an
agent-based run (the same run shown in the app), so what you see is exactly what the model did.

    html = farm_html(result)     # one self-contained HTML page (three.js is inlined, works offline)
"""
import json
import math
from pathlib import Path

from .model import COUNT_COLS, RunResult

STATIC = Path(__file__).parent / "static"


def control_lines(p) -> list[str]:
    lines = []
    if p.remove_sick:
        lines.append(f"Sick pigs found and removed ({p.p_remove:.0%} a day)")
    if p.biosecurity:
        lines.append(f"Biosecurity: between-pen contact cut by {p.e_bio:.0%}")
    if p.depopulate:
        lines.append("Every pig in a pen with an ASF death is culled")
    return lines


def replay_data(res: RunResult, subtitle: str = "") -> dict:
    p = res.p
    fin = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x)
    return {
        "n_pens": p.n_pens,
        "per_pen": p.pigs_per_pen,
        "history": res.history,
        "counts": {k: res.counts[k].astype(int).tolist() for k in COUNT_COLS},
        "detection_day": fin(res.detection_day),
        "first_death_day": fin(res.first_death_day),
        "controls": control_lines(p) if p.detection else [],
        "subtitle": subtitle or (f"β_w {p.beta_w}, β_b {p.beta_b}, "
                                 + (f"detected {p.detect_delay} d after 1st death" if p.detection else "no action")),
    }


def farm_html(res: RunResult, title: str = "ASF farm outbreak: 3D replay", subtitle: str = "") -> str:
    if not res.history:
        raise ValueError("run the ABM with record_history=True to build the 3D view")
    template = (STATIC / "farm3d.html").read_text(encoding="utf-8")
    three = (STATIC / "three.min.js").read_text(encoding="utf-8")
    data = json.dumps(replay_data(res, subtitle), separators=(",", ":")).replace("null", "NaN")
    return (template.replace("/*__THREE__*/", three)
                    .replace("/*__DATA__*/null", data)
                    .replace("__TITLE__", title))
