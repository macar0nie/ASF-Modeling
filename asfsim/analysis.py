"""Replications, scenario experiments and model validation."""
import numpy as np
import pandas as pd

from .config import Params, REFERENCE, SCENARIOS, Scenario
from .model import COUNT_COLS, run_abm, run_seir

KPI_LABELS = {
    "attack_rate": "Attack rate (share of pigs infected)",
    "lost": "Pigs lost (ASF deaths + culled)",
    "deaths": "ASF deaths",
    "culled": "Pigs culled",
    "survivors": "Pigs never infected",
    "peak_infectious": "Peak number infectious",
    "peak_day": "Day of peak",
    "duration": "Outbreak duration (days)",
    "detection_day": "Day ASF was confirmed",
    "minor_outbreak": "Share of minor outbreaks (<10% infected)",
}


def replicate(p: Params, n: int, model: str = "abm", seed: int | None = None, curves: bool = True):
    """Run n independent outbreaks. Returns (metrics per run, mean daily curve, 2.5/97.5% band of I)."""
    rng = np.random.default_rng(p.seed if seed is None else seed)
    runs, series = [], []
    for i in range(n):
        r = run_abm(p, record_history=False, rng=rng) if model == "abm" else run_seir(p, rng=rng)
        m = r.metrics()
        m["rep"] = i
        runs.append(m)
        if curves:
            series.append(r.counts[COUNT_COLS])
    metrics = pd.DataFrame(runs)
    if not curves:
        return metrics, None
    horizon = max(len(s) for s in series)
    stack = np.stack([s.reindex(range(horizon)).ffill().to_numpy() for s in series])  # runs x days x 5
    mean = pd.DataFrame(stack.mean(axis=0), columns=COUNT_COLS)
    mean["I_lo"] = np.percentile(stack[:, :, 2], 2.5, axis=0)
    mean["I_hi"] = np.percentile(stack[:, :, 2], 97.5, axis=0)
    mean.index.name = "day"
    return metrics, mean


def summarize(metrics: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Mean, SD, 95% CI of the mean, median and 2.5–97.5 percentile range for each KPI."""
    cols = cols or [c for c in KPI_LABELS if c in metrics]
    out = []
    for c in cols:
        x = metrics[c].astype(float).dropna()
        if x.empty:
            continue
        half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        out.append({"kpi": c, "label": KPI_LABELS.get(c, c), "mean": x.mean(), "sd": x.std(ddof=1),
                    "ci_low": x.mean() - half, "ci_high": x.mean() + half, "median": x.median(),
                    "p2.5": x.quantile(0.025), "p97.5": x.quantile(0.975), "n": len(x)})
    return pd.DataFrame(out)


def scenario_params(base: Params, sc: Scenario, level) -> Params:
    return base.with_(**sc.fixed, **{sc.param: level})


def run_scenario(base: Params, sc: Scenario, n: int, seed: int = 2026):
    """All three levels of one scenario. Returns (metrics with a 'level' column, mean curves per level)."""
    frames, curves = [], {}
    for lvl, name in zip(sc.levels, sc.level_names):
        m, c = replicate(scenario_params(base, sc, lvl), n, seed=seed)
        m["scenario"], m["level"], m["level_name"] = sc.key, lvl, name
        frames.append(m)
        curves[name] = c
    return pd.concat(frames, ignore_index=True), curves


def run_reference(base: Params, n: int, seed: int = 2026):
    m, c = replicate(base.with_(**REFERENCE), n, seed=seed)
    m["scenario"], m["level"], m["level_name"] = "REF", np.nan, "No action (reference)"
    return m, c


def run_all(base: Params, n: int, seed: int = 2026) -> pd.DataFrame:
    frames = [run_reference(base, n, seed)[0]]
    frames += [run_scenario(base, sc, n, seed)[0] for sc in SCENARIOS]
    return pd.concat(frames, ignore_index=True)


def scenario_table(all_runs: pd.DataFrame, kpis=("attack_rate", "lost", "deaths", "culled", "peak_day",
                                                 "duration", "detection_day")) -> pd.DataFrame:
    rows = []
    for (sc, name), g in all_runs.groupby(["scenario", "level_name"], sort=False):
        row = {"scenario": sc, "level": name, "runs": len(g)}
        for k in kpis:
            x = g[k].astype(float).dropna()
            if x.empty:
                row[k] = np.nan
                continue
            half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            row[k] = x.mean()
            row[k + "_ci"] = half
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def validate_abm_vs_seir(base: Params, n: int = 300, seed: int = 7):
    """Model-to-model check: ABM in SEIR-equivalent (exponential) mode vs the chain-binomial SEIR."""
    p = base.with_(**REFERENCE, durations="exponential")
    m_abm, c_abm = replicate(p, n, model="abm", seed=seed)
    m_seir, c_seir = replicate(p, n, model="seir", seed=seed + 1)
    rows = []
    for k in ("attack_rate", "deaths", "peak_infectious", "peak_day", "duration"):
        a, s = summarize(m_abm, [k]).iloc[0], summarize(m_seir, [k]).iloc[0]
        overlap = not (a.ci_high < s.ci_low or s.ci_high < a.ci_low)
        rows.append({"kpi": KPI_LABELS[k], "ABM mean": a["mean"], "ABM 95% CI": f"{a.ci_low:.2f} – {a.ci_high:.2f}",
                     "SEIR mean": s["mean"], "SEIR 95% CI": f"{s.ci_low:.2f} – {s.ci_high:.2f}",
                     "CIs overlap": "yes" if overlap else "no"})
    return pd.DataFrame(rows), c_abm, c_seir


def estimate_r0_within_pen(base: Params, n: int = 500, pen_size: int = 200, seed: int = 11) -> dict:
    """Secondary cases caused by the first pig in one large pen (no other pens, no control).

    Expected value is about beta_w x mean infectious period (0.6 x 4.5 = 2.7); Guinat et al. (2016)
    report 2.8 (95% CI 1.3–4.8)."""
    p = base.with_(**REFERENCE, n_pens=1, pigs_per_pen=pen_size)
    rng = np.random.default_rng(seed)
    sec = np.array([run_abm(p, record_history=False, rng=rng).index_secondary for _ in range(n)])
    half = 1.96 * sec.std(ddof=1) / np.sqrt(n)
    return {"simulated": sec.mean(), "ci_low": sec.mean() - half, "ci_high": sec.mean() + half,
            "theory": p.beta_w * p.mean_inf, "published": 2.8, "published_ci": (1.3, 4.8), "runs": n}


def detection_check(base: Params, n: int = 300, seed: int = 13) -> dict:
    """Nielsen et al. (2017): in a 360-pig, 12-pen house the 10th death came 13–19 days after introduction."""
    p = base.with_(**REFERENCE, n_pens=12, pigs_per_pen=30)
    m, _ = replicate(p, n, seed=seed, curves=False)
    d = m["day_10th_death"].dropna()
    return {"median": d.median(), "p25": d.quantile(0.25), "p75": d.quantile(0.75),
            "share_reaching_10": len(d) / n, "published": (13, 19), "runs": n}
