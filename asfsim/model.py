"""The two ASF within-farm models.

* ``run_abm``  – agent-based model: every pig is an individual with a pen, a disease state and timers.
* ``run_seir`` – pen-structured stochastic SEIR (discrete-time chain binomial), the baseline model.

Both use the same force of infection for a pig in pen k (frequency dependent, Guinat et al. 2016):

    lambda_k = beta_w * I_k / N_k  +  beta_b * sum_{j adjacent to k} I_j / N_j

where N is the number of live pigs in a pen and adjacent pens are k-1 and k+1 (pens stand in a row).
Time step = 1 day. A pig is infected on a day with probability 1 - exp(-lambda_k).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Params, S, E, I, D, R

COUNT_COLS = ["S", "E", "I", "D", "R"]


def adjacency(n_pens: int) -> np.ndarray:
    """A[k, j] = 1 when pens k and j stand next to each other."""
    a = np.zeros((n_pens, n_pens))
    for k in range(n_pens - 1):
        a[k, k + 1] = a[k + 1, k] = 1.0
    return a


def force_of_infection(inf: np.ndarray, live: np.ndarray, beta_w: float, beta_b: float,
                       adj: np.ndarray) -> np.ndarray:
    """lambda for every pen from the number of infectious (inf) and live pigs per pen."""
    frac = np.divide(inf, live, out=np.zeros(len(inf)), where=live > 0)
    return beta_w * frac + beta_b * (adj @ frac)


@dataclass
class RunResult:
    p: Params
    counts: pd.DataFrame                 # one row per day: S E I D R, new_inf, new_dead, new_removed
    pen_infected: np.ndarray             # days x pens: E + I
    pen_dead: np.ndarray                 # days x pens: D
    history: list = field(default_factory=list)   # per day, a string of digits (one per pig); ABM only
    detection_day: float = np.nan
    first_death_day: float = np.nan
    index_secondary: int = 0             # pigs infected directly by the first pig (ABM only)
    infected_by: np.ndarray = None       # ABM only: id of the pig that infected each pig (-1 = none / index)

    def metrics(self) -> dict:
        c, n = self.counts, self.p.n_pigs
        ever_infected = int(c["new_inf"].sum()) + 1
        dead = c["D"].iloc[-1]
        tenth = c.index[c["D"] >= 10]
        return {
            "attack_rate": ever_infected / n,
            "infected": ever_infected,
            "deaths": int(dead),
            "culled": int(c["R"].iloc[-1]),
            "lost": int(dead + c["R"].iloc[-1]),
            "survivors": int(c["S"].iloc[-1]),
            "peak_infectious": int(c["I"].max()),
            "peak_day": int(c["I"].idxmax()),
            "duration": int(c.index[-1]),          # day the last infected pig died or was removed
            "detection_day": self.detection_day,
            "first_death_day": self.first_death_day,
            "day_10th_death": float(tenth[0]) if len(tenth) else np.nan,
            "minor_outbreak": ever_infected < 0.1 * n,
            "index_secondary": self.index_secondary,
        }


# --------------------------------------------------------------------------------------------------
# Agent-based model
# --------------------------------------------------------------------------------------------------
def _draw_latent(p: Params, rng, k: int) -> np.ndarray:
    if p.durations == "exponential":
        return rng.geometric(1 - np.exp(-1 / p.latent_days), size=k)
    return np.full(k, int(round(p.latent_days)))


def _draw_infectious(p: Params, rng, k: int) -> np.ndarray:
    if p.durations == "exponential":
        return rng.geometric(1 - np.exp(-1 / p.mean_inf), size=k)
    return rng.integers(p.inf_min, p.inf_max + 1, size=k)


def run_abm(p: Params, record_history: bool = True, rng: np.random.Generator | None = None) -> RunResult:
    """One stochastic outbreak, starting from one infectious pig in pen 1 on day 0."""
    rng = rng if rng is not None else np.random.default_rng(p.seed)
    n, n_pens = p.n_pigs, p.n_pens
    pen = np.repeat(np.arange(n_pens), p.pigs_per_pen)
    adj = adjacency(n_pens)

    state = np.full(n, S, dtype=np.int8)
    timer = np.zeros(n, dtype=np.int32)
    infected_by = np.full(n, -1, dtype=np.int32)
    index = 0
    state[index] = I
    timer[index] = _draw_infectious(p, rng, 1)[0]

    detection_day = np.nan
    first_death_day = np.nan
    rows, pen_inf, pen_dead, history = [], [], [], []

    def snapshot(day, new_inf, new_dead, new_rem):
        cnt = np.bincount(state, minlength=5)
        rows.append([day, *cnt.tolist(), new_inf, new_dead, new_rem])
        pen_inf.append(np.bincount(pen[(state == E) | (state == I)], minlength=n_pens))
        pen_dead.append(np.bincount(pen[state == D], minlength=n_pens))
        if record_history:
            history.append("".join(map(str, state.tolist())))

    snapshot(0, 0, 0, 0)
    for day in range(1, p.max_days + 1):
        controls_on = p.detection and not np.isnan(detection_day) and day >= detection_day
        beta_b = p.beta_b * (1 - p.e_bio) if (controls_on and p.biosecurity) else p.beta_b

        # 1. infection pressure from the state at the start of the day
        is_inf = state == I
        live_mask = state <= I
        inf_pen = np.bincount(pen[is_inf], minlength=n_pens).astype(float)
        live_pen = np.bincount(pen[live_mask], minlength=n_pens).astype(float)
        lam = force_of_infection(inf_pen, live_pen, p.beta_w, beta_b, adj)
        sus = np.flatnonzero(state == S)
        newly = sus[rng.random(len(sus)) < 1 - np.exp(-lam[pen[sus]])]

        # who infected whom (for the R0 check): pick a source in proportion to its contribution
        if len(newly):
            src_ids = np.flatnonzero(is_inf)
            for v in newly:
                k = pen[v]
                w = np.where(pen[src_ids] == k, p.beta_w, np.where(adj[k, pen[src_ids]] > 0, beta_b, 0.0))
                w = w / np.maximum(live_pen[pen[src_ids]], 1)
                if w.sum() > 0:
                    infected_by[v] = src_ids[rng.choice(len(src_ids), p=w / w.sum())]

        # 2. progression of pigs infected on earlier days
        timer[(state == E) | (state == I)] -= 1
        to_inf = np.flatnonzero((state == E) & (timer <= 0))
        to_dead = np.flatnonzero((state == I) & (timer <= 0))
        state[to_dead] = D
        state[to_inf] = I
        timer[to_inf] = _draw_infectious(p, rng, len(to_inf))
        # 3. new infections from step 1
        state[newly] = E
        timer[newly] = _draw_latent(p, rng, len(newly))

        # 4. detection: first ASF death + d days
        if len(to_dead) and np.isnan(first_death_day):
            first_death_day = float(day)
            if p.detection:
                detection_day = float(day + p.detect_delay)

        # 5. control measures
        removed = 0
        if p.detection and not np.isnan(detection_day) and day >= detection_day:
            if p.remove_sick:
                sick = np.flatnonzero(state == I)
                found = sick[rng.random(len(sick)) < p.p_remove]
                state[found] = R
                removed += len(found)
            if p.depopulate:
                hit = np.unique(pen[state == D])
                cull = np.flatnonzero(np.isin(pen, hit) & (state <= I))
                state[cull] = R
                removed += len(cull)

        snapshot(day, len(newly), len(to_dead), removed)
        if not ((state == E) | (state == I)).any():
            break

    counts = pd.DataFrame(rows, columns=["day", *COUNT_COLS, "new_inf", "new_dead", "new_removed"]).set_index("day")
    return RunResult(p=p, counts=counts, pen_infected=np.array(pen_inf), pen_dead=np.array(pen_dead),
                     history=history, detection_day=detection_day, first_death_day=first_death_day,
                     index_secondary=int((infected_by == index).sum()), infected_by=infected_by)


# --------------------------------------------------------------------------------------------------
# Stochastic SEIR (chain binomial)
# --------------------------------------------------------------------------------------------------
def run_seir(p: Params, rng: np.random.Generator | None = None) -> RunResult:
    """Pen-structured stochastic SEIR with exponential (memoryless) latent and infectious periods.

        new exposed_k    ~ Binomial(S_k, 1 - exp(-lambda_k))
        new infectious_k ~ Binomial(E_k, 1 - exp(-1/L))
        new dead_k       ~ Binomial(I_k, 1 - exp(-1/T))
    No detection or control: it is the reference model used to validate the agent-based model.
    """
    rng = rng if rng is not None else np.random.default_rng(p.seed)
    n_pens = p.n_pens
    adj = adjacency(n_pens)
    Sx = np.full(n_pens, p.pigs_per_pen)
    Ex = np.zeros(n_pens, dtype=int)
    Ix = np.zeros(n_pens, dtype=int)
    Dx = np.zeros(n_pens, dtype=int)
    Sx[0] -= 1
    Ix[0] += 1
    p_ei = 1 - np.exp(-1 / p.latent_days)
    p_id = 1 - np.exp(-1 / p.mean_inf)

    rows = [[0, Sx.sum(), 0, 1, 0, 0, 0, 0, 0]]
    pen_inf, pen_dead = [Ex + Ix], [Dx.copy()]
    first_death = np.nan
    for day in range(1, p.max_days + 1):
        lam = force_of_infection(Ix.astype(float), (Sx + Ex + Ix).astype(float), p.beta_w, p.beta_b, adj)
        new_e = rng.binomial(Sx, 1 - np.exp(-lam))
        new_i = rng.binomial(Ex, p_ei)
        new_d = rng.binomial(Ix, p_id)
        Sx, Ex, Ix, Dx = Sx - new_e, Ex + new_e - new_i, Ix + new_i - new_d, Dx + new_d
        if new_d.sum() and np.isnan(first_death):
            first_death = float(day)
        rows.append([day, Sx.sum(), Ex.sum(), Ix.sum(), Dx.sum(), 0, new_e.sum(), new_d.sum(), 0])
        pen_inf.append(Ex + Ix)
        pen_dead.append(Dx.copy())
        if Ex.sum() + Ix.sum() == 0:
            break
    counts = pd.DataFrame(rows, columns=["day", *COUNT_COLS, "new_inf", "new_dead", "new_removed"]).set_index("day")
    return RunResult(p=p, counts=counts, pen_infected=np.array(pen_inf), pen_dead=np.array(pen_dead),
                     first_death_day=first_death)
