"""Verification tests for the ASF within-farm models (run: python -m pytest)."""
import numpy as np
import pytest

from asfsim import analysis as an
from asfsim.config import BASELINE, REFERENCE, SCENARIOS, D, E, I, R, S
from asfsim.model import COUNT_COLS, adjacency, run_abm, run_seir
from asfsim.viz3d import farm_html

NO_ACTION = BASELINE.with_(**REFERENCE)


@pytest.mark.parametrize("seed", range(5))
def test_pig_total_constant_every_day(seed):
    for p in (NO_ACTION.with_(seed=seed), BASELINE.with_(**SCENARIOS[2].fixed, seed=seed)):
        res = run_abm(p)
        assert (res.counts[COUNT_COLS].sum(axis=1) == p.n_pigs).all()
        assert all(len(h) == p.n_pigs for h in res.history)


def test_seir_total_constant():
    res = run_seir(NO_ACTION)
    assert (res.counts[COUNT_COLS].sum(axis=1) == NO_ACTION.n_pigs).all()


def test_no_transmission_only_index_pig_dies():
    res = run_abm(NO_ACTION.with_(beta_w=0.0, beta_b=0.0))
    m = res.metrics()
    assert m["infected"] == 1 and m["deaths"] == 1 and m["survivors"] == NO_ACTION.n_pigs - 1


def test_no_between_pen_contact_stays_in_pen_1():
    for seed in range(5):
        res = run_abm(NO_ACTION.with_(beta_b=0.0, seed=seed))
        assert res.pen_infected[:, 1:].sum() == 0 and res.pen_dead[:, 1:].sum() == 0


def test_same_seed_same_result():
    a, b = run_abm(BASELINE.with_(seed=42)), run_abm(BASELINE.with_(seed=42))
    assert a.history == b.history
    assert a.counts.equals(b.counts)


def test_realistic_durations_follow_the_rules():
    p = NO_ACTION.with_(seed=3)
    res = run_abm(p)
    hist = np.array([[int(c) for c in h] for h in res.history])        # days x pigs
    # index pig is infectious from day 0 and dies after 3-6 days
    death_day = np.flatnonzero(hist[:, 0] == D)[0]
    assert p.inf_min <= death_day <= p.inf_max
    for pig in range(1, p.n_pigs):
        col = hist[:, pig]
        if (col == E).any():
            e0 = np.flatnonzero(col == E)[0]
            i0 = np.flatnonzero(col == I)[0]
            d0 = np.flatnonzero(col == D)[0]
            assert i0 - e0 == p.latent_days                              # fixed latent period L
            assert p.inf_min <= d0 - i0 <= p.inf_max                     # infectious period T
            assert ((col[d0:] == D)).all()                               # dead pigs stay dead


def test_states_only_move_forward():
    res = run_abm(BASELINE.with_(**SCENARIOS[1].fixed, seed=5))
    hist = np.array([[int(c) for c in h] for h in res.history])
    order = {S: 0, E: 1, I: 2, D: 3, R: 3}
    ranks = np.vectorize(order.get)(hist)
    assert (np.diff(ranks, axis=0) >= 0).all()


def test_no_control_before_detection_and_detection_rule():
    for seed in range(5):
        p = BASELINE.with_(detection=True, detect_delay=4, remove_sick=True, depopulate=True, seed=seed)
        res = run_abm(p)
        assert res.detection_day == res.first_death_day + 4
        before = res.counts.loc[: res.detection_day - 1]
        assert before["R"].max() == 0


def test_no_action_never_detects_or_culls():
    res = run_abm(NO_ACTION)
    assert np.isnan(res.detection_day) and res.counts["R"].max() == 0


def test_adjacency_is_a_row():
    a = adjacency(4)
    assert a.sum() == 6 and a[0, 1] == a[1, 0] == 1 and a[0, 2] == 0


def test_scenarios_middle_level_is_baseline():
    for sc in SCENARIOS:
        assert len(sc.levels) == 3
        assert sc.levels[1] == getattr(BASELINE, sc.param)


def test_controls_reduce_losses():
    m_ref, _ = an.replicate(NO_ACTION, 60, seed=1, curves=False)
    m_ctl, _ = an.replicate(an.scenario_params(BASELINE, SCENARIOS[1], 1), 60, seed=1, curves=False)
    assert m_ctl["lost"].mean() < m_ref["lost"].mean()


def test_within_pen_r0_matches_guinat():
    r0 = an.estimate_r0_within_pen(BASELINE, n=200)
    assert 1.3 <= r0["simulated"] <= 4.8                                   # Guinat et al. 2016 95% CI
    assert abs(r0["simulated"] - r0["theory"]) < 0.5


def test_abm_agrees_with_seir():
    table, _, _ = an.validate_abm_vs_seir(BASELINE, n=150)
    assert (table["CIs overlap"] == "yes").all()


def test_3d_html_is_self_contained():
    html = farm_html(run_abm(BASELINE.with_(n_pens=2, pigs_per_pen=10)))
    assert "/*__DATA__*/" not in html and "/*__THREE__*/" not in html and "__TITLE__" not in html
    assert '"history":[' in html and "THREE" in html
