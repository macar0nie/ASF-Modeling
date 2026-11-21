"""PigPen ASF: a Streamlit app for the African Swine Fever within-farm simulation (CS 3110, Case 5).

    streamlit run app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from asfsim import analysis as an
from asfsim.config import (BASELINE, FARMS, PARAM_TABLE, REFERENCE, SCENARIO_BY_KEY, SCENARIOS, STATE_NAMES,
                           Params)
from asfsim.model import run_abm
from asfsim.viz3d import farm_html

st.set_page_config(page_title="PigPen ASF Simulator", page_icon="🐷", layout="wide")

COLORS = {"S": "#ff9ebb", "E": "#f2b705", "I": "#e0457b", "D": "#7d7b8a", "R": "#5b8def"}
LEVEL_COLORS = ["#4dabf7", "#845ef7", "#ff6b3d"]
REF_COLOR = "#7d7b8a"

# --------------------------------------------------------------------------------------------------
# Sidebar: inputs and scenario presets
# --------------------------------------------------------------------------------------------------
PRESETS = {"Custom (keep my values)": None, "No action (reference)": dict(REFERENCE)}
for sc in SCENARIOS:
    for lvl, name in zip(sc.levels, sc.level_names):
        PRESETS[f"{sc.key} {sc.name.split(': ')[1]} – {name}"] = {**sc.fixed, sc.param: lvl}
FARM_PRESETS = {"Custom (keep my values)": None, **FARMS}

DEFAULTS = dict(n_pens=BASELINE.n_pens, pigs_per_pen=BASELINE.pigs_per_pen, beta_w=BASELINE.beta_w,
                beta_b=BASELINE.beta_b, latent=int(BASELINE.latent_days), inf_min=BASELINE.inf_min,
                inf_max=BASELINE.inf_max, durations="realistic", detection=False,
                detect_delay=BASELINE.detect_delay, remove_sick=False, p_remove=BASELINE.p_remove,
                biosecurity=False, e_bio=BASELINE.e_bio, depopulate=False, seed=1, n_reps=200)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    preset = PRESETS[st.session_state["preset"]]
    if preset is None:
        return
    for k in ("beta_w", "beta_b", "latent", "inf_min", "inf_max", "detect_delay", "p_remove", "e_bio"):
        st.session_state[k] = DEFAULTS[k]
    for k, v in preset.items():
        st.session_state[k] = v


def apply_farm():
    farm = FARM_PRESETS[st.session_state["farm_preset"]]
    if farm is not None:
        st.session_state.update(farm)


with st.sidebar:
    st.title("🐷 PigPen ASF")
    st.caption("Type in your values, then press **▶ Start simulation**. Nothing runs until you do.")
    st.selectbox("Fill in a scenario preset (optional)", list(PRESETS), key="preset", on_change=apply_preset)
    st.selectbox("Fill in a farm size (optional)", list(FARM_PRESETS), key="farm_preset", on_change=apply_farm)
    with st.form("inputs"):
        st.subheader("Farm")
        c1, c2 = st.columns(2)
        c1.number_input("Number of pens", 1, 30, step=1, key="n_pens")
        c2.number_input("Pigs per pen", 2, 60, step=1, key="pigs_per_pen")
        st.subheader("Disease")
        st.number_input("β_w within-pen transmission rate (per day)", 0.0, 5.0, step=0.05, format="%.2f", key="beta_w")
        st.number_input("β_b between adjacent pens (per day)", 0.0, 5.0, step=0.05, format="%.2f", key="beta_b")
        st.number_input("Latent period L (days)", 1, 21, step=1, key="latent")
        c1, c2 = st.columns(2)
        c1.number_input("Infectious period T, min (days)", 1, 30, step=1, key="inf_min")
        c2.number_input("T, max (days)", 1, 30, step=1, key="inf_max")
        st.radio("Disease durations", ["realistic", "exponential"], key="durations", horizontal=True,
                 help="realistic: fixed L, T uniform between min and max (Guinat et al. 2016). "
                      "exponential: memoryless, the same as the SEIR model (used for validation).")
        st.subheader("Detection and control")
        st.checkbox("Farmer reports ASF (detection)", key="detection",
                    help="Off = no action: nothing is detected and no control measure runs.")
        st.number_input("Detection delay d (days after the first death)", 0, 60, step=1, key="detect_delay")
        st.checkbox("Remove sick pigs after detection", key="remove_sick")
        st.number_input("Daily chance a sick pig is found (0–1)", 0.0, 1.0, step=0.05, format="%.2f", key="p_remove")
        st.checkbox("Biosecurity between pens after detection", key="biosecurity")
        st.number_input("Cut in between-pen contact e_bio (0–1)", 0.0, 1.0, step=0.05, format="%.2f", key="e_bio")
        st.checkbox("Cull every pig in pens with an ASF death", key="depopulate")
        st.subheader("Runs")
        st.number_input("Replications (for the Replications tab)", 10, 2000, step=10, key="n_reps")
        st.number_input("Random seed", 0, 1_000_000, step=1, key="seed")
        started = st.form_submit_button("▶ Start simulation", type="primary", width="stretch")

ss = st.session_state
if started:
    if ss.inf_max < ss.inf_min:
        st.sidebar.error("T max must be at least T min.")
    else:
        ss["P"] = Params(n_pens=int(ss.n_pens), pigs_per_pen=int(ss.pigs_per_pen), beta_w=float(ss.beta_w),
                         beta_b=float(ss.beta_b), latent_days=float(ss.latent), inf_min=int(ss.inf_min),
                         inf_max=int(ss.inf_max), durations=ss.durations, detection=bool(ss.detection),
                         detect_delay=int(ss.detect_delay), remove_sick=bool(ss.remove_sick),
                         p_remove=float(ss.p_remove), biosecurity=bool(ss.biosecurity), e_bio=float(ss.e_bio),
                         depopulate=bool(ss.depopulate), seed=int(ss.seed))
        ss["P_reps"] = int(ss.n_reps)
        ss["P_label"] = ss.preset if PRESETS.get(ss.preset) is not None else "your values"
P = ss.get("P")
if P is not None:
    st.sidebar.success(f"Showing the run started with: {P.n_pigs} pigs, β_w {P.beta_w}, β_b {P.beta_b}, seed {P.seed}. "
                       "Change values and press Start again to rerun.")


def need_start():
    st.info("👈 Enter your input values in the sidebar, then press **▶ Start simulation**.")


# --------------------------------------------------------------------------------------------------
# Cached runs
# --------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=32)
def single_run(p: Params):
    return run_abm(p)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_3d(p: Params, subtitle: str):
    return farm_html(single_run(p), subtitle=subtitle)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_replicate(p: Params, n: int):
    return an.replicate(p, n, seed=p.seed * 1000 + 7)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_scenario(base: Params, key: str, n: int):
    return an.run_scenario(base, SCENARIO_BY_KEY[key], n, seed=2026)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_reference(base: Params, n: int):
    return an.run_reference(base, n, seed=2026)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_validation(base: Params, n: int):
    table, c_abm, c_seir = an.validate_abm_vs_seir(base, n)
    return table, c_abm, c_seir, an.estimate_r0_within_pen(base, n=n), an.detection_check(base, n=n)


def fmt_day(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"day {int(x)}"


def epi_figure(counts: pd.DataFrame, detection_day=None, title="Pigs in each state, day by day"):
    fig = go.Figure()
    for k, name in zip(["S", "E", "I", "D", "R"], STATE_NAMES):
        if k == "R" and counts[k].max() == 0:
            continue
        fig.add_trace(go.Scatter(x=counts.index, y=counts[k], name=name, mode="lines",
                                 line=dict(color=COLORS[k], width=3)))
    if detection_day is not None and not math.isnan(detection_day):
        fig.add_vline(x=detection_day, line_dash="dash", line_color="#e03131",
                      annotation_text=f"ASF confirmed (day {int(detection_day)})", annotation_position="top")
    fig.update_layout(title=title, xaxis_title="Day", yaxis_title="Pigs", height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=60, l=10, r=10))
    return fig


tabs = st.tabs(["🐷 Simulation run", "🧱 3D farm", "🧪 Experimental scenarios", "🔁 Replications",
                "✅ Model validation", "📖 About the model"])

# --------------------------------------------------------------------------------------------------
# 1. One run
# --------------------------------------------------------------------------------------------------
with tabs[0]:
    if P is None:
        need_start()
    else:
        res = single_run(P)
        m = res.metrics()
        st.subheader(f"One outbreak on a {P.n_pigs}-pig farm ({P.n_pens} pens × {P.pigs_per_pen} pigs)")
        c = st.columns(6)
        c[0].metric("Pigs infected", f"{m['infected']}", f"{m['attack_rate']:.0%} of the herd", delta_color="off")
        c[1].metric("Died of ASF", m["deaths"])
        c[2].metric("Culled", m["culled"])
        c[3].metric("Peak sick at once", m["peak_infectious"], f"on day {m['peak_day']}", delta_color="off")
        c[4].metric("Outbreak over on", f"day {m['duration']}")
        c[5].metric("ASF confirmed", fmt_day(m["detection_day"]),
                    f"first death {fmt_day(m['first_death_day'])}", delta_color="off")
        st.plotly_chart(epi_figure(res.counts, res.detection_day), width="stretch")

        left, right = st.columns(2)
        with left:
            z = res.pen_infected.T
            fig = go.Figure(go.Heatmap(z=z, x=list(range(z.shape[1])), y=[f"Pen {k + 1}" for k in range(z.shape[0])],
                                       colorscale=[[0, "#fff5f8"], [0.4, "#ff9ebb"], [1, "#a61e4d"]],
                                       colorbar=dict(title="Infected")))
            fig.update_layout(title="Where the virus is: infected pigs (E + I) per pen", xaxis_title="Day", height=360,
                              margin=dict(t=60, l=10, r=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        with right:
            first_inf = []
            for k in range(P.n_pens):
                days = np.flatnonzero(res.pen_infected[:, k] > 0)
                first_inf.append(days[0] if len(days) else np.nan)
            fig = go.Figure(go.Bar(x=[f"Pen {k + 1}" for k in range(P.n_pens)], y=first_inf, marker_color="#e0457b"))
            fig.update_layout(title="Day the virus reached each pen (blank = never)", yaxis_title="Day", height=360,
                              margin=dict(t=60, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
        st.caption("Pens stand in a row: pen k only touches pens k−1 and k+1, so the virus moves pen by pen. "
                   "Open the 3D farm tab to watch this run.")
        with st.expander("Daily counts (table and CSV)"):
            st.dataframe(res.counts, width="stretch")
            st.download_button("Download daily_counts.csv", res.counts.to_csv().encode(), "daily_counts.csv")

# --------------------------------------------------------------------------------------------------
# 2. 3D farm
# --------------------------------------------------------------------------------------------------
with tabs[1]:
    if P is None:
        need_start()
    else:
        sub = ss.get("P_label", "your values")
        st.markdown("A blocky replay of **the same run** as the first tab. Press **▶ Play**, drag to turn the farm, "
                    "scroll to zoom and hover a pen to see its pigs.")
        html3d = cached_3d(P, sub)
        if hasattr(st, "iframe"):                       # Streamlit 1.5x+
            st.iframe(html3d, height=720)
        else:                                           # older Streamlit
            components.html(html3d, height=720, scrolling=False)
        st.download_button("⬇ Download this 3D replay (one HTML file, works offline)", html3d.encode(),
                           "asf_farm_3d.html", mime="text/html")

# --------------------------------------------------------------------------------------------------
# 3. Experimental scenarios
# --------------------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Three experimental scenarios")
    st.markdown("Each scenario varies **one parameter** at three levels (the middle one is the baseline) while "
                "everything else stays at the baseline values on the farm size from your inputs. The no-action "
                "reference (no detection, no control) is shown in grey for comparison.")
    st.dataframe(pd.DataFrame([{"Scenario": s.name, "Parameter varied": s.label,
                                "Levels": ", ".join(s.level_names), "Question": s.question} for s in SCENARIOS]),
                 hide_index=True, width="stretch")
    farm = P if P is not None else BASELINE
    base = BASELINE.with_(n_pens=farm.n_pens, pigs_per_pen=farm.pigs_per_pen)
    st.caption(f"Farm used: {base.n_pigs} pigs in {base.n_pens} pens (from your inputs; change it in the sidebar and press Start).")
    n = st.select_slider("Replications per level", [50, 100, 200, 500, 1000], value=200,
                         help="Each level is run this many times with different random numbers. 1,000 is used in the report.")
    if st.button("▶ Run all three scenarios", type="primary"):
        ss["scen_n"] = n
    if "scen_n" in ss:
        n_run = ss["scen_n"]
        with st.spinner(f"Running {n_run} outbreaks × 10 settings…"):
            ref_m, ref_c = cached_reference(base, n_run)
            results = {sc.key: cached_scenario(base, sc.key, n_run) for sc in SCENARIOS}
        all_runs = pd.concat([ref_m] + [results[k][0] for k in results], ignore_index=True)
        table = an.scenario_table(all_runs)

        kpi = st.radio("Compare by", ["lost", "attack_rate", "deaths", "culled", "duration", "peak_infectious"],
                       format_func=lambda k: an.KPI_LABELS[k], horizontal=True)
        for sc in SCENARIOS:
            m_sc, curves = results[sc.key]
            st.markdown(f"#### {sc.name}")
            st.caption(f"Varies {sc.label}. {sc.question}")
            l, r = st.columns(2)
            groups = [("No action (reference)", ref_m, REF_COLOR)] + \
                     [(name, m_sc[m_sc.level_name == name], LEVEL_COLORS[i]) for i, name in enumerate(sc.level_names)]
            means = [g[kpi].astype(float).mean() for _, g, _ in groups]
            cis = [1.96 * g[kpi].astype(float).std(ddof=1) / np.sqrt(len(g)) for _, g, _ in groups]
            fig = go.Figure(go.Bar(x=[g[0] for g in groups], y=means, marker_color=[g[2] for g in groups],
                                   error_y=dict(type="data", array=cis, visible=True, color="#2b2340"),
                                   text=[f"{v:.0%}" if kpi == "attack_rate" else f"{v:.1f}" for v in means],
                                   textposition="outside"))
            fig.update_layout(title=f"{an.KPI_LABELS[kpi]}: mean and 95% CI", height=380,
                              margin=dict(t=60, l=10, r=10), yaxis_tickformat=".0%" if kpi == "attack_rate" else None)
            l.plotly_chart(fig, width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ref_c.index, y=ref_c["I"], name="No action", line=dict(color=REF_COLOR, dash="dot")))
            for i, name in enumerate(sc.level_names):
                cc = curves[name]
                fig.add_trace(go.Scatter(x=cc.index, y=cc["I"], name=name, line=dict(color=LEVEL_COLORS[i], width=3)))
            fig.update_layout(title="Average number of sick (infectious) pigs per day", xaxis_title="Day",
                              yaxis_title="Pigs", height=380, margin=dict(t=60, l=10, r=10),
                              legend=dict(orientation="h", y=-0.25))
            r.plotly_chart(fig, width="stretch")

        st.markdown("#### Summary table (means with ± 95% CI half-width)")
        show = table.copy()
        for k in ["attack_rate", "lost", "deaths", "culled", "peak_day", "duration", "detection_day"]:
            if k in show:
                pct = k == "attack_rate"
                show[k] = [("–" if pd.isna(v) else (f"{v:.1%} ± {ci:.1%}" if pct else f"{v:.1f} ± {ci:.1f}"))
                           for v, ci in zip(show[k], show.get(k + "_ci", pd.Series([0] * len(show))).fillna(0))]
        show = show[["scenario", "level", "runs", "attack_rate", "lost", "deaths", "culled", "peak_day", "duration",
                     "detection_day"]].rename(columns={k: an.KPI_LABELS.get(k, k) for k in show.columns})
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button("Download all runs (CSV)", all_runs.to_csv(index=False).encode(), "scenario_runs.csv")
    else:
        st.info("Press **Run all three scenarios** to start. 200 replications take about 20 seconds.")

# --------------------------------------------------------------------------------------------------
# 4. Replications of the sidebar settings
# --------------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Replications of your inputs")
    if P is None:
        need_start()
    else:
        n_rep = ss.get("P_reps", 200)
        st.markdown(f"One run is just one possible outbreak. Here your inputs are run **{n_rep} times** "
                    "(set *Replications* in the sidebar) to see the average and the spread.")
        with st.spinner(f"Running {n_rep} outbreaks…"):
            m_rep, c_rep = cached_replicate(P, n_rep)
        summ = an.summarize(m_rep)
        st.dataframe(summ.drop(columns=["kpi"]).round(3), hide_index=True, width="stretch")
        l, r = st.columns(2)
        fig = go.Figure(go.Histogram(x=m_rep["lost"], marker_color="#e0457b", nbinsx=40))
        fig.update_layout(title="Pigs lost per outbreak", xaxis_title="Pigs lost", yaxis_title="Runs", height=360)
        l.plotly_chart(fig, width="stretch")
        fig = go.Figure([
            go.Scatter(x=c_rep.index, y=c_rep["I_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=c_rep.index, y=c_rep["I_lo"], fill="tonexty", fillcolor="rgba(224,69,123,0.2)",
                       line=dict(width=0), name="95% of runs"),
            go.Scatter(x=c_rep.index, y=c_rep["I"], line=dict(color="#e0457b", width=3), name="Mean sick pigs")])
        fig.update_layout(title="Sick (infectious) pigs per day across runs", xaxis_title="Day", height=360)
        r.plotly_chart(fig, width="stretch")
        st.download_button("Download replication results (CSV)", m_rep.to_csv(index=False).encode(), "replications.csv")

# --------------------------------------------------------------------------------------------------
# 5. Validation
# --------------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Verification and validation")
    st.markdown("1. **Model-to-model:** the agent-based model in SEIR-equivalent mode against the stochastic SEIR model.\n"
                "2. **Within-pen R0:** pigs infected by the first pig, against Guinat et al. (2016): 2.8 (95% CI 1.3–4.8).\n"
                "3. **Detection timing:** day of the 10th death in a 360-pig, 12-pen house, against Nielsen et al. (2017): 13–19 days.")
    nv = st.select_slider("Runs per check", [100, 200, 500], value=200)
    if st.button("▶ Run validation", type="primary"):
        ss["val_n"] = nv
    if "val_n" in ss:
        farm = P if P is not None else BASELINE
        base = BASELINE.with_(n_pens=farm.n_pens, pigs_per_pen=farm.pigs_per_pen)
        with st.spinner("Running validation…"):
            vt, c_abm, c_seir, r0, det = cached_validation(base, ss["val_n"])
        st.markdown("##### 1. Agent-based vs SEIR (no control, exponential durations)")
        st.dataframe(vt.round(2), hide_index=True, width="stretch")
        fig = go.Figure()
        for k in ["S", "I", "D"]:
            fig.add_trace(go.Scatter(x=c_abm.index, y=c_abm[k], name=f"ABM {k}", line=dict(color=COLORS[k], width=3)))
            fig.add_trace(go.Scatter(x=c_seir.index, y=c_seir[k], name=f"SEIR {k}",
                                     line=dict(color=COLORS[k], width=2, dash="dash")))
        fig.update_layout(title="Mean curves: agent-based (solid) vs SEIR (dashed)", xaxis_title="Day", height=400)
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("##### 2. Within-pen R0")
        c1.metric("Simulated", f"{r0['simulated']:.2f}", f"95% CI {r0['ci_low']:.2f} – {r0['ci_high']:.2f}",
                  delta_color="off")
        c1.caption(f"Theory β_w × mean T = {r0['theory']:.2f}. Published: 2.8 (1.3–4.8). "
                   f"{'✅ inside' if 1.3 <= r0['simulated'] <= 4.8 else '❌ outside'} the published interval.")
        c2.markdown("##### 3. Day of the 10th death (360 pigs)")
        c2.metric("Simulated median", f"day {det['median']:.0f}", f"middle half: days {det['p25']:.0f} – {det['p75']:.0f}",
                  delta_color="off")
        c2.caption(f"Published: days 13–19. {'✅ inside' if 13 <= det['median'] <= 19 else '❌ outside'} that range.")
    st.caption("Code-level verification (pig totals stay constant, no spread when β = 0, same seed → same result, "
               "and more) is in tests/test_model.py: run `python -m pytest`.")

# --------------------------------------------------------------------------------------------------
# 6. About
# --------------------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("How the model works")
    st.markdown("""
**What is simulated.** African Swine Fever (ASF) spreading inside **one pig farm** after **one infected pig** comes
into contact with healthy pigs. Pens stand in a row; pigs in the same pen are in full contact, pigs in neighbouring
pens have less contact, and pens further apart have none.

**Disease states.** Susceptible → Exposed (infected, not yet spreading) → Infectious (sick, spreading) → Dead.
Pigs taken out by control measures become Removed (culled). Acute ASF is assumed, so every infectious pig dies
unless it is removed first.

**Each day** (time step 1 day):
1. For every pen, the force of infection is λ_k = β_w·I_k/N_k + β_b·Σ_adjacent I_j/N_j, and each healthy pig is infected with probability 1 − e^(−λ_k).
2. Exposed pigs become infectious after the latent period L; infectious pigs die after their infectious period T.
3. ASF is confirmed d days after the first death (if the farmer reports it).
4. After that, the chosen control measures run: remove sick pigs, cut between-pen contact, or cull whole pens.

**Two models.** The *agent-based model* follows each pig and drives the app and the 3D farm. The *stochastic SEIR
model* counts pigs per pen and is used to validate the agent-based one.

**Assumptions.** All pigs start healthy and unvaccinated; one infectious pig in pen 1 on day 0; no pigs enter or
leave except by death or control; no spread between farms, ticks or wild boar.
""")
    st.markdown("**Parameters**")
    st.dataframe(pd.DataFrame(PARAM_TABLE, columns=["Parameter", "Symbol", "Baseline", "Range", "Source"]),
                 hide_index=True, width="stretch")
    st.markdown("""
**References**
- Guinat, C. et al. (2016). Experimental pig-to-pig transmission dynamics for African swine fever virus, Georgia 2007/1 strain. *Epidemiology and Infection* 144(1), 25–34.
- Nielsen, J. P. et al. (2017). Estimation of the transmission dynamics of African swine fever virus within a swine house. *Epidemiology and Infection* 145(13), 2787–2796.
- World Organisation for Animal Health (2019). *African swine fever technical disease card*.
""")
