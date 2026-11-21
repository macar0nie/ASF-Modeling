# PigPen ASF: African Swine Fever spread on a pig farm

A simulation of how **African Swine Fever (ASF)** spreads inside one pig farm after **one infected pig** comes into
contact with healthy pigs. Made for CS 3110 Modelling & Simulation, Case 5: Infectious Disease Spread Simulation.

It uses a **hybrid design**:

* an **agent-based model** where every pig has a pen, a disease state (Susceptible → Exposed → Infectious → Dead,
  or Removed by culling) and its own timers. This model drives the app and the 3D farm.
* a **stochastic SEIR model** (pen-structured chain binomial) used as the baseline that validates the agent-based model.

Transmission values come from Guinat et al. (2016): β_w = 0.6/day within a pen, β_b = 0.3/day between adjacent
pens, latent period 4 days and infectious period 3–6 days (within-pen R0 ≈ 2.8).

## Run it on your computer

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

Other commands:

```bash
python -m pytest                # 19 verification and validation tests
python run_experiments.py       # 3 scenarios + reference, 1,000 replications each, saved to results/
python run_experiments.py --farm backyard
python make_3d.py --scenario S2 --level 0 --out fast_detection.html   # standalone 3D replay (offline HTML)
```

## Deploy on Streamlit Community Cloud (free)

1. Create a GitHub repository and upload everything in this folder (keep the folder layout; `app.py` must be at the top level).
2. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app**.
3. Pick your repository, branch `main`, and set **Main file path** to `app.py`.
4. Optional: under **Advanced settings** choose Python 3.12.
5. Click **Deploy**. The first build takes a few minutes; you get a public `*.streamlit.app` link.

Streamlit installs the packages from `requirements.txt` automatically. The 3D view needs no extra package because
three.js is bundled in `asfsim/static/`.

## What the app shows

| Tab | What it does |
|---|---|
| 🐷 Simulation run | One outbreak: KPI tiles, pigs in each state per day, an infected-per-pen heatmap and the day the virus reached each pen |
| 🧱 3D farm | A blocky toy farm that replays the same run. Pigs wander their pens, get a yellow **!** when infected, turn dark with a red **+** when sick, lie down grey with a halo when they die, and vanish when culled. Red flags mark pens with deaths, and a vet walks in when ASF is confirmed. Play, pause, change speed, scrub days, drag to rotate, scroll to zoom, hover a pen for its counts. Downloadable as one offline HTML file |
| 🧪 Experimental scenarios | Runs the three scenarios below and compares them with 95% confidence intervals |
| 🔁 Replications | Runs the sidebar settings many times: summary table, histogram and a band of sick pigs per day |
| ✅ Model validation | ABM vs SEIR, simulated within-pen R0 vs Guinat et al., and 10th-death timing vs Nielsen et al. |
| 📖 About the model | Rules, assumptions, parameter table and references |

**You enter the inputs, then start it yourself.** Nothing runs when the app opens. Type your values in the sidebar
and press **▶ Start simulation**. The inputs are: number of pens and pigs per pen, β_w, β_b, latent period,
infectious period (min and max), duration mode, detection on/off and delay, removing sick pigs (and the daily chance
of finding one), biosecurity between pens (and how much contact it cuts), culling whole pens, the number of
replications and the random seed. To change anything, edit the values and press Start again. The two optional
"Fill in" boxes at the top copy a scenario level or a farm size into the inputs for you; they still wait for Start.

## The three experimental scenarios

Each scenario changes **one** parameter at three levels. The middle level is the baseline.

| Scenario | Parameter varied | Levels | Other settings |
|---|---|---|---|
| 1 Contact intensity | β_w within-pen rate | 0.3, **0.6**, 1.05 per day | No detection or control |
| 2 Detection speed | d, days from first death to confirmed ASF | 1, **3**, 7 days | Sick pigs then removed (50% a day) |
| 3 Between-pen biosecurity | e_bio, cut in between-pen contact | 0%, **50%**, 90% | Detection at 3 days, sick pigs removed |

Results with 1,000 replications are in `results/` (`scenario_summary_commercial.md` and `_backyard.md`).
On the 200-pig farm, doing nothing loses about 194 pigs. Confirming ASF 1 day after the first death instead of 7
cuts losses from about 66 to 29 pigs. Adding 50% biosecurity between pens cuts losses from about 40 to 19 pigs.

## Validation results

* ABM (exponential mode) vs SEIR: attack rate, deaths, peak size, peak day and duration all agree (95% CIs overlap).
* Within-pen R0 from the simulation is about 2.8, against 2.8 (95% CI 1.3–4.8) in Guinat et al. (2016).
* In a 360-pig, 12-pen house the 10th death comes around day 17 (median), within the 13–19 days of Nielsen et al. (2017).

## Project structure

```
app.py                  Streamlit app
asfsim/config.py        parameters, farm presets and the three scenarios
asfsim/model.py         agent-based model and stochastic SEIR model
asfsim/analysis.py      replications, scenarios, validation
asfsim/viz3d.py         builds the 3D replay page
asfsim/static/          farm3d.html (3D page) and three.min.js (MIT licence)
run_experiments.py      batch experiments into results/
make_3d.py              standalone 3D replay
tests/test_model.py     verification tests
.streamlit/config.toml  app colours
```

## Assumptions and limits

One farm only (no spread between farms, ticks or wild boar); pens stand in a row and only neighbours touch; every
pig starts healthy and unvaccinated; acute ASF (every infectious pig dies unless culled first); parameters come
from European experiments with the Georgia 2007/1 strain; detection and control values are group assumptions.

## References

* Guinat, C. et al. (2016). Experimental pig-to-pig transmission dynamics for African swine fever virus, Georgia 2007/1 strain. *Epidemiology and Infection*, 144(1), 25–34.
* Nielsen, J. P. et al. (2017). Estimation of the transmission dynamics of African swine fever virus within a swine house. *Epidemiology and Infection*, 145(13), 2787–2796.
* World Organisation for Animal Health (2019). *African swine fever technical disease card*.

The art is original blocky "toy" style built from simple 3D boxes in three.js; it uses no brand assets.
