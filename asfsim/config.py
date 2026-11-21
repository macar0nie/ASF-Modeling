"""Parameters, farm presets and the three experimental scenarios.

Baseline disease values come from Guinat et al. (2016), the best-fitting model (L = 4 days):
    beta_w = 0.6 /day within a pen, beta_b = 0.3 /day between adjacent pens,
    infectious period 3-6 days, R0 within a pen = 2.8 (95% CI 1.3-4.8).
Upper beta_w (1.05) and the 5-day latent period come from Nielsen et al. (2017).
Control parameters (detection delay, removal chance, biosecurity effect) are group assumptions.
"""
from dataclasses import dataclass, field, replace, asdict


@dataclass(frozen=True)
class Params:
    # farm
    n_pens: int = 10
    pigs_per_pen: int = 20
    # disease (Guinat et al. 2016)
    beta_w: float = 0.6          # within-pen transmission rate, per day
    beta_b: float = 0.3          # between adjacent pens, per day
    latent_days: float = 4.0     # L
    inf_min: int = 3             # infectious period until death, uniform [inf_min, inf_max] days (realistic mode)
    inf_max: int = 6
    durations: str = "realistic"  # "realistic" (fixed L, uniform T) or "exponential" (SEIR-equivalent)
    # detection and control
    detection: bool = False      # if False there is no detection and no control (the no-action reference)
    detect_delay: int = 3        # d: days after the first ASF death until ASF is confirmed
    remove_sick: bool = False    # isolate / cull infectious pigs after detection
    p_remove: float = 0.5        # daily chance each infectious pig is found and removed
    biosecurity: bool = False    # cut between-pen contact after detection
    e_bio: float = 0.5           # fraction of beta_b removed by biosecurity
    depopulate: bool = False     # cull every pig in a pen that has had an ASF death
    # run
    max_days: int = 365
    seed: int = 1

    @property
    def n_pigs(self) -> int:
        return self.n_pens * self.pigs_per_pen

    @property
    def mean_inf(self) -> float:
        return (self.inf_min + self.inf_max) / 2

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


FARMS = {
    "Small commercial (200 pigs, 10 pens)": dict(n_pens=10, pigs_per_pen=20),
    "Backyard (20 pigs, 2 pens)": dict(n_pens=2, pigs_per_pen=10),
}

BASELINE = Params()

# No-action reference: baseline parameters, nothing is detected and nothing is done.
REFERENCE = dict(detection=False, remove_sick=False, biosecurity=False, depopulate=False)


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    param: str            # the Params field that is varied
    label: str            # human name of the parameter
    levels: tuple         # three values; the middle one is the baseline
    level_names: tuple
    fixed: dict = field(default_factory=dict)   # other settings for this scenario
    question: str = ""


SCENARIOS = [
    Scenario(
        key="S1", name="Scenario 1: Contact intensity", param="beta_w",
        label="Within-pen transmission rate β_w (per day)",
        levels=(0.3, 0.6, 1.05), level_names=("Low density (0.3)", "Baseline (0.6)", "High density (1.05)"),
        fixed=dict(REFERENCE),
        question="How fast and how far does ASF spread when pigs are in closer contact?",
    ),
    Scenario(
        key="S2", name="Scenario 2: Detection speed", param="detect_delay",
        label="Detection delay d (days after the first death)",
        levels=(1, 3, 7), level_names=("Fast (1 day)", "Baseline (3 days)", "Slow (7 days)"),
        fixed=dict(detection=True, remove_sick=True, p_remove=0.5, biosecurity=False, depopulate=False),
        question="How many pigs are saved by spotting and reporting ASF sooner?",
    ),
    Scenario(
        key="S3", name="Scenario 3: Between-pen biosecurity", param="e_bio",
        label="Reduction in between-pen contact e_bio",
        levels=(0.0, 0.5, 0.9), level_names=("None (0%)", "Baseline (50%)", "Strict (90%)"),
        fixed=dict(detection=True, detect_delay=3, remove_sick=True, p_remove=0.5, biosecurity=True,
                   depopulate=False),
        question="Does separating pens keep the outbreak from reaching the rest of the farm?",
    ),
]
SCENARIO_BY_KEY = {s.key: s for s in SCENARIOS}

STATE_NAMES = ["Susceptible", "Exposed", "Infectious", "Dead (ASF)", "Removed (culled)"]
S, E, I, D, R = range(5)

PARAM_TABLE = [
    ("Within-pen transmission rate (per day)", "β_w", "0.6", "0.3 – 1.05", "Guinat et al. 2016; Nielsen et al. 2017"),
    ("Between-pen transmission rate (per day)", "β_b", "0.3", "0.1 – 0.5", "Guinat et al. 2016"),
    ("Latent period (days)", "L", "4", "3 – 5", "Guinat et al. 2016; Nielsen et al. 2017"),
    ("Infectious period until death (days)", "T", "3 – 6 (uniform, mean 4.5)", "3 – 14", "Guinat et al. 2016"),
    ("Case fatality, acute form", "–", "100%", "–", "WOAH 2019"),
    ("Initial infected pigs", "I₀", "1", "1", "Study design"),
    ("Detection delay after first death (days)", "d", "3", "0 – 10", "Group assumption"),
    ("Daily chance a sick pig is found and removed", "p_remove", "0.5", "0 – 0.9", "Group assumption"),
    ("Reduction in between-pen contact (biosecurity)", "e_bio", "0.5", "0 – 0.9", "Group assumption"),
    ("Simulation time step", "Δt", "1 day", "–", "Study design"),
]
