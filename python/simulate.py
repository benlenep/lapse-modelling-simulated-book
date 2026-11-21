"""Synthetic membership book with a known monthly lapse hazard.

You decide the truth (the constants below), then write the functions that simulate
members living under it. Because you know the truth, every model you fit later can
be scored on how well it recovers it.

The hazard is built on the complementary log-log (cloglog) scale:

    hazard = 1 - exp(-exp(eta))
    eta    = BASELINE + tenure effects + member effects

This is the discrete-time version of a proportional hazards model, so a Cox model
should recover exp(beta) as hazard ratios, and a binomial GLM with a cloglog link on
person-month rows should recover beta itself.

The constants are a sensible starting point. Change them if you like: the notebooks
and tests read them from here, so the checks follow your choices. Keep the
"column[level]" naming in TRUE_EFFECTS, and keep each reference level (no effect)
first in LEVELS.

Replace each `raise NotImplementedError`, run the tests after each one:

    uv run pytest -v

then generate the data from the repository root:

    uv run python python/simulate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- the truth
START = pd.Period("2019-01", freq="M")    # month index 0
N_MONTHS = 84                             # observation runs to Dec 2025 (index 83)
LAST_JOIN = 77                            # members join up to Jun 2025 (index 77)

LEVELS = {
    "age_band": (["18-30", "31-45", "46-60", "61+"], [0.25, 0.30, 0.25, 0.20]),
    "product_tier": (["basic", "mid", "top"], [0.40, 0.40, 0.20]),
    "channel": (["direct", "broker", "online"], [0.50, 0.20, 0.30]),
}

BASELINE = -4.4          # about 1.2% a month for the reference member
TENURE_EFFECTS = {
    "early": 0.30,       # tenure months 1 to 6
    "anniversary": 0.90, # every 12th month, when the renewal price lands
    "long": -0.40,       # tenure beyond 36 months
}
# Log hazard ratios against the reference level (the first level listed above)
TRUE_EFFECTS = {
    "age_band[31-45]": -0.25,
    "age_band[46-60]": -0.40,
    "age_band[61+]": -0.70,
    "product_tier[mid]": -0.20,
    "product_tier[top]": -0.45,
    "channel[broker]": 0.25,
    "channel[online]": 0.35,
    "premium_increase": 0.06,   # per percentage point of the last renewal increase
}


def make_members(n, seed):
    """One row per member, with columns:
    member_id (1 to n), join_idx (an integer from 0 to LAST_JOIN inclusive),
    age_band, product_tier, channel (drawn from LEVELS with their probabilities),
    premium_increase (uniform between 0 and 12, rounded to 1 decimal place).

    Use rng = np.random.default_rng(seed), then rng.integers, rng.choice(levels, size=n, p=probs)
    and rng.uniform. The same seed must always give the same members.
    """
    raise NotImplementedError


def member_eta(members):
    """Return a NumPy array: each member's summed covariate effects from TRUE_EFFECTS.

    Reference levels add nothing. For every other level, add its effect where the member
    has that level (np.where helps). Then add premium_increase times its effect.
    """
    raise NotImplementedError


def tenure_eta(tenure):
    """The tenure part of eta for tenure month(s) t = 1, 2, 3, ... (a number or an array).

    early applies when t <= 6, anniversary when t is a multiple of 12, long when t > 36.
    They can overlap: month 48 is both an anniversary and long tenure.
    """
    raise NotImplementedError


def hazard(eta):
    """Monthly lapse probability from eta: 1 - exp(-exp(eta))."""
    raise NotImplementedError


def simulate(n=40_000, seed=2026):
    """Simulate each member month by month until they lapse or observation ends.

    Return one row per member with: member_id, join_month (text like "2021-03"),
    age_band, product_tier, channel, premium_increase,
    duration (months observed, at least 1) and lapsed (1 if they left, 0 if still a member).

    Suggested approach (vectorised over members, looping over tenure months):
    1. members = make_members(n, seed); base = BASELINE + member_eta(members)
    2. max_tenure = N_MONTHS - join_idx: how many months each member can be observed
    3. Start every member active, with duration = max_tenure and lapsed = 0
    4. For t = 1, 2, ...: members who are active and have t <= max_tenure are at risk.
       Draw a uniform number for each; those below hazard(base + tenure_eta(t)) lapse
       now: set their duration to t, lapsed to 1, and make them inactive.
    5. join_month = str(START + join_idx)

    Use a second generator for the lapse draws, such as np.random.default_rng(seed + 1).
    """
    raise NotImplementedError


def person_months(members):
    """Expand to one row per member per month at risk.

    Keep the member columns (except duration and lapsed) and add:
    tenure (1 to duration), lapse (1 only in the month a lapsed member leaves),
    calendar_month (text like "2024-07"), and 0/1 indicators early, anniversary and long.

    Hint: members.loc[members.index.repeat(members["duration"])] repeats each row,
    and .groupby("member_id").cumcount() + 1 numbers the months.
    """
    raise NotImplementedError


def main():
    members = simulate()
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    members.to_parquet(out / "members.parquet", index=False)
    print(f"{len(members):,} members, {members['lapsed'].mean():.1%} lapsed, "
          f"median duration {members['duration'].median():.0f} months")


if __name__ == "__main__":
    main()
