"""Tests for the simulator. Run from the repository root:  uv run pytest -v"""
import numpy as np
import pandas as pd
import pytest

import simulate as s


def test_make_members_columns_and_levels():
    m = s.make_members(500, seed=1)
    assert list(m.columns) == ["member_id", "join_idx", "age_band", "product_tier", "channel", "premium_increase"]
    assert m["member_id"].tolist() == list(range(1, 501))
    assert m["join_idx"].between(0, s.LAST_JOIN).all()
    for col, (levels, _) in s.LEVELS.items():
        assert set(m[col]) <= set(levels)
    assert m["premium_increase"].between(0, 12).all()


def test_same_seed_same_members():
    pd.testing.assert_frame_equal(s.make_members(200, seed=7), s.make_members(200, seed=7))


def test_reference_member_has_no_member_effect():
    ref = pd.DataFrame({"age_band": ["18-30"], "product_tier": ["basic"],
                        "channel": ["direct"], "premium_increase": [0.0]})
    assert s.member_eta(ref)[0] == pytest.approx(0.0)


def test_member_effects_add_up():
    m = pd.DataFrame({"age_band": ["61+"], "product_tier": ["top"],
                      "channel": ["online"], "premium_increase": [10.0]})
    expected = (s.TRUE_EFFECTS["age_band[61+]"] + s.TRUE_EFFECTS["product_tier[top]"]
                + s.TRUE_EFFECTS["channel[online]"] + 10 * s.TRUE_EFFECTS["premium_increase"])
    assert s.member_eta(m)[0] == pytest.approx(expected)


def test_tenure_effects():
    e = s.TENURE_EFFECTS
    got = s.tenure_eta(np.array([1, 7, 12, 37, 48]))
    want = [e["early"], 0.0, e["anniversary"], e["long"], e["anniversary"] + e["long"]]
    assert got == pytest.approx(want)


def test_hazard_is_a_probability_that_rises_with_eta():
    h = s.hazard(np.array([-8.0, -4.0, 0.0]))
    assert ((h > 0) & (h < 1)).all()
    assert h[0] < h[1] < h[2]


@pytest.fixture(scope="module")
def sim():
    return s.simulate(n=5_000, seed=11)


def test_simulate_shape_and_values(sim):
    assert len(sim) == 5_000
    assert set(sim["lapsed"]) <= {0, 1}
    assert (sim["duration"] >= 1).all()
    join = pd.PeriodIndex(sim["join_month"], freq="M")
    max_tenure = s.N_MONTHS - (join - s.START).map(lambda d: d.n)
    assert (sim["duration"] <= max_tenure).all()


def test_simulated_lapse_rate_is_plausible(sim):
    assert 0.2 < sim["lapsed"].mean() < 0.7


def test_person_months_add_up(sim):
    pm = s.person_months(sim)
    assert len(pm) == sim["duration"].sum()
    assert pm["lapse"].sum() == sim["lapsed"].sum()
    assert (pm.groupby("member_id")["lapse"].sum() <= 1).all()
    last = pm.groupby("member_id")["tenure"].max()
    assert (last.sort_index().to_numpy() == sim.set_index("member_id")["duration"].sort_index().to_numpy()).all()
