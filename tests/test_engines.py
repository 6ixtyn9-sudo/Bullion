"""Engine math and calibration honesty tests."""

from __future__ import annotations

import pytest

from bullion.calibration import CalibrationBook, fit_platt, logit, sigmoid
from bullion.engines import Result, normal_line_probs
from bullion.engines import football as football_engine
from bullion.engines import nhl as nhl_engine
from bullion.engines import soccer as soccer_engine
from bullion.models import BookQuote, Game, Status


def _results(league: str, *, n: int = 40, home_goals: float = 1.6, away_goals: float = 1.2):
    rows = []
    for index in range(n):
        day = f"2026-10-{(index % 28) + 1:02d}"
        first, second = f"t{index % 4}", f"t{(index + 1) % 4}"
        # Alternate home/away so ratings are identifiable, as a real schedule is.
        home_first = index % 2 == 0
        rows.append(
            Result(
                home_id=first if home_first else second,
                away_id=second if home_first else first,
                home_score=home_goals if home_first else away_goals,
                away_score=away_goals if home_first else home_goals,
                day=day,
            )
        )
    return rows


def _game(league: str = "nhl") -> Game:
    return Game(
        game_id="g1",
        league=league,
        season="2026",
        home_team_id="h0",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-30T00:00:00Z",
        status=Status.SCHEDULED,
    )


def _quotes(league: str, *, total_line: float = 5.5, spread_line: float = -1.5):
    return [
        BookQuote(
            game_id="g1",
            book="DraftKings",
            market="total",
            observed_utc="2026-11-29T12:00:00Z",
            side_prices={"over": 1.95, "under": 1.87},
            line=total_line,
            pre_match=True,
        ),
        BookQuote(
            game_id="g1",
            book="DraftKings",
            market="spread",
            observed_utc="2026-11-29T12:00:00Z",
            side_prices={"home": 1.91, "away": 1.91},
            line=spread_line,
            pre_match=True,
        ),
        BookQuote(
            game_id="g1",
            book="DraftKings",
            market="moneyline",
            observed_utc="2026-11-29T12:00:00Z",
            side_prices={"home": 1.85, "away": 1.95},
            pre_match=True,
        ),
    ]


# ---------------------------------------------------------------------------
# NHL
# ---------------------------------------------------------------------------


def test_nhl_moneyline_is_two_way_and_includes_overtime():
    forecasts = nhl_engine.forecast(
        _game(), _quotes("nhl"), results=_results("nhl"), reference_day="2026-11-30"
    )
    moneyline = [row for row in forecasts if row.market == "moneyline"]
    assert len(moneyline) == 2
    assert sum(row.raw_probability for row in moneyline) == pytest.approx(1.0, abs=1e-9)
    assert all(row.raw_probability > 0 for row in moneyline)
    # The official final contains the OT/SO goal, so the expected total must
    # exceed the regulation-only expectation.
    home_rate, away_rate = 1.6, 1.2
    margin, total, _draw = nhl_engine._final_distributions(home_rate, away_rate)
    assert sum(margin.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(total.values()) == pytest.approx(1.0, abs=1e-9)
    # A regulation tie always becomes a one-goal final (OT/SO), never a draw.
    assert 1 in margin and -1 in margin and max(margin) >= 2
    assert margin[1] > 0 and margin[-1] > 0


def test_nhl_puckline_and_total_probabilities_are_complementary():
    forecasts = nhl_engine.forecast(
        _game(),
        _quotes("nhl", total_line=6.0, spread_line=-1.5),
        results=_results("nhl"),
        reference_day="2026-11-30",
    )
    by_market = {}
    for row in forecasts:
        by_market.setdefault(row.market, {})[row.selection] = row
    total = by_market["total"]
    assert total["over"].raw_probability + total["under"].raw_probability == pytest.approx(
        1.0, abs=1e-9
    )
    assert total["over"].push_probability == pytest.approx(total["under"].push_probability)
    assert total["over"].push_probability > 0, "an integer total carries real push mass"
    spread = by_market["spread"]
    assert spread["home"].raw_probability + spread["away"].raw_probability == pytest.approx(
        1.0, abs=1e-9
    )
    assert spread["home"].line == -1.5


def test_nhl_without_history_emits_nothing():
    assert nhl_engine.forecast(_game(), _quotes("nhl"), results=[], reference_day="2026-11-30") == []


# ---------------------------------------------------------------------------
# Football
# ---------------------------------------------------------------------------


def test_football_fpi_surface_requires_a_real_predictor():
    game = _game("nfl")
    assert (
        football_engine.forecast(
            game,
            _quotes("nfl"),
            results=_results("nfl"),
            reference_day="2026-11-30",
            surface=football_engine.FPI_SOURCE,
            predictor=None,
        )
        == []
    ), "no silent fallback from a missing provider prediction"


def test_football_surfaces_are_distinct_and_normalise():
    game = _game("nfl")
    empirical = football_engine.forecast(
        game,
        _quotes("nfl", total_line=44.5, spread_line=-2.5),
        results=_results("nfl", n=30, home_goals=24, away_goals=20),
        reference_day="2026-11-30",
    )
    moneyline = [row for row in empirical if row.market == "moneyline"]
    assert sum(row.raw_probability for row in moneyline) == pytest.approx(1.0, abs=1e-9)
    total = [row for row in empirical if row.market == "total"]
    assert sum(row.raw_probability for row in total) == pytest.approx(1.0, abs=1e-9)
    integer_over, integer_push = normal_line_probs(44.0, 13.0, 44.0, "over")
    assert integer_push > 0
    assert 0.0 < integer_over < 1.0


def test_football_surface_resolution_uses_brier_skill_only():
    fpi = football_engine.FPI_SOURCE
    empirical = football_engine.EMPIRICAL_SOURCE
    assert football_engine.resolve_surface({fpi: 0.20, empirical: 0.24}) == fpi
    assert football_engine.resolve_surface({fpi: 0.24, empirical: 0.20}) == empirical
    assert football_engine.resolve_surface({fpi: 0.21, empirical: 0.211}) is None
    assert football_engine.resolve_surface({fpi: 0.21}) is None


# ---------------------------------------------------------------------------
# Soccer
# ---------------------------------------------------------------------------


def test_soccer_one_x_two_sums_to_one_and_draw_is_present():
    forecasts = soccer_engine.forecast(
        _game("eng.1"), _quotes("eng.1", total_line=2.5, spread_line=-0.5),
        results=_results("eng.1", home_goals=1.6, away_goals=1.2),
        reference_day="2026-11-30",
    )
    probabilities = {row.selection: row.raw_probability for row in forecasts if row.market == "moneyline"}
    assert set(probabilities) == {"home", "draw", "away"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert probabilities["draw"] > 0.15


def test_soccer_score_matrix_is_normalised_and_rho_moves_low_scores():
    base = soccer_engine.score_matrix(1.5, 1.1, 0.0)
    positive = soccer_engine.score_matrix(1.5, 1.1, 0.15)
    negative = soccer_engine.score_matrix(1.5, 1.1, -0.15)
    assert sum(sum(row) for row in base) == pytest.approx(1.0, abs=1e-9)
    assert sum(sum(row) for row in positive) == pytest.approx(1.0, abs=1e-9)
    # Dixon-Coles: positive rho shifts low-score mass from the diagonal
    # (0-0, 1-1) to one-goal results; negative rho does the reverse.
    assert positive[0][0] < base[0][0]
    assert positive[1][1] < base[1][1]
    assert positive[0][1] > base[0][1]
    assert negative[0][0] > base[0][0]


def test_soccer_totals_have_half_line_and_integer_behaviour():
    forecasts = soccer_engine.forecast(
        _game("eng.1"),
        _quotes("eng.1", total_line=2.0, spread_line=-0.5),
        results=_results("eng.1"),
        reference_day="2026-11-30",
    )
    total = {row.selection: row for row in forecasts if row.market == "total"}
    assert total["over"].push_probability > 0
    assert total["over"].raw_probability + total["under"].raw_probability == pytest.approx(
        1.0, abs=1e-9
    )
    half = soccer_engine.forecast(
        _game("eng.1"),
        _quotes("eng.1", total_line=2.5, spread_line=-0.5),
        results=_results("eng.1"),
        reference_day="2026-11-30",
    )
    half_total = {row.selection: row for row in half if row.market == "total"}
    assert half_total["over"].push_probability == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_excludes_same_day_and_future_results():
    book = CalibrationBook()
    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
    for index in range(40):
        book.add(segment, 0.6, 1, "2026-11-01")
    book.add(segment, 0.6, 0, "2026-11-28")  # same day as the scoring day
    book.add(segment, 0.6, 0, "2026-12-05")  # future
    report = book.report(segment, "2026-11-28")
    assert report.n == 40
    assert report.sufficient, report.reasons
    assert report.brier == pytest.approx(0.16, abs=1e-9)


def test_calibration_requires_thirty_samples():
    book = CalibrationBook()
    segment = ("nhl", "total", "full_game", "nhl_poisson_v1")
    for _ in range(29):
        book.add(segment, 0.5, 1, "2026-11-01")
    report = book.report(segment, "2026-11-28")
    assert not report.sufficient
    assert "insufficient_walk_forward_samples" in report.reasons
    assert report.calibrate(0.55) == 0.55, "provisional rows keep the raw probability"


def test_calibration_brier_cap_rejects_a_bad_segment():
    book = CalibrationBook()
    segment = ("eng.1", "moneyline", "full_game", "dixon_coles_v1")
    for index in range(40):
        book.add(segment, 0.9, 1 if index % 5 == 0 else 0, "2026-11-01")
    report = book.report(segment, "2026-11-28", brier_cap=0.22, cap_vs_base_rate=True)
    assert not report.sufficient
    assert "brier_above_cap" in report.reasons or "not_better_than_base_rate" in report.reasons


def test_platt_shrinks_overconfidence_and_stays_monotone():
    pairs = [(0.9, 1)] * 20 + [(0.9, 0)] * 40
    a, b = fit_platt(pairs)
    shrunk = sigmoid(a * logit(0.9) + b)
    assert 0.3 < shrunk < 0.9
    assert sigmoid(a * logit(0.6) + b) < shrunk


def test_calibration_counts_unique_games_not_correlated_sides():
    """A fixture with three quoted sides is one observation, not three."""
    segment = ("nfl", "moneyline", "full_game", "espn_fpi_v1")
    book = CalibrationBook()
    for index in range(10):
        for side, probability, outcome in (("home", 0.6, 1), ("away", 0.35, 0), ("draw", 0.05, 0)):
            book.add(segment, probability, outcome, "2026-11-01", f"game{index}")
    report = book.report(segment, "2026-11-28")

    assert report.n == 30, "thirty rows were recorded"
    assert report.n_games == 10, "but only ten fixtures were actually settled"
    assert not report.sufficient
    assert "insufficient_walk_forward_samples" in report.reasons

    for index in range(10, 30):
        book.add(segment, 0.6, 1, "2026-11-01", f"game{index}")
        book.add(segment, 0.35, 0, "2026-11-01", f"game{index}")
    report = book.report(segment, "2026-11-28")
    assert report.n_games == 30
    assert report.to_dict()["n_games"] == 30


def test_calibration_weights_every_fixture_equally():
    """One row in game A must weigh as much as three rows in game B."""
    segment = ("nhl", "moneyline", "full_game", "nhl_poisson_v1")
    book = CalibrationBook()
    book.add(segment, 1.0, 1, "2026-11-01", "quiet")  # single-row fixture, perfect
    for _ in range(3):
        book.add(segment, 0.0, 1, "2026-11-01", "busy")  # three-row fixture, all wrong
    report = book.report(segment, "2026-11-28")

    # Unweighted this is 0.75 (three of four rows are wrong); per fixture it is
    # half of the games right and half wrong, i.e. 0.5.
    assert report.brier == pytest.approx(0.5)
    assert report.n == 4
    assert report.n_games == 2


def test_nhl_overtime_always_adds_exactly_one_goal():
    """NHL OT is sudden death and a shootout winner is credited one goal."""
    from bullion.engines import MAX_GOALS, poisson_pmf

    assert nhl_engine.OT_GOAL_DISTRIBUTION == {1: 1.0}, (
        "a drawn regulation game cannot also be drawn after overtime"
    )
    home_rate = away_rate = 1.35
    _margin, total, regulation_draw = nhl_engine._final_distributions(home_rate, away_rate)

    expected: dict[int, float] = {}
    for home_goals in range(MAX_GOALS + 1):
        for away_goals in range(MAX_GOALS + 1):
            probability = poisson_pmf(home_rate, home_goals) * poisson_pmf(away_rate, away_goals)
            extra = 1 if home_goals == away_goals else 0
            expected[home_goals + away_goals + extra] = (
                expected.get(home_goals + away_goals + extra, 0.0) + probability
            )
    assert regulation_draw > 0.2, "the OT rule has to matter at these rates"
    # The production distribution renormalises the truncated Poisson grid.
    total_sum = sum(expected.values())
    expected = {goals: value / total_sum for goals, value in expected.items()}
    assert set(total) == set(expected)
    for goals, probability in expected.items():
        assert total[goals] == pytest.approx(probability, abs=1e-12)


def test_fpi_surface_uses_the_projected_margin_without_expected_scores():
    """ESPN FPI publishes ``teamPredPtDiff`` and no expected score."""
    from bullion.espn import PredictorEvidence

    game = Game(
        game_id="401",
        league="nfl",
        season="2026",
        home_team_id="10",
        away_team_id="20",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-28T23:00:00Z",
    )
    predictor_evidence = PredictorEvidence(
        prob_home=0.62,
        expected_home=None,
        expected_away=None,
        stats_side="home",
        predicted_margin=-3.5,
    )
    quotes = [
        BookQuote(
            game_id="401",
            book="DraftKings",
            market="spread",
            observed_utc="2026-11-28T12:00:00Z",
            side_prices={"home": 1.9, "away": 1.9},
            line=-2.5,
            pre_match=True,
        ),
        BookQuote(
            game_id="401",
            book="DraftKings",
            market="total",
            observed_utc="2026-11-28T12:00:00Z",
            side_prices={"over": 1.9, "under": 1.9},
            line=45.0,
            pre_match=True,
        ),
    ]
    forecasts = football_engine.forecast(
        game,
        quotes,
        results=[],
        reference_day="2026-11-28",
        surface=football_engine.FPI_SOURCE,
        predictor=predictor_evidence,
    )
    spreads = [row for row in forecasts if row.market == "spread"]
    assert spreads, "a projected margin is enough to price a spread"
    assert all(row.expected_margin == pytest.approx(-3.5) for row in spreads)
    # No expected score means no expected total, so totals stay unpriced.
    assert not [row for row in forecasts if row.market == "total"]
    home = next(row for row in forecasts if row.market == "moneyline" and row.selection == "home")
    assert home.expected_margin == pytest.approx(-3.5)


def test_surface_tiebreak_is_deterministic_when_briers_are_indistinguishable():
    assert football_engine.resolve_surface(
        {football_engine.FPI_SOURCE: 0.201, football_engine.EMPIRICAL_SOURCE: 0.202}
    ) is None, "a coin flip is not a measurement"
    assert (
        football_engine.tiebreak_surface(
            {football_engine.FPI_SOURCE: 0.201, football_engine.EMPIRICAL_SOURCE: 0.202}
        )
        == football_engine.FPI_SOURCE
    )
    assert (
        football_engine.tiebreak_surface(
            {football_engine.FPI_SOURCE: 0.2, football_engine.EMPIRICAL_SOURCE: 0.2}
        )
        == football_engine.EMPIRICAL_SOURCE
    ), "an exact tie falls back to the empirical surface"
    assert football_engine.tiebreak_surface({football_engine.FPI_SOURCE: 0.2}) == (
        football_engine.FPI_SOURCE
    )
    assert football_engine.tiebreak_surface({}) is None
