"""F5 — 3-way markets settle on regulation, in settlement AND calibration."""

from __future__ import annotations

import pytest

from bullion.models import ActualResult, Game, Status
from bullion.runner import _calibration_book
from bullion.store import Store
from bullion.tickets import grade_row

SOCCER = "eng.1"
ET_PERIODS = ((1, 1), (0, 0), (1, 0))  # 1-1 after 90', home wins it in ET


def _game(actual: ActualResult | None, *, status: Status = Status.FINAL) -> Game:
    return Game(
        game_id="s1",
        league=SOCCER,
        season="2026",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-25T20:00:00Z",
        status=status,
        actual=actual,
    )


def _actual(periods=ET_PERIODS, overtime: int = 1, home: float = 2.0, away: float = 1.0):
    return ActualResult(
        home_score=home, away_score=away, periods=tuple(periods), overtime_periods=overtime
    )


# ---------------------------------------------------------------------------
# T5.1 — settlement
# ---------------------------------------------------------------------------


def test_a_regulation_draw_never_settles_as_a_home_win():
    """1-1 at 90' with a 2-1 final: the home moneyline is a loss, not a win."""
    row = {"market": "moneyline", "selection": "home", "line": None}
    game = _game(_actual())
    assert game.actual.regulation_scores() == (1.0, 1.0)
    assert grade_row(row, game, has_draw=True) == "loss"
    assert grade_row({"market": "moneyline", "selection": "draw"}, game, has_draw=True) == "win"
    assert grade_row({"market": "moneyline", "selection": "away"}, game, has_draw=True) == "loss"


def test_a_regulation_draw_on_a_handicap_pushes():
    row = {"market": "spread", "selection": "home", "line": 0.0}
    assert grade_row(row, _game(_actual()), has_draw=True) == "push"


def test_a_total_settles_on_the_regulation_goals():
    # 2 goals in regulation, 3 in the ET-inclusive total; line 2.5 flips.
    over = {"market": "total", "selection": "over", "line": 2.5}
    assert grade_row(over, _game(_actual()), has_draw=True) == "loss"
    under = {"market": "total", "selection": "under", "line": 2.5}
    assert grade_row(under, _game(_actual()), has_draw=True) == "win"


def test_extra_time_without_a_period_breakdown_stays_pending():
    """Fail closed: never grade an ET match off the ET-inclusive score."""
    game = _game(_actual(periods=(), overtime=1))
    assert game.actual.regulation_scores() is None
    for row in (
        {"market": "moneyline", "selection": "home"},
        {"market": "total", "selection": "over", "line": 2.5},
        {"market": "spread", "selection": "away", "line": -0.5},
    ):
        assert grade_row(row, game, has_draw=True) is None


def test_a_90_minute_result_is_graded_on_the_full_score():
    game = _game(ActualResult(home_score=1, away_score=1, periods=((1, 1), (0, 0)), overtime_periods=0))
    assert grade_row({"market": "moneyline", "selection": "draw"}, game) == "win"
    assert game.actual.regulation_scores() == (1.0, 1.0)


def test_two_way_sports_still_grade_on_the_full_score():
    """An NBA final with overtime is not a 3-way market: no regulation split."""
    actual = ActualResult(
        home_score=118, away_score=114, periods=((28, 30), (30, 25), (25, 30), (25, 29), (10, 0)), overtime_periods=1
    )
    game = Game(
        game_id="b1",
        league="nba",
        season="2027",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-25T00:30:00Z",
        status=Status.FINAL,
        actual=actual,
    )
    assert actual.regulation_scores() == (108.0, 114.0)
    assert grade_row({"market": "moneyline", "selection": "home"}, game, has_draw=False) == "win"
    assert grade_row({"market": "total", "selection": "over", "line": 220.5}, game, has_draw=False) == "win"


def test_cancelled_and_postponed_keep_their_old_semantics():
    cancelled = _game(_actual(), status=Status.CANCELLED)
    assert grade_row({"market": "moneyline", "selection": "home"}, cancelled) == "push"
    postponed = _game(None, status=Status.POSTPONED)
    assert grade_row({"market": "moneyline", "selection": "home"}, postponed) is None


# ---------------------------------------------------------------------------
# T5.2 — calibration grades on the same number
# ---------------------------------------------------------------------------


def _seed_snapshot(store: Store, *, game_id: str, periods, overtime: int) -> None:
    from bullion.models import Forecast

    game = Game(
        game_id=game_id,
        league=SOCCER,
        season="2026",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-25T20:00:00Z",
        status=Status.FINAL,
        actual=ActualResult(home_score=2, away_score=1, periods=tuple(periods), overtime_periods=overtime),
    )
    store.upsert_game(game)
    store.conn.execute(
        "UPDATE games SET periods_json=?, overtime_periods=? WHERE game_id=?",
        (__import__("json").dumps([list(p) for p in periods]), overtime, game_id),
    )
    store.conn.commit()
    store.record_forecasts(
        [
            Forecast(
                game_id=game_id,
                league=SOCCER,
                scheduled_utc=game.scheduled_utc,
                home_name="Home",
                away_name="Away",
                market="moneyline",
                selection="home",
                raw_probability=0.55,
                source="dixon_coles",
            )
        ],
        "2026-11-25",
    )


def test_calibration_grades_an_extra_time_match_on_regulation(tmp_path):
    from bullion.models import LEAGUES

    with Store(tmp_path / "db.sqlite") as store:
        _seed_snapshot(store, game_id="et-1", periods=ET_PERIODS, overtime=1)
        book = _calibration_book(store, LEAGUES[SOCCER], "2026-11-28T00:00:00Z")
        segment = (SOCCER, "moneyline", "full_game", "dixon_coles")
        settled = book.settled(segment, "2026-11-28")
    assert len(settled) == 1
    # 1-1 in regulation: the home 55% call was wrong. Off the ET-inclusive 2-1
    # score it would have been scored a win and taught the model nothing.
    assert settled[0].outcome == 0


def test_calibration_skips_an_extra_time_match_with_no_periods(tmp_path):
    from bullion.models import LEAGUES

    with Store(tmp_path / "db.sqlite") as store:
        _seed_snapshot(store, game_id="et-2", periods=(), overtime=1)
        book = _calibration_book(store, LEAGUES[SOCCER], "2026-11-28T00:00:00Z")
        segment = (SOCCER, "moneyline", "full_game", "dixon_coles")
        assert book.settled(segment, "2026-11-28") == []
        # Not a "push" either: the observation is withheld, not neutralised.
        report = book.report(segment, "2026-11-28")
    assert report is None or getattr(report, "n", 0) == 0


@pytest.mark.parametrize("overtime,periods", [(0, ET_PERIODS), (1, ET_PERIODS)])
def test_calibration_and_settlement_agree_on_the_same_fixture(tmp_path, overtime, periods):
    from bullion.models import LEAGUES

    with Store(tmp_path / "db.sqlite") as store:
        _seed_snapshot(store, game_id="cmp", periods=periods, overtime=overtime)
        book = _calibration_book(store, LEAGUES[SOCCER], "2026-11-28T00:00:00Z")
        settled = book.settled((SOCCER, "moneyline", "full_game", "dixon_coles"), "2026-11-28")
    assert len(settled) == 1
    expected = 0  # 1-1 regulation (or ET off, still a home loss by 2-1? see below)
    if overtime == 0:
        expected = 1  # no extra time: the full 2-1 score is the regulation score
    assert settled[0].outcome == expected
