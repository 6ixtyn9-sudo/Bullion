"""F4 — timezone-proof UTC-hour fatigue, and an aligned schedule day.

A US night game is filed under the *previous* ET day (a 19:30 EST tip is
``00:30Z`` the next calendar day). Bucketing by SAST made a Friday-night to
Saturday-afternoon back-to-back read as a gap of 2 and silently miss the
penalty. Everything below is synthetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bullion.engines import Result, rest_adjustment
from bullion.engines import basketball as bb
from bullion.models import LEAGUES, et_day, sast_day
from bullion.store import Store

NB = LEAGUES["nba"]
NCAA_M = LEAGUES["ncaa_m"]
NHL = LEAGUES["nhl"]

FRI_NIGHT = "2026-11-28T00:30:00Z"  # Fri 19:30 EST
SAT_AFTERNOON = "2026-11-28T18:00:00Z"  # Sat 13:00 EST
SAT_NIGHT = "2026-11-29T00:30:00Z"  # Sat 19:30 EST


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def backfill():
    spec = importlib.util.spec_from_file_location(
        "bullion_backfill_fatigue", ROOT / "scripts" / "backfill.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(home: str, away: str, day: str, *, instant: str = "", hs=112, as_=104) -> Result:
    return Result(home, away, hs, as_, day, scheduled_utc=instant)


# ---------------------------------------------------------------------------
# T4.1 — the night-game back-to-back, through both paths
# ---------------------------------------------------------------------------


def _seed_night_history(store: Store) -> None:
    """14 finals over distinct days, then the Friday-night B2B game.

    The target matchday is the first day with ``>= MIN_SEED_HISTORY`` priors, so
    the seeder's refit window opens exactly there — which is what makes the
    seeded number and a live tick's number comparable at 1e-9.
    """
    from bullion.models import Game, Status

    def final(game_id: str, instant: str, home: str, away: str, hs: int, as_: int) -> None:
        store.upsert_game(
            Game(
                game_id=game_id,
                league="nba",
                season="2027",
                home_team_id=home,
                away_team_id=away,
                home_name=f"Home {home}",
                away_name=f"Away {away}",
                scheduled_utc=instant,
                status=Status.FINAL,
            )
        )
        store.conn.execute(
            "UPDATE games SET home_score=?, away_score=? WHERE game_id=?", (hs, as_, game_id)
        )
        store.conn.commit()

    # 14 finals on 14 distinct days (18:00Z = 13:00 EST, so the ET label is the
    # calendar date) plus the Friday-night game = 15 priors: exactly the floor,
    # crossed by the target matchday itself. That makes the day the seeder's
    # refit window opens on, which is what makes its number comparable to a live
    # tick's on the same fixture.
    for index in range(14):
        day = f"2026-11-{index + 1:02d}"
        final(
            f"hist{index}",
            f"{day}T18:00:00Z",
            "h1" if index % 2 else "h2",
            "a1" if index % 3 else "a2",
            108 + index,
            104 - (index % 4),
        )
    final("fri-night", FRI_NIGHT, "h1", "a9", 110, 101)

    # Target matchday: Saturday afternoon, h1 on the back of Friday night. The
    # row is a final because that is the only shape the seeder replays; the live
    # tick below prices it pre-tip, at an instant before its own result exists.
    final("target", SAT_AFTERNOON, "h1", "a2", 110, 103)


def test_fatigue_multiplier_counts_a_night_game_to_afternoon_back_to_back():
    prior = _result("h1", "a9", et_day(FRI_NIGHT), instant=FRI_NIGHT)
    assert et_day(FRI_NIGHT) == "2026-11-27", "19:30 EST is the previous ET day"
    assert sast_day(FRI_NIGHT) == "2026-11-28", "...and the next SAST day"
    assert bb.fatigue_multiplier(
        [prior], "h1", et_day(SAT_AFTERNOON), game_scheduled_utc=SAT_AFTERNOON
    ) == pytest.approx(bb.REST_B2B_PENALTY)
    # The same pair through the calendar path the old code used: ET day gap 1.
    assert bb.fatigue_multiplier([prior], "h1", et_day(SAT_AFTERNOON)) == pytest.approx(
        bb.REST_B2B_PENALTY
    )
    # SAST labels for the same pair are a gap of 1 too; the failure mode was the
    # *night* target below, not this one.
    assert bb.fatigue_multiplier(
        [prior], "h1", "2026-11-28", game_scheduled_utc=SAT_NIGHT
    ) == pytest.approx(bb.REST_B2B_PENALTY)


def test_saturday_night_back_to_back_is_seen_where_the_sast_math_missed_it(tmp_path):
    """Fri 19:30 EST -> Sat 19:30 EST: 24h of rest, SAST days a gap of 1."""
    prior = _result("h1", "a9", et_day(FRI_NIGHT), instant=FRI_NIGHT)
    assert bb.fatigue_multiplier(
        [prior], "h1", et_day(SAT_NIGHT), game_scheduled_utc=SAT_NIGHT
    ) == pytest.approx(bb.REST_B2B_PENALTY)
    # 42h later is not a back-to-back.
    sunday_afternoon = "2026-11-29T18:00:00Z"
    assert bb.fatigue_multiplier(
        [prior], "h1", et_day(sunday_afternoon), game_scheduled_utc=sunday_afternoon
    ) == pytest.approx(1.0)
    # A same-ET-day doubleheader (9h apart) is a back-to-back too.
    evening = "2026-11-28T09:30:00Z"
    assert bb.fatigue_multiplier(
        [_result("h1", "a9", et_day("2026-11-28T00:30:00Z"), instant="2026-11-28T00:30:00Z")],
        "h1",
        et_day(evening),
        game_scheduled_utc=evening,
    ) == pytest.approx(bb.REST_B2B_PENALTY)


def test_third_in_four_stacks_on_top_of_the_back_to_back():
    priors = [
        _result("h1", "a9", et_day("2026-11-26T00:30:00Z"), instant="2026-11-26T00:30:00Z"),
        _result("h1", "a9", et_day(FRI_NIGHT), instant=FRI_NIGHT),
    ]
    assert bb.fatigue_multiplier(
        priors, "h1", et_day(SAT_NIGHT), game_scheduled_utc=SAT_NIGHT
    ) == pytest.approx(bb.REST_B2B_PENALTY * bb.REST_3IN4_PENALTY)
    # Two games inside 84h with none of them a back-to-back: 3-in-4 only.
    spaced = [
        _result("h1", "a9", "2026-11-26", instant="2026-11-26T12:00:00Z"),
        _result("h1", "a9", "2026-11-27", instant="2026-11-27T12:00:00Z"),
    ]
    target = "2026-11-28T23:00:00Z"  # 35h and 59h after them
    assert bb.fatigue_multiplier(
        spaced, "h1", et_day(target), game_scheduled_utc=target
    ) == pytest.approx(bb.REST_3IN4_PENALTY)


def test_rest_adjustment_boundary_is_34_hours():
    prior = _result("h1", "a9", "2026-11-27", instant="2026-11-27T00:00:00Z")
    at_34 = "2026-11-28T10:00:00Z"
    past_34 = "2026-11-28T10:30:00Z"
    assert rest_adjustment([prior], "h1", "2026-11-28", game_scheduled_utc=at_34) == pytest.approx(0.975)
    assert rest_adjustment([prior], "h1", "2026-11-28", game_scheduled_utc=past_34) == pytest.approx(1.0)


def test_legacy_results_without_instants_keep_the_calendar_fallback():
    prior = _result("h1", "a9", "2026-11-27")
    assert prior.scheduled_utc == ""
    assert bb.fatigue_multiplier([prior], "h1", "2026-11-28") == pytest.approx(0.975)
    assert bb.fatigue_multiplier([prior], "h1", "2026-11-29") == pytest.approx(1.0)
    assert rest_adjustment([prior], "h1", "2026-11-28") == pytest.approx(0.975)
    assert rest_adjustment([prior], "h1", "2026-11-30") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T4.2 — schedule_day is one convention per sport
# ---------------------------------------------------------------------------


def test_schedule_day_buckets_us_sports_by_et_and_soccer_by_sast():
    for league in (LEAGUES["nba"], LEAGUES["wnba"], NCAA_M, LEAGUES["ncaa_w"], NHL):
        assert league.schedule_day(FRI_NIGHT) == "2026-11-27"
    assert LEAGUES["nfl"].schedule_day(FRI_NIGHT) == "2026-11-27"
    assert LEAGUES["cfb"].schedule_day(FRI_NIGHT) == "2026-11-27"
    soccer = LEAGUES["eng.1"]
    assert soccer.schedule_day(FRI_NIGHT) == "2026-11-28"
    assert soccer.schedule_day(FRI_NIGHT) == sast_day(FRI_NIGHT)


def test_runner_results_and_the_seeder_agree_on_the_day_per_game(tmp_path):
    from bullion.runner import _results

    with Store(tmp_path / "db.sqlite") as store:
        _seed_night_history(store)
        results = _results(store, NB, "2026-11-28T16:00:00Z")
        assert results
        for row in results:
            assert row.day == NB.schedule_day(row.scheduled_utc)
        # The Friday-night game is filed under ET Friday, not SAST Saturday.
        fri = [r for r in _results(store, NB, "2026-11-28T16:00:00Z") if r.scheduled_utc == FRI_NIGHT]
        assert fri and fri[0].day == "2026-11-27"


# ---------------------------------------------------------------------------
# T4.3 — seeded probabilities equal the live engine's, to 1e-9
# ---------------------------------------------------------------------------


def test_seeded_probability_matches_a_live_tick_on_the_same_fixture(tmp_path, backfill, monkeypatch):
    """The strongest form of the alignment claim: same number, both paths."""
    from bullion.models import BookQuote
    from bullion.runner import _results

    with Store(tmp_path / "bullion.db") as store:
        _seed_night_history(store)
        backfill.seed_walk_forward_ml(store, leagues=("nba",))
        seeded = {
            row["selection"]: row["raw_probability"]
            for row in store.conn.execute(
                "SELECT selection, raw_probability FROM forecast_snapshots "
                "WHERE game_id='target' AND market='moneyline'"
            )
        }
        assert set(seeded) == {"home", "away"}

        as_of = "2026-11-28T16:00:00Z"
        results = _results(store, NB, as_of)
        quote = BookQuote(
            game_id="target",
            book="DraftKings",
            market="moneyline",
            observed_utc="2026-11-28T15:00:00Z",
            side_prices={"home": 1.91, "away": 1.91},
            pre_match=True,
        )
        from bullion.models import Game

        game = Game(
            game_id="target",
            league="nba",
            season="2027",
            home_team_id="h1",
            away_team_id="a2",
            home_name="Home h1",
            away_name="Away a2",
            scheduled_utc=SAT_AFTERNOON,
        )
        live = [
            f
            for f in bb.forecast(
                game,
                [quote],
                results=results,
                reference_day=NB.schedule_day(game.scheduled_utc),
            )
            if f.market == "moneyline"
        ]
        live_home = next(f for f in live if f.selection == "home")
        assert seeded["home"] == pytest.approx(live_home.raw_probability, abs=1e-9)
        assert seeded["home"] + seeded["away"] == pytest.approx(1.0)

        # ... and the penalty is genuinely in the number: switch it off and the
        # fatigued home side gains probability.
        monkeypatch.setattr(bb, "fatigue_multiplier", lambda *a, **k: 1.0)
        rested = next(
            f
            for f in bb.forecast(
                game,
                [quote],
                results=results,
                reference_day=NB.schedule_day(game.scheduled_utc),
            )
            if f.market == "moneyline" and f.selection == "home"
        )
        assert rested.raw_probability > live_home.raw_probability
