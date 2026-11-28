"""The four US basketball lanes: registration, ingestion and seeding contracts.

What these tests are actually protecting:

* the lanes are *registered* correctly (ESPN ids, groups, regulation periods,
  calendar-year seasons) — a wrong id silently prices a different competition;
* one main line per ``(game, market, book)``, and the college anchor uses that
  line exactly once;
* scoreboard finals never return a scheduled or in-progress fixture, so a
  caller cannot mistake a tip for a final;
* the basketball walk-forward seeder is leak-free, idempotent, and refits on a
  weekly cadence **without** paying for fits on a warm cache;
* a subset probe or a subset daily run cannot erase other lanes' evidence.

Everything here is synthetic. No test touches the network (``conftest`` blocks
sockets outright), and no number below is a claim about a real ESPN payload.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bullion.engines import Result
from bullion.engines import basketball as bb
from bullion.espn import (
    EspnIdentityError,
    _normalize_main_lines,
    parse_scoreboard,
    scoreboard_finals,
    scoreboard_url,
)
from bullion.models import (
    BASKETBALL_LEAGUES,
    LEAGUES,
    BookQuote,
    CoverageGap,
    Game,
    Sport,
    Status,
    sast_day,
    to_utc_z,
)
from bullion.store import Store
from tests.factories import close, event, offer

ROOT = Path(__file__).resolve().parents[1]
SCHEDULED = "2026-11-29T00:30:00Z"
DAY = "2026-11-28"


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_four_basketball_lanes_are_registered_as_live_predictor_lanes():
    assert BASKETBALL_LEAGUES == ("nba", "ncaa_m", "ncaa_w", "wnba")
    assert len(LEAGUES) == 13
    for key in BASKETBALL_LEAGUES:
        league = LEAGUES[key]
        assert league.predictor is True, key
        assert league.cutoff_hour_sast == 20, key
        assert league.espn_league_id is not None, key
    assert [LEAGUES[k].espn_league_id for k in BASKETBALL_LEAGUES] == ["46", "41", "54", "59"]
    # Men's and women's D1 are served only under groups=50.
    assert LEAGUES["ncaa_m"].scoreboard_groups == 50
    assert LEAGUES["ncaa_w"].scoreboard_groups == 50
    assert LEAGUES["nba"].scoreboard_groups is None


def test_regulation_periods_are_per_lane_not_per_sport():
    # Men's D1 plays two 20-minute halves; the women's game plays four
    # 10-minute quarters, so a four-period linescore is regulation there.
    assert LEAGUES["ncaa_m"].regulation_periods == 2
    assert LEAGUES["ncaa_w"].regulation_periods == 4
    assert LEAGUES["nba"].regulation_periods == 4
    assert LEAGUES["wnba"].regulation_periods == 4


def test_only_the_wnba_reads_seasons_as_calendar_years():
    assert LEAGUES["wnba"].calendar_year_season is True
    for key in ("nba", "ncaa_m", "ncaa_w"):
        assert LEAGUES[key].calendar_year_season is False, key


def test_existing_nine_lanes_are_untouched_by_basketball_registration():
    for key in (
        "nhl",
        "nfl",
        "cfb",
        "eng.1",
        "esp.1",
        "ger.1",
        "ita.1",
        "fra.1",
        "uefa.champions",
    ):
        assert key in LEAGUES, key
        assert LEAGUES[key].sport is not Sport.BASKETBALL, key


# --------------------------------------------------------------------------
# scoreboard URL and identity
# --------------------------------------------------------------------------


def test_basketball_scoreboard_url_uses_the_wide_window_and_groups():
    url = scoreboard_url(LEAGUES["ncaa_m"], "20261128")
    assert "limit=500" in url
    assert "groups=50" in url
    assert "dates=20261128" in url
    nba = scoreboard_url(LEAGUES["nba"], "20261128")
    assert "limit=500" in nba
    assert "groups=" not in nba
    # The nine existing lanes keep the 100-event window.
    assert "limit=100" in scoreboard_url(LEAGUES["nhl"], "20261128")
    assert "groups=" not in scoreboard_url(LEAGUES["nhl"], "20261128")


def test_identity_guard_raises_only_on_contradiction():
    with pytest.raises(EspnIdentityError):
        scoreboard_finals({"leagues": [{"id": "42"}], "events": []}, LEAGUES["ncaa_w"])
    # An absent or empty leagues[] block is not evidence of anything.
    assert scoreboard_finals({"events": []}, LEAGUES["ncaa_w"]) == ([], [])
    assert scoreboard_finals({"leagues": [], "events": []}, LEAGUES["ncaa_m"]) == ([], [])
    games, gaps = scoreboard_finals(
        {"leagues": [{"id": "54"}, {"id": "54"}], "events": []}, LEAGUES["ncaa_w"]
    )
    assert (games, gaps) == ([], [])


# --------------------------------------------------------------------------
# main-line normalisation
# --------------------------------------------------------------------------


def _quote(market: str, line, *, book: str = "DraftKings", game: str = "g1", main=True):
    sides = {"home": 1.91, "away": 1.91} if market != "total" else {"over": 1.91, "under": 1.91}
    if market == "moneyline":
        sides = {"home": 1.7, "away": 2.1}
        line = None
    return BookQuote(
        game_id=game,
        book=book,
        market=market,
        observed_utc="2026-11-28T12:00:00Z",
        side_prices=sides,
        line=line,
        pre_match=True,
        main_line=main,
    )


def test_normalisation_keeps_exactly_one_main_line_per_group():
    quotes = [
        _quote("spread", -3.5),
        _quote("spread", -4.5),
        _quote("spread", -5.5),
        _quote("total", 148.5),
        _quote("total", 151.5),
        _quote("total", 154.5),
        _quote("moneyline", None),
    ]
    _normalize_main_lines(quotes)
    spreads = [q for q in quotes if q.market == "spread"]
    totals = [q for q in quotes if q.market == "total"]
    assert sum(q.main_line for q in spreads) == 1
    assert sum(q.main_line for q in totals) == 1
    # Deterministic without a provider flag: the closest handicap to zero, and
    # the middle total (not the first or last row of the payload).
    assert [q.line for q in spreads if q.main_line] == [-3.5]
    assert [q.line for q in totals if q.main_line] == [151.5]
    # A moneyline has no line to choose between, so it is always main.
    assert all(q.main_line for q in quotes if q.market == "moneyline")


def test_provider_declaration_wins_over_the_deterministic_rule():
    quotes = [_quote("spread", -3.5), _quote("spread", -4.5), _quote("total", 150.5)]
    _normalize_main_lines(quotes, [False, True, None])
    assert [q.line for q in quotes if q.market == "spread" and q.main_line] == [-4.5]


def test_normalisation_never_leaves_two_mains_in_one_group():
    quotes = [_quote("total", 150.5), _quote("total", 152.5), _quote("total", 154.5)]
    _normalize_main_lines(quotes, [True, True, True])
    assert sum(q.main_line for q in quotes) == 1


def test_scoreboard_parse_marks_one_main_line_per_market():
    spread_offer = offer(
        spread={
            "home": {"close": {"line": "-4.5", "odds": -110}, "open": {"line": "-4.5", "odds": -110}},
            "away": {"close": {"line": "4.5", "odds": -110}, "open": {"line": "4.5", "odds": -110}},
        },
        total={
            "over": {"close": {"line": "150.5", "odds": -110}, "open": {"line": "150.5", "odds": -110}},
            "under": {"close": {"line": "150.5", "odds": -110}, "open": {"line": "150.5", "odds": -110}},
        },
    )
    payload = {"events": [event("401", offers=[spread_offer])]}
    board = parse_scoreboard(payload, LEAGUES["ncaa_m"], observed_utc="2026-11-28T12:00:00Z")
    for market in ("spread", "total"):
        rows = [q for q in board.quotes if q.market == market]
        assert len(rows) == 1, market
        assert rows[0].main_line is True
    assert any(q.market == "spread" for q in board.quotes)


# --------------------------------------------------------------------------
# scoreboard finals
# --------------------------------------------------------------------------


def test_scoreboard_finals_returns_post_events_only():
    payload = {
        "events": [
            event("1", state="post", home_score="78", away_score="71"),
            event("2", state="pre"),
            event("3", state="in", home_score="40", away_score="38"),
        ]
    }
    games, _gaps = scoreboard_finals(payload, LEAGUES["ncaa_m"])
    assert [game.game_id for game in games] == ["1"]
    assert games[0].status is Status.FINAL
    assert (games[0].actual.home_score, games[0].actual.away_score) == (78.0, 71.0)


def test_scoreboard_finals_tolerate_missing_linescores():
    payload = {"events": [event("1", state="post", home_score="78", away_score="71")]}
    games, gaps = scoreboard_finals(payload, LEAGUES["nba"])
    assert len(games) == 1
    assert games[0].actual.periods == ()
    assert [gap.field for gap in gaps] == ["boxscore.linescores"]


def test_non_empty_linescore_that_contradicts_the_final_is_dropped():
    payload = {
        "events": [
            event(
                "1",
                state="post",
                home_score="90",
                away_score="70",
                home_lines=[{"value": 20}, {"value": 20}],
                away_lines=[{"value": 30}, {"value": 30}],
            )
        ]
    }
    games, gaps = scoreboard_finals(payload, LEAGUES["ncaa_m"])
    assert games[0].actual.periods == ()
    assert any(gap.field == "scoreboard_finals.linescore_mismatch" for gap in gaps)


def test_college_regulation_halves_do_not_count_as_overtime():
    payload = {
        "events": [
            event(
                "1",
                state="post",
                home_score="80",
                away_score="75",
                home_lines=[{"value": 41}, {"value": 39}],
                away_lines=[{"value": 38}, {"value": 37}],
            )
        ]
    }
    games, _gaps = scoreboard_finals(payload, LEAGUES["ncaa_w"])
    assert len(games[0].actual.periods) == 2
    assert games[0].actual.overtime_periods == 0
    assert games[0].actual.regulation_scores() == (80.0, 75.0)


def test_overtime_is_counted_against_the_lane_regulation_periods():
    payload = {
        "events": [
            event(
                "1",
                state="post",
                home_score="88",
                away_score="84",
                home_lines=[{"value": 38}, {"value": 38}, {"value": 12}],
                away_lines=[{"value": 36}, {"value": 36}, {"value": 12}],
            )
        ]
    }
    games, _gaps = scoreboard_finals(payload, LEAGUES["ncaa_m"])
    assert len(games[0].actual.periods) == 3
    assert games[0].actual.overtime_periods == 1
    assert games[0].actual.regulation_scores() == (76.0, 72.0)


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def _history(days: int = 45, seed: int = 11):
    import random

    rng = random.Random(seed)
    results = []
    for index in range(days):
        day = f"2026-{9 + index // 28:02d}-{1 + index % 28:02d}"
        strength = 80 + (index % 6) * 3
        results.append(
            Result(
                home_id=f"h{index % 8}",
                away_id=f"a{index % 8}",
                home_score=round(strength + rng.gauss(0, 7)),
                away_score=round(strength - 3 + rng.gauss(0, 7)),
                day=day,
            )
        )
    return results


def _game(league: str = "ncaa_m", game_id: str = "g1") -> Game:
    return Game(
        game_id=game_id,
        league=league,
        season="2027",
        home_team_id="h1",
        away_team_id="a2",
        home_name="Home",
        away_name="Away",
        scheduled_utc=SCHEDULED,
        status=Status.SCHEDULED,
    )


def _board(league: str, game_id: str = "g1"):
    def quote(market, line, main=True):
        return _quote(market, line, game=game_id, main=main)

    return [
        quote("moneyline", None),
        quote("spread", -5.5, main=False),
        quote("spread", -6.5),
        quote("spread", -7.5, main=False),
        quote("total", 148.5, main=False),
        quote("total", 150.5),
        quote("total", 152.5, main=False),
    ]


def test_college_spread_is_anchored_once_at_the_main_line():
    results = _history()
    quotes = _board("ncaa_m")
    forecasts = bb.forecast(
        _game(), quotes, results=results, reference_day=DAY, surface=bb.EMPIRICAL_SOURCE
    )
    spreads = [f for f in forecasts if f.market == "spread"]
    assert spreads, "synthetic board must produce a spread ladder"
    anchored = spreads[0].expected_margin
    model_margin, _total, _sigma_margin, _sigma_total, _prob = bb.empirical_expectations(
        "ncaa_m",
        results,
        bb.fit_lane_ratings("ncaa_m", results, reference_day=DAY),
        home_id="h1",
        away_id="a2",
        reference_day=DAY,
        neutral_site=False,
    )
    expected = 0.20 * model_margin + 0.80 * 6.5
    assert anchored == pytest.approx(expected, abs=1e-3)
    # Anchored once: every alt line is scored against the same mean.
    assert all(f.expected_margin == anchored for f in spreads)


def test_spread_ladder_is_monotonic_across_alternate_lines():
    forecasts = bb.forecast(
        _game(),
        _board("ncaa_m"),
        results=_history(),
        reference_day=DAY,
        surface=bb.EMPIRICAL_SOURCE,
    )
    home_rows = sorted(
        (f for f in forecasts if f.market == "spread" and f.selection == "home"),
        key=lambda f: f.line,
    )
    probabilities = [f.raw_probability for f in home_rows]
    # A shorter home handicap must never be less likely than a longer one.
    assert probabilities == sorted(probabilities)
    assert len(home_rows) == 3


def test_nba_and_wnba_are_not_anchored():
    for league in ("nba", "wnba"):
        forecasts = bb.forecast(
            _game(league),
            _board(league),
            results=_history(),
            reference_day=DAY,
            surface=bb.EMPIRICAL_SOURCE,
        )
        model_margin = next(f for f in forecasts if f.market == "spread").expected_margin
        spread_rows = [f for f in forecasts if f.market == "spread"]
        assert all(f.expected_margin == model_margin for f in spread_rows)


def test_moneyline_is_never_anchored():
    results = _history()
    forecasts = bb.forecast(
        _game("ncaa_m"),
        _board("ncaa_m"),
        results=results,
        reference_day=DAY,
        surface=bb.EMPIRICAL_SOURCE,
    )
    ml = {f.selection: f.raw_probability for f in forecasts if f.market == "moneyline"}
    anchor_free = bb.forecast(
        _game("ncaa_m"),
        # A board with no main spread at all: nothing to anchor to.
        [_quote("moneyline", None, game="g1")],
        results=results,
        reference_day=DAY,
        surface=bb.EMPIRICAL_SOURCE,
    )
    bare = {f.selection: f.raw_probability for f in anchor_free if f.market == "moneyline"}
    assert ml["home"] == pytest.approx(bare["home"], abs=1e-12)
    assert ml["home"] + ml["away"] == pytest.approx(1.0)


def test_fatigue_multiplier_only_counts_strictly_past_days():
    results = [
        Result("h1", "a1", 90, 80, "2026-11-26"),
        Result("h1", "a1", 90, 80, "2026-11-27"),
        Result("h1", "a1", 90, 80, "2026-11-28"),  # same day as reference
        Result("h1", "a1", 90, 80, "2026-11-29"),  # after the reference day
    ]
    # The 27th is a back-to-back before the 28th; the 26th is the third game in
    # four days. The same-day and later finals are invisible to both.
    assert bb.fatigue_multiplier(results, "h1", DAY) == pytest.approx(
        bb.REST_B2B_PENALTY * bb.REST_3IN4_PENALTY
    )
    # A day with no strictly-past game is a neutral multiplier.
    assert bb.fatigue_multiplier(results, "h1", "2026-11-01") == 1.0
    # Back-to-back only.
    b2b = [Result("h1", "a1", 90, 80, "2026-11-27")]
    assert bb.fatigue_multiplier(b2b, "h1", DAY) == pytest.approx(bb.REST_B2B_PENALTY)


def test_empirical_surface_is_empty_without_results_and_bpi_without_a_predictor():
    assert bb.forecast(_game(), _board("nba"), results=[], reference_day=DAY) == []
    assert (
        bb.forecast(
            _game(),
            _board("nba"),
            results=_history(),
            reference_day=DAY,
            surface=bb.BPI_SOURCE,
        )
        == []
    )


def test_bpi_surface_emits_no_total_when_the_payload_has_no_expected_points():
    from bullion.espn import PredictorEvidence

    evidence = PredictorEvidence(
        prob_home=0.62, expected_home=None, expected_away=None, stats_side="home"
    )
    forecasts = bb.forecast(
        _game("nba"),
        _board("nba"),
        results=_history(),
        reference_day=DAY,
        surface=bb.BPI_SOURCE,
        predictor=evidence,
    )
    assert forecasts, "a win probability alone is enough to price a moneyline"
    assert not [f for f in forecasts if f.market == "total"]
    assert [f for f in forecasts if f.market == "spread"]


def test_post_tip_quotes_are_refused():
    quotes = _board("nba")
    quotes[0].pre_match = False
    with pytest.raises(Exception):
        bb.forecast(_game("nba"), quotes, results=_history(), reference_day=DAY)


def test_surface_resolution_refuses_a_coin_flip_and_tiebreaks_deterministically():
    assert bb.resolve_surface({bb.BPI_SOURCE: 0.21, bb.EMPIRICAL_SOURCE: 0.211}) is None
    assert bb.resolve_surface({bb.BPI_SOURCE: 0.20, bb.EMPIRICAL_SOURCE: 0.24}) == bb.BPI_SOURCE
    assert (
        bb.resolve_surface({bb.BPI_SOURCE: 0.24, bb.EMPIRICAL_SOURCE: 0.20})
        == bb.EMPIRICAL_SOURCE
    )
    assert bb.tiebreak_surface({bb.BPI_SOURCE: 0.23, bb.EMPIRICAL_SOURCE: 0.23}) == (
        bb.EMPIRICAL_SOURCE
    )
    assert bb.tiebreak_surface({}) is None


# --------------------------------------------------------------------------
# scripts: season keys, seeding, subset merges, cold lanes
# --------------------------------------------------------------------------


@pytest.fixture()
def backfill():
    spec = importlib.util.spec_from_file_location(
        "bullion_backfill_basketball", ROOT / "scripts" / "backfill.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def probe_script():
    spec = importlib.util.spec_from_file_location(
        "bullion_probe_basketball", ROOT / "scripts" / "probe_espn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_espn_season_keys_expands_a_straddling_season_only(backfill):
    assert backfill.espn_season_keys(LEAGUES["ncaa_m"], [2026]) == [2027, 2026]
    assert backfill.espn_season_keys(LEAGUES["ncaa_w"], [2025, 2026]) == [2027, 2026, 2025]
    # The WNBA plays inside its calendar year and every other lane is untouched.
    assert backfill.espn_season_keys(LEAGUES["wnba"], [2026]) == [2026]
    assert backfill.espn_season_keys(LEAGUES["nhl"], [2026]) == [2026]


def test_select_leagues_understands_basketball_everywhere(backfill):
    assert [league.key for league in backfill.select_leagues("basketball")] == list(
        BASKETBALL_LEAGUES
    )
    from bullion.runner import select_leagues as runner_select

    assert [league.key for league in runner_select("basketball")] == list(BASKETBALL_LEAGUES)
    # A comma list (what --detect-cold emits) resolves lane by lane, once each.
    for select in (backfill.select_leagues, runner_select):
        assert [league.key for league in select("nba,ncaa_m,nba")] == ["nba", "ncaa_m"]
        with pytest.raises(ValueError):
            select("nba,not_a_lane")


def _basketball_final(store: Store, game_id: str, day: str, home: str, away: str, hs, as_):
    game = Game(
        game_id=game_id,
        league="nba",
        season="2027",
        home_team_id=home,
        away_team_id=away,
        home_name=f"Home {home}",
        away_name=f"Away {away}",
        # 18:00Z is the same calendar day in both ET and SAST, so a test that
        # labels a game by ``day`` means the same thing under
        # League.schedule_day (ET for US sports) and under SAST helpers.
        scheduled_utc=f"{day}T18:00:00Z",
        status=Status.FINAL,
        actual=None,
    )
    store.upsert_game(game)
    store.conn.execute(
        "UPDATE games SET home_score=?, away_score=? WHERE game_id=?", (hs, as_, game_id)
    )
    store.conn.commit()


def _seed_nba_history(store: Store, *, days: int = 24) -> None:
    import random

    rng = random.Random(3)
    for index in range(days):
        day = f"2026-{10 + index // 28:02d}-{1 + index % 28:02d}"
        for game_index in range(2):
            strength = 108 + (index + game_index) % 5
            _basketball_final(
                store,
                f"nba-{index}-{game_index}",
                day,
                f"h{(index + game_index) % 6}",
                f"a{(index * 2 + game_index) % 6}",
                round(strength + rng.gauss(0, 9)),
                round(strength - 2 + rng.gauss(0, 9)),
            )


def test_basketball_seeder_is_leak_free_idempotent_and_refits_weekly(tmp_path, backfill):
    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_history(store)
        first = backfill.seed_walk_forward_ml(store, leagues=("nba",))
        lane = first["lanes"]["nba"]
        assert lane["rows"] > 0 and lane["inserted"] == lane["rows"]
        assert lane["refit_cadence_days"] == bb.SEED_REFIT_EVERY_DAYS
        scored_days = lane["matchdays_scored"]
        # Weekly cadence, not per-day: at most one fit per seven scored days
        # (plus the window that opens on the first scored day).
        assert lane["refits"] <= (scored_days + bb.SEED_REFIT_EVERY_DAYS - 1) // (
            bb.SEED_REFIT_EVERY_DAYS
        )

        # Idempotent: a warm cache inserts nothing and pays for no fits.
        second = backfill.seed_walk_forward_ml(store, leagues=("nba",))
        assert second["lanes"]["nba"]["inserted"] == 0
        assert second["lanes"]["nba"]["rows"] == 0
        assert second["lanes"]["nba"]["refits"] == 0
        assert second["fit_seconds"] == 0.0


def test_basketball_seeded_probability_matches_the_frozen_fit_and_never_sees_itself(
    tmp_path, backfill, monkeypatch
):
    """Two things, both about honesty rather than accuracy.

    Leak firewall: every fit the seeder performs may only see finals whose SAST
    day is strictly before the day the fit is for. Parity: the row it persists
    is exactly what the production expectation function produces from that
    frozen fit, so the seeded probability is reproducible rather than invented.
    """
    calls = []
    real_fit = bb.fit_lane_ratings

    def spy(league, results, *, reference_day):
        calls.append({"reference_day": reference_day, "results": list(results)})
        return real_fit(league, results, reference_day=reference_day)

    monkeypatch.setattr(bb, "fit_lane_ratings", spy)

    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_history(store)
        summary = backfill.seed_walk_forward_ml(store, leagues=("nba",))
        assert summary["lanes"]["nba"]["rows"] > 0
        assert calls, "seeding must fit at least once"

        # 1. Leak firewall, checked on every fit, not just the last one.
        for call in calls:
            assert call["results"], "a fit with no history cannot seed anything"
            assert max(result.day for result in call["results"]) < call["reference_day"], (
                "a seeded fit saw a final from its own reference day or later"
            )

        # 2. Parity with the production function on the frozen fit.
        rows = store.conn.execute(
            """
            SELECT game_id, selection, raw_probability, observed_day, source
            FROM forecast_snapshots
            WHERE league='nba' AND market='moneyline'
            ORDER BY observed_day DESC, game_id, selection
            LIMIT 2
            """
        ).fetchall()
        assert len(rows) == 2
        observed_day = rows[0]["observed_day"]
        used = [call for call in calls if call["reference_day"] <= observed_day][-1]
        target = store.conn.execute(
            "SELECT * FROM games WHERE game_id=?", (rows[0]["game_id"],)
        ).fetchone()
        ratings = real_fit("nba", used["results"], reference_day=used["reference_day"])
        sigmas = bb.lane_sigmas("nba", used["results"], ratings, used["reference_day"])
        # Fatigue reads every final strictly before the matchday — including
        # those that settled inside the refit window — while ratings and sigmas
        # stay frozen at the window's start.
        fatigue_visible = [
            Result(
                home_id=row["home_team_id"],
                away_id=row["away_team_id"],
                home_score=float(row["home_score"]),
                away_score=float(row["away_score"]),
                day=LEAGUES["nba"].schedule_day(row["scheduled_utc"]),
                scheduled_utc=to_utc_z(row["scheduled_utc"]),
            )
            for row in store.conn.execute(
                "SELECT * FROM games WHERE league='nba' AND status='final' "
                "AND home_score IS NOT NULL ORDER BY scheduled_utc"
            )
            if LEAGUES["nba"].schedule_day(row["scheduled_utc"]) < observed_day
        ]
        _margin, _total, _sigma_margin, _sigma_total, prob_home = bb.empirical_expectations(
            "nba",
            used["results"],
            ratings,
            home_id=target["home_team_id"],
            away_id=target["away_team_id"],
            reference_day=used["reference_day"],
            sigmas=sigmas,
            prior_days=bb.prior_days_index(fatigue_visible),
            # Fatigue is evaluated on the matchday, not on the window's start,
            # and on the exact UTC instant so the seeder and a live tick agree.
            fatigue_day=observed_day,
            game_scheduled_utc=target["scheduled_utc"],
            prior_times=bb.prior_times_index(fatigue_visible),
        )
        seeded = {row["selection"]: row["raw_probability"] for row in rows}
        assert seeded["home"] == pytest.approx(prob_home, abs=1e-9)
        assert seeded["home"] + seeded["away"] == pytest.approx(1.0)
        assert rows[0]["source"] == bb.EMPIRICAL_SOURCE
        assert all(row["observed_day"] == observed_day for row in rows)
        # A frozen weekly fit is *not* the same number a per-day refit would
        # produce; the recorded reference day is what makes that difference
        # auditable instead of invisible.
        assert used["reference_day"] <= observed_day


def test_only_basketball_refits_on_a_cadence(tmp_path, backfill):
    with Store(tmp_path / "bullion.db") as store:
        _seed_cfb_history(store)
        summary = backfill.seed_walk_forward_ml(store, leagues=("cfb",))
        assert summary["lanes"]["cfb"]["refit_cadence_days"] == 1
        assert summary["lanes"]["cfb"]["refits"] == 0


def _seed_cfb_history(store: Store) -> None:
    import random

    rng = random.Random(5)
    for index in range(18):
        day = f"2026-{10 + index // 28:02d}-{1 + index % 28:02d}"
        game = Game(
            game_id=f"cfb-{index}",
            league="cfb",
            season="2026",
            home_team_id=f"h{index % 5}",
            away_team_id=f"a{index % 5}",
            home_name=f"Home {index % 5}",
            away_name=f"Away {index % 5}",
            scheduled_utc=f"{day}T18:00:00Z",
            status=Status.FINAL,
        )
        store.upsert_game(game)
        store.conn.execute(
            "UPDATE games SET home_score=?, away_score=? WHERE game_id=?",
            (round(28 + rng.gauss(0, 7)), round(24 + rng.gauss(0, 7)), game.game_id),
        )
    store.conn.commit()


def test_probe_subset_merges_into_the_existing_report(tmp_path, probe_script):
    path = tmp_path / "espn_probe_latest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bullion.espn_probe",
                "mode": "live",
                "captured_utc": "2026-11-20T00:00:00Z",
                "leagues": {"cfb": {"status": "ok"}, "nfl": {"status": "ok"}},
                "backfill": {"cfb": {"events": 10}},
            }
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": "2026-11-28T00:00:00Z",
        "leagues": {"nba": {"status": "warn"}},
        "backfill": {"nba": {"events": 4}},
    }
    merged = probe_script._merge_previous_report(path, report, probed=["nba"])
    assert sorted(merged["leagues"]) == ["cfb", "nba", "nfl"]
    # A legacy report has no per-lane stamp, so carried lanes inherit the
    # PREVIOUS file's timestamp — never this run's, which would date old
    # evidence as new.
    assert merged["leagues"]["cfb"] == {
        "status": "ok",
        "captured_utc": "2026-11-20T00:00:00Z",
    }
    assert merged["leagues"]["nba"] == {"status": "warn"}
    assert merged["backfill"]["cfb"] == {"events": 10}
    assert merged["probed_lanes"] == ["nba"]
    assert merged["carried_over_lanes"] == ["cfb", "nfl"]
    assert merged["carried_over_from_utc"] == "2026-11-20T00:00:00Z"
    # A probe of every lane carries nothing over and adds no stale keys.
    full = probe_script._merge_previous_report(
        path,
        {
            "captured_utc": "2026-11-29T00:00:00Z",
            "leagues": {"cfb": {"status": "ok"}, "nfl": {"status": "ok"}, "nba": {"status": "ok"}},
        },
        probed=["cfb", "nfl", "nba"],
    )
    assert "carried_over_lanes" not in full


def test_daily_subset_run_cannot_erase_other_lanes(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "bullion_daily_basketball", ROOT / "scripts" / "daily.py"
    )
    daily = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daily)
    path = tmp_path / "daily_2026-11-28.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bullion.daily_run",
                "generated_utc": "2026-11-28T06:00:00Z",
                "sast_day": "2026-11-28",
                "runs": [{"lane": "cfb", "tickets": 2}, {"lane": "nhl", "tickets": 1}],
            }
        ),
        encoding="utf-8",
    )
    merged = daily._merge_previous_run(
        path,
        {
            "schema_version": "bullion.daily_run",
            "sast_day": "2026-11-28",
            "runs": [{"lane": "nba", "tickets": 3}],
        },
    )
    assert [run["lane"] for run in merged["runs"]] == ["cfb", "nhl", "nba"]
    assert merged["carried_over_lanes"] == ["cfb", "nhl"]
    # Re-running a lane replaces it in place instead of duplicating it.
    again = daily._merge_previous_run(
        path,
        {"runs": [{"lane": "cfb", "tickets": 5}]},
    )
    assert [run["lane"] for run in again["runs"]] == ["cfb", "nhl"]
    assert again["runs"][0]["tickets"] == 5


def test_cold_lanes_reports_short_lanes_and_skips_completed_sweeps(tmp_path, backfill):
    from datetime import date, timedelta

    today = date.today()
    past_days = [
        (today - timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(1, backfill.SWEEP_COMPLETE_DAYS + 2)
    ]
    with Store(tmp_path / "bullion.db") as store:
        assert backfill.cold_lanes(store, min_finals=50) == list(LEAGUES)
        _seed_nba_history(store, days=30)  # 60 finals clears the bar
        cold = backfill.cold_lanes(store, min_finals=50)
        assert "nba" not in cold
        assert "wnba" in cold
        # The WNBA is out of season: a completed sweep with too few finals is
        # reported as sparse and excluded, so no tick re-walks 400 empty days.
        for et_date in past_days:
            store.record_scoreboard_day("wnba", et_date, 0)
        cold = backfill.cold_lanes(store, min_finals=50)
        assert "wnba" not in cold
        assert "ncaa_m" in cold


def test_evicted_warehouse_cannot_inherit_a_completed_sweep(tmp_path, backfill):
    """Resume state must never outlive the finals it describes.

    The warehouse is gitignored and restored from an Actions cache; anything
    that records "already swept" outside it (a committed JSON manifest, say)
    survives a cache eviction and then tells the backfill that 400 past days of
    an EMPTY database are done. The lane would sit at 0 finals forever. This
    test simulates exactly that: a full sweep recorded in one DB, then a fresh
    empty DB, then a backfill that must refetch.
    """
    with Store(tmp_path / "evicted.db") as stale:
        for et_date in backfill.et_days_back("20261101", 30)[1:]:
            stale.record_scoreboard_day("nba", et_date, 0)
        for lane in BASKETBALL_LEAGUES:
            stale.record_scoreboard_day(lane, "20261001", 0)
        # 30 swept days plus the extra one recorded for every lane below.
        assert stale.scoreboard_days_captured("nba", before="20261101") == 31

    with Store(tmp_path / "fresh.db") as fresh:
        # Same lane, brand-new file: nothing is captured, nothing is skipped.
        assert fresh.scoreboard_days_captured("nba", before="20261101") == 0
        assert not fresh.scoreboard_day_done("nba", "20261001")
        assert [key for key in backfill.cold_lanes(fresh, min_finals=50)] == list(LEAGUES)
        fetched = []

        def fake(url):
            fetched.append(url)
            et_date = url.split("dates=")[1][:8]
            return {
                "events": [
                    event(
                        f"nba-{et_date}-{index}",
                        state="post",
                        home_score="110",
                        away_score="104",
                        date=f"{et_date[:4]}-{et_date[4:6]}-{et_date[6:]}T23:30:00Z",
                    )
                    for index in range(2)
                ]
            }

        from bullion.espn import EspnClient

        counters = backfill.backfill_scoreboard_days(
            fresh, EspnClient(fetch=fake), LEAGUES["nba"], days_back=3, workers=1
        )
        assert counters["days"] == 4
        assert counters["skipped"] == 0, "an empty warehouse must refetch every day"
        assert counters["stored"] == 8
        assert len(fetched) == 4
        assert backfill.lane_finals_count(fresh, "nba") == 8
        # And a second pass skips exactly the past days, keeping today live.
        before = len(fetched)
        again = backfill.backfill_scoreboard_days(
            fresh, EspnClient(fetch=fake), LEAGUES["nba"], days_back=3, workers=1
        )
        assert again["skipped"] == 3
        assert len(fetched) - before == 1


def test_et_days_back_is_inclusive_newest_first(backfill):
    assert backfill.et_days_back("20260103", 2) == ["20260103", "20260102", "20260101"]


def test_lane_permission_still_fails_closed():
    from bullion.espn import (
        PROBE_MIN_COVERAGE,
        PROBE_MIN_SAMPLE,
        PROBE_SCHEMA,
        lane_permission,
    )

    def market(status, coverage=0.0, sample=0):
        return {"status": status, "event_coverage": coverage, "sample_size": sample}

    empty_window = {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": "2026-11-28T15:00:00Z",
        "leagues": {
            "wnba": {
                "coverage": {
                    "named_book_pre_tip": {
                        "moneyline": market("no_pre_match_events"),
                        "spread": market("no_pre_match_events"),
                        "total": market("no_pre_match_events"),
                    }
                }
            }
        },
    }
    # An out-of-season lane reports zero pre-match events, which is an absent
    # measurement, not a measured zero: it stays research_only with no market.
    verdict = lane_permission(empty_window, LEAGUES["wnba"])
    assert verdict.research_only is True
    assert verdict.ticket_markets == frozenset()
    assert verdict.reason == "espn_probe_no_pre_match_events"

    thin_sample = {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": "2026-11-28T15:00:00Z",
        "leagues": {
            "nba": {
                "coverage": {
                    "named_book_pre_tip": {
                        "moneyline": market("insufficient_sample", 0.9, PROBE_MIN_SAMPLE - 1),
                        "spread": market("insufficient_sample", 0.9, 1),
                        "total": market("insufficient_sample", 0.9, 1),
                    }
                }
            }
        },
    }
    assert lane_permission(thin_sample, LEAGUES["nba"]).research_only is True

    covered = {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": "2026-11-28T15:00:00Z",
        "leagues": {
            "nba": {
                "coverage": {
                    "named_book_pre_tip": {
                        "moneyline": market("ok", PROBE_MIN_COVERAGE, PROBE_MIN_SAMPLE),
                        "spread": market("ok", 0.9, 40),
                        "total": market("ok", 0.9, 40),
                    }
                }
            }
        },
    }
    allowed = lane_permission(covered, LEAGUES["nba"])
    assert allowed.research_only is False
    assert allowed.allows("moneyline") and allowed.allows("spread") and allowed.allows("total")
    assert (PROBE_MIN_SAMPLE, PROBE_MIN_COVERAGE) == (3, 0.5)
    assert CoverageGap("nba", "pre_match", "no pre-tip odds", None).field == "pre_match"


# --------------------------------------------------------------------------
# end-to-end lane run
# --------------------------------------------------------------------------


def _basketball_offers():
    return [
        offer(
            moneyline={"home": close(-150), "away": close(130)},
            spread={"home": close(-110, "-4.5"), "away": close(-110, "+4.5")},
            total={"over": close(-110, "o220.5"), "under": close(-110, "u220.5")},
        )
    ]


def _nba_scoreboard():
    return {
        "events": [
            event(
                event_id=f"nba-pre-{index}",
                home="h1",
                away="a1",
                # 19:30Z-20:30Z on the 28th: same SAST evening, before the
                # 22:00 SAST cutoff the lanes use.
                date=f"2026-11-28T{19 + index}:{'30' if index == 0 else '00'}:00Z",
                offers=_basketball_offers(),
            )
            for index in range(3)
        ]
    }


class _BasketballEspn:
    def __init__(self):
        self.calls = []

    def __call__(self, url: str):
        self.calls.append(url)
        if "/summary?event=" in url:
            from tests.factories import summary

            return summary(
                event_id="nba-past",
                home="h1",
                away="a1",
                home_score="112",
                away_score="104",
                home_lines=[{"value": 56}, {"value": 56}],
                away_lines=[{"value": 52}, {"value": 52}],
            )
        return _nba_scoreboard()


def _nba_probe_report():
    from bullion.espn import PROBE_SCHEMA

    market = {"event_coverage": 0.9, "sample_size": 24, "status": "ok"}
    return {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": "2026-11-28T06:00:00Z",
        "leagues": {
            "nba": {
                "coverage": {
                    "named_book_pre_tip": {
                        "moneyline": dict(market),
                        "spread": dict(market),
                        "total": dict(market),
                    }
                }
            }
        },
    }


def _seed_nba_calibration(store: Store, count: int = 40) -> None:
    from bullion.models import Forecast

    by_day = {}
    for index in range(count):
        day = f"2026-10-{(index % 28) + 1:02d}"
        game = Game(
            game_id=f"nba-past-{index}",
            league="nba",
            season="2027",
            home_team_id="h1",
            away_team_id="a1",
            home_name="Home h1",
            away_name="Away a1",
            scheduled_utc=f"{day}T00:30:00Z",
            status=Status.FINAL,
        )
        store.upsert_game(game)
        store.conn.execute(
            "UPDATE games SET home_score=112, away_score=104 WHERE game_id=?", (game.game_id,)
        )
        by_day.setdefault(game.game_id, [])
        by_day[game.game_id].append(
            Forecast(
                game_id=game.game_id,
                league="nba",
                scheduled_utc=game.scheduled_utc,
                home_name=game.home_name,
                away_name=game.away_name,
                market="moneyline",
                selection="home",
                raw_probability=0.62,
                source=bb.EMPIRICAL_SOURCE,
            )
        )
    store.conn.commit()
    for rows in by_day.values():
        store.record_forecasts(rows, sast_day(rows[0].scheduled_utc))


def test_nba_lane_runs_end_to_end_and_invents_no_edge_on_a_fair_board(tmp_path):
    """The whole live lane, on a synthetic board, with a granted permission.

    Three fixtures x four markets are priced by the empirical surface and land
    in the candidate pool; nothing becomes a leg, because every quote is a fair
    -110/-110 price and the model has no edge on it. That is the property worth
    pinning: the lane reaches the ticket stage and then declines, rather than
    manufacturing an edge to fill a card.
    """
    from bullion.espn import EspnClient
    from bullion.runner import run_lane

    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_calibration(store)
        run = run_lane(
            league=LEAGUES["nba"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=_BasketballEspn()),
            as_of_utc="2026-11-28T19:00:00Z",
            probe_report=_nba_probe_report(),
        )
        sources = {
            row[0]
            for row in store.conn.execute(
                "SELECT DISTINCT source FROM forecast_snapshots WHERE league='nba'"
            )
        }
    assert run.status == "frozen_no_bet", run.reason
    assert run.candidate_pool > 0, "the priced board must reach the candidate pool"
    assert run.eligible_legs == 0
    assert run.tickets == 0
    assert set(run.ticket_markets) == {"moneyline", "spread", "total"}
    # No predictor payload was served, so no BPI row may exist: a lane never
    # invents provider evidence it did not receive.
    assert sources == {bb.EMPIRICAL_SOURCE}
    assert not (tmp_path / "nba" / f"auto_tickets_{DAY}_decision.json").exists()
    assert all("identity" not in gap.field for gap in run.gaps)


def test_wnba_stays_research_only_without_a_probe(tmp_path):
    from bullion.espn import EspnClient
    from bullion.runner import run_lane

    with Store(tmp_path / "bullion.db") as store:
        run = run_lane(
            league=LEAGUES["wnba"],
            day=DAY,
            store=store,
            localdata=tmp_path,
            client=EspnClient(fetch=_BasketballEspn()),
            as_of_utc="2026-11-28T19:00:00Z",
        )
    assert run.status == "frozen_no_bet"
    assert "espn_probe_evidence_missing" in run.reason
    assert not (tmp_path / "wnba" / f"auto_tickets_{DAY}_decision.json").exists()


# --------------------------------------------------------------------------
# seeder cost + fatigue accuracy (regressions found in audit)
# --------------------------------------------------------------------------


def _spy_on_expectations(monkeypatch):
    """Capture every empirical_expectations call the seeder makes."""
    calls = []
    real = bb.empirical_expectations

    def spy(league, results, ratings, **kwargs):
        calls.append({"results": results, "ratings": ratings, **kwargs})
        return real(league, results, ratings, **kwargs)

    monkeypatch.setattr(bb, "empirical_expectations", spy)
    return calls


def _seed_sparse_then_consecutive(store: Store) -> tuple:
    """15 daily finals to clear the warm-up floor, then a long gap, then B2B.

    Lane days are ``2026-10-01 … 2026-10-16`` (16 finals), then a three-week
    gap, then ``2026-11-05`` and ``2026-11-06``. With a 7-day refit window over
    lane *positions*, the window containing both November days opened at
    position 14 (2026-10-15), so the frozen fit's reference day is three weeks
    before the November games.
    """
    import random

    rng = random.Random(9)
    days = [f"2026-10-{day:02d}" for day in range(1, 17)] + ["2026-11-05", "2026-11-06"]
    for index, day in enumerate(days):
        strength = 108 + rng.gauss(0, 4)
        _basketball_final(
            store,
            f"nba-{index}",
            day,
            "h1",
            "a1",
            round(strength + rng.gauss(0, 8)),
            round(strength - 2 + rng.gauss(0, 8)),
        )
    return days


def test_b2b_fatigue_reads_the_matchday_not_the_refit_window(tmp_path, backfill, monkeypatch):
    """Two ways a stale fatigue day goes wrong, both pinned here.

    With a weekly refit window the frozen fit's reference day can be weeks
    before the matchday. Pricing fatigue against it (a) invents a back-to-back
    for a team that has not played in a month, and (b) misses a real one the
    night before D. Fatigue must be evaluated on D, and the team index must
    include finals that settled inside the window.
    """
    calls = _spy_on_expectations(monkeypatch)
    nights = ("2026-11-05", "2026-11-06")
    with Store(tmp_path / "bullion.db") as store:
        _seed_sparse_then_mostly_rested = _seed_sparse_then_consecutive(store)
        summary = backfill.seed_walk_forward_ml(store, leagues=("nba",))
        seeded = {
            (row["observed_day"], row["selection"]): row["raw_probability"]
            for row in store.conn.execute(
                "SELECT observed_day, selection, raw_probability FROM forecast_snapshots "
                "WHERE league='nba' AND market='moneyline'"
            )
        }
    assert summary["lanes"]["nba"]["matchdays_scored"] == 3
    assert seeded, "the two November matchdays must have been seeded"

    by_day = {call["fatigue_day"]: call for call in calls}
    rested, back_to_back = by_day[nights[0]], by_day[nights[1]]

    # (a) No stale penalty: 20 days after the last game, fatigue is neutral...
    window_start = rested["reference_day"]
    assert window_start != nights[0], "the window must have opened before D"
    assert bb.fatigue_multiplier(
        rested["results"], "h1", window_start, prior_days=rested["prior_days"]
    ) < 1.0, "the window start sits right after a game, so it would have applied one"
    assert bb.fatigue_multiplier(
        rested["results"], "h1", nights[0], prior_days=rested["prior_days"]
    ) == pytest.approx(1.0)

    # (b) ...while the real back-to-back the night before D is seen, even though
    # that game settled inside the refit window and is not in the frozen fit.
    assert nights[0] in back_to_back["prior_days"]["h1"]
    assert nights[0] not in [row.day for row in back_to_back["results"]], (
        "the D-1 final belongs to the window, not to the frozen fit"
    )
    assert bb.fatigue_multiplier(
        back_to_back["results"], "h1", nights[1], prior_days=back_to_back["prior_days"]
    ) == pytest.approx(bb.REST_B2B_PENALTY)

    # Frozen ratings, frozen sigmas: one fit and one sigma scan per window.
    assert rested["ratings"] is back_to_back["ratings"]
    assert rested["sigmas"] is back_to_back["sigmas"]
    assert rested["sigmas"] is not None


def test_seeding_scans_sigmas_once_per_window_not_once_per_fixture(
    tmp_path, backfill, monkeypatch
):
    """A full NCAA backfill must stay linear, not quadratic.

    ``residual_sigmas`` is O(results); calling it per fixture is what made a
    cold NCAA seeding run take minutes. One call per refit window — and none at
    all for a warm cache — is the contract.
    """
    sigma_calls = []
    real_sigmas = bb.lane_sigmas

    def spy_sigmas(league, results, ratings, reference_day):
        sigma_calls.append((league, reference_day))
        return real_sigmas(league, results, ratings, reference_day)

    monkeypatch.setattr(bb, "lane_sigmas", spy_sigmas)
    with Store(tmp_path / "bullion.db") as store:
        _seed_nba_history(store, days=40)  # 80 finals, ~25 scored matchdays
        summary = backfill.seed_walk_forward_ml(store, leagues=("nba",))
    lane = summary["lanes"]["nba"]
    assert lane["matchdays_scored"] > 5
    assert 0 < len(sigma_calls) <= lane["refits"], (
        f"{len(sigma_calls)} sigma scans for {lane['refits']} windows / "
        f"{lane['matchdays_scored']} matchdays"
    )
    # Warm cache: no fits, no sigma scans.
    sigma_calls.clear()
    with Store(tmp_path / "bullion.db") as store:
        warm = backfill.seed_walk_forward_ml(store, leagues=("nba",))
    assert warm["lanes"]["nba"]["inserted"] == 0
    assert sigma_calls == []


def test_history_crossing_the_floor_mid_window_is_not_skipped(tmp_path, backfill):
    """The warm-up floor can be crossed on a non-boundary day.

    The refit window object used to be snapshotted at position 0 (empty, below
    MIN_SEED_HISTORY) and only refreshed every 7th position, so matchdays
    3..6 after the floor was reached were scored from an empty fit and silently
    produced zero rows.
    """
    import random

    rng = random.Random(21)
    with Store(tmp_path / "bullion.db") as store:
        days = []
        for index in range(10):  # 3 finals/day: floor (15) crossed on day 4
            day = f"2026-10-{index + 1:02d}"
            days.append(day)
            for game_index in range(3):
                _basketball_final(
                    store,
                    f"nba-{index}-{game_index}",
                    day,
                    f"h{game_index}",
                    f"a{game_index}",
                    round(105 + rng.gauss(0, 6)),
                    round(101 + rng.gauss(0, 6)),
                )
        summary = backfill.seed_walk_forward_ml(store, leagues=("nba",))
        lane = summary["lanes"]["nba"]
        scored_days = {
            row[0]
            for row in store.conn.execute(
                "SELECT DISTINCT observed_day FROM forecast_snapshots WHERE league='nba'"
            )
        }
    # 15 priors are reached after day index 4, so days 5..9 must all be scored.
    assert lane["matchdays_scored"] == 5, lane
    assert scored_days == set(days[5:]), sorted(scored_days)
    # 3 games x 2 selections per scored day.
    assert lane["rows"] == 5 * 3 * 2


def test_probe_uses_the_live_season_key(probe_script, monkeypatch):
    """A probe of the finished season is a probe of nothing.

    November 2026 must ask ESPN for season 2027 on the three straddling
    basketball lanes, and 2026 for the WNBA.
    """
    from bullion.models import espn_season_keys as model_season_keys

    asked = {}

    class _Client:
        def get(self, url):
            asked[url.split("/leagues/")[-1].split("/")[0]] = url
            return {"count": 0, "pageCount": 0, "items": []}

    for key in ("nba", "ncaa_m", "ncaa_w", "wnba"):
        league = LEAGUES[key]
        season = model_season_keys(league, [2026])[0]
        report = probe_script.probe_backfill(_Client(), league, season)
        assert report["season"] == (2027 if key != "wnba" else 2026), key
        assert str(report["season"]) in report["url"], key
        assert report["season_key_source"] == "bullion.models.espn_season_keys"


def test_fresh_probe_stamps_every_probed_lane_with_its_own_capture_time(
    tmp_path, probe_script, monkeypatch
):
    """A lane's timestamp must describe when THAT lane was measured.

    The report-level ``captured_utc`` covers only the lanes probed now; a
    reader that falls back to it after a subset merge would report stale
    coverage as fresh.
    """

    def fake_probe_league(client, league, sast_days, **kwargs):
        return {"sport": league.sport.value, "coverage": {}}

    def fake_probe_backfill(client, league, season):
        return {"season": season, "count": 0}

    monkeypatch.setattr(probe_script, "probe_league", fake_probe_league)
    monkeypatch.setattr(probe_script, "probe_backfill", fake_probe_backfill)
    monkeypatch.setattr(probe_script, "EspnClient", lambda *a, **k: object())
    out = tmp_path / "probe.json"
    code = probe_script.main(
        ["--out", str(out), "--leagues", "nba,wnba", "--date", "2026-11-28", "--days-ahead", "0"]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["probed_lanes"] == ["nba", "wnba"]
    assert sorted(report["leagues"]) == ["nba", "wnba"]
    for lane, entry in report["leagues"].items():
        assert entry["captured_utc"] == report["captured_utc"], lane
    assert report["captured_utc"].endswith("Z")


def test_probe_merge_never_redates_carried_lanes(tmp_path, probe_script):
    """A subset probe must not touch the age of the lanes it did not measure."""
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bullion.espn_probe",
                "mode": "live",
                "captured_utc": "2026-11-20T00:00:00Z",
                "leagues": {
                    "cfb": {"status": "ok", "captured_utc": "2026-11-18T09:00:00Z"},
                    "nhl": {"status": "ok"},  # legacy: no per-lane stamp
                    "nba": {"status": "ok", "captured_utc": "2026-11-20T00:00:00Z"},
                },
                "backfill": {},
            }
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": "2026-11-28T07:12:44Z",
        "leagues": {"ncaa_w": {"status": "ok", "captured_utc": "2026-11-28T07:12:44Z"}},
        "backfill": {},
    }
    merged = probe_script._merge_previous_report(path, report, probed=["ncaa_w"])
    assert merged["captured_utc"] == "2026-11-28T07:12:44Z"
    assert merged["leagues"]["ncaa_w"]["captured_utc"] == "2026-11-28T07:12:44Z"
    # A lane with its own older stamp keeps it exactly.
    assert merged["leagues"]["cfb"]["captured_utc"] == "2026-11-18T09:00:00Z"
    assert merged["leagues"]["nba"]["captured_utc"] == "2026-11-20T00:00:00Z"
    # A legacy lane is dated by the previous FILE stamp, not by this run.
    assert merged["leagues"]["nhl"]["captured_utc"] == "2026-11-20T00:00:00Z"
    assert merged["carried_over_lanes"] == ["cfb", "nba", "nhl"]
    assert merged["carried_over_from_utc"] == "2026-11-20T00:00:00Z"
    # And the carried entries are copies, not references into the old file.
    assert merged["leagues"]["cfb"] is not json.loads(path.read_text())["leagues"]["cfb"]


def test_full_probe_clears_stale_carry_over_metadata(tmp_path, probe_script):
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bullion.espn_probe",
                "mode": "live",
                "captured_utc": "2026-11-20T00:00:00Z",
                "leagues": {
                    "cfb": {"status": "ok"},
                    "nba": {"status": "ok", "captured_utc": "2026-11-27T00:00:00Z"},
                },
                "carried_over_lanes": ["nba"],
                "carried_over_from_utc": "2026-11-19T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": "2026-11-28T07:12:44Z",
        "leagues": {
            "cfb": {"status": "ok", "captured_utc": "2026-11-28T07:12:44Z"},
            "nba": {"status": "ok", "captured_utc": "2026-11-28T07:12:44Z"},
        },
    }
    merged = probe_script._merge_previous_report(path, report, probed=["cfb", "nba"])
    assert "carried_over_lanes" not in merged
    assert "carried_over_from_utc" not in merged
    assert merged["probed_lanes"] == ["cfb", "nba"]
