"""LIVE_VERIFIED 2026-11-26T12:13:48Z: what the second probe proved, and what it did not.

The capture is a facts file: per-lane pre-match counts, the book labels ESPN
returned, the odds key paths it recorded, and the predictor key names per sport.
Nothing here invents a price. Where a test needs a payload it builds one *from
the recorded paths* and labels it SHAPE_VERIFIED, so "we know where the draw
price lives" never gets confused with "we know the draw price".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bullion.espn import (
    canonical_book,
    lane_permission,
    parse_pickcenter,
    parse_predictor,
    parse_scoreboard,
    quote_market_presence,
)
from bullion.models import LEAGUES, Game, Status

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/espn_live_2026-11-26T12-13-48Z.json"
CAPTURE_UTC = "2026-11-26T12:13:48Z"


@pytest.fixture(scope="module")
def live() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _game(league: str = "eng.1") -> Game:
    return Game(
        game_id="401879225",
        league=league,
        season="2026",
        home_team_id="393",
        away_team_id="363",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-27T20:00:00Z",
        status=Status.SCHEDULED,
    )


# ---------------------------------------------------------------------------
# Facts the capture establishes
# ---------------------------------------------------------------------------


def test_capture_covers_a_pre_match_window(live):
    assert live["captured_utc"] == CAPTURE_UTC
    assert live["sast_days"] == ["2026-11-26", "2026-11-27", "2026-11-28"]
    pre = {lane: entry["pre_events"] for lane, entry in live["leagues"].items()}
    assert pre["cfb"] == 67 and pre["nfl"] == 4
    assert pre["eng.1"] == 4 and pre["esp.1"] == 5 and pre["ita.1"] == 4
    assert pre["ger.1"] == 6 and pre["fra.1"] == 6
    assert pre["nhl"] == 28
    assert pre["uefa.champions"] == 0, "UCL was between matchdays"


def test_soccer_lanes_are_priced_by_named_books(live):
    for lane, expected in (
        ("eng.1", 1.0),
        ("esp.1", 1.0),
        ("fra.1", 1.0),
        ("ita.1", 1.0),
        ("ger.1", 0.8333),
    ):
        coverage = live["leagues"][lane]["coverage"]
        assert coverage["moneyline"]["event_coverage"] == expected, lane
        assert coverage["moneyline"]["status"] == "ok", lane
        assert coverage["moneyline"]["sample_size"] == live["leagues"][lane]["pre_events"]
        assert coverage["total"]["event_coverage"] == expected, lane
        assert coverage["spread"]["event_coverage"] == expected, lane


def test_books_arrive_with_inconsistent_spacing(live):
    """The same book must not count twice because ESPN pads its name."""
    assert live["leagues"]["nfl"]["books_raw"] == ["Draft Kings"]
    assert live["leagues"]["cfb"]["books_raw"] == ["Draft Kings"]
    assert live["leagues"]["eng.1"]["books_raw"] == ["DraftKings"]
    assert live["leagues"]["ger.1"]["books_raw"] == ["Bet 365", "DraftKings"]
    assert canonical_book("Draft Kings") == "DraftKings"
    assert canonical_book("Bet 365") == "Bet365"
    assert canonical_book("DraftKings") == "DraftKings"
    assert canonical_book("SomeNewBook") == "SomeNewBook", "unknown labels pass through"
    assert canonical_book("") == ""


def test_draw_prices_exist_on_soccer_and_the_probe_used_to_miss_them(live):
    draw_paths = live["leagues"]["eng.1"]["draw_paths"]
    assert draw_paths["moneyline.draw.close.odds"] is True
    assert draw_paths["moneyline.draw.open.odds"] is True
    assert draw_paths["drawOdds.moneyLine"] is True
    assert live["leagues"]["eng.1"]["coverage_denominator"] == "summary_pickcenter_pre_match_events"
    # The capture itself still reports 0.0 because that probe counted market
    # labels, not priced sides — the defect this commit fixes.
    assert live["leagues"]["eng.1"]["coverage"]["moneyline_draw"]["event_coverage"] == 0.0


def test_nfl_predictor_failed_only_on_the_tie_mass(live):
    predictor = live["leagues"]["nfl"]["predictor"]
    assert predictor["attempts"] == 3
    assert predictor["errors"][0]["error"] == (
        "event 401873081: predictor probabilities sum to 99.6480"
    )
    stats = predictor["samples"][0]["stats"]["homeTeam"]
    assert "teamChanceTie" in stats
    assert "teamPredPtDiff" in stats
    assert "teamExpectedPts" not in stats, "expected-score keys were never live-confirmed"


def test_cfb_predictor_parsed_without_a_tie(live):
    predictor = live["leagues"]["cfb"]["predictor"]
    assert predictor["errors"] == []
    assert predictor["samples"][0]["prob_home"] == pytest.approx(0.8228)
    stats = predictor["samples"][0]["stats"]["homeTeam"]
    assert "teamChanceTie" not in stats
    assert "teamPctHfa" in stats


def test_nhl_had_no_offers_and_no_predictor_route(live):
    nhl = live["leagues"]["nhl"]
    assert nhl["odds_offers"] == 0
    assert nhl["coverage"]["moneyline"]["event_coverage"] == 0.0
    assert nhl["summary_priced"][0]["pickcenter_entries"] == 0
    error = nhl["predictor"]["errors"][0]["error"]
    assert "/predictor" in error
    assert "HTTP Error 400" in error, "the route exists for football, not for hockey"


def test_ucl_is_unmeasured_not_zero(live):
    coverage = live["leagues"]["uefa.champions"]["coverage"]
    assert coverage["moneyline"]["status"] == "no_pre_match_events"
    assert coverage["moneyline"]["event_coverage"] is None


# ---------------------------------------------------------------------------
# SHAPE_VERIFIED: extraction built from the recorded paths
# ---------------------------------------------------------------------------


def _soccer_offer() -> dict:
    """Offer shaped exactly like the recorded ``offer.*`` paths (SHAPE_VERIFIED)."""
    return {
        "provider": {"id": "2", "name": "DraftKings", "displayName": "DraftKings"},
        "details": "DraftKings",
        "overUnder": 2.5,
        "drawOdds": {"moneyLine": 255, "link": {"rel": ["draw"]}},
        "moneyline": {
            "displayName": "Moneyline",
            "shortDisplayName": "ML",
            "home": {"close": {"odds": -125}, "open": {"odds": -120}},
            "away": {"close": {"odds": 320}, "open": {"odds": 300}},
            "draw": {"close": {"odds": 255}, "open": {"odds": 250}},
        },
        "pointSpread": {
            "displayName": "Spread",
            "home": {"close": {"odds": -110, "line": "-0.5"}},
            "away": {"close": {"odds": -110, "line": "+0.5"}},
        },
        "total": {
            "displayName": "Total",
            # Soccer totals use overUnder as the line; sides carry the prices.
            "over": {"close": {"odds": -115, "line": "o2.5"}},
            "under": {"close": {"odds": -105, "line": "u2.5"}},
        },
    }


def test_scoreboard_soccer_moneyline_now_carries_the_draw_price():
    payload = {
        "events": [
            {
                "id": "401879225",
                "date": "2026-11-27T20:00:00Z",
                "competitions": [
                    {
                        "date": "2026-11-27T20:00:00Z",
                        "status": {"type": {"state": "pre"}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "id": "393",
                                "team": {"id": "393", "displayName": "Home"},
                            },
                            {
                                "homeAway": "away",
                                "id": "363",
                                "team": {"id": "363", "displayName": "Away"},
                            },
                        ],
                        "odds": [_soccer_offer()],
                    }
                ],
            }
        ]
    }
    parsed = parse_scoreboard(payload, LEAGUES["eng.1"], CAPTURE_UTC)
    moneyline = next(quote for quote in parsed.quotes if quote.market == "moneyline")
    assert set(moneyline.side_prices) == {"home", "away", "draw"}
    assert moneyline.side_prices["draw"] == pytest.approx(3.55)
    assert moneyline.book == "DraftKings"
    assert "moneyline_draw" in quote_market_presence([moneyline])
    assert not [
        gap for gap in parsed.gaps if gap.field == "pickcenter.moneyline.draw"
    ], "the three-way gap must disappear once a draw price is extracted"


def test_draw_also_extracts_from_the_flat_summary_drawOdds_path():
    """Some summary containers expose only ``drawOdds.moneyLine`` (SHAPE_VERIFIED)."""
    payload = {
        "header": {"competitions": [{"status": {"type": {"state": "pre"}}}]},
        "pickcenter": [
            {
                "provider": {"displayName": "DraftKings"},
                "homeTeamOdds": {"moneyLine": -140},
                "awayTeamOdds": {"moneyLine": 120},
                "drawOdds": {"moneyLine": 260},
            }
        ],
    }
    parsed = parse_pickcenter(payload, _game(), CAPTURE_UTC)
    moneyline = next(quote for quote in parsed.quotes if quote.market == "moneyline")
    assert moneyline.side_prices["draw"] == pytest.approx(3.6)
    assert parsed.gaps == []


def test_two_way_quote_is_not_counted_as_three_way():
    payload = {
        "header": {"competitions": [{"status": {"type": {"state": "pre"}}}]},
        "pickcenter": [
            {
                "provider": {"displayName": "Draft Kings"},
                "moneyline": {"home": {"close": {"odds": -110}}, "away": {"close": {"odds": -110}}},
            }
        ],
    }
    parsed = parse_pickcenter(payload, _game("nfl"), CAPTURE_UTC)
    moneyline = next(quote for quote in parsed.quotes if quote.market == "moneyline")
    assert moneyline.book == "DraftKings"
    presence = quote_market_presence([moneyline])
    assert "moneyline" in presence and "moneyline_draw" not in presence


# ---------------------------------------------------------------------------
# NFL tie handling
# ---------------------------------------------------------------------------


def _nfl_predictor(
    home: float,
    away: float,
    tie: float,
    *,
    home_margin: float = 2.4,
    away_margin: float = -2.4,
) -> dict:
    """SHAPE_VERIFIED NFL predictor: the keys the live payload actually carried."""

    def side(projection: float, loss: float, margin: float) -> dict:
        return {
            "statistics": [
                {"name": "gameProjection", "value": projection},
                {"name": "teamChanceLoss", "value": loss},
                {"name": "teamChanceTie", "value": tie},
                {"name": "teamPredPtDiff", "value": margin},
            ]
        }

    return {
        "homeTeam": {**side(home, away, home_margin), "team": {"id": "h1"}},
        "awayTeam": {**side(away, home, away_margin), "team": {"id": "a1"}},
    }


def test_nfl_tie_mass_is_accepted_and_conditioned_out():
    payload = _nfl_predictor(61.93, 37.72, 0.35)
    game = Game(
        game_id="401873081",
        league="nfl",
        season="2026",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-27T18:00:00Z",
        status=Status.SCHEDULED,
    )
    evidence = parse_predictor(payload, game)
    assert evidence.prob_tie == pytest.approx(0.0035)
    assert evidence.tie_rule == "push_conditional"
    conditional_home = 0.6193 / (0.6193 + 0.3772)
    assert evidence.prob_home == pytest.approx(conditional_home)
    assert evidence.prob_home > 0.6193, (
        "tie mass is removed from the opponent side, so the conditional home "
        "probability rises slightly above the raw gameProjection"
    )
    assert evidence.prob_home + (1.0 - evidence.prob_home) == pytest.approx(1.0)
    assert evidence.predicted_margin == pytest.approx(2.4)


def test_predictor_still_rejects_a_sum_that_ignores_real_mass():
    payload = _nfl_predictor(61.0, 37.0, 0.0)  # 98.0
    game = Game(
        game_id="401873081",
        league="nfl",
        season="2026",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-27T18:00:00Z",
        status=Status.SCHEDULED,
    )
    with pytest.raises(Exception) as excinfo:
        parse_predictor(payload, game)
    assert "sum to 98.0000" in str(excinfo.value)


def test_contradictory_predicted_margin_is_dropped_not_guessed():
    payload = _nfl_predictor(61.93, 37.72, 0.35, home_margin=2.4, away_margin=-5.0)
    game = Game(
        game_id="401873081",
        league="nfl",
        season="2026",
        home_team_id="h1",
        away_team_id="a1",
        home_name="Home",
        away_name="Away",
        scheduled_utc="2026-11-27T18:00:00Z",
        status=Status.SCHEDULED,
    )
    assert parse_predictor(payload, game).predicted_margin is None


# ---------------------------------------------------------------------------
# Permission consequences
# ---------------------------------------------------------------------------


def _report(**coverage) -> dict:
    """A probe report whose soccer draw coverage is measured, not label-counted."""
    base = {"moneyline": 1.0, "moneyline_draw": 1.0, "spread": 1.0, "total": 1.0}
    base.update(coverage)
    markets = {
        market: {
            "event_coverage": value,
            "sample_size": 6,
            "status": "ok" if value is not None else "no_pre_match_events",
        }
        for market, value in base.items()
    }
    nhl = {
        market: {"event_coverage": 0.0, "sample_size": 8, "status": "ok"}
        for market in ("moneyline", "moneyline_draw", "spread", "total")
    }
    return {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": CAPTURE_UTC,
        "leagues": {
            "eng.1": {"coverage": {"named_book_pre_tip": markets}},
            "nhl": {"coverage": {"named_book_pre_tip": nhl}},
        },
    }


def test_soccer_tickets_open_only_once_the_draw_is_measured():
    report = _report()
    permission = lane_permission(report, LEAGUES["eng.1"])
    assert not permission.research_only
    assert permission.allows("moneyline") and permission.allows("total")
    assert permission.allows("spread")

    unmeasured_draw = _report(moneyline_draw=0.0)
    permission = lane_permission(unmeasured_draw, LEAGUES["eng.1"])
    assert not permission.allows("moneyline"), "a 1X2 ticket without a draw price is blind"
    assert permission.allows("total"), "totals do not depend on the draw"


def test_nhl_stays_research_only_on_this_capture():
    permission = lane_permission(_report(), LEAGUES["nhl"])
    assert permission.research_only
    assert permission.reason == "named_book_coverage_below_50pct"
