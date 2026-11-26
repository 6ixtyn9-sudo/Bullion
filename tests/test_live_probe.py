"""LIVE_VERIFIED audit tests against the 2026-11-26T12:00:17Z probe capture.

These tests never invent payload data. They read:
* ``tests/fixtures/espn_probe_2026-11-26T12-00-17Z.json`` — a sanitised extract of
  a real ``probe-espn`` GitHub Actions run (counts, flags, ids, key paths);
* ``localdata/_probes/espn_probe_latest.json`` when present — the committed raw
  probe output from that same run.

What the capture proves, and what it does not, is asserted explicitly so the
next reader cannot mistake "the route resolved" for "the market is priced".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bullion.espn import (
    PROBE_SCHEMA,
    et_days_for_sast_day,
    event_id_from_ref,
    lane_permission,
)
from bullion.models import LEAGUES

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/espn_probe_2026-11-26T12-00-17Z.json"
LATEST_PROBE = ROOT / "localdata/_probes/espn_probe_latest.json"

CAPTURE_UTC = "2026-11-26T12:00:17Z"


@pytest.fixture(scope="module")
def live() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def latest() -> dict:
    if not LATEST_PROBE.exists():
        pytest.skip("committed probe output is not present on this branch")
    return json.loads(LATEST_PROBE.read_text(encoding="utf-8"))


def test_capture_is_labelled_live_and_dated(live):
    assert live["fixture_label"] == "LIVE_VERIFIED"
    assert live["schema_version"] == PROBE_SCHEMA
    assert live["mode"] == "live"
    assert live["captured_utc"] == CAPTURE_UTC
    assert live["sast_day"] == "2026-11-26"
    assert set(live["leagues"]) == {
        "nhl",
        "nfl",
        "cfb",
        "eng.1",
        "esp.1",
        "ger.1",
        "ita.1",
        "fra.1",
        "uefa.champions",
    }


def test_probed_window_was_one_day_early_which_is_now_fixed(latest):
    """The captured run queried ET D-2/D-1 and saw only completed games."""
    urls = latest["leagues"]["nhl"]["scoreboard_urls"]
    captured_et_days = sorted(
        {url.split("dates=")[1].split("&")[0] for url in urls if "dates=" in url}
    )
    assert captured_et_days == ["20261124", "20261125"], (
        "the 2026-11-26 capture must show the one-day-early ET window"
    )
    assert et_days_for_sast_day("2026-11-26") == ["20261125", "20261126"], (
        "the corrected mapping must ask for ET D-1 and D"
    )
    assert all(
        entry["pre_events"] == 0 for entry in latest["leagues"].values()
    ), "no pre-match event was capturable in that window"


@pytest.mark.parametrize(
    "lane",
    ["nhl", "nfl", "cfb", "eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "uefa.champions"],
)
def test_every_lane_measured_zero_pre_match_events(live, lane):
    entry = live["leagues"][lane]
    assert entry["pre_events"] == 0
    assert entry["events"] >= 0
    for market, coverage in entry["coverage"].items():
        assert coverage["events"] == 0
        assert coverage["event_coverage"] == 0.0


def test_no_lane_may_price_a_ticket_from_this_capture(live, latest):
    for lane in LEAGUES:
        permission = lane_permission(latest, LEAGUES[lane])
        assert permission.research_only, lane
        assert permission.ticket_markets == frozenset(), lane
        assert permission.captured_utc == CAPTURE_UTC
    # The sanitised fixture lacks the per-market status/sample fields and must
    # therefore also be refused: a number without a sample is not evidence.
    for lane in LEAGUES:
        assert lane_permission(live, LEAGUES[lane]).research_only


def test_live_summaries_carry_pickcenter_but_no_predictor(live):
    sampled = {
        lane: entry["summary"]
        for lane, entry in live["leagues"].items()
        if entry["summary"]
    }
    assert set(sampled) == {"nhl", "nfl", "cfb", "uefa.champions"}
    for lane, facts in sampled.items():
        assert facts, lane
        for fact in facts:
            assert fact["has_pickcenter"] is True, (lane, fact["game_id"])
            assert fact["has_predictor"] is False, (lane, fact["game_id"])
            assert "pickcenter" in fact["top_level_keys"]
            assert "header" in fact["top_level_keys"]
            assert "boxscore" in fact["top_level_keys"]
    assert sampled["nhl"][0]["has_injuries"] is True
    assert sampled["nfl"][0]["has_injuries"] is True


def test_live_scoreboard_odds_container_exists_but_was_empty(live):
    for lane, entry in live["leagues"].items():
        if entry["events"] == 0:
            continue
        assert entry["parse"]["quotes"] == 0, lane
        assert entry["parse"]["gap_fields"] in (
            ["pickcenter"],
            [],
        ), (lane, entry["parse"]["gap_fields"])
    assert live["leagues"]["uefa.champions"]["scoreboard_key_paths_with_odds"] == [
        "events[].competitions[].odds"
    ]


def test_live_core_event_refs_parse_including_query_suffix(live):
    for lane, backfill in live["backfill"].items():
        if not backfill["first_ref"]:
            continue
        assert backfill["first_ref"].endswith("?lang=en&region=us")
        assert backfill["ref_id_from_ref"]
        assert backfill["ref_id_from_ref"].isdigit()
        assert event_id_from_ref(backfill["first_ref"]) == backfill["ref_id_from_ref"]
        assert backfill["count"] > 0
        assert "lang=en" not in (backfill["ref_id_from_ref"] or "")


def test_backfill_season_index_counts_are_real(live):
    counts = {lane: entry["count"] for lane, entry in live["backfill"].items()}
    assert counts["eng.1"] == 374
    assert counts["esp.1"] == 381
    assert counts["ger.1"] == 297
    assert counts["fra.1"] == 288
    assert counts["nhl"] == 1478
    assert counts["nfl"] == 320
    assert counts["cfb"] == 950
    assert counts["uefa.champions"] == 189
    assert all(entry["page_count"] >= 1 for entry in live["backfill"].values())


def test_known_gaps_are_stated_in_the_fixture(live):
    gaps = " ".join(live["known_gaps"])
    for phrase in (
        "pre-tip coverage was not measured",
        "NHL puckline",
        "soccer three-way draw",
        "NFL/CFB FPI keys",
    ):
        assert phrase in gaps
