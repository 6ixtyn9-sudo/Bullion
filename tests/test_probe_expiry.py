"""F2 — per-league probe expiry (a lapsed probe stops granting tickets)."""

from __future__ import annotations

import pytest

from bullion.espn import PROBE_MAX_AGE_HOURS, lane_permission
from bullion.models import LEAGUES


def _market(coverage: float = 0.9, sample: int = 8) -> dict:
    return {"event_coverage": coverage, "sample_size": sample, "status": "ok"}


def _report(captured: str = "2026-11-28T06:00:00Z", lanes: dict | None = None) -> dict:
    lanes = lanes if lanes is not None else {"nhl": {"captured_utc": captured}}
    report = {
        "schema_version": "bullion.espn_probe",
        "mode": "live",
        "captured_utc": captured,
        "leagues": {},
    }
    for key, entry in lanes.items():
        report["leagues"][key] = {
            "captured_utc": entry.get("captured_utc", captured),
            "coverage": {
                "named_book_pre_tip": {
                    "moneyline": _market(),
                    "spread": _market(),
                    "total": _market(),
                }
            },
        }
    return report


def test_a_probe_older_than_the_limit_fails_closed():
    report = _report("2026-11-20T06:00:00Z")  # 8 days before the decision instant
    permission = lane_permission(
        report, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z"
    )
    assert permission.research_only is True
    assert permission.reason == "espn_probe_stale"
    assert permission.ticket_markets == frozenset()
    assert permission.captured_utc == "2026-11-20T06:00:00Z"


def test_a_probe_inside_the_window_keeps_its_permission():
    report = _report("2026-11-22T06:00:00Z")  # 6 days old
    permission = lane_permission(
        report, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z"
    )
    assert permission.research_only is False
    assert permission.reason == "probe_covered"
    assert set(permission.ticket_markets) == {"moneyline", "spread", "total"}


@pytest.mark.parametrize("captured", ["", "not-a-timestamp", "2026-12-05T00:00:00Z"])
def test_missing_unparseable_and_future_stamps_all_fail_closed(captured):
    report = _report("2026-11-28T06:00:00Z", lanes={"nhl": {"captured_utc": captured}})
    if not captured:
        # No stamp anywhere is the "missing" case; an empty per-lane stamp with a
        # fresh file stamp is a legacy report and correctly falls back to it.
        report["captured_utc"] = ""
    permission = lane_permission(
        report, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z"
    )
    assert permission.research_only is True
    assert permission.reason == "espn_probe_stale"


def test_per_league_stamps_never_mask_each_other():
    """The anti-mask rule: a fresh file stamp cannot rescue a stale lane entry."""
    report = _report(
        "2026-11-28T06:00:00Z",
        lanes={
            "nhl": {"captured_utc": "2026-11-20T06:00:00Z"},  # 8 days stale
            "nba": {"captured_utc": "2026-11-28T06:00:00Z"},  # fresh
        },
    )
    nhl = lane_permission(report, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z")
    nba = lane_permission(report, LEAGUES["nba"], as_of_utc="2026-11-28T15:00:00Z")
    assert nhl.reason == "espn_probe_stale"
    assert nhl.captured_utc == "2026-11-20T06:00:00Z"
    assert nba.reason == "probe_covered"
    assert nba.research_only is False


def test_a_legacy_report_without_per_lane_stamps_uses_the_file_stamp():
    report = _report("2026-11-28T06:00:00Z", lanes={"nhl": {}})
    del report["leagues"]["nhl"]["captured_utc"]
    permission = lane_permission(
        report, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z"
    )
    assert permission.reason == "probe_covered"
    assert permission.captured_utc == "2026-11-28T06:00:00Z"


def test_offline_callers_get_no_age_check():
    """``as_of_utc=None`` keeps committed historical fixtures usable forever."""
    ancient = _report("2026-01-01T00:00:00Z")
    assert lane_permission(ancient, LEAGUES["nhl"]).research_only is False
    # ... while supplying the instant makes the same report stale.
    stale = lane_permission(ancient, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z")
    assert stale.reason == "espn_probe_stale"


def test_the_limit_is_a_week_and_is_overridable():
    assert PROBE_MAX_AGE_HOURS == 168.0
    report = _report("2026-11-20T06:00:00Z")
    assert lane_permission(
        report, LEAGUES["nhl"], as_of_utc="2026-11-28T15:00:00Z", max_age_hours=1e6
    ).research_only is False
