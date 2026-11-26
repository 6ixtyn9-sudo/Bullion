"""Probe script smoke tests on SYNTHETIC_FIXTURE payloads (no network).

The probe is the only live-evidence path, so a runtime error inside it would
silently produce no report at all — the worst failure mode in this repo. These
tests drive ``probe_league``/``probe_backfill`` end to end against a fake fetch
and assert the report has the shape ``lane_permission`` requires.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bullion.espn import PROBE_MIN_SAMPLE, lane_permission
from bullion.models import LEAGUES

from .factories import close, event, offer, summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "bullion_probe_espn", ROOT / "scripts" / "probe_espn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pre_offers() -> list:
    return [
        offer(
            moneyline={"home": close(-140), "away": close(120)},
            spread={"home": close(-110, "-1.5"), "away": close(-110, "+1.5")},
            total={"over": close(-110, "o6.5"), "under": close(-110, "u6.5")},
        )
    ]


def _live_scoreboard(day: str) -> dict:
    """SYNTHETIC_FIXTURE shaped like a real pre-match scoreboard day."""
    events = [
        event(
            event_id=f"pre{index}",
            home="h1",
            away="a1",
            date=f"{day}T{18 + index}:00:00Z",
            offers=_pre_offers(),
        )
        for index in range(4)
    ]
    events.append(
        event(
            event_id="done1",
            home="h1",
            away="a1",
            date=f"{day}T06:00:00Z",
            state="post",
            home_score="3",
            away_score="2",
            offers=[],
        )
    )
    return {"events": events}


def _predictor_payload() -> dict:
    """SYNTHETIC_FIXTURE: dual-sided predictor with team ids and win chances."""
    return {
        "homeTeam": {
            "team": {"id": "h1"},
            "statistics": [
                {"name": "gameProjection", "value": 62.0},
                {"name": "teamChanceLoss", "value": 38.0},
                {"name": "teamPredictedPointDifferential", "value": 2.4},
            ],
        },
        "awayTeam": {
            "team": {"id": "a1"},
            "statistics": [
                {"name": "gameProjection", "value": 38.0},
                {"name": "teamChanceLoss", "value": 62.0},
                {"name": "teamPredictedPointDifferential", "value": -2.4},
            ],
        },
    }


class FakeProbeClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str) -> dict:
        self.urls.append(url)
        if "/events?dates=" in url:  # core season index
            return {
                "count": 12,
                "pageCount": 1,
                "pageIndex": 1,
                "pageSize": 1000,
                "items": [
                    {"$ref": "http://sports.core.api.espn.com/v2/sports/hockey/leagues/"
                             "nhl/events/401856879?lang=en&region=us"},
                ],
            }
        if "/predictor" in url:
            return _predictor_payload()
        if "/summary" in url:
            event_id = url.split("event=")[1].split("&")[0]
            if event_id == "done1":
                return summary(
                    event_id="done1", home="h1", away="a1", home_score="3", away_score="2"
                )
            payload = summary(event_id=event_id, home="h1", away="a1")
            payload["pickcenter"] = [
                {
                    "provider": {"displayName": "DraftKings"},
                    "moneyline": {"home": close(-140), "away": close(120)},
                    "total": {
                        "over": close(-110, "o6.5"),
                        "under": close(-110, "u6.5"),
                    },
                }
            ]
            return payload
        return _live_scoreboard("2026-12-02")


def test_probe_report_unlocks_a_lane_only_when_it_measured_pre_match_coverage(probe):
    client = FakeProbeClient()
    entry = probe.probe_league(
        client, LEAGUES["nhl"], ["2026-12-02"], summary_samples=4, predictor_samples=2
    )
    report = {
        "schema_version": probe.PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": entry["summary"]["field_presence"] and "2026-12-02T12:00:00Z",
        "leagues": {"nhl": entry},
    }
    json.dumps(report)  # the probe output must always be serialisable

    assert entry["events"] == 5
    assert entry["pre_events"] == 4
    coverage = entry["coverage"]["named_book_pre_tip"]
    assert coverage["moneyline"]["status"] == "ok"
    assert coverage["moneyline"]["sample_size"] == 4 >= PROBE_MIN_SAMPLE
    assert coverage["moneyline"]["event_coverage"] == 1.0
    assert entry["coverage"]["named_book_pre_tip_scoreboard"]["moneyline"]["events"] == 4
    assert entry["summary"]["field_presence"], "summaries must be sampled"
    assert entry["predictor"]["attempts"] >= 1
    sample = entry["predictor"]["samples"][0]
    assert sample["sample_context"] == "pre"
    assert sample["prob_home"] == pytest.approx(0.62)
    assert "teamChanceLoss" in sample["sides"]["homeTeam"]["statistics"]
    assert sample["paths"], "the probe must record the predictor key paths"

    permission = lane_permission(report, LEAGUES["nhl"])
    assert not permission.research_only
    assert permission.allows("moneyline")
    assert permission.allows("total")


def test_probe_reports_an_empty_window_as_unmeasured_not_as_zero_coverage(probe):
    class EmptyClient(FakeProbeClient):
        def get(self, url: str) -> dict:
            if "/events?dates=" in url:
                return {"count": 0, "pageCount": 0, "items": []}
            if "/summary" in url or "/predictor" in url:
                raise AssertionError("no events to sample")
            return {"events": []}

    entry = probe.probe_league(EmptyClient(), LEAGUES["eng.1"], ["2026-12-02"])
    coverage = entry["coverage"]["named_book_pre_tip"]
    assert coverage["moneyline"]["status"] == "no_pre_match_events"
    assert coverage["moneyline"]["event_coverage"] is None
    report = {
        "schema_version": probe.PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": "2026-12-02T12:00:00Z",
        "leagues": {"eng.1": entry},
    }
    permission = lane_permission(report, LEAGUES["eng.1"])
    assert permission.research_only
    assert permission.reason == "espn_probe_no_pre_match_events"


def test_backfill_probe_parses_real_style_refs(probe):
    report = probe.probe_backfill(FakeProbeClient(), LEAGUES["nhl"], 2026)
    assert report["count"] == 12
    assert report["refs_parse_ok"] is True
    assert report["ref_ids_parsed"] == ["401856879"]
    assert "lang=en" not in report["ref_ids_parsed"][0]


def test_probe_records_payload_shape_even_when_the_parser_rejects_it(probe):
    """A rejected predictor must still yield the evidence needed to fix it."""

    class OddPredictorClient(FakeProbeClient):
        def get(self, url: str) -> dict:
            if "/predictor" in url:
                return {
                    "homeTeam": {
                        "team": {"id": "h1"},
                        "statistics": [{"name": "winProbability", "value": 0.61}],
                    }
                }
            return super().get(url)

    entry = probe.probe_league(
        OddPredictorClient(), LEAGUES["nhl"], ["2026-12-02"], summary_samples=1,
        predictor_samples=1,
    )
    sample = entry["predictor"]["samples"][0]
    assert "parse_error" in sample
    assert sample["sides"]["homeTeam"]["statistics"] == ["winProbability"]
    assert sample["payload_keys"]
    assert entry["predictor"]["errors"][0]["sides"]["homeTeam"]["team_id"] == "h1"


def test_probe_window_defaults_to_the_sast_calendar_day():
    from bullion.espn import et_days_for_sast_day

    assert et_days_for_sast_day("2026-12-02") == ["20261201", "20261202"]
