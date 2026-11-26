"""Per-lane isolation in the daily runner script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bullion.runner import LaneRun
from bullion.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def daily():
    spec = importlib.util.spec_from_file_location(
        "bullion_daily_script", ROOT / "scripts" / "daily.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_failing_lane_does_not_abort_the_other_lanes(daily, tmp_path, monkeypatch):
    def fake_run_lane(*, league, day, **kwargs):
        if league.key == "eng.1":
            raise RuntimeError("provider shape changed")
        return LaneRun(lane=league.key, day=day, status="pre_cutoff_scan")

    monkeypatch.setattr(daily, "run_lane", fake_run_lane)
    localdata = tmp_path / "localdata"
    code = daily.main(
        [
            "--league",
            "soccer",
            "--date",
            "2026-11-28",
            "--localdata",
            str(localdata),
            "--db",
            str(tmp_path / "db.sqlite"),
        ]
    )

    assert code == 1, "the run still fails, but only after every lane was attempted"
    summary = json.loads((localdata / "_runs" / "daily_2026-11-28.json").read_text())
    statuses = {run["lane"]: run["status"] for run in summary["runs"]}
    assert len(statuses) == 6, "all six soccer lanes ran"
    assert statuses["eng.1"] == "lane_error"
    failed = next(run for run in summary["runs"] if run["lane"] == "eng.1")
    assert "RuntimeError: provider shape changed" in failed["reason"]
    assert all(status == "pre_cutoff_scan" for lane, status in statuses.items() if lane != "eng.1")


def test_every_daily_run_rehydrates_a_warehouse_that_already_exists(daily, tmp_path, monkeypatch):
    """P1-N1: the workflow only backfills a *missing* database, so a restored-
    but-stale warehouse used to stay short of history forever. The run itself
    now heals it, and healing twice must not double anything."""
    monkeypatch.setattr(
        daily,
        "run_lane",
        lambda *, league, day, **kwargs: LaneRun(lane=league.key, day=day, status="pre_cutoff_scan"),
    )
    localdata = tmp_path / "localdata"
    lane = localdata / "nhl"
    lane.mkdir(parents=True)
    (lane / "auto_tickets_2026-11-28_decision.json").write_text(
        json.dumps(
            {
                "schema_version": "magolide.paper_ticket_receipt",
                "date": "2026-11-28",
                "league": "NHL",
                "frozen_at": "2026-11-28T15:00:00Z",
                "candidate_pool": [
                    {
                        "forecast_id": "fid-frozen",
                        "game_id": "401",
                        "league": "nhl",
                        "market": "moneyline",
                        "period": "full_game",
                        "source": "nhl_poisson_v1",
                        "selection": "home",
                        "line": None,
                        "raw_probability": 0.62,
                        "push_probability": 0.0,
                    }
                ],
            }
        )
    )
    db = tmp_path / "warm.sqlite"
    with Store(db):
        pass  # the warehouse exists, so the workflow skips its backfill step
    argv = [
        "--league", "nhl",
        "--date", "2026-11-28",
        "--localdata", str(localdata),
        "--db", str(db),
    ]

    assert daily.main(argv) == 0
    summary = json.loads((localdata / "_runs" / "daily_2026-11-28.json").read_text())
    assert summary["rehydrated"]["restored"] == 1
    with Store(db) as store:
        assert store.counts()["forecast_snapshots"] == 1

    assert daily.main(argv) == 0
    summary = json.loads((localdata / "_runs" / "daily_2026-11-28.json").read_text())
    assert summary["rehydrated"]["restored"] == 0
    with Store(db) as store:
        assert store.counts()["forecast_snapshots"] == 1, "a second run adds nothing"
