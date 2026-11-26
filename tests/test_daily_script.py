"""Per-lane isolation in the daily runner script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bullion.runner import LaneRun

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
