"""The engine diagnostic must be read-only, and must actually measure things.

Two properties matter. It has to leave the warehouse byte-identical (a
diagnostic that mutates the evidence it is diagnosing is worthless), and it has
to emit the memo sections the review asked for — baseline decomposition,
chronological ablation and the cross-lane table — from data alone.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from bullion.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def diagnose():
    spec = importlib.util.spec_from_file_location(
        "bullion_diagnose_engines", ROOT / "scripts" / "diagnose_engines.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve ``cls.__module__`` through ``sys.modules``
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _periods(home: int, away: int, overtime: int) -> list:
    """Three regulation periods summing to the non-overtime score."""
    if overtime:
        home -= 1 if home > away else 0
        away -= 1 if away > home else 0
    first = [home // 3, away // 3]
    second = [home // 3, away // 3]
    third = [home - first[0] - second[0], away - first[1] - second[1]]
    return [first, second, third]


def _warehouse(path: Path) -> Path:
    """A small synthetic warehouse: 12 NHL matchdays and 8 cfb matchdays."""
    with Store(path):
        pass  # production schema, created read-write, then left alone
    conn = sqlite3.connect(path)
    rows = []
    index = 0
    for day_no in range(12):
        day = f"2026-01-{day_no + 1:02d}"
        for slot in range(4):
            home, away = 3 + index % 5, 1 + index % 4
            overtime = 1 if index % 7 == 0 else 0
            if overtime:
                home += 1
            rows.append(
                (
                    f"nhl-{index:03d}", "nhl", "2026", str(index % 9), str((index + 4) % 9),
                    f"H{index % 9}", f"A{(index + 4) % 9}", f"{day}T{18 + slot}:00:00Z",
                    "final", 0, "", float(home), float(away), overtime,
                    json.dumps(_periods(home, away, overtime)), "2026-02-01T00:00:00Z",
                )
            )
            index += 1
    index = 0
    for day_no in range(8):
        day = f"2026-01-{day_no + 1:02d}"
        for slot in range(3):
            rows.append(
                (
                    f"cfb-{index:03d}", "cfb", "2026", str(index % 7), str((index + 3) % 7),
                    f"CH{index % 7}", f"CA{(index + 3) % 7}", f"{day}T{16 + slot}:00:00Z",
                    "final", 0, "", float(24 + index % 11), float(20 + index % 9), 0,
                    "[]", "2026-02-01T00:00:00Z",
                )
            )
            index += 1
    conn.executemany(
        "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_measures_the_baseline_ablation_and_cross_lane_sections(
    diagnose, tmp_path, capsys
):
    db = _warehouse(tmp_path / "w.sqlite")
    code = diagnose.main(
        ["--db", str(db), "--lanes", "nhl", "cfb", "--fast", "--max-days", "12"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "# Engine diagnostic" in out
    assert "## 1. Baseline `nhl_poisson_v1`" in out
    assert "| production `home_factor` |" in out, "the home-factor defect must be reported"
    assert "## 2. Chronological ablation" in out
    assert "| A_home_only |" in out and "| B_ipf_prior_home |" in out
    assert "| D_ipf_regulation_b2b |" in out
    assert "## 3. Cross-lane side effects" in out
    assert "| cfb |" in out
    assert "opened `mode=ro`" in out


def test_diagnostic_json_memo_is_machine_readable(diagnose, tmp_path, capsys):
    db = _warehouse(tmp_path / "w.sqlite")
    assert diagnose.main(["--db", str(db), "--lanes", "nhl", "--fast", "--json"]) == 0
    memo = json.loads(capsys.readouterr().out)
    assert memo["schema"] == "bullion.engine_diagnostic"
    nhl = memo["lanes"]["nhl"]
    assert nhl["baseline"]["rows"] > 0
    assert nhl["baseline"]["production_home_factor"] is not None
    assert nhl["ablation"]["variants"]["production"]["holdout"]["brier"] is not None


def test_diagnostic_leaves_the_warehouse_byte_identical(diagnose, tmp_path, capsys):
    db = _warehouse(tmp_path / "w.sqlite")
    before = _digest(db)
    assert diagnose.main(["--db", str(db), "--lanes", "nhl", "--fast"]) == 0
    capsys.readouterr()
    assert _digest(db) == before, "the diagnostic must never write to the warehouse"
    assert not (tmp_path / "w.sqlite-wal").exists()
    assert not (tmp_path / "w.sqlite-shm").exists()


def test_read_only_connection_refuses_a_write(diagnose, tmp_path):
    db = _warehouse(tmp_path / "w.sqlite")
    conn = diagnose.open_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM games")
    finally:
        conn.close()


def test_missing_warehouse_is_a_clean_refusal(diagnose, tmp_path, capsys):
    assert diagnose.main(["--db", str(tmp_path / "absent.sqlite")]) == 2
    assert "nothing to diagnose" in capsys.readouterr().err
