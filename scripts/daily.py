#!/usr/bin/env python3
"""Daily end-to-end runner: ingest -> forecast -> freeze -> grade -> card.

Examples
--------
    python3 scripts/daily.py --league all --date 2026-11-28
    python3 scripts/daily.py --league nhl --probe-report localdata/_probes/espn_probe_latest.json

Without a commit-ed live probe report every lane stays ``research_only``: the
runner forecasts and grades, but freezes no ticket. That is the anti-
hallucination rule, not an inconvenience.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.espn import EspnClient  # noqa: E402
from bullion.models import sast_day, now_utc  # noqa: E402
from bullion.runner import run_lane, select_leagues  # noqa: E402
from bullion.store import Store  # noqa: E402
from bullion.tickets import GatePolicy  # noqa: E402


def load_probe(path: Optional[str]) -> Optional[Dict[str, Any]]:
    candidate = Path(path) if path else ROOT / "localdata/_probes/espn_probe_latest.json"
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="all", help="lane key, 'soccer', or 'all'")
    parser.add_argument("--date", default=None, help="SAST day (YYYY-MM-DD); default today")
    parser.add_argument("--as-of", default=None, help="decision-time UTC override (testing)")
    parser.add_argument("--db", default="localdata/bullion.db")
    parser.add_argument("--localdata", default="localdata")
    parser.add_argument("--probe-report", default=None)
    parser.add_argument("--legs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--min-leg-odds", type=float, default=1.20)
    parser.add_argument("--max-leg-odds", type=float, default=3.00)
    parser.add_argument("--min-odds", type=float, default=3.00, help="combined acca band (0 disables)")
    parser.add_argument("--max-odds", type=float, default=4.50, help="combined acca band")
    parser.add_argument(
        "--no-combined-band", action="store_true", help="accept any combined price"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    day = args.date or sast_day(now_utc())
    localdata = Path(args.localdata)
    if not localdata.is_absolute():
        localdata = ROOT / localdata
    localdata.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    probe = load_probe(args.probe_report)
    policy = GatePolicy(min_leg_odds=args.min_leg_odds, max_leg_odds=args.max_leg_odds)
    runs = []
    with Store(db_path) as store:
        for league in select_leagues(args.league):
            run = run_lane(
                league=league,
                day=day,
                store=store,
                localdata=localdata,
                client=EspnClient(),
                as_of_utc=args.as_of,
                probe_report=probe,
                legs=args.legs,
                limit=args.limit,
                policy=policy,
                min_combined_odds=None if args.no_combined_band else args.min_odds,
                max_combined_odds=None if args.no_combined_band else args.max_odds,
            )
            runs.append(run)
            print(
                f"[daily] {run.lane} {run.day}: {run.status} "
                f"({run.games} games, {run.forecasts} forecasts, "
                f"{run.candidate_pool} pool, {run.eligible_legs} eligible, "
                f"{run.tickets} tickets){' — ' + run.reason if run.reason else ''}",
                file=sys.stderr,
            )
    summary = {
        "schema_version": "bullion.daily_run",
        "generated_utc": now_utc(),
        "sast_day": day,
        "probe_report_used": probe is not None,
        "runs": [run.to_dict() for run in runs],
    }
    out_dir = localdata / "_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"daily_{day}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    failures = [run for run in runs if run.status == "ingest_failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
