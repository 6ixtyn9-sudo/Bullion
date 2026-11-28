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
sys.path.insert(0, str(ROOT / "scripts"))

from bullion.espn import EspnClient  # noqa: E402
from bullion.models import sast_day, now_utc  # noqa: E402
from bullion.runner import LaneRun, run_lane, select_leagues  # noqa: E402
from bullion.store import Store  # noqa: E402
from bullion.tickets import (  # noqa: E402
    DEFAULT_ACCA_LIMIT,
    DEFAULT_MAX_COMBINED_ODDS,
    DEFAULT_MAX_LEGS,
    DEFAULT_MIN_COMBINED_ODDS,
    DEFAULT_MIN_LEGS,
    EDGE_FLAG_THRESHOLD,
    MAX_CLAIMED_EDGE,
    MAX_LEG_ODDS,
    MIN_LEG_ODDS,
    RANKING_METRIC_KELLY,
    VALID_RANKING_METRICS,
    GatePolicy,
)

from backfill import rehydrate_forecasts, seed_walk_forward_ml  # noqa: E402  (same directory)


def load_probe(path: Optional[str]) -> Optional[Dict[str, Any]]:
    candidate = Path(path) if path else ROOT / "localdata/_probes/espn_probe_latest.json"
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _merge_previous_run(path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Keep lanes this invocation did not run in ``daily_{day}.json``.

    Two ticks a day with different ``--league`` values (the basketball lanes run
    on a different cadence) must both survive: overwriting the file would erase
    the other lanes' tickets for that day, and the day receipt is the only
    record of what was published. Lanes run now replace their own entry in
    place; untouched lanes keep their previous entry and position.
    """
    if not path.exists():
        return summary
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return summary
    if not isinstance(previous, dict):
        return summary
    previous_runs = [
        run for run in (previous.get("runs") or []) if isinstance(run, dict) and run.get("lane")
    ]
    if not previous_runs:
        return summary
    fresh = {run["lane"]: run for run in summary.get("runs") or [] if run.get("lane")}
    merged_runs = [fresh.get(run["lane"], run) for run in previous_runs]
    seen = {run["lane"] for run in merged_runs}
    merged_runs.extend(run for run in summary.get("runs") or [] if run.get("lane") not in seen)
    carried = sorted({run["lane"] for run in previous_runs} - set(fresh))
    summary["runs"] = merged_runs
    if carried:
        summary["carried_over_lanes"] = carried
        summary["carried_over_generated_utc"] = previous.get("generated_utc")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league",
        default="all",
        help="lane key (or comma list), 'soccer', 'basketball', or 'all'",
    )
    parser.add_argument("--date", default=None, help="SAST day (YYYY-MM-DD); default today")
    parser.add_argument("--as-of", default=None, help="decision-time UTC override (testing)")
    parser.add_argument("--db", default="localdata/bullion.db")
    parser.add_argument("--localdata", default="localdata")
    parser.add_argument("--probe-report", default=None)
    parser.add_argument(
        "--legs",
        type=int,
        default=None,
        help="pin both ends of the leg range (legacy override); empty = use --min-legs/--max-legs",
    )
    parser.add_argument("--min-legs", type=int, default=DEFAULT_MIN_LEGS)
    parser.add_argument("--max-legs", type=int, default=DEFAULT_MAX_LEGS)
    parser.add_argument("--limit", type=int, default=DEFAULT_ACCA_LIMIT)
    parser.add_argument("--min-leg-odds", type=float, default=MIN_LEG_ODDS)
    parser.add_argument("--max-leg-odds", type=float, default=MAX_LEG_ODDS)
    parser.add_argument(
        "--min-odds", type=float, default=DEFAULT_MIN_COMBINED_ODDS, help="combined acca band"
    )
    parser.add_argument(
        "--max-odds", type=float, default=DEFAULT_MAX_COMBINED_ODDS, help="combined acca band"
    )
    parser.add_argument(
        "--ranking-metric",
        default=RANKING_METRIC_KELLY,
        choices=list(VALID_RANKING_METRICS),
        help="acca ranking objective",
    )
    parser.add_argument(
        "--max-claimed-edge",
        type=float,
        default=MAX_CLAIMED_EDGE,
        help="hard sanity cap on p*O-1 (0 or negative disables)",
    )
    parser.add_argument(
        "--edge-flag-threshold",
        type=float,
        default=EDGE_FLAG_THRESHOLD,
        help="non-blocking audit tag above this claimed edge (0 or negative disables)",
    )
    parser.add_argument(
        "--forbid-alt-line-reuse",
        dest="forbid_alt_line_reuse",
        action="store_true",
        default=True,
        help="no (game, market, selection) reuse across accas at another line",
    )
    parser.add_argument(
        "--allow-alt-line-reuse",
        dest="forbid_alt_line_reuse",
        action="store_false",
        help="allow the same side at a different line in a second acca",
    )
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
    policy = GatePolicy(
        min_leg_odds=args.min_leg_odds,
        max_leg_odds=args.max_leg_odds,
        max_claimed_edge=args.max_claimed_edge if args.max_claimed_edge > 0 else None,
        edge_flag_threshold=(
            args.edge_flag_threshold if args.edge_flag_threshold > 0 else None
        ),
    )
    runs = []
    with Store(db_path) as store:
        # Repairs a partially lost cache before anything reads it: the workflow
        # only runs the backfill when the warehouse is missing entirely, so a
        # restored-but-stale warehouse would otherwise stay short of history
        # forever. Idempotent (`INSERT OR IGNORE`) and cheap.
        rehydrated = rehydrate_forecasts(store, localdata)
        # Walk-forward moneyline seeding, strictly AFTER the receipt restore so
        # live provenance always claims its wager id first. On a warm cache the
        # per-game_id existence check makes this a no-op costing a handful of
        # indexed queries and zero engine fits; a seeder failure must not cost
        # the run its lanes, so it is isolated like a lane error.
        try:
            seeded = seed_walk_forward_ml(store)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            seeded = {"error": f"{type(exc).__name__}: {exc}"}
        for league in select_leagues(args.league):
            # One lane's exception must not cost the other eight. A provider
            # shape change in, say, UCL used to abort the whole run before the
            # summary, the parity gate, and the localdata commit ever happened.
            try:
                run = run_lane(
                    league=league,
                    day=day,
                    store=store,
                    localdata=localdata,
                    client=EspnClient(),
                    as_of_utc=args.as_of,
                    probe_report=probe,
                    legs=args.legs,
                    min_legs=args.min_legs,
                    max_legs=args.max_legs,
                    limit=args.limit,
                    policy=policy,
                    min_combined_odds=None if args.no_combined_band else args.min_odds,
                    max_combined_odds=None if args.no_combined_band else args.max_odds,
                    ranking_metric=args.ranking_metric,
                    forbid_alt_line_reuse=args.forbid_alt_line_reuse,
                )
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                run = LaneRun(
                    lane=league.key,
                    day=day,
                    status="lane_error",
                    reason=f"{type(exc).__name__}: {exc}",
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
        "rehydrated": rehydrated,
        "seeded_ml": seeded,
    }
    out_dir = localdata / "_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _merge_previous_run(out_dir / f"daily_{day}.json", summary)
    (out_dir / f"daily_{day}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    failures = [
        run for run in runs if run.status in {"ingest_failed", "lane_error"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
