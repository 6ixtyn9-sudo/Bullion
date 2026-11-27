#!/usr/bin/env python3
"""Replay frozen decision pools: parity, variants, sizing, paired A/B.

Examples
--------
    python3 scripts/replay_harness.py --parity
    python3 scripts/replay_harness.py --mode cross-league --legs 2 --limit 3
    python3 scripts/replay_harness.py --stake-mode kelly --kelly 0.25 --slots 3
    python3 scripts/replay_harness.py --variant legs=3 --against legs=2 --ab

Zero settled days prints zeros and a low-sample flag — never a flat curve that
looks like a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.replay import (  # noqa: E402
    VALID_RANKING_METRICS,
    ReplaySpec,
    check_parity,
    compare_variants,
    load_evidence,
    merge_days,
    replay,
)

VALID_KEYS = {
    "legs",
    "min_legs",
    "max_legs",
    "limit",
    "min_leg_odds",
    "max_leg_odds",
    "price_weight",
    "min_odds",
    "max_odds",
    "ranking_metric",
    "max_claimed_edge",
    "edge_flag_threshold",
    "forbid_alt_line_reuse",
    "stake_mode",
    "daily_pct",
    "kelly_fraction",
    "slots",
}

# CLI spellings that mean a ReplaySpec field of a different name. Resolved
# BEFORE the VALID_KEYS check, which used to reject ``min_odds=`` for a key it
# then went on to accept.
ALIASES = {
    "min_odds": "min_combined_odds",
    "max_odds": "max_combined_odds",
}

BOOL_FIELDS = {"forbid_alt_line_reuse"}
INT_FIELDS = {"legs", "min_legs", "max_legs", "limit", "slots"}


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"expected a boolean (true/false), got {raw!r}")


def parse_variant(raw: Optional[str], base: ReplaySpec) -> ReplaySpec:
    if not raw:
        return base
    overrides: Dict[str, Any] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"variant entry must be key=value (got {chunk!r})")
        key, value = (part.strip() for part in chunk.split("=", 1))
        if key not in VALID_KEYS:
            raise SystemExit(f"unknown variant key {key!r}; known: {sorted(VALID_KEYS)}")
        field = ALIASES.get(key, key)
        if field in {"stake_mode", "ranking_metric"}:
            overrides[field] = value
        elif field in BOOL_FIELDS:
            overrides[field] = _parse_bool(value)
        elif field in INT_FIELDS:
            overrides[field] = int(value)
        else:
            overrides[field] = float(value)
    metric = overrides.get("ranking_metric")
    if metric is not None and metric not in VALID_RANKING_METRICS:
        raise SystemExit(
            f"unknown ranking_metric {metric!r}; known: {sorted(VALID_RANKING_METRICS)}"
        )
    return ReplaySpec(**{**base.__dict__, **overrides})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localdata", default="localdata")
    parser.add_argument("--lane", default=None, help="comma list of lanes to include")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--mode", choices=["per-league", "cross-league"], default="per-league")
    parser.add_argument(
        "--parity",
        action="store_true",
        help="rebuild receipts and diff; exits 1 on any mismatch (0 when there is no receipt to check)",
    )
    parser.add_argument("--legs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-leg-odds", type=float, default=None)
    parser.add_argument("--max-leg-odds", type=float, default=None)
    parser.add_argument("--price-weight", type=float, default=None)
    parser.add_argument("--stake-mode", choices=["flat", "daily_pct", "kelly", "skim"], default="flat")
    parser.add_argument("--daily-pct", type=float, default=0.25)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--variant", default=None, help="A: key=value,key=value")
    parser.add_argument("--against", default=None, help="B for --ab")
    parser.add_argument("--ab", action="store_true", help="paired A/B bootstrap")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    localdata = Path(args.localdata)
    if not localdata.is_absolute():
        localdata = ROOT / localdata
    lanes = [item.strip() for item in (args.lane or "").split(",") if item.strip()]
    days = load_evidence(localdata, lanes=lanes, since=args.since, until=args.until)
    if args.mode == "cross-league":
        days = merge_days(days)

    base = ReplaySpec(
        legs=args.legs,
        limit=args.limit,
        min_leg_odds=args.min_leg_odds,
        max_leg_odds=args.max_leg_odds,
        price_weight=args.price_weight,
        stake_mode=args.stake_mode,
        daily_pct=args.daily_pct,
        kelly_fraction=args.kelly,
        slots=args.slots,
    )
    report: Dict[str, Any] = {
        "schema_version": "bullion.replay_report",
        "mode": args.mode,
        "lanes": lanes or "all",
        "days_with_pool": len(days),
        # No-bet locks replay as 0-ticket days: their frozen pool is still
        # evidence, and ``--parity`` still has to agree no ticket was due.
        "no_bet_days": sum(
            1 for day in days if day.receipt.get("schema_version") == "magolide.no_bet_decision"
        ),
    }
    if args.parity:
        report["parity"] = check_parity(days)
    if args.ab:
        spec_a = parse_variant(args.variant, base)
        spec_b = parse_variant(args.against or args.variant, base)
        report["ab"] = compare_variants(days, spec_a, spec_b)
        report["variant_a"] = {**spec_a.__dict__}
        report["variant_b"] = {**spec_b.__dict__}
    else:
        spec = parse_variant(args.variant, base)
        report["replay"] = replay(days, spec)
        report["spec"] = {**spec.__dict__}

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key in ("parity", "replay", "ab"):
            if key in report:
                print(f"— {key} —")
                print(json.dumps(report[key], indent=2, sort_keys=True))
        if "replay" in report:
            print(report["replay"]["low_sample_flag"], "|", report["replay"]["note"])
    if args.parity and report["parity"]["mismatched"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
