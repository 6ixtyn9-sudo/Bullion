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
    ReplaySpec,
    check_parity,
    compare_variants,
    load_evidence,
    merge_days,
    replay,
)

VALID_KEYS = {
    "legs",
    "limit",
    "min_leg_odds",
    "max_leg_odds",
    "price_weight",
    "min_odds",
    "max_odds",
    "stake_mode",
    "daily_pct",
    "kelly_fraction",
    "slots",
}


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
        if key == "min_odds":
            key = "min_combined_odds"
        if key == "max_odds":
            key = "max_combined_odds"
        if key not in VALID_KEYS:
            raise SystemExit(f"unknown variant key {key!r}; known: {sorted(VALID_KEYS)}")
        field = key
        if field in {"stake_mode"}:
            overrides[field] = value
        elif field in {"legs", "limit", "slots"}:
            overrides[field] = int(value)
        else:
            overrides[field] = float(value)
    return ReplaySpec(**{**base.__dict__, **overrides})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localdata", default="localdata")
    parser.add_argument("--lane", default=None, help="comma list of lanes to include")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--mode", choices=["per-league", "cross-league"], default="per-league")
    parser.add_argument("--parity", action="store_true", help="rebuild receipts and diff")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
