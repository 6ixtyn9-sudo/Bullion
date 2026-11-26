#!/usr/bin/env python3
"""Live ESPN probe: capture what the endpoints actually return, then stop.

This script is the ONLY place allowed to claim live evidence, and it may only
do so by writing payload facts into a JSON report. It never marks a field
"verified" because a URL was constructible.

Run it in GitHub Actions (``docs/workflows/probe_espn.yml``) and commit
``localdata/_probes/espn_probe_latest.json``. Every parser gap that matters —
NHL puckline nesting, soccer draw paths, NFL/CFB predictor statistics keys,
per-league named-book coverage, and historical backfill routes — is recorded
there so the next change is a measurement, not a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.espn import (  # noqa: E402
    PROBE_SCHEMA,
    EspnClient,
    et_days_for_sast_day,
    line_number,
    nested,
    parse_predictor,
    parse_scoreboard,
    parse_summary,
    scoreboard_url,
    season_events_url,
    summary_url,
    predictor_url,
)
from bullion.models import LEAGUES, Sport, to_utc_z  # noqa: E402

MAX_KEY_PATHS = 400


def _key_paths(value: Any, prefix: str = "", depth: int = 0, out: Optional[set] = None) -> set:
    """Dotted key paths, bounded in depth and size (never full payloads)."""
    out = out if out is not None else set()
    if len(out) >= MAX_KEY_PATHS:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.add(path)
            if depth < 4:
                _key_paths(item, path, depth + 1, out)
    elif isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            _key_paths(first, f"{prefix}[]", depth + 1, out)
    return out


def _offer_shape(offer: Dict[str, Any]) -> Dict[str, Any]:
    moneyline = offer.get("moneyline") or {}
    point_spread = offer.get("pointSpread") or {}
    total = offer.get("total") or {}
    return {
        "provider": nested(offer, "provider", "displayName")
        or nested(offer, "provider", "name")
        or nested(offer, "provider", "id"),
        "moneyline_sides": sorted(moneyline) if isinstance(moneyline, dict) else [],
        "draw_paths": {
            "moneyline.draw.close.odds": nested(moneyline, "draw", "close", "odds") is not None,
            "moneyline.draw.open.odds": nested(moneyline, "draw", "open", "odds") is not None,
            "drawOdds": offer.get("drawOdds") is not None,
            "moneyline.drawOdds": nested(moneyline, "drawOdds") is not None,
        },
        "point_spread_home_line": line_number(nested(point_spread, "home", "close", "line")),
        "point_spread_sides": sorted(point_spread) if isinstance(point_spread, dict) else [],
        "total_over_line": line_number(nested(total, "over", "close", "line")),
        "overUnder": offer.get("overUnder"),
        "homeTeamOdds_moneyLine": nested(offer, "homeTeamOdds", "moneyLine") is not None,
    }


def probe_league(
    client: EspnClient,
    league,
    day: str,
    *,
    predictor_samples: int = 3,
    summary_samples: int = 2,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "label": league.label,
        "sport": league.sport.value,
        "espn_path": league.espn_path,
        "scoreboard_urls": [],
        "events": 0,
        "pre_events": 0,
        "odds_offers": 0,
        "books": [],
        "markets": {"moneyline": 0, "moneyline_draw": 0, "spread": 0, "total": 0},
        "coverage": {},
        "offer_samples": [],
        "parse": {"games": 0, "quotes": 0, "gaps": []},
        "predictor": {"attempts": 0, "samples": [], "errors": []},
        "summary": {"attempts": 0, "field_presence": [], "errors": []},
        "key_paths": [],
        "errors": [],
    }
    events: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for et_date in et_days_for_sast_day(day):
        url = scoreboard_url(league, et_date)
        entry["scoreboard_urls"].append(url)
        try:
            payload = client.get(url)
        except Exception as exc:  # probe reports every failure instead of hiding it
            entry["errors"].append({"url": url, "error": str(exc)})
            continue
        entry["key_paths"] = sorted(set(entry["key_paths"]) | _key_paths(payload))
        for event in payload.get("events") or []:
            event_id = str(event.get("id") or "")
            if event_id and event_id not in seen:
                seen.add(event_id)
                events.append(event)

    entry["events"] = len(events)
    market_counts = {"moneyline": 0, "moneyline_draw": 0, "spread": 0, "total": 0}
    books: set[str] = set()
    pre_events = 0
    market_event_presence: Dict[str, int] = {
        "moneyline": 0,
        "moneyline_draw": 0,
        "spread": 0,
        "total": 0,
    }
    for event in events:
        competitions = event.get("competitions") or [{}]
        competition = competitions[0]
        state = nested(competition, "status", "type", "state")
        if state == "pre":
            pre_events += 1
        offers = [offer for offer in competition.get("odds") or [] if isinstance(offer, dict)]
        entry["odds_offers"] += len(offers)
        seen_markets: set[str] = set()
        for offer in offers:
            shape = _offer_shape(offer)
            if shape["provider"]:
                books.add(str(shape["provider"]))
            if len(entry["offer_samples"]) < 3:
                entry["offer_samples"].append(shape)
            if "home" in shape["moneyline_sides"] and "away" in shape["moneyline_sides"]:
                market_counts["moneyline"] += 1
                seen_markets.add("moneyline")
            if any(shape["draw_paths"].values()):
                market_counts["moneyline_draw"] += 1
                seen_markets.add("moneyline_draw")
            if shape["point_spread_home_line"] is not None:
                market_counts["spread"] += 1
                seen_markets.add("spread")
            if shape["total_over_line"] is not None or shape["overUnder"] is not None:
                market_counts["total"] += 1
                seen_markets.add("total")
        if state == "pre":
            for market in seen_markets:
                market_event_presence[market] += 1

    entry["markets"] = market_counts
    entry["books"] = sorted(books)
    entry["pre_events"] = pre_events
    denominator = max(pre_events, 1)
    entry["coverage"] = {
        "denominator": "pre_match_events",
        "pre_match_events": pre_events,
        "named_book_pre_tip": {
            market: {
                "events": market_event_presence[market],
                "event_coverage": round(market_event_presence[market] / denominator, 4),
            }
            for market in ("moneyline", "moneyline_draw", "spread", "total")
        },
    }

    captured = to_utc_z(datetime.now(timezone.utc))
    parsed = parse_scoreboard({"events": events}, league, captured)
    entry["parse"] = {
        "games": len(parsed.games),
        "quotes": len(parsed.quotes),
        "gaps": [gap.to_dict() for gap in parsed.gaps[:20]],
        "quote_markets": sorted({quote.market for quote in parsed.quotes}),
    }

    for game in [g for g in parsed.games if g.status.value == "scheduled"][:predictor_samples]:
        entry["predictor"]["attempts"] += 1
        try:
            payload = client.get(predictor_url(league, game.game_id))
            evidence = parse_predictor(payload, game)
            entry["predictor"]["samples"].append(
                {
                    "game_id": game.game_id,
                    "prob_home": round(evidence.prob_home, 6),
                    "expected_home": evidence.expected_home,
                    "expected_away": evidence.expected_away,
                    "stats_side": evidence.stats_side,
                    "statistics_keys": sorted(_key_paths(payload, "predictor")),
                }
            )
        except Exception as exc:
            entry["predictor"]["errors"].append({"game_id": game.game_id, "error": str(exc)})

    candidates = sorted(parsed.games, key=lambda g: g.status.value != "final")
    for game in candidates[:summary_samples]:
        entry["summary"]["attempts"] += 1
        try:
            payload = client.get(summary_url(league, game.game_id))
            entry["summary"]["field_presence"].append(
                {
                    "game_id": game.game_id,
                    "keys": sorted(payload)[:40],
                    "has_pickcenter": "pickcenter" in payload,
                    "has_predictor": "predictor" in payload,
                    "has_injuries": "injuries" in payload,
                    "boxscore_keys": sorted(nested(payload, "boxscore") or {})
                    if isinstance(nested(payload, "boxscore"), dict)
                    else [],
                    "summary_keys": sorted(_key_paths(payload, "summary"))[:120],
                }
            )
            if game.status.value == "final":
                parse_summary(payload, game, league)
        except Exception as exc:
            entry["summary"]["errors"].append({"game_id": game.game_id, "error": str(exc)})

    return entry


def probe_backfill(client: EspnClient, league, season: int) -> Dict[str, Any]:
    url = season_events_url(league, season, 1)
    report: Dict[str, Any] = {"url": url, "season": season}
    try:
        payload = client.get(url)
    except Exception as exc:
        report["error"] = str(exc)
        return report
    items = payload.get("items") or []
    report.update(
        {
            "count": payload.get("count"),
            "page_count": payload.get("pageCount"),
            "first_refs": [str(item.get("$ref")) for item in items[:3]],
            "keys": sorted(payload),
        }
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="localdata/_probes/espn_probe_latest.json")
    parser.add_argument("--date", default=None, help="SAST day (YYYY-MM-DD); default today")
    parser.add_argument(
        "--leagues",
        default="all",
        help="comma list of lane keys, 'soccer', or 'all'",
    )
    parser.add_argument("--season", type=int, default=None, help="season year for backfill probe")
    parser.add_argument("--predictor-samples", type=int, default=3)
    args = parser.parse_args(argv)

    if args.date:
        day = args.date
    else:
        from zoneinfo import ZoneInfo

        day = datetime.now(ZoneInfo("Africa/Johannesburg")).strftime("%Y-%m-%d")

    selector = args.leagues.strip().lower()
    if selector == "all":
        leagues = list(LEAGUES.values())
    elif selector == "soccer":
        leagues = [league for league in LEAGUES.values() if league.sport is Sport.SOCCER]
    else:
        leagues = [LEAGUES[key.strip()] for key in selector.split(",") if key.strip()]

    client = EspnClient()
    report: Dict[str, Any] = {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": to_utc_z(datetime.now(timezone.utc)),
        "sast_day": day,
        "note": (
            "Captured payload facts only. Coverage below 0.5 keeps a market/lane in "
            "research_only; no forecast or ticket is implied by this file."
        ),
        "leagues": {},
    }
    season = args.season or datetime.now(timezone.utc).year
    for league in leagues:
        print(f"[probe] {league.key} {day}", file=sys.stderr)
        report["leagues"][league.key] = probe_league(
            client, league, day, predictor_samples=args.predictor_samples
        )
    report["backfill"] = {
        league.key: probe_backfill(client, league, season) for league in leagues
    }
    report["warnings"] = [
        "A scoreboard route that resolves does not imply a predictor or pickcenter contract.",
        "Coverage is computed over the probed SAST day only; re-run before enabling a new lane.",
    ]

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
