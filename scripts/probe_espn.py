#!/usr/bin/env python3
"""Live ESPN probe: capture what the endpoints actually return, then stop.

This script is the ONLY place allowed to claim live evidence, and it may only
do so by writing payload facts into a JSON report. It never marks a field
"verified" because a URL was constructible.

Run it in GitHub Actions (``.github/workflows/probe_espn.yml``) and commit
``localdata/_probes/espn_probe_latest.json``. Every parser gap that matters —
NHL puckline nesting, soccer draw paths, NFL/CFB predictor statistics keys,
per-league named-book coverage, and historical backfill routes — is recorded
there so the next change is a measurement, not a guess.

What the 2026-11-26 run taught this script (and why it changed):

* it asked ESPN for ET days ``D-2/D-1`` instead of ``D-1/D``, so every probed
  event was already completed, the scoreboard odds arrays were empty, and no
  predictor could be attempted. The window and the ET dates are now recorded.
* it truncated an alphabetically sorted path list, which lost
  ``pickcenter``/``odds`` paths entirely. Paths are now captured per targeted
  subtree in payload order.
* every sampled summary DID carry ``pickcenter`` (plus ``odds``/``hasOdds`` for
  soccer) while scoreboard odds were empty, so summary quotes are now probed
  and counted as coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.espn import (  # noqa: E402
    PROBE_MIN_SAMPLE,
    PROBE_SCHEMA,
    EspnClient,
    et_days_for_sast_day,
    event_id_from_ref,
    line_number,
    nested,
    parse_pickcenter,
    canonical_book,
    parse_predictor,
    parse_scoreboard,
    parse_summary,
    predictor_url,
    quote_market_presence,
    scoreboard_url,
    season_events_url,
    summary_url,
)
from bullion.models import (  # noqa: E402
    LEAGUES,
    Sport,
    espn_season_keys,
    sast_day,
    to_utc_z,
)

MAX_PATHS = 120
MARKETS = ("moneyline", "moneyline_draw", "spread", "total")


def _paths(value: Any, prefix: str = "", limit: int = MAX_PATHS) -> List[str]:
    """Dotted paths of one targeted subtree, in payload order, capped by count."""
    out: List[str] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if len(out) >= limit or depth > 4:
            return
        if isinstance(node, dict):
            for key, item in node.items():
                child = f"{path}.{key}" if path else str(key)
                out.append(child)
                if len(out) >= limit:
                    return
                walk(item, child, depth + 1)
        elif isinstance(node, list) and node and isinstance(node[0], dict):
            walk(node[0], f"{path}[]", depth + 1)

    walk(value, prefix, 0)
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
            "drawOdds.moneyLine": nested(offer, "drawOdds", "moneyLine") is not None,
            "moneyline.drawOdds": nested(moneyline, "drawOdds") is not None,
        },
        "point_spread_home_line": line_number(nested(point_spread, "home", "close", "line")),
        "point_spread_sides": sorted(point_spread) if isinstance(point_spread, dict) else [],
        "total_over_line": line_number(nested(total, "over", "close", "line")),
        "overUnder": offer.get("overUnder"),
        "spread": offer.get("spread"),
        "flat_moneyline": nested(offer, "homeTeamOdds", "moneyLine") is not None,
        "keys": sorted(str(key) for key in offer)[:24],
    }


def _state(competition: Dict[str, Any]) -> str:
    return str(nested(competition, "status", "type", "state") or "")


def _coverage_block(counts: Dict[str, int], sample: int, denominator: str) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "denominator": denominator,
        "pre_match_events": sample,
        "sample_size": sample,
        "named_book_pre_tip": {},
    }
    for market in MARKETS:
        if sample <= 0:
            status = "no_pre_match_events"
            coverage = None
        elif sample < PROBE_MIN_SAMPLE:
            status = "insufficient_sample"
            coverage = round(counts[market] / sample, 4)
        else:
            status = "ok"
            coverage = round(counts[market] / sample, 4)
        block["named_book_pre_tip"][market] = {
            "events": counts[market],
            "event_coverage": coverage,
            "sample_size": sample,
            "status": status,
        }
    return block


def _probe_summary(
    client: EspnClient, league, game, captured: str, entry: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Fetch one summary and record parse evidence plus observed paths."""
    entry["summary"]["attempts"] += 1
    try:
        payload = client.get(summary_url(league, game.game_id))
    except Exception as exc:
        entry["summary"]["errors"].append({"game_id": game.game_id, "error": str(exc)})
        return None
    competitions = nested(payload, "header", "competitions") or []
    competition = competitions[0] if competitions else {}
    competitors = competition.get("competitors") or []
    pickcenter_rows = payload.get("pickcenter")
    pickcenter_entry = (
        pickcenter_rows[0]
        if isinstance(pickcenter_rows, list) and pickcenter_rows and isinstance(pickcenter_rows[0], dict)
        else {}
    )
    odds_rows = payload.get("odds")
    odds_entry = (
        odds_rows[0]
        if isinstance(odds_rows, list) and odds_rows and isinstance(odds_rows[0], dict)
        else {}
    )
    record = {
        "game_id": game.game_id,
        "status": _state(competition),
        "scheduled_utc": game.scheduled_utc,
        "top_level_keys": sorted(str(key) for key in payload)[:40],
        "has_pickcenter": isinstance(pickcenter_rows, list),
        "pickcenter_entries": len(pickcenter_rows) if isinstance(pickcenter_rows, list) else 0,
        "has_odds": isinstance(odds_rows, list),
        "has_odds_flag": payload.get("hasOdds"),
        "has_predictor": "predictor" in payload,
        "has_injuries": "injuries" in payload,
        "pickcenter_paths": _paths(pickcenter_entry, "pickcenter[0]"),
        "pickcenter_keys": sorted(str(key) for key in pickcenter_entry)[:24],
        "odds_paths": _paths(odds_entry, "odds[0]") if odds_entry else [],
        "header_paths": _paths(competition, "header.competitions[0]"),
        "competitor_linescores": [
            {
                "homeAway": competitor.get("homeAway"),
                "periods": len(competitor.get("linescores") or []),
            }
            for competitor in competitors
            if isinstance(competitor, dict)
        ],
    }
    parsed_pickcenter = parse_pickcenter(payload, game, captured)
    record["pickcenter_parse"] = {
        "quotes": len(parsed_pickcenter.quotes),
        "markets": sorted({quote.market for quote in parsed_pickcenter.quotes}),
        "markets_priced": sorted(quote_market_presence(parsed_pickcenter.quotes)),
        "side_keys": {
            quote.market: sorted(quote.side_prices) for quote in parsed_pickcenter.quotes
        },
        "pre_match_quotes": sum(1 for quote in parsed_pickcenter.quotes if quote.pre_match),
        "gaps": [gap.field for gap in parsed_pickcenter.gaps],
        "observed_entry_keys": parsed_pickcenter.observed_entry_keys[:24],
    }
    if record["status"] == "post":
        try:
            parse_summary(payload, game, league)
            record["summary_parse"] = "ok"
        except Exception as exc:
            record["summary_parse"] = f"error: {exc}"
    entry["summary"]["field_presence"].append(record)
    return record


def probe_league(
    client: EspnClient,
    league,
    sast_days: List[str],
    *,
    summary_samples: int = 8,
    predictor_samples: int = 3,
) -> Dict[str, Any]:
    captured = to_utc_z(datetime.now(timezone.utc))
    entry: Dict[str, Any] = {
        "label": league.label,
        "sport": league.sport.value,
        "espn_path": league.espn_path,
        "sast_days_probed": sast_days,
        "et_dates_requested": {},
        "scoreboard_urls": [],
        "events": 0,
        "pre_events": 0,
        "state_histogram": {},
        "events_with_odds_key": 0,
        "events_with_offers": 0,
        "odds_offers": 0,
        "books": [],
        "markets": {market: 0 for market in MARKETS},
        "offer_samples": [],
        "scoreboard_offer_paths": [],
        "coverage": {},
        "parse": {"games": 0, "quotes": 0, "gaps": []},
        "predictor": {"attempts": 0, "samples": [], "errors": [], "note": ""},
        "summary": {"attempts": 0, "field_presence": [], "errors": []},
        "errors": [],
    }
    seen: Dict[str, Dict[str, Any]] = {}
    for day in sast_days:
        et_days = et_days_for_sast_day(day)
        entry["et_dates_requested"][day] = et_days
        for et_date in et_days:
            url = scoreboard_url(league, et_date)
            entry["scoreboard_urls"].append(url)
            try:
                payload = client.get(url)
            except Exception as exc:
                entry["errors"].append({"url": url, "error": str(exc)})
                continue
            for event in payload.get("events") or []:
                event_id = str(event.get("id") or "")
                if not event_id or event_id in seen:
                    continue
                competitions = event.get("competitions") or [{}]
                seen[event_id] = {
                    "event": event,
                    "competition": competitions[0],
                    "state": _state(competitions[0]),
                    "sast_day": sast_day(competitions[0].get("date") or event.get("date")),
                }

    events = list(seen.values())
    entry["events"] = len(events)
    entry["et_dates_only"] = sorted(
        {date for dates in entry["et_dates_requested"].values() for date in dates}
    )
    histogram: Dict[str, int] = {}
    market_counts = {market: 0 for market in MARKETS}
    books: set[str] = set()
    raw_books: set[str] = set()
    for item in events:
        competition = item["competition"]
        state = item["state"]
        histogram[state or "unknown"] = histogram.get(state or "unknown", 0) + 1
        if "odds" in competition:
            entry["events_with_odds_key"] += 1
        offers = [offer for offer in (competition.get("odds") or []) if isinstance(offer, dict)]
        if offers:
            entry["events_with_offers"] += 1
        entry["odds_offers"] += len(offers)
        for offer in offers:
            shape = _offer_shape(offer)
            if shape["provider"]:
                books.add(str(shape["provider"]))
                raw_books.add(str(shape["provider"]))
            if len(entry["offer_samples"]) < 3:
                entry["offer_samples"].append(shape)
            if not entry["scoreboard_offer_paths"]:
                entry["scoreboard_offer_paths"] = _paths(offer, "offer")
    entry["state_histogram"] = histogram
    entry["books"] = sorted({canonical_book(name) for name in books})
    entry["book_normalisation"] = {
        name: canonical_book(name) for name in sorted(raw_books) if canonical_book(name) != name
    }
    entry["markets"] = market_counts
    pre_events = [item for item in events if item["state"] == "pre"]
    entry["pre_events"] = len(pre_events)

    parsed = parse_scoreboard(
        {"events": [item["event"] for item in events]}, league, captured
    )
    games_by_id = {game.game_id: game for game in parsed.games}
    # Coverage is judged by priced *sides*, not market labels: a soccer
    # moneyline quote that carries ``draw`` is a three-way quote even though its
    # market is still "moneyline".
    presence_by_game: Dict[str, set] = {}
    for quote in parsed.quotes:
        presence_by_game.setdefault(quote.game_id, set()).update(
            quote_market_presence([quote])
        )
    market_counts = {
        market: sum(
            1
            for item in pre_events
            if market in presence_by_game.get(str(item["event"].get("id")), set())
        )
        for market in MARKETS
    }
    entry["markets"] = market_counts
    entry["coverage"] = _coverage_block(
        market_counts, len(pre_events), "scoreboard_pre_match_events"
    )
    entry["coverage"]["named_book_pre_tip_scoreboard"] = dict(
        entry["coverage"]["named_book_pre_tip"]
    )
    entry["parse"] = {
        "games": len(parsed.games),
        "quotes": len(parsed.quotes),
        "gaps": [gap.to_dict() for gap in parsed.gaps[:20]],
        "quote_markets": sorted({quote.market for quote in parsed.quotes}),
    }

    # Summary sample: upcoming fixtures first (they are the only ones that can
    # prove pre-tip coverage), then recent finals for field/period evidence.
    upcoming = sorted(
        (item for item in pre_events if item["event"]["id"] in games_by_id),
        key=lambda item: str(item["competition"].get("date") or ""),
    )[:summary_samples]
    recent = sorted(
        (
            item
            for item in events
            if item["state"] == "post" and item["event"]["id"] in games_by_id
        ),
        key=lambda item: str(item["competition"].get("date") or ""),
        reverse=True,
    )[: max(summary_samples - len(upcoming), 1)]
    summary_market_counts = {market: 0 for market in MARKETS}
    summary_sample = 0
    for item in [*upcoming, *recent]:
        game = games_by_id[item["event"]["id"]]
        record = _probe_summary(client, league, game, captured, entry)
        if record is None:
            continue
        if record["status"] == "pre":
            summary_sample += 1
            markets_with_quotes = set(record["pickcenter_parse"]["markets_priced"])
            if record["pickcenter_parse"]["pre_match_quotes"] <= 0:
                markets_with_quotes = set()
            for market in markets_with_quotes:
                summary_market_counts[market] += 1
    if summary_sample:
        summary_block = _coverage_block(
            summary_market_counts, summary_sample, "summary_pickcenter_pre_match_events"
        )
        entry["coverage"]["named_book_pre_tip"] = summary_block["named_book_pre_tip"]
        entry["coverage"]["denominator"] = summary_block["denominator"]
        entry["coverage"]["summary_sample_size"] = summary_sample

    # Predictor: upcoming fixtures first; if the window has none, one finals
    # sample is still worth capturing to learn the key names (never pre-tip
    # availability — the sample context is recorded).
    if not league.predictor:
        entry["predictor"]["skipped"] = True
        entry["predictor"]["note"] = (
            "no /predictor route for this lane (LIVE_VERIFIED 2026-11-26: HTTP 400 for "
            "NHL and every soccer league); not requested"
        )
        return entry
    predictor_targets = [
        (games_by_id[item["event"]["id"]], "pre") for item in upcoming[:predictor_samples]
    ]
    if not predictor_targets:
        predictor_targets = [
            (games_by_id[item["event"]["id"]], "post") for item in recent[:1]
        ]
        entry["predictor"]["note"] = (
            "no pre-match events in the probe window; predictor keys sampled on a "
            "completed fixture and do NOT establish pre-tip availability"
        )
    for game, context in predictor_targets:
        entry["predictor"]["attempts"] += 1
        record: Dict[str, Any] = {
            "game_id": game.game_id,
            "sample_context": context,
            "home_team_id": game.home_team_id,
            "away_team_id": game.away_team_id,
        }
        try:
            payload = client.get(predictor_url(league, game.game_id))
        except Exception as exc:
            entry["predictor"]["errors"].append({**record, "error": f"fetch: {exc}"})
            continue
        # Capture the payload shape BEFORE parsing: if the parser rejects the
        # payload, these keys are the evidence needed to fix it. An error string
        # alone would waste the probe run.
        record["payload_keys"] = sorted(str(key) for key in payload)[:24]
        record["sides"] = {
            side: {
                "team_ref": nested(payload, side, "team", "$ref"),
                "team_id": nested(payload, side, "team", "id"),
                "statistics": [
                    str(item.get("name"))
                    for item in (nested(payload, side, "statistics") or [])
                    if isinstance(item, dict)
                ][:24],
                # Values for the keys that decide two-way/tie handling. The
                # next probe can then check ``teamPredPtDiff`` reconciliation
                # instead of trusting a key name.
                "stat_values": {
                    str(item.get("name")): item.get("value")
                    for item in (nested(payload, side, "statistics") or [])
                    if isinstance(item, dict)
                    and item.get("name")
                    in (
                        "gameProjection",
                        "teamChanceLoss",
                        "teamChanceTie",
                        "teamPredPtDiff",
                    )
                },
            }
            for side in ("homeTeam", "awayTeam")
        }
        record["paths"] = _paths(payload, "predictor")
        try:
            evidence = parse_predictor(payload, game)
        except Exception as exc:
            record["parse_error"] = str(exc)
            entry["predictor"]["errors"].append({**record, "error": str(exc)})
        else:
            record.update(
                {
                    "prob_home": round(evidence.prob_home, 6),
                    "prob_tie": round(evidence.prob_tie, 6),
                    "tie_rule": evidence.tie_rule,
                    "predicted_margin": evidence.predicted_margin,
                    "expected_home": evidence.expected_home,
                    "expected_away": evidence.expected_away,
                    "stats_side": evidence.stats_side,
                }
            )
        entry["predictor"]["samples"].append(record)
    return entry


def probe_backfill(client: EspnClient, league, season: int) -> Dict[str, Any]:
    url = season_events_url(league, season, 1)
    report: Dict[str, Any] = {
        "url": url,
        "season": season,
        "season_key_source": "bullion.models.espn_season_keys",
    }
    try:
        payload = client.get(url)
    except Exception as exc:
        report["error"] = str(exc)
        return report
    refs = [str(item.get("$ref")) for item in payload.get("items") or [] if isinstance(item, dict)]
    ids = [event_id_from_ref(ref) for ref in refs[:5]]
    report.update(
        {
            "count": payload.get("count"),
            "page_count": payload.get("pageCount"),
            "first_refs": refs[:3],
            "ref_ids_parsed": ids,
            "refs_parse_ok": bool(ids) and all(ids),
            "keys": sorted(str(key) for key in payload),
        }
    )
    return report


def _merge_previous_report(
    path: Path, report: Dict[str, Any], *, probed: List[str]
) -> Dict[str, Any]:
    """Fold this run's lanes into the previous report instead of replacing it.

    Probing a subset (``--leagues basketball``) used to overwrite the file and
    erase every other lane's captured coverage, which would silently drop those
    lanes back to ``research_only`` on the next daily run — a coverage file that
    forgets evidence is worse than no coverage file. Lanes probed now win; every
    other lane keeps its last captured block verbatim, and the receipt says
    which lanes came from when.
    """
    if not path.exists():
        report["probed_lanes"] = probed
        return report
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report["probed_lanes"] = probed
        return report
    if not isinstance(previous, dict):
        report["probed_lanes"] = probed
        return report
    merged = dict(previous)
    merged.update(report)
    merged_leagues = dict(previous.get("leagues") or {})
    merged_leagues.update(report.get("leagues") or {})
    merged["leagues"] = merged_leagues
    merged_backfill = dict(previous.get("backfill") or {})
    merged_backfill.update(report.get("backfill") or {})
    merged["backfill"] = merged_backfill
    merged["probed_lanes"] = probed
    carried = sorted(set(merged_leagues) - set(probed))
    if carried:
        merged["carried_over_lanes"] = carried
        merged["carried_over_from_utc"] = (previous.get("captured_utc") or previous.get("generated_utc"))
    return merged


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="localdata/_probes/espn_probe_latest.json")
    parser.add_argument("--date", default=None, help="first SAST day (YYYY-MM-DD); default today")
    parser.add_argument("--days-ahead", type=int, default=2, help="probe this many extra SAST days")
    parser.add_argument(
        "--leagues", default="all", help="comma list of lane keys, 'soccer', or 'all'"
    )
    parser.add_argument("--season", type=int, default=None, help="season year for backfill probe")
    parser.add_argument("--predictor-samples", type=int, default=3)
    parser.add_argument("--summary-samples", type=int, default=8)
    args = parser.parse_args(argv)

    if args.date:
        first_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        from zoneinfo import ZoneInfo

        first_day = datetime.now(ZoneInfo("Africa/Johannesburg")).date()
    sast_days = [(first_day + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(args.days_ahead + 1)]

    selector = args.leagues.strip().lower()
    if selector == "all":
        leagues = list(LEAGUES.values())
    elif selector == "soccer":
        leagues = [league for league in LEAGUES.values() if league.sport is Sport.SOCCER]
    elif selector == "basketball":
        leagues = [league for league in LEAGUES.values() if league.sport is Sport.BASKETBALL]
    else:
        leagues = [LEAGUES[key.strip()] for key in selector.split(",") if key.strip()]

    client = EspnClient()
    report: Dict[str, Any] = {
        "schema_version": PROBE_SCHEMA,
        "mode": "live",
        "captured_utc": to_utc_z(datetime.now(timezone.utc)),
        "sast_days": sast_days,
        "sast_day": sast_days[0],
        "note": (
            "Captured payload facts only. Coverage below 0.5 keeps a market/lane in "
            "research_only; a coverage number without status='ok' and a sample of at "
            "least 3 pre-match events is not a measurement and cannot gate tickets."
        ),
        "leagues": {},
    }
    season = args.season or datetime.now(timezone.utc).year
    for league in leagues:
        print(f"[probe] {league.key} {sast_days}", file=sys.stderr)
        report["leagues"][league.key] = probe_league(
            client,
            league,
            sast_days,
            summary_samples=args.summary_samples,
            predictor_samples=args.predictor_samples,
        )
    # Ask for the season a lane is actually playing: ESPN numbers a straddling
    # basketball season by the year it ends in, so a raw 2026 in November 2026
    # probes the finished 2025-26 season and reports on history no lane prices.
    report["backfill"] = {
        league.key: probe_backfill(
            client, league, espn_season_keys(league, [season])[0]
        )
        for league in leagues
    }
    report["warnings"] = [
        "A scoreboard route that resolves does not imply a predictor or pickcenter contract.",
        "Coverage is measured over the probed SAST window only; re-run before enabling a lane.",
        "Predictor samples taken on completed fixtures do not establish pre-tip availability.",
    ]

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    report = _merge_previous_report(out, report, probed=[league.key for league in leagues])
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
