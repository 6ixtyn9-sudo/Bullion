#!/usr/bin/env python3
"""ESPN historical backfill for settled finals (2024-25 onward, bounded).

What this does: stores official finals and their native period linescores so
the NHL/Soccer/Football rating fits start warm on day one.

What this deliberately does NOT do: fabricate pre-tip quotes for old games.
Historical scoreboards are served today, so their prices are closing/after-the-
fact numbers. Using them as ticket evidence would be a look-ahead fantasy, and
using them to "warm calibration" would smuggle post-game information into a
signal. Calibration therefore starts cold and must earn 30 settled
observations per segment exactly like the live lane does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.espn import (  # noqa: E402
    EspnClient,
    EspnFetchError,
    event_id_from_ref,
    game_from_summary,
    parse_summary,
    season_events_url,
)
from bullion.models import LEAGUES, Sport, now_utc  # noqa: E402
from bullion.store import Store  # noqa: E402

REINDEX_PAGES = 25


def select_leagues(selector: str):
    selector = (selector or "all").strip().lower()
    if selector == "all":
        return list(LEAGUES.values())
    if selector == "soccer":
        return [league for league in LEAGUES.values() if league.sport is Sport.SOCCER]
    if selector not in LEAGUES:
        raise ValueError(f"unknown league {selector!r}")
    return [LEAGUES[selector]]


def season_event_ids(client: EspnClient, league, season: int) -> List[str]:
    ids: List[str] = []
    for page in range(1, REINDEX_PAGES + 1):
        payload = client.get(season_events_url(league, season, page))
        for item in payload.get("items") or []:
            # LIVE_VERIFIED 2026-11-26: refs arrive as
            # ``.../events/401856879?lang=en&region=us`` — the query suffix must
            # be stripped, otherwise every ref fails ``isdigit()`` and the
            # backfill silently stores nothing.
            event_id = event_id_from_ref(item.get("$ref"))
            if event_id and event_id not in ids:
                ids.append(event_id)
        page_count = int(payload.get("pageCount") or 1)
        if page >= page_count:
            break
    return ids


def run(
    *,
    store: Store,
    client: EspnClient,
    leagues: Iterable,
    seasons: Iterable[int],
    max_events: int,
    delay_seconds: float = 0.15,
) -> dict:
    manifest = {
        "schema_version": "bullion.backfill_manifest",
        "generated_utc": now_utc(),
        "seasons": list(seasons),
        "max_events_per_season": max_events,
        "leagues": {},
        "note": (
            "Finals only. No historical prices are used as ticket or calibration "
            "evidence; calibration must accrue from live pre-tip forecasts."
        ),
    }
    for league in leagues:
        league_manifest = {}
        for season in seasons:
            try:
                ids = season_event_ids(client, league, season)[:max_events]
            except (EspnFetchError, ValueError) as exc:
                league_manifest[str(season)] = {"error": str(exc)}
                continue
            stored = skipped = failed = 0
            for event_id in ids:
                if store.get_game(event_id) is not None:
                    skipped += 1
                    continue
                try:
                    payload = client.summary(league, event_id)
                    game = game_from_summary(payload, league, event_id)
                    final_game, _gaps = parse_summary(payload, game, league)
                    store.upsert_game(final_game)
                    stored += 1
                except (EspnFetchError, ValueError) as exc:
                    failed += 1
                    if failed <= 3:
                        league_manifest.setdefault("sample_errors", []).append(
                            {"event_id": event_id, "error": str(exc)}
                        )
                time.sleep(delay_seconds)
            league_manifest[str(season)] = {
                "indexed": len(ids),
                "stored": stored,
                "skipped_existing": skipped,
                "failed": failed,
            }
            print(f"[backfill] {league.key} {season}: {league_manifest[str(season)]}", file=sys.stderr)
        manifest["leagues"][league.key] = league_manifest
    manifest["counts"] = store.counts()
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="all")
    parser.add_argument("--seasons", default="2024,2025,2026")
    parser.add_argument("--max-events", type=int, default=400)
    parser.add_argument("--db", default="localdata/bullion.db")
    parser.add_argument("--out", default="localdata/_backfill/manifest.json")
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args(argv)

    seasons = [int(item) for item in args.seasons.split(",") if item.strip()]
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    with Store(db_path) as store:
        manifest = run(
            store=store,
            client=EspnClient(),
            leagues=select_leagues(args.league),
            seasons=seasons,
            max_events=args.max_events,
            delay_seconds=args.delay,
        )
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
