"""Keyless ESPN client and fail-closed parsers.

Provenance of the patterns here
-------------------------------
The URL shapes, ``homeAway`` side resolution, odds nesting
(``moneyline|pointSpread|total`` -> ``side`` -> ``open|close`` -> ``line|odds``),
American->decimal conversion, and the predictor ``$ref`` orientation rule are
all taken from the LIVE_VERIFIED basketball adapter in
``Ma_Golide_Satellites/ingestion/providers/espn.py`` (NBA/WNBA/NCAA captured
2026-11-06).

What is NOT yet proven
----------------------
No hockey, football, or soccer ESPN response has been captured by this repo.
Extrapolating a proven pattern is not verification:

* hockey ``pointSpread`` puckline nesting (``±1.5``) and NHL ``pickcenter``
  book coverage are unconfirmed;
* football ``/predictor`` statistics keys (``gameProjection`` etc.) are
  unconfirmed for NFL/CFB — they are proven for basketball BPI only;
* soccer three-way ``moneyline.draw`` / ``drawOdds`` paths are unconfirmed;
* pre-kickoff named-book coverage percentages per soccer league are unknown.

``scripts/probe_espn.py`` exists to replace every "unconfirmed" above with a
captured fact. Until it has run and its JSON has been inspected, treat all of
it as an assumption, never as ``LIVE_VERIFIED``.

Fail-closed contract
--------------------
Missing or contradictory fields never cause an imputed value, invented price,
or a swapped home/away. They produce a :class:`CoverageGap` (machine-readable)
or raise :class:`EspnIdentityError`.
"""

from __future__ import annotations

import gzip
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    LEAGUES,
    ActualResult,
    BookQuote,
    CoverageGap,
    Game,
    League,
    Sport,
    Status,
    american_to_decimal,
    parse_utc,
    to_utc_z,
)

SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports"
CORE_BASE = "https://sports.core.api.espn.com/v2/sports"

FetchFn = Callable[[str], Any]


class EspnFetchError(RuntimeError):
    """A request failed after retries; never silently substituted."""


class EspnIdentityError(ValueError):
    """A payload contradicted the requested league/game identity."""


class EspnParseError(ValueError):
    """A payload was malformed for the field being read."""


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def scoreboard_url(league: League, et_date: str) -> str:
    """Scoreboard URL. ``et_date`` is a US Eastern calendar day (YYYYMMDD).

    The scoreboard ``dates=`` parameter selects the US Eastern day of the
    slate while event timestamps are UTC — a ``20261105`` slate contains games
    stamped ``2026-11-06T02:00Z``. Callers must not treat ``dates=`` as UTC.
    """
    sport, slug = league.espn_path.split("/", 1)
    # A Saturday NCAA slate is 100+ fixtures across two ET days; the default
    # 100-event window silently drops the tail, and a dropped fixture is a
    # missed pre-tip price rather than a visible error.
    limit = 500 if league.sport in (Sport.FOOTBALL, Sport.BASKETBALL) else 100
    url = f"{SITE_BASE}/{sport}/{slug}/scoreboard?dates={et_date}&limit={limit}"
    if league.scoreboard_groups is not None:
        # NCAA basketball needs groups=50 (D1); without it the same league id
        # serves every division's games.
        url += f"&groups={int(league.scoreboard_groups)}"
    return url


def summary_url(league: League, event_id: str) -> str:
    sport, slug = league.espn_path.split("/", 1)
    return f"{SITE_BASE}/{sport}/{slug}/summary?event={event_id}"


def predictor_url(league: League, event_id: str) -> str:
    sport, slug = league.espn_path.split("/", 1)
    return (
        f"{CORE_BASE}/{sport}/leagues/{slug}/events/{event_id}"
        f"/competitions/{event_id}/predictor"
    )


def season_events_url(league: League, season: int, page: int) -> str:
    sport, slug = league.espn_path.split("/", 1)
    return (
        f"{CORE_BASE}/{sport}/leagues/{slug}/events"
        f"?dates={int(season)}&limit=1000&page={int(page)}"
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


#: Errors that mean "the network or the CDN hiccuped", not "the request was
#: wrong". ``http.client.HTTPException`` is the important addition: a chunked
#: summary response can end mid-stream and raise ``IncompleteRead``, which is a
#: ``HTTPException`` — *not* an ``OSError`` or ``URLError`` — so it used to escape
#: the retry loop entirely and kill the backfill (live 2026-11-26:
#: ``IncompleteRead(397213 bytes read)`` on the first cold run).
RETRYABLE_FETCH_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    json.JSONDecodeError,
    OSError,
)

#: ESPN summary payloads are several hundred KB. Requesting gzip measurably
#: shortens the transfer, which is the main practical defence against the
#: mid-stream drops above.
FETCH_HEADERS = {
    "User-Agent": "bullion/0.1 (keyless ESPN reader; contact: operator)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
}


def _decode_body(raw: bytes, content_encoding: str) -> bytes:
    """Undo ``Content-Encoding: gzip``; tolerate servers that ignore it."""
    if "gzip" in (content_encoding or "").lower() or raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def http_fetch(url: str, *, timeout: float = 20.0, retries: int = 3) -> Any:
    """Fetch JSON with stdlib only. Raises :class:`EspnFetchError` after retries.

    Every attempt is a full retry: a failed download never leaves partial state,
    and a gzip body is decompressed before parsing. ``retries=3`` (four attempts
    with linear backoff) is enough for a CDN blip without turning a genuinely
    dead route into a multi-minute hang.
    """
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=FETCH_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                content_encoding = response.headers.get("Content-Encoding") or ""
            return json.loads(_decode_body(raw, content_encoding).decode("utf-8"))
        except RETRYABLE_FETCH_ERRORS as exc:
            last_error = exc
            if attempt < retries:
                delay = 1.5 * (attempt + 1)
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                    # A 429 carries its own pacing. Honour it (clamped) instead of
                    # blindly sleeping 1.5s and making the throttle worse.
                    try:
                        retry_after = exc.headers.get("Retry-After")
                    except AttributeError:
                        retry_after = None
                    try:
                        delay = min(max(float(retry_after), 1.0), 30.0)
                    except (TypeError, ValueError):
                        delay = 30.0
                    exc.close()
                time.sleep(delay)
    raise EspnFetchError(f"{url}: {last_error}") from last_error


@dataclass
class EspnClient:
    """Thin injectable client. Tests pass a fixture ``fetch``; live passes HTTP."""

    fetch: FetchFn = http_fetch

    def get(self, url: str) -> Any:
        payload = self.fetch(url)
        if not isinstance(payload, dict):
            raise EspnParseError(f"{url}: payload is not a JSON object")
        return payload

    def scoreboard(self, league: League, et_date: str) -> Dict[str, Any]:
        return self.get(scoreboard_url(league, et_date))

    def summary(self, league: League, event_id: str) -> Dict[str, Any]:
        return self.get(summary_url(league, event_id))

    def predictor(self, league: League, event_id: str) -> Dict[str, Any]:
        return self.get(predictor_url(league, event_id))

    @classmethod
    def live(cls, *, timeout: float = 12.0, retries: int = 2) -> "EspnClient":
        """Client for a scheduled tick: a tighter budget than the backfill.

        Worst case per dead call is ``12s*3 + 1.5s + 3s ~= 40s`` (vs ~80s at the
        backfill defaults), which keeps a degraded-ESPN day inside the job's
        wall clock instead of losing the cache save and the commit.
        """

        def fetch(url: str) -> Any:
            return http_fetch(url, timeout=timeout, retries=retries)

        return cls(fetch=fetch)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def nested(mapping: Any, *path: str) -> Any:
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def line_number(value: Any) -> Optional[float]:
    """Parse a handicap/total line. ``o6.5``/``u6.5`` prefixes are stripped."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if text[:1].lower() in {"o", "u", "+"} and len(text) > 1:
        text = text[1:] if text[0].lower() in {"o", "u"} else text
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def decimal_price(value: Any) -> Optional[float]:
    """American odds -> decimal. Returns ``None`` for anything unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        american = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    try:
        price = american_to_decimal(american)
    except (ValueError, ZeroDivisionError):
        return None
    return price if 1.0 < price <= 1000.0 else None


def _state(competition: Dict[str, Any]) -> str:
    return str(nested(competition, "status", "type", "state") or "")


def _verify_league_identity(payload: Dict[str, Any], league: League) -> None:
    """Refuse a payload whose ``leagues[]`` block names a different league.

    A scoreboard URL can resolve to a payload for another competition (ESPN
    serves ``basketball/womens-college-basketball`` and men's D1 from sibling
    ids, and a mistyped ``groups=`` returns the wrong division). The check is
    deliberately contradiction-only: an absent or empty ``leagues[]`` block is
    not evidence of anything, so it does not raise. Failing on field absence
    would quarantine every payload that omits an optional block.
    """
    expected = league.espn_league_id
    if expected is None:
        return
    identified = {
        str(entry.get("id"))
        for entry in (payload.get("leagues") or [])
        if isinstance(entry, dict) and entry.get("id") is not None
    }
    if identified and expected not in identified:
        raise EspnIdentityError(
            f"{league.key}: scoreboard payload identifies leagues "
            f"{sorted(identified)}, expected {expected!r}"
        )


#: Status names ESPN uses for an event that is finished as a fixture but never
#: played to a result. ``state`` alone cannot separate them from a final: canceled
#: and postponed events arrive as ``state="post"`` and, worse, frequently carry
#: ``completed=false`` with placeholder ``"0"`` scores. Grading those as a 0-0
#: final invents a result; refusing them forever strands the refresh queue.
_CANCELLED_TOKENS = ("CANCEL", "ABANDON", "FORFEIT")
_POSTPONED_TOKENS = ("POSTPON",)


def _competition_status(competition: Dict[str, Any]) -> Status:
    """Map one competition's ``status.type`` block to a :class:`Status`.

    Order matters. The *name* is checked first because ESPN reuses
    ``state="post"`` for a canceled or postponed fixture; ``completed is False``
    is then honoured only as an explicit boolean, because minimal payloads (and
    synthetic tests) omit ``completed`` on genuine finals and ``None`` is not
    evidence that the game was unplayed.
    """
    name = str(nested(competition, "status", "type", "name") or "").upper()
    state = _state(competition)
    if any(token in name for token in _CANCELLED_TOKENS):
        return Status.CANCELLED
    if any(token in name for token in _POSTPONED_TOKENS):
        return Status.POSTPONED
    if state == "post":
        if nested(competition, "status", "type", "completed") is False:
            return Status.UNKNOWN
        return Status.FINAL
    if state == "pre":
        return Status.SCHEDULED
    if state == "in":
        return Status.LIVE
    return Status.UNKNOWN


_TEAM_REF_RE = re.compile(r"/teams/(\d+)")


def team_ref_id(team: Any) -> Optional[str]:
    """Team id from an explicit ``id`` or a core-API ``.../teams/{id}`` ref."""
    if not isinstance(team, dict):
        return None
    if team.get("id") is not None:
        return str(team["id"])
    match = _TEAM_REF_RE.search(str(team.get("$ref") or ""))
    return match.group(1) if match else None


def sides(competition: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Home/away competitors via ``homeAway`` markers — never array order."""
    competitors = competition.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if home is None or away is None:
        raise EspnIdentityError(
            "competitors lack homeAway markers; refusing to infer sides from array order"
        )
    return home, away


def _competition(event: Dict[str, Any]) -> Dict[str, Any]:
    competitions = event.get("competitions") or []
    if not competitions:
        raise EspnParseError(f"event {event.get('id')} carries no competitions")
    return competitions[0]


def is_placeholder(competition: Dict[str, Any]) -> bool:
    for competitor in competition.get("competitors") or []:
        name = str(nested(competitor, "team", "displayName") or "").strip()
        team_id = team_ref_id(competitor.get("team") or {})
        if team_id and team_id.startswith("-"):
            return True
        if re.match(r"^(?:tbd|tba)(?:\s+#?\d+)?$|^(?:winner|loser)\s+of\b", name, re.I):
            return True
    return False


# ---------------------------------------------------------------------------
# Scoreboard parsing
# ---------------------------------------------------------------------------


@dataclass
class Scoreboard:
    games: List[Game] = field(default_factory=list)
    quotes: List[BookQuote] = field(default_factory=list)
    gaps: List[CoverageGap] = field(default_factory=list)


def parse_scoreboard(
    payload: Dict[str, Any], league: League, observed_utc: str
) -> Scoreboard:
    """Parse one scoreboard payload into games plus pre-tip named-book quotes."""
    _verify_league_identity(payload, league)
    observed = to_utc_z(observed_utc)
    result = Scoreboard()
    seen_events: set[str] = set()
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or not event.get("id"):
            continue
        event_id = str(event["id"])
        if event_id in seen_events:
            continue
        try:
            competition = _competition(event)
        except EspnParseError as exc:
            result.gaps.append(CoverageGap(league.key, "competition", str(exc), event_id))
            continue
        if is_placeholder(competition):
            result.gaps.append(
                CoverageGap(league.key, "competitor", "unresolved bracket placeholder", event_id)
            )
            continue
        try:
            game = _game_from_event(event, competition, league)
        except (EspnIdentityError, EspnParseError, ValueError) as exc:
            result.gaps.append(CoverageGap(league.key, "game", str(exc), event_id))
            continue
        seen_events.add(event_id)
        result.games.append(game)
        quotes, gaps = _quotes_from_event(event, competition, game, observed)
        result.quotes.extend(quotes)
        result.gaps.extend(gaps)
    return result


def scoreboard_finals(
    payload: Dict[str, Any], league: League
) -> Tuple[List[Game], List[CoverageGap]]:
    """Official finals straight off a scoreboard payload — nothing else.

    The summary route costs one request per game; a 130-fixture NCAA Saturday
    needs ~130 of them to learn scores the scoreboard already published. This
    parser reads the same payload the live scan already fetched and returns
    **only** ``post`` events: a ``pre``/``in`` event is skipped silently, never
    returned as a scheduled game, so a caller cannot mistake a fixture for a
    final.

    Missing ``linescores`` do not disqualify a final. Plenty of ESPN mid-major
    and small-conference scoreboards publish an official final score with no
    period breakdown, and discarding those would throw away most of the
    history the rating fit needs. Period *contradictions* are different: if a
    non-empty breakdown does not add up to the final, the periods are dropped
    and a gap is recorded so ``/summary`` can repair it later.
    """
    _verify_league_identity(payload, league)
    games: List[Game] = []
    gaps: List[CoverageGap] = []
    seen_events: set[str] = set()
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or not event.get("id"):
            continue
        event_id = str(event["id"])
        if event_id in seen_events:
            continue
        try:
            competition = _competition(event)
        except EspnParseError as exc:
            gaps.append(CoverageGap(league.key, "competition", str(exc), event_id))
            continue
        if _competition_status(competition) is not Status.FINAL:
            # Pre-tip, in-progress, or terminal-but-unplayed: this function's
            # contract is finals only, so a canceled 0-0 placeholder is skipped.
            continue
        if is_placeholder(competition):
            gaps.append(
                CoverageGap(league.key, "competitor", "unresolved bracket placeholder", event_id)
            )
            continue
        try:
            game = _game_from_event(event, competition, league)
            home, away = sides(competition)
            home_score = _score(home, event_id)
            away_score = _score(away, event_id)
        except (EspnIdentityError, EspnParseError, ValueError) as exc:
            gaps.append(CoverageGap(league.key, "game", str(exc), event_id))
            continue
        periods, period_gap = _periods(home, away, league, event_id)
        if period_gap:
            gaps.append(period_gap)
        if periods and (
            abs(sum(home_period for home_period, _ in periods) - home_score) > 0.5
            or abs(sum(away_period for _, away_period in periods) - away_score) > 0.5
        ):
            gaps.append(
                CoverageGap(
                    league.key,
                    "scoreboard_finals.linescore_mismatch",
                    "period sum contradicts final score; periods dropped",
                    event_id,
                )
            )
            periods = ()
        overtime = max(len(periods) - league.regulation_periods, 0) if periods else 0
        seen_events.add(event_id)
        game.status = Status.FINAL
        game.actual = ActualResult(
            home_score=home_score,
            away_score=away_score,
            periods=periods,
            overtime_periods=overtime,
        )
        games.append(game)
    return games, gaps


def _game_from_event(
    event: Dict[str, Any], competition: Dict[str, Any], league: League
) -> Game:
    home, away = sides(competition)
    home_id = team_ref_id(home.get("team") or {})
    away_id = team_ref_id(away.get("team") or {})
    if not home_id or not away_id:
        raise EspnIdentityError("competitor team reference cannot be resolved")
    raw_date = competition.get("date") or event.get("date")
    if not raw_date:
        raise EspnParseError("event carries no date")
    season = str(nested(event, "season", "year") or "")
    return Game(
        game_id=str(event["id"]),
        league=league.key,
        season=season,
        home_team_id=home_id,
        away_team_id=away_id,
        home_name=str(nested(home, "team", "displayName") or "").strip(),
        away_name=str(nested(away, "team", "displayName") or "").strip(),
        scheduled_utc=to_utc_z(raw_date),
        status=_competition_status(competition),
        neutral_site=bool(competition.get("neutralSite")),
        venue=str(nested(competition, "venue", "fullName") or ""),
    )


def _quotes_from_event(
    event: Dict[str, Any],
    competition: Dict[str, Any],
    game: Game,
    observed_utc: str,
) -> Tuple[List[BookQuote], List[CoverageGap]]:
    quotes: List[BookQuote] = []
    gaps: List[CoverageGap] = []
    offers = [offer for offer in (competition.get("odds") or []) if isinstance(offer, dict)]
    if not offers:
        gaps.append(CoverageGap(game.league, "pickcenter", "no odds offers on event", game.game_id))
        return quotes, gaps
    pre_match = game.status is Status.SCHEDULED and (
        parse_utc(observed_utc) < parse_utc(game.scheduled_utc)
    )
    declared_flags: List[Optional[bool]] = []
    for offer in offers:
        provider = offer.get("provider") or {}
        book = canonical_book(
            provider.get("displayName") or provider.get("name") or provider.get("id") or ""
        )
        if not book:
            gaps.append(
                CoverageGap(game.league, "pickcenter.provider", "offer without book", game.game_id)
            )
            continue
        common = {
            "game_id": game.game_id,
            "book": book,
            "observed_utc": observed_utc,
            "provider": "espn",
            "pre_match": pre_match,
        }
        market_quotes, market_gaps, market_declared = _book_market_quotes(
            offer, common, game.league
        )
        quotes.extend(market_quotes)
        gaps.extend(market_gaps)
        declared_flags.extend(market_declared)
    _normalize_main_lines(quotes, declared_flags)
    return quotes, gaps


def _is_soccer(league_key: str) -> bool:
    return league_key in LEAGUES and LEAGUES[league_key].sport is Sport.SOCCER


#: Canonical book names keyed by a punctuation/space-insensitive token.
#: LIVE_VERIFIED 2026-11-26: the same book arrives as ``"Draft Kings"`` on
#: NFL/CFB and ``"DraftKings"`` on soccer, and Bundesliga adds ``"Bet 365"``.
#: Without this the same price source would look like two different books in
#: coverage accounting, calibration segmentation, and receipts.
BOOK_ALIASES = {
    "draftkings": "DraftKings",
    "draftkingssportsbook": "DraftKings",
    "bet365": "Bet365",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "caesarsportsbook": "Caesars",
    "fanduel": "FanDuel",
    "espnbet": "ESPN BET",
    "pointsbet": "PointsBet",
    "betrivers": "BetRivers",
    "williamhill": "William Hill",
}


def canonical_book(raw: Any) -> str:
    """Canonical book name for an ESPN provider label; unknown names pass through.

    The raw label is preserved elsewhere (the probe records every provider
    string it saw), so normalisation here never destroys the audit trail.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return BOOK_ALIASES.get(re.sub(r"[^a-z0-9]", "", text.lower()), text)


def _phased_prices(
    offer: Dict[str, Any], market: str, sides: Tuple[str, ...]
) -> Tuple[Dict[str, float], Optional[str]]:
    """Side prices for one market, all taken from the **same** phase.

    Reading each side independently (``close`` else ``open``, per side) let an
    offer with only an away ``close`` and a home ``open`` produce a quote that
    mixed two snapshots — a book that never existed. A phase is used only when
    it prices every side the market needs; otherwise the market is not quoted.
    """
    for phase in ("close", "open"):
        prices: Dict[str, float] = {}
        for side in sides:
            price = decimal_price(nested(offer, market, side, phase, "odds"))
            if price is None:
                break
            prices[side] = price
        else:
            return prices, phase
    return {}, None


def _moneyline_prices(
    offer: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    prices, phase = _phased_prices(offer, "moneyline", ("home", "away"))
    home = prices.get("home")
    away = prices.get("away")
    if home is None or away is None:
        # Flat variant: one value per side, so there is no phase to mix.
        home = decimal_price(nested(offer, "homeTeamOdds", "moneyLine"))
        away = decimal_price(nested(offer, "awayTeamOdds", "moneyLine"))
        phase = None
    draw = None
    if phase is not None:
        draw = decimal_price(nested(offer, "moneyline", "draw", phase, "odds"))
    # Draw paths: proven for no sport yet. Candidate shapes are tried in a
    # fixed order and any single hit is used; absence stays absence. These are
    # single values, so they cannot mix a line phase either.
    if draw is None:
        draw = decimal_price(nested(offer, "drawOdds", "moneyLine"))
    if draw is None:
        draw = decimal_price(offer.get("drawOdds"))
    if draw is None:
        draw = decimal_price(nested(offer, "moneyline", "drawOdds"))
    return home, away, draw


def _phase_line(
    offer: Dict[str, Any], market: str, sides: Tuple[str, ...], phase: Optional[str]
) -> Optional[float]:
    if phase is None:
        return None
    for side in sides:
        line = line_number(nested(offer, market, side, phase, "line"))
        if line is not None:
            return line
    return None


def _spread_line(offer: Dict[str, Any], phase: Optional[str]) -> Optional[float]:
    """Home-oriented handicap taken from the phase the prices came from."""
    home = _phase_line(offer, "pointSpread", ("home",), phase)
    if home is not None:
        return home
    away = _phase_line(offer, "pointSpread", ("away",), phase)
    if away is not None:
        return -away
    return line_number(offer.get("spread"))


def _total_line(offer: Dict[str, Any], phase: Optional[str]) -> Optional[float]:
    line = _phase_line(offer, "total", ("over", "under"), phase)
    if line is not None:
        return line
    return line_number(offer.get("overUnder"))


def _is_split_line(line: Optional[float]) -> bool:
    """True for .25/.75 Asian split lines: two half-stakes, not a binary bet.

    A quarter line is half-pushed/half-won by construction, so grading it with
    the binary :func:`resolve_outcome` is wrong by design. Pricing it as binary
    is equally wrong. Skipping it with a gap is the only honest option until a
    split-stake settlement rule exists.
    """
    if line is None:
        return False
    try:
        value = float(line)
    except (TypeError, ValueError):
        return False
    return round(value * 4) % 2 != 0


def _declared_main(entry: Dict[str, Any], market: str, sides_wanted: Tuple[str, ...]) -> Optional[bool]:
    """A provider's own "this is the main line" flag, if it published one.

    ESPN has carried ``main``/``mainLine`` at both the market and the side/phase
    level across sports. Every shape is read defensively; an absent flag stays
    absent (``None``) so :func:`_normalize_main_lines` can choose deterministically
    instead of guessing which of two published lines the provider meant.
    """
    if market == "moneyline":
        return True
    containers = [entry.get(market)]
    for container in list(containers):
        if not isinstance(container, dict):
            continue
        for side in sides_wanted:
            side_block = container.get(side)
            if not isinstance(side_block, dict):
                continue
            containers.append(side_block)
            for phase in ("close", "open"):
                phase_block = side_block.get(phase)
                if isinstance(phase_block, dict):
                    containers.append(phase_block)
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("main", "mainLine", "isMain"):
            if key in container:
                return bool(container.get(key))
    return None


def _normalize_main_lines(
    quotes: List["BookQuote"], offers_main_flags: Optional[List[Optional[bool]]] = None
) -> None:
    """Exactly one ``main_line=True`` per ``(game, market, book)``.

    Downstream, ``main_line`` decides which single line a college spread/total
    mean is anchored to, and which line a card shows as the headline price. Two
    main lines in one group would make that choice depend on payload order.

    Order of preference inside a group: the provider's own declaration, else a
    deterministic rule (closest handicap to zero for spreads, the middle line
    for totals). Moneyline quotes carry no line, so every one of them is main.
    """
    flags = offers_main_flags or [None] * len(quotes)
    groups: Dict[Tuple[str, str, str], List[int]] = {}
    for index, quote in enumerate(quotes):
        groups.setdefault((quote.game_id, quote.market, quote.book), []).append(index)
    for (_game_id, market, _book), indexes in groups.items():
        if market == "moneyline":
            for index in indexes:
                quotes[index].main_line = True
            continue
        declared = [
            index
            for index in indexes
            if index < len(flags) and flags[index]
        ]
        if declared:
            chosen = declared[0]
        else:
            with_lines = [index for index in indexes if quotes[index].line is not None]
            if not with_lines:
                for index in indexes:
                    quotes[index].main_line = False
                continue
            if market == "spread":
                chosen = min(
                    with_lines,
                    key=lambda index: (abs(float(quotes[index].line)), float(quotes[index].line)),
                )
            else:
                ordered = sorted(float(quotes[i].line) for i in with_lines)
                middle = ordered[len(ordered) // 2]
                chosen = min(
                    with_lines,
                    key=lambda index: (abs(float(quotes[index].line) - middle), float(quotes[index].line)),
                )
        for index in indexes:
            quotes[index].main_line = index == chosen


def _book_market_quotes(
    entry: Dict[str, Any],
    common: Dict[str, Any],
    league_key: str,
) -> Tuple[List["BookQuote"], List["CoverageGap"], List[Optional[bool]]]:
    """Every recognised market in one book entry, under the fail-closed rules.

    Shared by the scoreboard ``odds`` path and the summary ``pickcenter`` path
    so a rule cannot hold on one and silently not on the other. The third
    element is the provider's own main-line declaration per emitted quote
    (``None`` when it published none); callers pass it to
    :func:`_normalize_main_lines` once the whole payload is parsed.
    """
    quotes: List["BookQuote"] = []
    gaps: List["CoverageGap"] = []
    declared: List[Optional[bool]] = []
    game_id = str(common.get("game_id") or "")
    three_way = _is_soccer(league_key)

    home_price, away_price, draw_price = _moneyline_prices(entry)
    prices = {
        key: price
        for key, price in (("home", home_price), ("away", away_price), ("draw", draw_price))
        if price is not None
    }
    if three_way and home_price is not None and draw_price is None:
        gaps.append(
            CoverageGap(
                league_key,
                "pickcenter.moneyline.draw",
                f"{common.get('book')}: three-way moneyline quote without a draw price",
                game_id,
            )
        )
    # A soccer moneyline without its draw price is a 2-way (draw-no-bet) price.
    # Grading it as 1X2 would settle level games as losses, so it is not quoted.
    if {"home", "away"} <= set(prices) and not (three_way and "draw" not in prices):
        quotes.append(
            BookQuote(
                **common,
                market="moneyline",
                side_prices=prices,
                main_line=True,
            )
        )
        declared.append(_declared_main(entry, "moneyline", ("home", "away")))

    spread_prices, spread_phase = _phased_prices(entry, "pointSpread", ("home", "away"))
    spread_line = _spread_line(entry, spread_phase)
    if spread_line is not None and {"home", "away"} <= set(spread_prices):
        # Every league, not just soccer: a quarter line splits the stake no
        # matter which sport quoted it, and a binary grader settles it wrong.
        if _is_split_line(spread_line):
            gaps.append(
                CoverageGap(
                    league_key,
                    "pickcenter.line.split",
                    f"{common.get('book')}: quarter spread line {spread_line} is a split stake",
                    game_id,
                )
            )
        else:
            quotes.append(
                BookQuote(**common, market="spread", line=spread_line, side_prices=spread_prices)
            )
            declared.append(_declared_main(entry, "pointSpread", ("home", "away")))

    total_prices, total_phase = _phased_prices(entry, "total", ("over", "under"))
    total_line = _total_line(entry, total_phase)
    if total_line is not None and {"over", "under"} <= set(total_prices):
        if _is_split_line(total_line):
            gaps.append(
                CoverageGap(
                    league_key,
                    "pickcenter.line.split",
                    f"{common.get('book')}: quarter total line {total_line} is a split stake",
                    game_id,
                )
            )
        else:
            quotes.append(
                BookQuote(**common, market="total", line=total_line, side_prices=total_prices)
            )
            declared.append(_declared_main(entry, "total", ("over", "under")))
    return quotes, gaps, declared


def quote_market_presence(quotes: Any) -> set:
    """Markets a quote set actually prices, judged by priced *sides*.

    A soccer moneyline quote can carry ``home``/``away``/``draw``; reading only
    ``quote.market`` reports the draw as unmeasured even when every book quotes
    it. LIVE_VERIFIED 2026-11-26: ``pickcenter[0].moneyline.draw.close.odds`` and
    ``pickcenter[0].drawOdds.moneyLine`` were present on 100% of sampled EPL,
    La Liga, Ligue 1 and Serie A fixtures while the probe reported 0% draw
    coverage, because it counted market names instead of prices.
    """
    presence: set = set()
    for quote in quotes or ():
        sides = set(getattr(quote, "side_prices", {}) or {})
        market = getattr(quote, "market", "")
        if market == "moneyline":
            if {"home", "away"} <= sides:
                presence.add("moneyline")
            if "draw" in sides:
                presence.add("moneyline_draw")
        elif market == "spread" and {"home", "away"} <= sides:
            presence.add("spread")
        elif market == "total" and {"over", "under"} <= sides:
            presence.add("total")
    return presence


# ---------------------------------------------------------------------------
# Summary / results parsing
# ---------------------------------------------------------------------------


def parse_summary(
    payload: Dict[str, Any], game: Game, league: League
) -> Tuple[Game, List[CoverageGap]]:
    """Return a copy of ``game`` with its official final, or raise.

    Raises :class:`EspnIdentityError` unless the summary is ``post``, carries
    two resolvable competitors that match the scoreboard game, and has numeric
    final scores. Missing linescores keep the final score but record a gap.
    """
    header = payload.get("header") or {}
    competitions = header.get("competitions") or []
    if not competitions:
        raise EspnParseError(f"event {game.game_id}: summary carries no competition")
    competition = competitions[0]
    status = _competition_status(competition)
    if status is not Status.FINAL:
        raise EspnIdentityError(
            f"event {game.game_id}: summary status={status.value!r}; result quarantined"
        )
    home, away = sides(competition)
    ids = {team_ref_id(home.get("team") or {}), team_ref_id(away.get("team") or {})}
    if ids != {game.home_team_id, game.away_team_id}:
        raise EspnIdentityError(
            f"event {game.game_id}: summary competitors {sorted(ids)} contradict "
            f"scoreboard teams {game.home_team_id}/{game.away_team_id}"
        )
    gaps: List[CoverageGap] = []
    home_score = _score(home, game.game_id)
    away_score = _score(away, game.game_id)
    periods, period_gap = _periods(home, away, league, game.game_id)
    if period_gap:
        gaps.append(period_gap)
    overtime = max(len(periods) - league.regulation_periods, 0) if periods else 0
    actual = ActualResult(
        home_score=home_score,
        away_score=away_score,
        periods=periods,
        overtime_periods=overtime,
    )
    return (
        Game(
            game_id=game.game_id,
            league=game.league,
            season=game.season,
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            home_name=game.home_name or str(nested(home, "team", "displayName") or "").strip(),
            away_name=game.away_name or str(nested(away, "team", "displayName") or "").strip(),
            scheduled_utc=game.scheduled_utc,
            status=Status.FINAL,
            neutral_site=game.neutral_site,
            venue=game.venue,
            actual=actual,
        ),
        gaps,
    )


def _score(competitor: Dict[str, Any], event_id: str) -> float:
    raw = competitor.get("score")
    if isinstance(raw, dict):  # some payloads nest {"value": ..., "displayValue": ...}
        raw = raw.get("value")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise EspnParseError(
            f"event {event_id}: competitor has no numeric final score"
        ) from None
    if not math.isfinite(value):
        raise EspnParseError(f"event {event_id}: non-finite final score")
    return value


def _linescores(competitor: Dict[str, Any]) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for element in competitor.get("linescores") or []:
        raw = element.get("value") if isinstance(element, dict) else element
        if raw is None and isinstance(element, dict):
            raw = element.get("displayValue")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(None)
    return values


def _periods(
    home: Dict[str, Any], away: Dict[str, Any], league: League, event_id: str
) -> Tuple[tuple[tuple[float, float], ...], Optional[CoverageGap]]:
    home_values = _linescores(home)
    away_values = _linescores(away)
    if not home_values and not away_values:
        return (), CoverageGap(league.key, "boxscore.linescores", "no period data", event_id)
    if len(home_values) != len(away_values):
        return (), CoverageGap(
            league.key,
            "boxscore.linescores",
            f"mismatched period counts home {len(home_values)} vs away {len(away_values)}",
            event_id,
        )
    if any(value is None for value in (*home_values, *away_values)):
        return (), CoverageGap(
            league.key, "boxscore.linescores", "unparseable period value", event_id
        )
    return (
        tuple((float(h), float(a)) for h, a in zip(home_values, away_values)),
        None,
    )


# ---------------------------------------------------------------------------
# Predictor (FPI/BPI-shaped) parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictorEvidence:
    """Normalised FPI/BPI evidence for one fixture.

    ``prob_home`` is conditioned on no tie whenever the payload reports a tie
    probability, because a two-way market has no draw selection to absorb that
    mass (LIVE_VERIFIED 2026-11-26: NFL ``pickcenter`` moneyline sides are
    ``home``/``away`` only, whereas every soccer offer carries an explicit
    ``draw`` side). ``prob_tie`` keeps the excluded mass visible so a receipt or
    coverage file never pretends the market was strictly two-outcome.
    """

    prob_home: float
    expected_home: Optional[float]
    expected_away: Optional[float]
    stats_side: str
    prob_tie: float = 0.0
    tie_rule: str = ""
    predicted_margin: Optional[float] = None


PROB_SUM_TOLERANCE = 0.01  # payload percentages
MARGIN_TOLERANCE = 0.25  # points


def _stats_by_name(statistics: Any) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for item in statistics or []:
        if not isinstance(item, dict) or item.get("name") is None:
            continue
        raw = item.get("value")
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            values[str(item["name"])] = float(raw)
        except (TypeError, ValueError):
            continue
    return values


def parse_predictor(payload: Dict[str, Any], game: Game) -> PredictorEvidence:
    """Normalise a predictor payload, reconciling sides by team id.

    ``homeTeam``/``awayTeam`` placement of ``statistics`` is not guaranteed
    (observed on ``awayTeam`` for basketball BPI), so the populated side is
    identified by its team ``$ref``/id and matched against the game's
    competitors. An unreconcilable side raises; home/away are never swapped to
    make data fit.
    """
    home_side = payload.get("homeTeam") or {}
    away_side = payload.get("awayTeam") or {}
    populated = [side for side in (home_side, away_side) if side.get("statistics")]
    if not populated:
        raise EspnParseError(f"event {game.game_id}: predictor carries no statistics")
    home_ref = team_ref_id(home_side.get("team") or {})
    away_ref = team_ref_id(away_side.get("team") or {})
    if home_ref and away_ref and home_ref == away_ref:
        raise EspnIdentityError(
            f"event {game.game_id}: predictor sides both reference team {home_ref}"
        )
    if len(populated) == 2:
        return _dual_sided(game, home_side, away_side, home_ref, away_ref)

    stats_side = populated[0]
    side_id = team_ref_id(stats_side.get("team") or {})
    if side_id is None:
        raise EspnIdentityError(
            f"event {game.game_id}: predictor stats side has no resolvable team ref"
        )
    if side_id == game.home_team_id:
        is_home = True
    elif side_id == game.away_team_id:
        is_home = False
    else:
        raise EspnIdentityError(
            f"event {game.game_id}: predictor stats side team {side_id} matches neither "
            f"competitor (home {game.home_team_id}, away {game.away_team_id})"
        )
    stats = _stats_by_name(stats_side.get("statistics"))
    prob_side, prob_opp, prob_tie = _probabilities(stats, game.game_id)
    expected_side = stats.get("teamExpectedPts")
    expected_opp = stats.get("oppExpectedPts")
    _check_margin(stats, expected_side, expected_opp, game.game_id)
    win_home, win_away = _two_way(
        prob_side if is_home else prob_opp,
        prob_opp if is_home else prob_side,
        prob_tie,
        game.game_id,
    )
    prob_home = win_home
    expected_home = (expected_side if is_home else expected_opp) if (
        expected_side is not None and expected_opp is not None
    ) else None
    expected_away = (expected_opp if is_home else expected_side) if (
        expected_side is not None and expected_opp is not None
    ) else None
    margin_side = stats.get("teamPredPtDiff")
    return PredictorEvidence(
        prob_home=prob_home,
        expected_home=expected_home,
        expected_away=expected_away,
        stats_side="home" if is_home else "away",
        prob_tie=prob_tie,
        tie_rule="push_conditional" if prob_tie else "",
        predicted_margin=None if margin_side is None else round(margin_side if is_home else -margin_side, 4),
    )


def _probabilities(stats: Dict[str, float], event_id: str) -> Tuple[float, float, float]:
    """(win, loss, tie) fractions from a predictor statistics block.

    LIVE_VERIFIED 2026-11-26: NFL payloads carry ``teamChanceTie`` alongside
    ``gameProjection``/``teamChanceLoss`` and sum to 100 only when the tie is
    included (``99.6480`` without it). CFB carries ``teamPctHfa`` instead of a
    tie, so a missing tie key is a genuine zero, not a malformed payload.
    """
    projection = stats.get("gameProjection")
    chance_loss = stats.get("teamChanceLoss")
    if projection is None or chance_loss is None:
        raise EspnParseError(f"event {event_id}: predictor missing win probabilities")
    raw_tie = stats.get("teamChanceTie")
    tie = 0.0 if raw_tie is None else float(raw_tie)
    for name, value in (
        ("gameProjection", projection),
        ("teamChanceLoss", chance_loss),
        ("teamChanceTie", tie),
    ):
        if not 0.0 <= value <= 100.0:
            raise EspnParseError(f"event {event_id}: {name} {value} outside 0-100")
    total = projection + chance_loss + tie
    if abs(total - 100.0) > PROB_SUM_TOLERANCE:
        raise EspnParseError(f"event {event_id}: predictor probabilities sum to {total:.4f}")
    return projection / 100.0, chance_loss / 100.0, tie / 100.0


def _two_way(
    win_side: float, win_opp: float, tie: float, event_id: str
) -> Tuple[float, float]:
    """Condition a three-outcome split onto the two priced sides.

    Only used when the payload reports tie mass. Both returned values are
    probabilities *given the game did not end level*, which is what a two-way
    market pays on; the excluded mass is reported separately as ``prob_tie``.
    """
    if win_side <= 0.0 or win_opp <= 0.0:
        raise EspnParseError(f"event {event_id}: predictor side probability is non-positive")
    denominator = win_side + win_opp
    return win_side / denominator, win_opp / denominator


def _check_margin(
    stats: Dict[str, float],
    expected_side: Optional[float],
    expected_opp: Optional[float],
    event_id: str,
) -> None:
    difference = stats.get("teamPredPtDiff")
    if (
        expected_side is not None
        and expected_opp is not None
        and difference is not None
        and abs((expected_side - expected_opp) - difference) > MARGIN_TOLERANCE
    ):
        raise EspnParseError(
            f"event {event_id}: expected-score margin contradicts teamPredPtDiff"
        )


def _dual_sided(
    game: Game,
    home_side: Dict[str, Any],
    away_side: Dict[str, Any],
    home_ref: Optional[str],
    away_ref: Optional[str],
) -> PredictorEvidence:
    if home_ref != game.home_team_id or away_ref != game.away_team_id:
        raise EspnIdentityError(
            f"event {game.game_id}: dual-sided predictor refs {home_ref}/{away_ref} "
            f"contradict scoreboard home/away {game.home_team_id}/{game.away_team_id}"
        )
    home_stats = _stats_by_name(home_side.get("statistics"))
    away_stats = _stats_by_name(away_side.get("statistics"))
    home_win, home_loss, home_tie = _probabilities(home_stats, game.game_id)
    away_win, away_loss, away_tie = _probabilities(away_stats, game.game_id)
    if abs(home_win + away_win + home_tie - 1.0) > PROB_SUM_TOLERANCE:
        raise EspnParseError(f"event {game.game_id}: home/away projections are not complementary")
    if abs(home_loss - away_win) > PROB_SUM_TOLERANCE:
        raise EspnParseError(f"event {game.game_id}: reciprocal probabilities contradict")
    if abs(home_tie - away_tie) > PROB_SUM_TOLERANCE:
        raise EspnParseError(f"event {game.game_id}: tie probabilities contradict")
    expected_home = home_stats.get("teamExpectedPts")
    expected_away = home_stats.get("oppExpectedPts")
    win_home, win_away = _two_way(home_win, away_win, home_tie, game.game_id)
    # ``teamPredPtDiff`` is the provider's own projected margin per side. It is
    # only exposed when the two sides reconcile (home ≈ -away); otherwise it is
    # dropped rather than guessed.
    predicted_margin = None
    home_margin = home_stats.get("teamPredPtDiff")
    away_margin = away_stats.get("teamPredPtDiff")
    if (
        home_margin is not None
        and away_margin is not None
        and abs(home_margin + away_margin) <= MARGIN_TOLERANCE
    ):
        predicted_margin = round(home_margin, 4)
    return PredictorEvidence(
        prob_home=win_home,
        expected_home=expected_home,
        expected_away=expected_away,
        stats_side="home",
        prob_tie=home_tie,
        tie_rule="push_conditional" if home_tie else "",
        predicted_margin=predicted_margin,
    )


# ---------------------------------------------------------------------------
# High-level discovery
# ---------------------------------------------------------------------------


def et_days_for_sast_day(day: str) -> List[str]:
    """US Eastern ``dates=`` values that a SAST calendar day's slate can occupy.

    SAST is UTC+2, US Eastern is UTC-5/-4, so a SAST day D spans
    ``D-1T22:00Z .. DT21:59Z`` — i.e. ET days **D-1 and D**. An earlier version
    anchored on SAST midnight and then subtracted a day, which requested ET
    days D-2/D-1: the 2026-11-26 live probe captured only completed games for
    exactly that reason, and no pre-match odds existed to measure.

    LIVE_VERIFIED 2026-11-26: the corrected mapping asks for the ET days whose
    events can carry a SAST-day timestamp; the probe now records the ET dates
    it requested so this can be re-checked without guessing.
    """
    from datetime import date, datetime, timedelta
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    sast = ZoneInfo("Africa/Johannesburg")
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=sast)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    first: date = start.astimezone(et).date()
    last: date = end.astimezone(et).date()
    days: List[str] = []
    cursor = first
    while cursor <= last:
        days.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return sorted(set(days))


def event_id_from_ref(ref: Any) -> Optional[str]:
    """Event id from a core-API event ``$ref``.

    LIVE_VERIFIED 2026-11-26: real season-index refs carry a query suffix, e.g.
    ``.../events/401856879?lang=en&region=us``. Splitting on ``/`` and asking
    ``isdigit()`` therefore returned nothing and the backfill would have
    silently stored zero historical games.
    """
    match = re.search(r"/events/(\d+)", str(ref or ""))
    return match.group(1) if match else None


def discover(client: EspnClient, league: League, day: str, observed_utc: str) -> Scoreboard:
    """Discover a SAST day's slate across the ET scoreboard days that can hold it."""
    combined = Scoreboard()
    seen: set[str] = set()
    for et_date in et_days_for_sast_day(day):
        payload = client.scoreboard(league, et_date)
        parsed = parse_scoreboard(payload, league, observed_utc)
        for game in parsed.games:
            if game.game_id in seen:
                continue
            seen.add(game.game_id)
            combined.games.append(game)
        combined.quotes.extend(parsed.quotes)
        combined.gaps.extend(parsed.gaps)
    return combined



# ---------------------------------------------------------------------------
# Summary pickcenter parsing
# ---------------------------------------------------------------------------

#: Containers inside a summary payload that can hold named-book offers. Ranked:
#: ``pickcenter`` is the book-quote list; ``odds`` is the alternate container
#: observed in the same live summaries.
PICKCENTER_CONTAINERS = ("pickcenter", "odds")


@dataclass
class PickcenterParse:
    quotes: List[BookQuote] = field(default_factory=list)
    gaps: List[CoverageGap] = field(default_factory=list)
    observed_entry_keys: List[str] = field(default_factory=list)


def summary_state(payload: Dict[str, Any]) -> str:
    """Status state of a summary's first competition (``pre``/``in``/``post``)."""
    competitions = nested(payload, "header", "competitions") or []
    if not competitions:
        return ""
    return _state(competitions[0])


def summary_status(payload: Dict[str, Any]) -> Status:
    """Full status of a summary's first competition.

    ``summary_state`` answers "is it pre/in/post"; this answers "is there a
    result", which is the only question grading may ask. A payload with no
    competition is ``Status.UNKNOWN`` — an absent header is not a final.
    """
    competitions = nested(payload, "header", "competitions") or []
    if not competitions:
        return Status.UNKNOWN
    return _competition_status(competitions[0])


def summary_scheduled_utc(payload: Dict[str, Any]) -> Optional[str]:
    """The tip time a summary reports, normalised to ``...Z`` (``None`` if absent)."""
    competitions = nested(payload, "header", "competitions") or []
    raw: Any = None
    if competitions:
        raw = competitions[0].get("date") if isinstance(competitions[0], dict) else None
    if not raw:
        raw = nested(payload, "header", "date")
    if not raw:
        return None
    try:
        return to_utc_z(raw)
    except (TypeError, ValueError):
        return None


def _provider_name(offer: Dict[str, Any]) -> str:
    provider = offer.get("provider")
    if isinstance(provider, dict):
        raw = (
            provider.get("displayName")
            or provider.get("name")
            or provider.get("id")
            or ""
        )
    elif isinstance(provider, str):
        raw = provider
    else:
        raw = offer.get("details") or offer.get("providerName") or ""
    return canonical_book(raw)


def parse_pickcenter(
    payload: Dict[str, Any], game: Game, observed_utc: str
) -> PickcenterParse:
    """Parse named-book quotes from a summary payload, failing closed.

    LIVE_VERIFIED 2026-11-26: every sampled summary (NHL, NFL, CFB, UCL) carried
    a ``pickcenter`` container, and the soccer summaries also carried an
    ``odds`` container plus a ``hasOdds`` flag — while the *scoreboard* odds
    arrays for the probed events were empty. So summary ``pickcenter`` is a real
    quote source and not an optional nicety.

    The *internal nesting* of a pickcenter entry is not yet captured, so this
    reads only extraction paths already proven on scoreboard offers
    (``moneyline|pointSpread|total`` -> side -> ``open|close`` -> ``odds``) plus
    the flat ``homeTeamOdds.moneyLine`` variant documented in the reference
    adapter. Anything else yields ``pickcenter.shape_unrecognised`` with the
    observed key list: evidence for the next probe, never a guess.
    """
    result = PickcenterParse()
    state = summary_state(payload)
    entries: List[Any] = []
    container_key = ""
    for key in PICKCENTER_CONTAINERS:
        candidate = payload.get(key)
        if isinstance(candidate, list) and candidate:
            entries = candidate
            container_key = key
            break
    if not entries:
        result.gaps.append(
            CoverageGap(game.league, "pickcenter.absent", "no pickcenter/odds entries", game.game_id)
        )
        return result
    pre_match = state == "pre" and parse_utc(observed_utc) < parse_utc(game.scheduled_utc)
    seen_keys: set[str] = set()
    declared_flags: List[Optional[bool]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keys = sorted(str(key) for key in entry)[:24]
        seen_keys.update(keys)
        book = _provider_name(entry)
        if not book:
            result.gaps.append(
                CoverageGap(
                    game.league,
                    f"{container_key}.provider",
                    "offer without a book identity",
                    game.game_id,
                )
            )
            continue
        common = {
            "game_id": game.game_id,
            "book": book,
            "observed_utc": to_utc_z(observed_utc),
            "provider": "espn",
            "pre_match": pre_match,
        }
        market_quotes, market_gaps, market_declared = _book_market_quotes(
            entry, common, game.league
        )
        result.quotes.extend(market_quotes)
        result.gaps.extend(market_gaps)
        declared_flags.extend(market_declared)
    # Normalise across the whole payload, not per entry: two entries for the
    # same book and market must not both claim to be the main line.
    _normalize_main_lines(result.quotes, declared_flags)
    result.observed_entry_keys = sorted(seen_keys)
    if not result.quotes:
        result.gaps.append(
            CoverageGap(
                game.league,
                f"{container_key}.shape_unrecognised",
                "no recognised price path in " + ", ".join(sorted(seen_keys)[:16]),
                game.game_id,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Probe report contract (the only evidence that live field coverage exists)
# ---------------------------------------------------------------------------

PROBE_SCHEMA = "bullion.espn_probe"
PROBE_MIN_COVERAGE = 0.5
#: A probe older than this stops granting ticketing permission (fail closed).
#: The probe workflow refreshes it Mon/Thu, so a healthy lane never trips this;
#: a lane whose evidence lapsed stops ticketing instead of quoting week-old books.
PROBE_MAX_AGE_HOURS = 168.0
#: Minimum pre-match events behind a coverage number before it may gate tickets.
PROBE_MIN_SAMPLE = 3


@dataclass(frozen=True)
class LanePermission:
    """Which markets may price a paper ticket, and why (or why not)."""

    ticket_markets: frozenset
    research_only: bool
    reason: str
    captured_utc: str = ""
    market_evidence: Dict[str, Any] = field(default_factory=dict)

    def allows(self, market: str) -> bool:
        return market in self.ticket_markets


def _coverage_status(entry: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """(coverage, status) for one market, defaulting to insufficient evidence."""
    if not isinstance(entry, dict):
        return None, "missing_market_entry"
    status = str(entry.get("status") or "")
    coverage = entry.get("event_coverage")
    sample = entry.get("sample_size")
    if status != "ok":
        return None, status or "unreported_status"
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
        return None, "missing_coverage"
    if not isinstance(sample, int) or sample < PROBE_MIN_SAMPLE:
        return None, "insufficient_sample"
    return float(coverage), "ok"


def lane_permission(
    report: Optional[Dict[str, Any]],
    league: League,
    *,
    as_of_utc: Optional[str] = None,
    max_age_hours: float = PROBE_MAX_AGE_HOURS,
) -> LanePermission:
    """Fail closed unless a real probe report proves named-book coverage.

    A lane with no captured probe is not "probably fine" -- it is unverified,
    so the mission's rule applies: no tickets. Below 50% pre-kickoff named-book
    coverage on a market keeps that market out of tickets; a lane with no market
    above the bar stays in ``research_only``: forecast and grade, never ticket.

    ``status``/``sample_size`` must come with the coverage number. The
    2026-11-26 probe reported ``0.0`` coverage over **zero** pre-match events
    (it had queried the wrong ET days), which is an absent measurement, not a
    measured zero; treating that as "0% covered" would have been a silent
    mislabel, and treating it as "fine" would have been a hallucination.
    """
    if not report:
        return LanePermission(frozenset(), True, "espn_probe_evidence_missing")
    if report.get("schema_version") != PROBE_SCHEMA:
        return LanePermission(frozenset(), True, "espn_probe_schema_mismatch")
    if report.get("mode") != "live":
        return LanePermission(frozenset(), True, "espn_probe_not_live")
    raw_entry = (report.get("leagues") or {}).get(league.key)
    entry = raw_entry if isinstance(raw_entry, dict) else {}
    # Per-league timestamp WINS, with the file stamp only as fallback. Never take
    # max(report, entry): a subset probe for ncaa_w would otherwise mask a stale
    # nhl entry whose own captured_utc is two weeks old.
    captured = str(entry.get("captured_utc") or report.get("captured_utc") or "")
    if as_of_utc is not None:
        stale = True
        if captured:
            try:
                age = parse_utc(as_of_utc) - parse_utc(captured)
                # A future stamp is a corrupt stamp, not fresh evidence.
                stale = age > timedelta(hours=max_age_hours) or age.total_seconds() < 0
            except (ValueError, TypeError, OverflowError):
                stale = True  # unparseable stamp: fail closed
        if stale:
            return LanePermission(
                frozenset(), True, "espn_probe_stale", captured_utc=captured
            )
    # ``as_of_utc is None`` (offline tests and fixtures): no age check, so a
    # committed historical probe never time-bombs in a future month.
    if not isinstance(raw_entry, dict):
        return LanePermission(frozenset(), True, "espn_probe_league_missing")
    coverage = (entry.get("coverage") or {}).get("named_book_pre_tip") or {}
    markets: set[str] = set()
    evidence: Dict[str, Any] = {}
    for market in ("moneyline", "total", "spread", "moneyline_draw"):
        value, status = _coverage_status(coverage.get(market) or {})
        evidence[market] = {
            "event_coverage": value,
            "status": status,
            "sample_size": (coverage.get(market) or {}).get("sample_size"),
        }
        if value is not None and value >= PROBE_MIN_COVERAGE:
            markets.add(market)
    if league.sport is Sport.SOCCER and "moneyline_draw" not in markets:
        markets.discard("moneyline")
    markets.discard("moneyline_draw")
    if not markets:
        statuses = {item["status"] for item in evidence.values()}
        if "ok" in statuses:
            reason = "named_book_coverage_below_50pct"
        elif evidence["moneyline"]["status"] == "no_pre_match_events":
            reason = "espn_probe_no_pre_match_events"
        else:
            reason = "espn_probe_coverage_insufficient_sample"
        return LanePermission(
            frozenset(), True, reason, captured_utc=captured, market_evidence=evidence
        )
    return LanePermission(
        frozenset(markets), False, "probe_covered", captured_utc=captured, market_evidence=evidence
    )


def game_from_summary(
    payload: Dict[str, Any], league: League, event_id: Optional[str] = None
) -> Game:
    """Build a Game from a summary header (used by historical backfill).

    Requires a ``post`` competition: a result row is never built from a live or
    partial payload. ``event_id`` is the id that was requested, and wins over an
    absent header id: a summary header's id shape is not proven for
    hockey/football/soccer, but the caller always knows which id it fetched.
    """
    competitions = nested(payload, "header", "competitions") or []
    if not competitions:
        raise EspnParseError("summary carries no competition")
    competition = competitions[0]
    status = _competition_status(competition)
    if status is not Status.FINAL:
        raise EspnIdentityError(
            f"summary status={status.value!r}; refusing to build a fixture"
        )
    home, away = sides(competition)
    home_id = team_ref_id(home.get("team") or {})
    away_id = team_ref_id(away.get("team") or {})
    if not home_id or not away_id:
        raise EspnIdentityError("summary competitor team reference cannot be resolved")
    scheduled = competition.get("date") or nested(payload, "header", "date")
    if not scheduled:
        raise EspnParseError("summary carries no date")
    event_id = str(
        event_id or payload.get("header", {}).get("id") or nested(competition, "id") or ""
    )
    if not event_id:
        raise EspnParseError("summary carries no event id")
    return Game(
        game_id=event_id,
        league=league.key,
        season=str(nested(payload, "header", "season", "year") or ""),
        home_team_id=home_id,
        away_team_id=away_id,
        home_name=str(nested(home, "team", "displayName") or "").strip(),
        away_name=str(nested(away, "team", "displayName") or "").strip(),
        scheduled_utc=to_utc_z(scheduled),
        status=Status.FINAL,
        neutral_site=bool(competition.get("neutralSite")),
        venue=str(nested(competition, "venue", "fullName") or ""),
    )
