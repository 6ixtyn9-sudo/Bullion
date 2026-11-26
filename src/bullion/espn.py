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

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
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
    limit = 500 if league.sport is Sport.FOOTBALL else 100
    return f"{SITE_BASE}/{sport}/{slug}/scoreboard?dates={et_date}&limit={limit}"


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


def http_fetch(url: str, *, timeout: float = 20.0, retries: int = 2) -> Any:
    """Fetch JSON with stdlib only. Raises :class:`EspnFetchError` after retries."""
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "bullion/0.1 (keyless ESPN reader; contact: operator)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            return json.loads(payload.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
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


_STATUS = {"pre": Status.SCHEDULED, "in": Status.LIVE, "post": Status.FINAL}

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
        status=_STATUS.get(_state(competition), Status.UNKNOWN),
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
    for offer in offers:
        provider = offer.get("provider") or {}
        book = str(
            provider.get("displayName") or provider.get("name") or provider.get("id") or ""
        ).strip()
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
        home_price, away_price, draw_price = _moneyline_prices(offer)
        if draw_price is None and game.league and _is_soccer(game.league) and home_price:
            gaps.append(
                CoverageGap(
                    game.league,
                    "pickcenter.moneyline.draw",
                    f"{book}: three-way moneyline quote without a draw price",
                    game.game_id,
                )
            )
        prices = {
            key: price
            for key, price in (
                ("home", home_price),
                ("away", away_price),
                ("draw", draw_price),
            )
            if price is not None
        }
        if {"home", "away"} <= set(prices):
            quotes.append(BookQuote(**common, market="moneyline", side_prices=prices))

        spread_price = _side_prices(offer, "pointSpread")
        home_line = line_number(nested(offer, "pointSpread", "home", "close", "line"))
        if home_line is None:
            home_line = line_number(nested(offer, "pointSpread", "home", "open", "line"))
        if home_line is None:
            away_line = line_number(nested(offer, "pointSpread", "away", "close", "line"))
            if away_line is None:
                away_line = line_number(nested(offer, "pointSpread", "away", "open", "line"))
            home_line = -away_line if away_line is not None else None
        if home_line is not None and {"home", "away"} <= set(spread_price):
            quotes.append(
                BookQuote(
                    **common,
                    market="spread",
                    line=home_line,
                    side_prices=spread_price,
                )
            )

        total_prices = _side_prices(offer, "total")
        total_line = line_number(nested(offer, "total", "over", "close", "line"))
        if total_line is None:
            total_line = line_number(nested(offer, "total", "under", "close", "line"))
        if total_line is None:
            total_line = line_number(offer.get("overUnder"))
        if total_line is not None and {"over", "under"} <= set(total_prices):
            quotes.append(
                BookQuote(
                    **common,
                    market="total",
                    line=total_line,
                    side_prices=total_prices,
                )
            )
    return quotes, gaps


def _is_soccer(league_key: str) -> bool:
    return league_key in LEAGUES and LEAGUES[league_key].sport is Sport.SOCCER


def _price_at(offer: Dict[str, Any], market: str, side: str) -> Optional[float]:
    for phase in ("close", "open"):
        price = decimal_price(nested(offer, market, side, phase, "odds"))
        if price is not None:
            return price
    return None


def _moneyline_prices(
    offer: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    home = _price_at(offer, "moneyline", "home")
    away = _price_at(offer, "moneyline", "away")
    if home is None:
        home = decimal_price(nested(offer, "homeTeamOdds", "moneyLine"))
    if away is None:
        away = decimal_price(nested(offer, "awayTeamOdds", "moneyLine"))
    # Draw paths: proven for no sport yet. Candidate shapes are tried in a
    # fixed order and any single hit is used; absence stays absence.
    draw = _price_at(offer, "moneyline", "draw")
    for key in ("drawOdds",):
        if draw is None:
            draw = decimal_price(offer.get(key))
    if draw is None:
        draw = decimal_price(nested(offer, "moneyline", "drawOdds"))
    return home, away, draw


def _side_prices(offer: Dict[str, Any], market: str) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    candidate_sides = ("home", "away") if market == "pointSpread" else ("over", "under")
    for side in candidate_sides:
        price = _price_at(offer, market, side)
        if price is not None:
            prices[side] = price
    return prices


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
    state = _state(competition)
    if state != "post":
        raise EspnIdentityError(
            f"event {game.game_id}: summary state={state!r}; result quarantined"
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
    prob_home: float
    expected_home: Optional[float]
    expected_away: Optional[float]
    stats_side: str


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
    prob_side, prob_opp = _probabilities(stats, game.game_id)
    expected_side = stats.get("teamExpectedPts")
    expected_opp = stats.get("oppExpectedPts")
    _check_margin(stats, expected_side, expected_opp, game.game_id)
    prob_home = prob_side if is_home else prob_opp
    expected_home = (expected_side if is_home else expected_opp) if (
        expected_side is not None and expected_opp is not None
    ) else None
    expected_away = (expected_opp if is_home else expected_side) if (
        expected_side is not None and expected_opp is not None
    ) else None
    return PredictorEvidence(
        prob_home=prob_home,
        expected_home=expected_home,
        expected_away=expected_away,
        stats_side="home" if is_home else "away",
    )


def _probabilities(stats: Dict[str, float], event_id: str) -> Tuple[float, float]:
    projection = stats.get("gameProjection")
    chance_loss = stats.get("teamChanceLoss")
    if projection is None or chance_loss is None:
        raise EspnParseError(f"event {event_id}: predictor missing win probabilities")
    for name, value in (("gameProjection", projection), ("teamChanceLoss", chance_loss)):
        if not 0.0 <= value <= 100.0:
            raise EspnParseError(f"event {event_id}: {name} {value} outside 0-100")
    if abs(projection + chance_loss - 100.0) > PROB_SUM_TOLERANCE:
        raise EspnParseError(
            f"event {event_id}: predictor probabilities sum to {projection + chance_loss:.4f}"
        )
    return projection / 100.0, chance_loss / 100.0


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
    home_projection, home_loss = _probabilities(home_stats, game.game_id)
    away_projection, away_loss = _probabilities(away_stats, game.game_id)
    if abs(home_projection + away_projection - 1.0) > PROB_SUM_TOLERANCE:
        raise EspnParseError(f"event {game.game_id}: home/away projections are not complementary")
    if abs(home_loss - away_projection) > PROB_SUM_TOLERANCE:
        raise EspnParseError(f"event {game.game_id}: reciprocal probabilities contradict")
    expected_home = home_stats.get("teamExpectedPts")
    expected_away = home_stats.get("oppExpectedPts")
    return PredictorEvidence(
        prob_home=home_projection,
        expected_home=expected_home,
        expected_away=expected_away,
        stats_side="home",
    )


# ---------------------------------------------------------------------------
# High-level discovery
# ---------------------------------------------------------------------------


def et_days_for_sast_day(day: str) -> List[str]:
    """US Eastern ``dates=`` values that can contain a SAST day's slate.

    SAST is UTC+2; US Eastern is UTC-5/-4. A SAST morning European kickoff and
    a SAST-morning US night game therefore sit on two different ET days, so
    both are requested.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    start = datetime.strptime(day, "%Y-%m-%d").replace(hour=0, tzinfo=ZoneInfo("Africa/Johannesburg"))
    days = []
    for offset in (-1, 0):
        local = (start + timedelta(days=offset)).astimezone(et)
        days.append(local.strftime("%Y%m%d"))
    return sorted(set(days))


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
# Probe report contract (the only evidence that live field coverage exists)
# ---------------------------------------------------------------------------

PROBE_SCHEMA = "bullion.espn_probe"
PROBE_MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class LanePermission:
    """Which markets may price a paper ticket, and why (or why not)."""

    ticket_markets: frozenset
    research_only: bool
    reason: str

    def allows(self, market: str) -> bool:
        return market in self.ticket_markets


def lane_permission(report: Optional[Dict[str, Any]], league: League) -> LanePermission:
    """Fail closed unless a real probe report proves named-book coverage.

    A lane with no captured probe is not "probably fine" -- it is unverified,
    so the mission's rule applies: no tickets. Below 50% pre-kickoff named-book
    coverage on a market keeps that market (and a lane with none) in
    ``research_only``: forecast and grade, never ticket.
    """
    if not report:
        return LanePermission(frozenset(), True, "espn_probe_evidence_missing")
    if report.get("schema_version") != PROBE_SCHEMA:
        return LanePermission(frozenset(), True, "espn_probe_schema_mismatch")
    if report.get("mode") != "live":
        return LanePermission(frozenset(), True, "espn_probe_not_live")
    entry = (report.get("leagues") or {}).get(league.key)
    if not isinstance(entry, dict):
        return LanePermission(frozenset(), True, "espn_probe_league_missing")
    coverage = (entry.get("coverage") or {}).get("named_book_pre_tip") or {}
    markets: set[str] = set()
    moneyline = (coverage.get("moneyline") or {}).get("event_coverage") or 0.0
    if moneyline >= PROBE_MIN_COVERAGE:
        markets.add("moneyline")
    total = (coverage.get("total") or {}).get("event_coverage") or 0.0
    if total >= PROBE_MIN_COVERAGE:
        markets.add("total")
    spread = (coverage.get("spread") or {}).get("event_coverage") or 0.0
    if spread >= PROBE_MIN_COVERAGE:
        markets.add("spread")
    if league.sport is Sport.SOCCER:
        draw = (coverage.get("moneyline_draw") or {}).get("event_coverage") or 0.0
        if draw < PROBE_MIN_COVERAGE:
            markets.discard("moneyline")
    if not markets:
        return LanePermission(frozenset(), True, "named_book_coverage_below_50pct")
    return LanePermission(frozenset(markets), False, "probe_covered")


def game_from_summary(payload: Dict[str, Any], league: League) -> Game:
    """Build a Game from a summary header (used by historical backfill).

    Requires a ``post`` competition: a result row is never built from a live or
    partial payload.
    """
    competitions = nested(payload, "header", "competitions") or []
    if not competitions:
        raise EspnParseError("summary carries no competition")
    competition = competitions[0]
    if _state(competition) != "post":
        raise EspnIdentityError(
            f"summary state={_state(competition)!r}; refusing to build a fixture"
        )
    home, away = sides(competition)
    home_id = team_ref_id(home.get("team") or {})
    away_id = team_ref_id(away.get("team") or {})
    if not home_id or not away_id:
        raise EspnIdentityError("summary competitor team reference cannot be resolved")
    scheduled = competition.get("date") or nested(payload, "header", "competitions")
    if not scheduled:
        raise EspnParseError("summary carries no date")
    event_id = str(payload.get("header", {}).get("id") or nested(competition, "id") or "")
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
