"""Sport-aware domain types, league registry, and UTC/SAST time helpers.

Canonical timestamps are UTC ISO-8601 with a ``Z`` suffix. SAST
(``Africa/Johannesburg``, UTC+2, no DST) is a presentation-only timezone used
on cards and for publication cutoffs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

SAST = ZoneInfo("Africa/Johannesburg")
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_utc(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Naive inputs are rejected: guessing a timezone silently shifts a cutoff or
    a post-tip check, which is exactly the class of bug this engine forbids.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
    else:
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}")
    if moment.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware (got {value!r})")
    return moment.astimezone(UTC)


def to_utc_z(value: Any) -> str:
    """Canonical UTC ISO-8601 string, second precision, ``Z`` suffix."""
    return parse_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc() -> str:
    return to_utc_z(datetime.now(UTC))


def sast_day(value: Any) -> str:
    """SAST calendar day (the day a lane's card belongs to)."""
    return parse_utc(value).astimezone(SAST).strftime("%Y-%m-%d")


def format_sast(value: Any) -> str:
    """Human-readable SAST timestamp, e.g. ``Sat 2026-11-28 16:30 SAST``."""
    local = parse_utc(value).astimezone(SAST)
    return local.strftime("%a %Y-%m-%d %H:%M SAST")


def sast_cutoff_utc(day: str, hour_sast: int) -> str:
    """UTC instant of ``hour_sast`` on SAST calendar ``day``."""
    naive = datetime.strptime(day, "%Y-%m-%d").replace(hour=hour_sast, tzinfo=SAST)
    return to_utc_z(naive)


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    """Convert an American price to decimal. Zero is not a price."""
    value = float(american)
    if value == 0 or not math.isfinite(value):
        raise ValueError(f"not an American price: {american!r}")
    if value > 0:
        return 1.0 + value / 100.0
    return 1.0 + 100.0 / abs(value)


# ---------------------------------------------------------------------------
# Leagues
# ---------------------------------------------------------------------------


class Sport(str, Enum):
    HOCKEY = "hockey"
    FOOTBALL = "football"
    SOCCER = "soccer"
    BASKETBALL = "basketball"


class Status(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class League:
    """One priced lane. ``lane`` doubles as the ``localdata/<lane>/`` folder."""

    key: str
    label: str
    sport: Sport
    espn_path: str  # e.g. "hockey/nhl"
    cutoff_hour_sast: int
    requires_named_book: bool = True
    research_only: bool = False
    #: Whether ESPN exposes a ``/predictor`` (FPI/BPI) route for this lane.
    #: LIVE_VERIFIED 2026-11-26: NFL and CFB return a payload; hockey and every
    #: soccer league return ``HTTP 400 Bad Request``, so asking is a wasted
    #: request and a fake coverage gap. Basketball routes return a BPI payload;
    #: whether it carries both sides or one is not assumed (see parse_predictor).
    predictor: bool = False
    #: Regulation periods when the sport default is wrong for a lane: NCAA men's
    #: basketball is two 20-minute halves, not four quarters.
    regulation_periods_override: Optional[int] = None
    #: ESPN's own numeric league id, used only as a fail-closed identity check
    #: against a scoreboard payload's ``leagues[]`` block.
    espn_league_id: Optional[str] = None
    #: ESPN's ``groups=`` selector. NCAA basketball needs ``50`` (D1) or the
    #: scoreboard returns every division's games under one league id.
    scoreboard_groups: Optional[int] = None
    #: True for leagues whose ESPN season year is the calendar year that starts
    #: the season (WNBA). False for the straddling US college/pro seasons, whose
    #: ESPN ``season.year`` is the calendar year the season ENDS in.
    calendar_year_season: bool = False

    @property
    def lane(self) -> str:
        return self.key

    @property
    def receipt_league(self) -> str:
        return self.key.upper()

    @property
    def regulation_periods(self) -> int:
        if self.regulation_periods_override is not None:
            return self.regulation_periods_override
        return {
            Sport.HOCKEY: 3,
            Sport.FOOTBALL: 4,
            Sport.SOCCER: 2,
            Sport.BASKETBALL: 4,
        }[self.sport]

    @property
    def has_draw(self) -> bool:
        return self.sport is Sport.SOCCER


LEAGUES: Dict[str, League] = {
    league.key: league
    for league in (
        League("nhl", "NHL", Sport.HOCKEY, "hockey/nhl", cutoff_hour_sast=16),
        League(
            "nfl", "NFL", Sport.FOOTBALL, "football/nfl", cutoff_hour_sast=16, predictor=True
        ),
        League(
            "cfb",
            "CFB",
            Sport.FOOTBALL,
            "football/college-football",
            cutoff_hour_sast=16,
            predictor=True,
        ),
        League("eng.1", "Premier League", Sport.SOCCER, "soccer/eng.1", cutoff_hour_sast=12),
        League("esp.1", "LaLiga", Sport.SOCCER, "soccer/esp.1", cutoff_hour_sast=12),
        League("ger.1", "Bundesliga", Sport.SOCCER, "soccer/ger.1", cutoff_hour_sast=12),
        League("ita.1", "Serie A", Sport.SOCCER, "soccer/ita.1", cutoff_hour_sast=12),
        League("fra.1", "Ligue 1", Sport.SOCCER, "soccer/fra.1", cutoff_hour_sast=12),
        League("uefa.champions", "UEFA Champions League", Sport.SOCCER, "soccer/uefa.champions",
               cutoff_hour_sast=12),
        # US basketball. Tips are 19:00-20:00 ET, i.e. 02:00-03:00 SAST on the
        # SAST day the card belongs to, so the 20:00 SAST cutoff is only ever
        # pulled *earlier* by the tip-minus-two-hours rule.
        League(
            "nba",
            "NBA",
            Sport.BASKETBALL,
            "basketball/nba",
            cutoff_hour_sast=20,
            predictor=True,
            espn_league_id="46",
        ),
        League(
            "ncaa_m",
            "NCAA Men D1",
            Sport.BASKETBALL,
            "basketball/mens-college-basketball",
            cutoff_hour_sast=20,
            predictor=True,
            espn_league_id="41",
            scoreboard_groups=50,
            regulation_periods_override=2,
        ),
        League(
            "ncaa_w",
            "NCAA Women D1",
            Sport.BASKETBALL,
            "basketball/womens-college-basketball",
            cutoff_hour_sast=20,
            predictor=True,
            #: Women's D1 is id 54. 42 is the men's NIT-era id some older
            #: references still carry; using it here would silently probe the
            #: wrong competition and the identity guard would never fire.
            espn_league_id="54",
            scoreboard_groups=50,
        ),
        League(
            "wnba",
            "WNBA",
            Sport.BASKETBALL,
            "basketball/wnba",
            cutoff_hour_sast=20,
            predictor=True,
            espn_league_id="59",
            calendar_year_season=True,
        ),
    )
}

SOCCER_LEAGUES = tuple(key for key, item in LEAGUES.items() if item.sport is Sport.SOCCER)
BASKETBALL_LEAGUES = tuple(
    key for key, item in LEAGUES.items() if item.sport is Sport.BASKETBALL
)


def espn_season_keys(league: League, requested) -> List[int]:
    """ESPN season years to ask for, newest first.

    ESPN numbers a straddling US season by the calendar year it *ends* in, so
    asking for 2026 in November 2026 returns the 2025-26 season that already
    finished. Basketball therefore expands each requested year to ``{y, y+1}``
    so the live season is included; the WNBA and every other lane keep the plain
    request, because their ``season.year`` is the year they play in.

    Lives here rather than in a script because the backfill and the probe both
    need it, and a probe that asks for the finished season reports on a season
    that no lane will price.
    """
    years = {int(year) for year in requested}
    if league.sport is not Sport.BASKETBALL or league.calendar_year_season:
        return sorted(years, reverse=True)
    expanded: set = set()
    for year in years:
        expanded.update({year, year + 1})
    return sorted(expanded, reverse=True)


def get_league(key: str) -> League:
    try:
        return LEAGUES[key.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown league {key!r}; known: {sorted(LEAGUES)}") from None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ActualResult:
    """Official final score plus whatever native period evidence existed."""

    home_score: float
    away_score: float
    periods: tuple[tuple[float, float], ...] = ()
    overtime_periods: int = 0
    source: str = "espn"

    @property
    def total(self) -> float:
        return self.home_score + self.away_score

    @property
    def margin(self) -> float:
        return self.home_score - self.away_score

    def regulation_scores(self) -> Optional[tuple[float, float]]:
        """Sum of the non-overtime periods, or ``None`` without period data."""
        regulation = len(self.periods) - self.overtime_periods
        if regulation < 1 or self.overtime_periods < 0:
            return None
        used = self.periods[:regulation]
        return (sum(p[0] for p in used), sum(p[1] for p in used))


@dataclass
class Game:
    game_id: str
    league: str
    season: str
    home_team_id: str
    away_team_id: str
    home_name: str
    away_name: str
    scheduled_utc: str
    status: Status = Status.SCHEDULED
    neutral_site: bool = False
    venue: str = ""
    actual: Optional[ActualResult] = None

    @property
    def final(self) -> bool:
        return self.status is Status.FINAL and self.actual is not None

    def side_name(self, selection: str) -> str:
        if selection == "home":
            return self.home_name
        if selection == "away":
            return self.away_name
        return selection


@dataclass
class BookQuote:
    """One named book's full-game quote. ``side_prices`` keys are selections."""

    game_id: str
    book: str
    market: str  # "moneyline" | "spread" | "total"
    observed_utc: str
    side_prices: Dict[str, float]
    line: Optional[float] = None  # home-oriented handicap, or total points
    provider: str = "espn"
    pre_match: bool = False
    main_line: bool = True
    provenance: str = "named_book"

    def price_for(self, selection: str) -> Optional[float]:
        price = self.side_prices.get(selection)
        return float(price) if isinstance(price, (int, float)) and price > 1.0 else None

    def require_pre_tip(self) -> None:
        if not self.pre_match:
            raise ValueError(
                f"post-tip quote rejected: {self.book} {self.market} observed "
                f"{self.observed_utc} is not pre-match"
            )


@dataclass
class CoverageGap:
    """Machine-readable statement that a field/market was absent or unusable."""

    league: str
    field: str
    reason: str
    game_id: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "league": self.league,
            "field": self.field,
            "reason": self.reason,
            "game_id": self.game_id,
        }


@dataclass
class Forecast:
    """A model expectation for one exact wager selection."""

    game_id: str
    league: str
    scheduled_utc: str
    home_name: str
    away_name: str
    market: str  # moneyline | spread | total
    selection: str  # home | away | draw | over | under
    raw_probability: float
    source: str  # model surface tag, e.g. "nhl_poisson"
    period: str = "full_game"
    line: Optional[float] = None
    push_probability: float = 0.0
    expected_margin: Optional[float] = None
    expected_total: Optional[float] = None
    alternative_probabilities: Dict[str, float] = field(default_factory=dict)
    book: Optional[BookQuote] = None
    calibrated_probability: Optional[float] = None
    calibration: Dict[str, Any] = field(default_factory=dict)
    eligibility: Dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("raw_probability", "push_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability (got {value!r})")
        if self.calibrated_probability is not None:
            value = self.calibrated_probability
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("calibrated_probability must be a finite probability")
        if self.book is not None and not self.book.pre_match:
            raise ValueError("a forecast cannot carry a post-tip book observation")

    @property
    def decimal_price(self) -> Optional[float]:
        return self.book.price_for(self.selection) if self.book else None

    @property
    def forecast_id(self) -> str:
        """Stable wager identity: the selection and the model, never the price.

        The book used to be part of this hash, but the same forecast is recorded
        *before* a book is attached and frozen *after*, so it produced two ids
        for one observation — one in ``forecast_snapshots``, a different one in
        the receipt's ``candidate_pool``. Restoring history from a receipt then
        duplicated every row it meant to heal. Price is evidence, not identity:
        it lives on the receipt row, where ``price_evidence`` already records
        the book and the observation time.
        """
        return forecast_id_for(
            league=self.league,
            game_id=self.game_id,
            market=self.market,
            period=self.period,
            selection=self.selection,
            line=self.line,
            source=self.source,
        )

    @property
    def wager_key(self) -> tuple:
        return (self.league, self.game_id, self.market, self.period, self.selection, self.line)


def forecast_id_for(
    *,
    league: str,
    game_id: str,
    market: str,
    period: str,
    selection: str,
    line: Any,
    source: str,
) -> str:
    """Hash a wager identity: ``(league, game, market, period, selection, line, source)``."""
    identity = "|".join(
        [
            str(league),
            str(game_id),
            str(market),
            str(period),
            str(selection),
            "" if line is None else f"{float(line):g}",
            str(source),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def utc_minus_hours(value: Any, hours: float) -> str:
    return to_utc_z(parse_utc(value) - timedelta(hours=hours))
