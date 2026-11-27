"""Read-only diagnostic of the rating fitter and the NHL moneyline model.

Answers one question with numbers instead of opinion: `nhl_poisson_v1` scores a
Brier of 0.2507 (skill -0.0007) over the backfilled NHL finals — is that missing
signal, or a broken estimator? The 2026-11-27 review suspected the latter and
named two defects in :func:`bullion.engines.fit_ratings`:

1. ``home_factor = home_goals / home_expected``, where ``home_expected`` already
   includes ``home_factor``. The update is therefore ``h <- K / h``, a two-cycle
   whose low phase is clamped to 1.0 — and with the shipped ``iterations=25``
   (odd) the fit lands on exactly 1.0, i.e. no home advantage at all.
2. ``attack * (target / attack) ** 0.5``: because ``target`` is itself
   proportional to ``1 / attack``, the fixed point of that update is
   ``sqrt(scored / C)``, i.e. the square root of the true rating.

This script measures both, then runs a chronological ablation of candidate
fixes. It never writes: the warehouse is opened ``mode=ro`` with
``PRAGMA query_only``, no engine, gate or schema code is touched, and every
candidate fix is a local reimplementation inside this file.

Deliberate scope limits, so the results cannot be read as more than they are:

* tuning happens on the earlier split only, and the later split is reported
  untouched;
* the leak rule is strict (fit on finals with SAST day < the scored day), the
  same firewall the seeder uses;
* no gate is read or relaxed anywhere — the gates are the subject of the
  measurement, not part of it.

Usage::

    python3 scripts/diagnose_engines.py --db localdata/bullion.db
    python3 scripts/diagnose_engines.py --db … --lanes nhl --fast --stride 1
    python3 scripts/diagnose_engines.py --db … --json > memo.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bullion.calibration import fit_platt, logit, sigmoid  # noqa: E402
from bullion.engines import (  # noqa: E402
    Result,
    decay_weight,
    fit_ratings,
    normal_cdf,
    rates_for,
)
from bullion.engines import nhl as nhl_engine  # noqa: E402
from bullion.engines import soccer as soccer_engine  # noqa: E402
from bullion.engines.football import SIGMA_PRIORS, residual_sigmas  # noqa: E402
from bullion.models import sast_day  # noqa: E402

SOCCER_LANES = frozenset({"eng.1", "esp.1", "ger.1", "ita.1", "fra.1"})
SEED_LANES = ("nhl", "nfl", "cfb", "eng.1", "esp.1", "ger.1", "ita.1", "fra.1")
MIN_HISTORY = 15
PLATT_MIN_ROWS = 30
TRUTH_HOME_FACTOR_NOTE = 1.10  # NHL-ish; only used as a reference in the memo


# ---------------------------------------------------------------------------
# Warehouse access (read-only by construction)
# ---------------------------------------------------------------------------


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open the warehouse so that a write is impossible, not merely unlikely."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    if not conn.execute("PRAGMA query_only").fetchone()[0]:
        raise RuntimeError("could not put the warehouse into query_only mode")
    return conn


def _regulation_scores(periods: Any, overtime_periods: Any) -> Optional[Tuple[float, float]]:
    """Sum of the non-overtime periods, mirroring ``ActualResult.regulation_scores``."""
    try:
        parsed = json.loads(periods or "[]")
    except (TypeError, ValueError):
        return None
    overtime = int(overtime_periods or 0)
    usable = len(parsed) - overtime
    if usable < 1 or overtime < 0:
        return None
    used = parsed[:usable]
    try:
        return (
            float(sum(pair[0] for pair in used)),
            float(sum(pair[1] for pair in used)),
        )
    except (TypeError, ValueError, IndexError):
        return None


@dataclass(frozen=True)
class FinalGame:
    game_id: str
    league: str
    day: str
    home_id: str
    away_id: str
    home_score: float
    away_score: float
    regulation: Optional[Tuple[float, float]]

    @property
    def outcome(self) -> int:
        return 1 if self.home_score > self.away_score else 0

    def result(self, *, regulation: bool = False, cap: Optional[float] = None) -> Optional[Result]:
        home, away = self.home_score, self.away_score
        if regulation:
            if self.regulation is None:
                return None
            home, away = self.regulation
        if cap is not None:
            home, away = min(home, cap), min(away, cap)
        return Result(
            home_id=self.home_id,
            away_id=self.away_id,
            home_score=float(home),
            away_score=float(away),
            day=self.day,
        )


def load_finals(conn: sqlite3.Connection, lane: str) -> List[FinalGame]:
    rows = conn.execute(
        """
        SELECT game_id, league, home_team_id, away_team_id, scheduled_utc,
               home_score, away_score, periods_json, overtime_periods
        FROM games
        WHERE league=? AND status='final' AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        ORDER BY scheduled_utc, game_id
        """,
        (lane,),
    ).fetchall()
    finals: List[FinalGame] = []
    for row in rows:
        finals.append(
            FinalGame(
                game_id=str(row["game_id"]),
                league=str(row["league"]),
                day=sast_day(row["scheduled_utc"]),
                home_id=str(row["home_team_id"]),
                away_id=str(row["away_team_id"]),
                home_score=float(row["home_score"]),
                away_score=float(row["away_score"]),
                regulation=_regulation_scores(
                    row["periods_json"], row["overtime_periods"]
                ),
            )
        )
    return finals


# ---------------------------------------------------------------------------
# Candidate estimators (local reimplementations; production code untouched)
# ---------------------------------------------------------------------------


def fit_home_factor_only(
    results: Sequence[Result], *, reference_day: str, **_: Any
) -> Dict[str, Any]:
    """(A) Production ratings, with the home factor refitted honestly.

    Keeps the shipped ``attack``/``defence`` exactly as they are, so this
    isolates the home-advantage defect from the compression defect.
    """
    ratings = dict(fit_ratings(results, reference_day=reference_day))
    rows = [row for row in results if row.day < reference_day]
    weights = [decay_weight(row.day, reference_day, 120.0) for row in rows]
    goals = sum(w * row.home_score for w, row in zip(weights, rows))
    expected = sum(
        w
        * float(ratings["league_rate"])
        * ratings["attack"][row.home_id]
        * ratings["defence"][row.away_id]
        for w, row in zip(weights, rows)
    )
    home = 1.0
    if expected > 0:
        home = float(ratings.get("home_factor") or 1.0)
        for _ in range(40):
            home = max(1.0, min(1.25, home * ((goals / expected) / home) ** 0.5))
    ratings["home_factor"] = home
    return ratings


def _b2b_flags(results: Sequence[Result]) -> Dict[Tuple[str, str], float]:
    """(team, day) -> 1.0 when the team played the previous day, else 0.0."""
    last: Dict[str, str] = {}
    flags: Dict[Tuple[str, str], float] = {}
    for row in sorted(results, key=lambda r: r.day):
        for team in (row.home_id, row.away_id):
            previous = last.get(team)
            flags[(team, row.day)] = _is_back_to_back(previous, row.day)
        last[row.home_id] = row.day
        last[row.away_id] = row.day
    return flags


def _is_back_to_back(previous: Optional[str], day: str) -> float:
    if previous is None:
        return 0.0
    from bullion.engines import total_days_between

    return 1.0 if total_days_between(previous, day) <= 1 else 0.0


def fit_ipf(
    results: Sequence[Result],
    *,
    reference_day: str,
    prior_games: float = 10.0,
    half_life_days: float = 120.0,
    iterations: int = 40,
    back_to_back: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """(B)/(D) p=1 multiplicative IPF, pseudo-game prior, honest home factor.

    ``back_to_back=(offence, concession)`` adds the asymmetric rest model: a
    team playing the second night defends worse more than it scores less.
    """
    rows = [row for row in results if row.day < reference_day]
    teams = sorted({team for row in rows for team in (row.home_id, row.away_id)})
    if not rows:
        return {"league_rate": 0.0, "home_factor": 1.0, "attack": {}, "defence": {}, "n": 0}
    weights = [decay_weight(row.day, reference_day, half_life_days) for row in rows]
    total_weight = sum(weights)
    league_rate = (
        sum(w * (row.home_score + row.away_score) / 2.0 for w, row in zip(weights, rows))
        / total_weight
    )
    league_rate = max(league_rate, 0.2)
    attack = {team: 1.0 for team in teams}
    defence = {team: 1.0 for team in teams}
    home = 1.10
    flags = _b2b_flags(rows) if back_to_back else {}
    for _ in range(iterations):
        scored = {team: 0.0 for team in teams}
        conceded = {team: 0.0 for team in teams}
        expected_scored = {team: 0.0 for team in teams}
        expected_conceded = {team: 0.0 for team in teams}
        for weight, row in zip(weights, rows):
            home_mult, away_mult, home_concede, away_concede = 1.0, 1.0, 1.0, 1.0
            if back_to_back:
                offence, concession = back_to_back
                if flags.get((row.home_id, row.day)):
                    home_mult, away_concede = offence, concession
                if flags.get((row.away_id, row.day)):
                    away_mult, home_concede = offence, concession
            lam_home = (
                league_rate
                * attack[row.home_id]
                * defence[row.away_id]
                * home
                * home_mult
                * home_concede
            )
            lam_away = (
                league_rate
                * attack[row.away_id]
                * defence[row.home_id]
                * away_mult
                * away_concede
            )
            scored[row.home_id] += weight * row.home_score
            conceded[row.away_id] += weight * row.home_score
            scored[row.away_id] += weight * row.away_score
            conceded[row.home_id] += weight * row.away_score
            expected_scored[row.home_id] += weight * lam_home
            expected_scored[row.away_id] += weight * lam_away
            expected_conceded[row.away_id] += weight * lam_home
            expected_conceded[row.home_id] += weight * lam_away
        prior = prior_games * league_rate
        for team in teams:
            attack[team] = _bounded(attack[team] * (scored[team] + prior) / (expected_scored[team] + prior))
            defence[team] = _bounded(defence[team] * (conceded[team] + prior) / (expected_conceded[team] + prior))
        mean_attack = sum(attack.values()) / len(attack)
        mean_defence = sum(defence.values()) / len(defence)
        for team in teams:
            attack[team] /= mean_attack
            defence[team] /= mean_defence
        goals = sum(w * row.home_score for w, row in zip(weights, rows))
        expected = sum(
            w * league_rate * attack[row.home_id] * defence[row.away_id]
            for w, row in zip(weights, rows)
        )
        if expected > 0:
            home = max(1.0, min(1.25, home * ((goals / expected) / home) ** 0.5))
    return {
        "league_rate": league_rate,
        "home_factor": home,
        "attack": attack,
        "defence": defence,
        "n": len(rows),
    }


def _bounded(value: float) -> float:
    return max(0.5, min(2.0, value))


# ---------------------------------------------------------------------------
# Probability per market surface (mirrors the seeder and the live engines)
# ---------------------------------------------------------------------------


def _sigma_for(lane: str, results: Sequence[Result], ratings: Dict[str, Any], day: str) -> float:
    prior_margin, _prior_total = SIGMA_PRIORS.get(lane, (14.0, 14.0))
    margin, _total = residual_sigmas(results, ratings, reference_day=day)
    return max(margin or prior_margin, 0.5)


def p_home(
    lane: str,
    ratings: Dict[str, Any],
    game: FinalGame,
    results: Sequence[Result],
    *,
    rho: Optional[float] = None,
) -> float:
    home_rate, away_rate = rates_for(ratings, game.home_id, game.away_id)
    if lane == "nhl":
        margin, _total, _draw = nhl_engine._final_distributions(home_rate, away_rate)
        probability = sum(p for m, p in margin.items() if m > 0)
    elif lane in SOCCER_LANES:
        # One rho per matchday: it depends on the fit, not on the fixture.
        if rho is None:
            rho = soccer_engine.fit_rho(
                results, ratings, reference_day=game.day, half_life_days=240.0
            )
        matrix = soccer_engine.score_matrix(home_rate, away_rate, rho)
        probability, _draw, _away = soccer_engine._one_x_two(matrix)
    else:
        sigma = _sigma_for(lane, results, ratings, game.day)
        probability = 1.0 - normal_cdf((0.0 - (home_rate - away_rate)) / sigma)
    return min(max(float(probability), 1e-6), 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Walk-forward driver and metrics
# ---------------------------------------------------------------------------


@dataclass
class Scored:
    day: str
    game_id: str
    probability: float
    outcome: int


def _by_day(finals: Sequence[FinalGame]) -> "Dict[str, List[FinalGame]]":
    grouped: Dict[str, List[FinalGame]] = {}
    for game in finals:
        grouped.setdefault(game.day, []).append(game)
    return grouped


def walk_forward(
    lane: str,
    finals: Sequence[FinalGame],
    fit: Callable[..., Dict[str, Any]],
    *,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    regulation_fit: bool = False,
    cap: Optional[float] = None,
    first_day: Optional[str] = None,
    last_day: Optional[str] = None,
    stride: int = 1,
) -> List[Scored]:
    """Score matchdays strictly from finals dated before them.

    ``stride`` scores every n-th matchday. The cost of a refit is linear in the
    history it consumes, so a full-history walk-forward is quadratic in games;
    on 1,600 finals the full pass takes ~5 minutes per lane. Every variant is
    strided identically, so the comparison stays apples to apples.
    """
    fit_kwargs = dict(fit_kwargs or {})
    grouped = _by_day(finals)
    prior: List[Result] = []
    scored: List[Scored] = []
    stride = max(1, int(stride))
    for index, day in enumerate(sorted(grouped)):
        if first_day is not None and day < first_day:
            prior.extend(_results_for(grouped[day], regulation_fit, cap))
            continue
        if last_day is not None and day > last_day:
            break
        if len(prior) >= MIN_HISTORY and index % stride == 0:
            ratings = fit(prior, reference_day=day, **fit_kwargs)
            if ratings.get("n"):
                rho = (
                    soccer_engine.fit_rho(
                        prior, ratings, reference_day=day, half_life_days=240.0
                    )
                    if lane in SOCCER_LANES
                    else None
                )
                for game in grouped[day]:
                    scored.append(
                        Scored(
                            day=day,
                            game_id=game.game_id,
                            probability=p_home(lane, ratings, game, prior, rho=rho),
                            outcome=game.outcome,
                        )
                    )
        prior.extend(_results_for(grouped[day], regulation_fit, cap))
    return scored


def _results_for(
    games: Iterable[FinalGame], regulation_fit: bool, cap: Optional[float]
) -> List[Result]:
    out: List[Result] = []
    for game in games:
        result = game.result(regulation=regulation_fit, cap=cap)
        if result is not None:
            out.append(result)
    return out


def brier(scored: Sequence[Scored]) -> Optional[float]:
    if not scored:
        return None
    return sum((row.probability - row.outcome) ** 2 for row in scored) / len(scored)


def auc(scored: Sequence[Scored]) -> Optional[float]:
    positives = [row.probability for row in scored if row.outcome == 1]
    negatives = [row.probability for row in scored if row.outcome == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for q in negatives:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(positives) * len(negatives))


def calibration_line(scored: Sequence[Scored]) -> Tuple[Optional[float], Optional[float]]:
    """Least-squares slope/intercept of outcome on logit(p) (1.0/0.0 is ideal)."""
    if len(scored) < 3:
        return None, None
    xs = [logit(row.probability) for row in scored]
    ys = [float(row.outcome) for row in scored]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0:
        return None, None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance / variance
    return slope, mean_y - slope * mean_x


def platt_prequential(scored: Sequence[Scored]) -> Dict[str, Any]:
    """Fit Platt on days < D, score day D. The only convincing Platt number."""
    grouped: Dict[str, List[Scored]] = {}
    for row in scored:
        grouped.setdefault(row.day, []).append(row)
    history: List[Tuple[float, int]] = []
    pairs: List[Tuple[float, int]] = []
    for day in sorted(grouped):
        if len(history) >= PLATT_MIN_ROWS:
            a, b = fit_platt(history)
            for row in grouped[day]:
                pairs.append((sigmoid(a * logit(row.probability) + b), row.outcome))
        history.extend((row.probability, row.outcome) for row in grouped[day])
    if not pairs:
        return {"rows": 0, "brier": None}
    return {
        "rows": len(pairs),
        "brier": sum((p - y) ** 2 for p, y in pairs) / len(pairs),
    }


def describe(lane: str, scored: Sequence[Scored]) -> Dict[str, Any]:
    raw = brier(scored)
    slope, intercept = calibration_line(scored)
    predicted = sum(row.probability for row in scored) / len(scored) if scored else None
    realised = sum(row.outcome for row in scored) / len(scored) if scored else None
    return {
        "lane": lane,
        "rows": len(scored),
        "brier": raw,
        "skill": None if raw is None else 1.0 - raw / 0.25,
        "auc": auc(scored),
        "mean_p_home": predicted,
        "realised_home_rate": realised,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "prequential_platt": platt_prequential(scored),
    }


# ---------------------------------------------------------------------------
# The memo
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _delta_fmt(value: Optional[float], digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _split_day(finals: Sequence[FinalGame], fraction: float) -> str:
    days = sorted({game.day for game in finals})
    if not days:
        return ""
    return days[max(0, min(len(days) - 1, int(len(days) * fraction)))]


def tune(
    lane: str,
    finals: Sequence[FinalGame],
    *,
    first_day: Optional[str],
    last_day: str,
    grid: Sequence[Tuple[float, float]],
    stride: int = 1,
) -> Tuple[Tuple[float, float], List[Dict[str, Any]]]:
    """Pick (prior_games, half_life) on the earlier split only."""
    table: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, Tuple[float, float]]] = None
    for prior_games, half_life in grid:
        scored = walk_forward(
            lane,
            finals,
            fit_ipf,
            fit_kwargs={"prior_games": prior_games, "half_life_days": half_life},
            first_day=first_day,
            last_day=last_day,
            stride=stride,
        )
        value = brier(scored)
        table.append(
            {"prior_games": prior_games, "half_life_days": half_life, "train_brier": value}
        )
        if value is not None and (best is None or value < best[0]):
            best = (value, (prior_games, half_life))
    return (best[1] if best else (10.0, 120.0)), table


def build_memo(
    db_path: Path,
    *,
    lanes: Sequence[str],
    fast: bool,
    max_days: Optional[int],
    stride: int = 2,
) -> Dict[str, Any]:
    conn = open_readonly(db_path)
    try:
        finals = {lane: load_finals(conn, lane) for lane in lanes}
    finally:
        conn.close()

    memo: Dict[str, Any] = {
        "schema": "bullion.engine_diagnostic",
        "db": str(db_path),
        "lanes": {},
    }

    nhl = finals.get("nhl", [])
    if max_days:
        days = sorted({game.day for game in nhl})
        if len(days) > max_days:
            cutoff = days[max_days]
            nhl = [game for game in nhl if game.day <= cutoff]
            finals["nhl"] = nhl

    # -- 1. baseline -------------------------------------------------------
    if nhl:
        baseline = describe("nhl", walk_forward("nhl", nhl, fit_ratings, stride=stride))
        production = fit_ratings(
            [game.result() for game in nhl if game.day < nhl[-1].day],
            reference_day=nhl[-1].day,
        )
        baseline["production_home_factor"] = production.get("home_factor")
        baseline["production_home_factor_refit"] = fit_home_factor_only(
            [game.result() for game in nhl if game.day < nhl[-1].day],
            reference_day=nhl[-1].day,
        ).get("home_factor")
        with_regulation = [game for game in nhl if game.regulation is not None]
        baseline["finals"] = len(nhl)
        baseline["regulation_available"] = len(with_regulation)
        baseline["regulation_share"] = (
            len(with_regulation) / len(nhl) if nhl else None
        )
        memo["lanes"]["nhl"] = {"baseline": baseline}

        # -- 2. chronological ablation ------------------------------------
        split = _split_day(nhl, 0.7)
        last = max(game.day for game in nhl)
        grid: Sequence[Tuple[float, float]] = (
            ((10.0, 120.0),)
            if fast
            else ((3.0, 60.0), (3.0, 120.0), (10.0, 60.0), (10.0, 120.0),
                  (10.0, 240.0), (25.0, 120.0))
        )
        chosen, tuning = tune(
            "nhl", nhl, first_day=None, last_day=split, grid=grid, stride=stride
        )
        prior_games, half_life = chosen
        variants: Dict[str, Dict[str, Any]] = {}
        for name, fit, kwargs, regulation, cap in (
            ("production", fit_ratings, {}, False, None),
            ("A_home_only", fit_home_factor_only, {}, False, None),
            ("B_ipf_prior_home", fit_ipf,
             {"prior_games": prior_games, "half_life_days": half_life}, False, None),
            ("C_ipf_regulation_cap", fit_ipf,
             {"prior_games": prior_games, "half_life_days": half_life}, True, 3.0),
            ("C_ipf_regulation_nocap", fit_ipf,
             {"prior_games": prior_games, "half_life_days": half_life}, True, None),
            ("D_ipf_regulation_b2b", fit_ipf,
             {"prior_games": prior_games, "half_life_days": half_life,
              "back_to_back": (0.975, 1.02)}, True, 3.0),
        ):
            train = walk_forward(
                "nhl", nhl, fit, fit_kwargs=dict(kwargs),
                regulation_fit=regulation, cap=cap, last_day=split, stride=stride,
            )
            holdout = walk_forward(
                "nhl", nhl, fit, fit_kwargs=dict(kwargs),
                regulation_fit=regulation, cap=cap, first_day=_next_day(split),
                stride=stride,
            )
            variants[name] = {
                "train": describe("nhl", train),
                "holdout": describe("nhl", holdout),
                "fit_kwargs": {k: v for k, v in kwargs.items()},
            }
        memo["lanes"]["nhl"]["ablation"] = {
            "split_day": split,
            "last_day": last,
            "chosen": {"prior_games": prior_games, "half_life_days": half_life},
            "tuning": tuning,
            "variants": variants,
        }

    # -- 3. cross-lane ----------------------------------------------------
    for lane in lanes:
        if lane == "nhl":
            continue
        games = finals.get(lane) or []
        if not games:
            memo["lanes"][lane] = {"baseline": None, "note": "no finals in warehouse"}
            continue
        memo["lanes"][lane] = {
            "baseline": describe(
                lane, walk_forward(lane, games, fit_ratings, stride=stride)
            ),
            "A_home_only": describe(
                lane, walk_forward(lane, games, fit_home_factor_only, stride=stride)
            ),
            "B_ipf_prior_home": describe(
                lane, walk_forward(lane, games, fit_ipf, stride=stride)
            ),
            "finals": len(games),
        }
    return memo


def _next_day(day: str) -> str:
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()


def render_markdown(memo: Dict[str, Any]) -> str:
    lines: List[str] = ["# Engine diagnostic — read-only memo", ""]
    lines.append(f"Warehouse: `{memo['db']}`")
    nhl = memo["lanes"].get("nhl") or {}
    baseline = nhl.get("baseline") or {}
    if baseline:
        lines += [
            "",
            "## 1. Baseline `nhl_poisson_v1` (walk-forward, no gate)",
            "",
            "| metric | value |",
            "|---|---|",
            f"| finals scored | {baseline['rows']} |",
            f"| Brier | {_fmt(baseline['brier'])} |",
            f"| skill vs 0.25 | {_delta_fmt(baseline['skill'])} |",
            f"| AUC | {_fmt(baseline['auc'])} |",
            f"| mean predicted p(home) | {_fmt(baseline['mean_p_home'])} |",
            f"| realised home win rate | {_fmt(baseline['realised_home_rate'])} |",
            f"| calibration slope | {_fmt(baseline['calibration_slope'])} |",
            f"| calibration intercept | {_fmt(baseline['calibration_intercept'])} |",
            f"| prequential Platt Brier | {_fmt(baseline['prequential_platt']['brier'])} "
            f"({baseline['prequential_platt']['rows']} rows) |",
            f"| production `home_factor` | {_fmt(baseline['production_home_factor'])} "
            f"(refit honestly: {_fmt(baseline['production_home_factor_refit'])}) |",
            f"| finals with regulation scores | {baseline['regulation_available']}"
            f"/{baseline['finals']} ({_fmt((baseline['regulation_share'] or 0) * 100, 1)}%) |",
        ]
    ablation = nhl.get("ablation")
    if ablation:
        lines += [
            "",
            "## 2. Chronological ablation (tuned on the earlier split only)",
            "",
            f"Split day `{ablation['split_day']}`, last day `{ablation['last_day']}`. "
            f"Chosen: `prior_games={ablation['chosen']['prior_games']}`, "
            f"`half_life_days={ablation['chosen']['half_life_days']}`.",
            "",
            "| variant | train Brier | train skill | holdout Brier | holdout skill | holdout AUC |",
            "|---|---|---|---|---|---|",
        ]
        for name, payload in ablation["variants"].items():
            train, holdout = payload["train"], payload["holdout"]
            lines.append(
                f"| {name} | {_fmt(train['brier'])} | {_delta_fmt(train['skill'])} "
                f"| {_fmt(holdout['brier'])} | {_delta_fmt(holdout['skill'])} "
                f"| {_fmt(holdout['auc'])} |"
            )
        lines += ["", "Tuning grid (train split, Brier):", "", "| prior_games | half_life | Brier |", "|---|---|---|"]
        for row in ablation["tuning"]:
            lines.append(
                f"| {row['prior_games']} | {row['half_life_days']} | {_fmt(row['train_brier'])} |"
            )
    others = {lane: payload for lane, payload in memo["lanes"].items() if lane != "nhl" and payload.get("baseline")}
    if others:
        lines += [
            "",
            "## 3. Cross-lane side effects (read-only; nothing shared was changed)",
            "",
            "| lane | finals | baseline Brier | A Brier | B Brier | baseline skill | B skill |",
            "|---|---|---|---|---|---|---|",
        ]
        for lane, payload in others.items():
            base = payload["baseline"]
            lines.append(
                f"| {lane} | {payload['finals']} | {_fmt(base['brier'])} "
                f"| {_fmt(payload['A_home_only']['brier'])} "
                f"| {_fmt(payload['B_ipf_prior_home']['brier'])} "
                f"| {_delta_fmt(base['skill'])} | {_delta_fmt(payload['B_ipf_prior_home']['skill'])} |"
            )
    lines += ["", "_Read-only: the warehouse was opened `mode=ro` with `query_only`; "
                  "no engine, gate, schema or localdata file was modified._"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "localdata" / "bullion.db"))
    parser.add_argument("--lanes", nargs="*", default=list(SEED_LANES))
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    parser.add_argument("--fast", action="store_true", help="skip the tuning grid")
    parser.add_argument("--max-days", type=int, default=None, help="cap NHL matchdays (smoke runs)")
    parser.add_argument(
        "--stride", type=int, default=2,
        help="score every n-th matchday (1 = every day, the slowest and most exact)",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no warehouse at {db_path}: nothing to diagnose", file=sys.stderr)
        return 2
    memo = build_memo(
        db_path,
        lanes=args.lanes,
        fast=args.fast,
        max_days=args.max_days,
        stride=args.stride,
    )
    if args.json:
        print(json.dumps(memo, indent=2, sort_keys=True))
    else:
        print(render_markdown(memo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
