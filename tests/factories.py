"""SYNTHETIC_FIXTURE builders.

Nothing here is a captured ESPN response. Shapes follow the proven basketball
contract (competitions -> competitors with ``homeAway``, ``odds`` offers with
``moneyline``/``pointSpread``/``total`` nested open/close values) and are
labelled synthetic everywhere they are used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYNTHETIC = "SYNTHETIC_FIXTURE"


def event(
    event_id: str = "401",
    *,
    home: str = "10",
    away: str = "20",
    home_name: str = "Home Club",
    away_name: str = "Away Club",
    date: str = "2026-11-28T23:00:00Z",
    state: str = "pre",
    status_name: Optional[str] = None,
    completed: Optional[bool] = None,
    offers: Optional[List[Dict[str, Any]]] = None,
    home_score: Optional[str] = None,
    away_score: Optional[str] = None,
    home_lines: Optional[List[Dict[str, Any]]] = None,
    away_lines: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    home_competitor: Dict[str, Any] = {
        "homeAway": "home",
        "team": {"id": home, "displayName": home_name},
    }
    away_competitor: Dict[str, Any] = {
        "homeAway": "away",
        "team": {"id": away, "displayName": away_name},
    }
    if home_score is not None:
        home_competitor["score"] = home_score
    if away_score is not None:
        away_competitor["score"] = away_score
    if home_lines is not None:
        home_competitor["linescores"] = home_lines
    if away_lines is not None:
        away_competitor["linescores"] = away_lines
    status_type: Dict[str, Any] = {"state": state}
    if status_name is not None:
        status_type["name"] = status_name
    if completed is not None:
        status_type["completed"] = completed
    return {
        "id": event_id,
        "date": date,
        "season": {"year": 2026},
        "competitions": [
            {
                "date": date,
                "status": {"type": status_type},
                "competitors": [away_competitor, home_competitor],
                "odds": offers or [],
                "neutralSite": False,
                "venue": {"fullName": "Synthetic Arena"},
            }
        ],
    }


def offer(
    *,
    book: str = "DraftKings",
    moneyline: Optional[Dict[str, Any]] = None,
    spread: Optional[Dict[str, Any]] = None,
    total: Optional[Dict[str, Any]] = None,
    draw_odds: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"provider": {"displayName": book}}
    if moneyline is not None:
        payload["moneyline"] = moneyline
    if spread is not None:
        payload["pointSpread"] = spread
    if total is not None:
        payload["total"] = total
    if draw_odds is not None:
        payload["drawOdds"] = draw_odds
    return payload


def close(odds: Optional[int] = None, line: Optional[str] = None) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    if odds is not None:
        value["odds"] = odds
    if line is not None:
        value["line"] = line
    return {"close": value, "open": dict(value)}


def summary(
    *,
    event_id: str = "401",
    home: str = "10",
    away: str = "20",
    state: str = "post",
    status_name: Optional[str] = None,
    completed: Optional[bool] = None,
    date: str = "2026-11-28T23:00:00Z",
    home_score: str = "3",
    away_score: str = "2",
    home_lines: Optional[List[Dict[str, Any]]] = None,
    away_lines: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    status_type: Dict[str, Any] = {"state": state}
    if status_name is not None:
        status_type["name"] = status_name
    if completed is not None:
        status_type["completed"] = completed
    return {
        "header": {
            "id": event_id,
            "season": {"year": 2026},
            "competitions": [
                {
                    "date": date,
                    "status": {"type": status_type},
                    "competitors": [
                        {
                            "homeAway": "away",
                            "team": {"id": away, "displayName": "Away Club"},
                            "score": away_score,
                            "linescores": away_lines
                            or [{"value": 1}, {"value": 1}, {"value": 0}],
                        },
                        {
                            "homeAway": "home",
                            "team": {"id": home, "displayName": "Home Club"},
                            "score": home_score,
                            "linescores": home_lines
                            or [{"value": 1}, {"value": 1}, {"value": 1}],
                        },
                    ],
                }
            ],
        }
    }


def predictor(
    *,
    home_ref: str = "10",
    away_ref: str = "20",
    stats_on: str = "away",
    projection: float = 55.0,
    expected: bool = True,
) -> Dict[str, Any]:
    statistics = [
        {"name": "gameProjection", "value": projection},
        {"name": "teamChanceLoss", "value": 100.0 - projection},
    ]
    if expected:
        statistics += [
            {"name": "teamExpectedPts", "value": 3.2},
            {"name": "oppExpectedPts", "value": 2.6},
            {"name": "teamPredPtDiff", "value": 0.6},
        ]
    home_side: Dict[str, Any] = {"team": {"$ref": f"http://x/teams/{home_ref}"}}
    away_side: Dict[str, Any] = {"team": {"$ref": f"http://x/teams/{away_ref}"}}
    if stats_on == "home":
        home_side["statistics"] = statistics
    else:
        away_side["statistics"] = statistics
    return {"homeTeam": home_side, "awayTeam": away_side, "lastModified": "2026-11-27T12:00:00Z"}
