"""Hermetic test setup: local ``src`` import path and no network sockets.

Ordinary tests must never touch ESPN. Any accidental live call raises, which
keeps "synthetic fixture" tests from quietly becoming unverifiable live claims.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _NoNetwork(socket.socket):
    def __init__(self, *args, **kwargs):  # noqa: D401 - hard guard, not a feature
        raise RuntimeError("network access is blocked in unit tests")


@pytest.fixture(autouse=True)
def _block_sockets(monkeypatch):
    monkeypatch.setattr(socket, "socket", _NoNetwork)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _raise())
    yield


def _raise():
    raise RuntimeError("network access is blocked in unit tests")
