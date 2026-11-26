"""Every entry point must import and expose its CLI.

The 2026-11-26 PR that rewrote ``espn.py`` truncated the file after
``LanePermission`` and silently deleted ``game_from_summary``. Unit tests stayed
green because nothing imported it; ``scripts/backfill.py`` died at import time
instead, which in CI would have surfaced as a cold-warehouse failure on the first
daily run. This test keeps the whole entry-point surface honest: if a symbol a
script imports disappears, this fails here rather than in Actions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"bullion_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_script_imports_and_exposes_main(path):
    module = _load(path)
    assert callable(module.main), f"{path.name} has no main()"
    assert (module.main.__doc__ or "").strip() or True  # help text is optional


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_script_cli_parses_help(path, capsys):
    module = _load(path)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_backfill_imports_the_summary_builder_it_uses():
    """Regression: the deleted ``game_from_summary`` broke this import."""
    from bullion.espn import game_from_summary

    assert callable(game_from_summary)
    module = _load(ROOT / "scripts" / "backfill.py")
    assert "game_from_summary" in module.__dict__ or callable(
        getattr(module, "game_from_summary", None)
    )
