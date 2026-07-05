"""Fixed, versioned seen/unseen game split for the learned-abstraction pivot.

The split is intentionally stable (not random per run) so that learning progress
can be compared without experimental noise. ``ar25`` is always evaluated
separately as a regression benchmark, never as the final objective.

Short game ids (e.g. ``"wa30"``) are used as the canonical, hash-independent
keys. Full game ids (e.g. ``"wa30-ee6fef47"``) are resolved by scanning
``environment_files/`` so the split survives re-downloads with new hashes.

Mechanic diversity of the held-out unseen set (one per family):
    wa30  warehouse        -> push / navigation
    tn36  true_naming      -> selection / click
    ft09  flip_tiles       -> transformation
    cn04  connector        -> multi-object / correspondence
    sb26  subroutine       -> weird / unknown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / "environment_files"

# ar25 is the regression benchmark, evaluated on its own.
AR25_EVAL: List[str] = ["ar25"]

# Five mechanically diverse games never used to train the models.
PUBLIC_UNSEEN: List[str] = ["wa30", "tn36", "ft09", "cn04", "sb26"]


def _discover_short_ids() -> List[str]:
    """Return every short game id that has a local environment directory."""

    if not ENV_DIR.exists():
        return []
    return sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir())


# Every game with a local offline environment.
ALL_PUBLIC_GAMES: List[str] = _discover_short_ids()

# Everything that is neither the ar25 benchmark nor a held-out unseen game.
PUBLIC_SEEN: List[str] = [
    game
    for game in ALL_PUBLIC_GAMES
    if game not in AR25_EVAL and game not in PUBLIC_UNSEEN
]


_FULL_ID_CACHE: Dict[str, str] = {}


def resolve_full_game_id(short_id: str) -> str:
    """Resolve a short id (``"wa30"``) to its full id (``"wa30-ee6fef47"``).

    Falls back to the short id itself if no local metadata is found.
    """

    if "-" in short_id:
        return short_id
    if short_id in _FULL_ID_CACHE:
        return _FULL_ID_CACHE[short_id]

    game_dir = ENV_DIR / short_id
    full_id = short_id
    if game_dir.is_dir():
        for hash_dir in sorted(game_dir.iterdir()):
            metadata = hash_dir / "metadata.json"
            if metadata.is_file():
                try:
                    payload = json.loads(metadata.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                candidate = payload.get("game_id")
                if candidate:
                    full_id = str(candidate)
                    break
            # Fallback: derive from directory name if no metadata game_id.
            full_id = f"{short_id}-{hash_dir.name}"
            break

    _FULL_ID_CACHE[short_id] = full_id
    return full_id


_SPLIT_ALIASES: Dict[str, List[str]] = {
    "ar25": AR25_EVAL,
    "public": ALL_PUBLIC_GAMES,
    "public_seen": PUBLIC_SEEN,
    "public_unseen": PUBLIC_UNSEEN,
    "public_unseen_split": PUBLIC_UNSEEN,
    "all": ALL_PUBLIC_GAMES,
}


def resolve(split_name: str, *, full_ids: bool = True) -> List[str]:
    """Return the list of game ids for a named split.

    ``split_name`` accepts the aliases used by the CLI tools
    (``ar25``, ``public``, ``public_seen``, ``public_unseen_split``) or a
    comma-separated list of explicit short/full game ids.
    """

    key = (split_name or "").strip().lower()
    if key in _SPLIT_ALIASES:
        games = list(_SPLIT_ALIASES[key])
    else:
        games = [part.strip() for part in split_name.split(",") if part.strip()]
    if full_ids:
        return [resolve_full_game_id(game) for game in games]
    return games


def split_for_game(short_or_full_id: str) -> Optional[str]:
    """Return which split a game belongs to (``ar25``/``seen``/``unseen``)."""

    short = short_or_full_id.split("-", 1)[0]
    if short in AR25_EVAL:
        return "ar25"
    if short in PUBLIC_UNSEEN:
        return "unseen"
    if short in PUBLIC_SEEN:
        return "seen"
    return None


if __name__ == "__main__":
    print(f"ALL_PUBLIC_GAMES ({len(ALL_PUBLIC_GAMES)}): {ALL_PUBLIC_GAMES}")
    print(f"AR25_EVAL: {[resolve_full_game_id(g) for g in AR25_EVAL]}")
    print(f"PUBLIC_UNSEEN ({len(PUBLIC_UNSEEN)}): {[resolve_full_game_id(g) for g in PUBLIC_UNSEEN]}")
    print(f"PUBLIC_SEEN ({len(PUBLIC_SEEN)}): {[resolve_full_game_id(g) for g in PUBLIC_SEEN]}")
