"""Pygame-based human recorder over the ARC-AGI-3 `EnvironmentWrapper`.

Works with either transport via the `arc_agi.Arcade` SDK:
  - OFFLINE: drives the local Python simulator in `environment_files/`
  - NORMAL / ONLINE: hits the ARC-AGI-3 API (same as the Agent framework)

Controls
========
Actions:
  1..5, 7           : ACTION1..ACTION5, ACTION7
  Left mouse click  : ACTION6 at clicked (x, y) (grid coords)
  R                 : RESET
  Space             : skip (record no action, useful for thinking / tagging)

Annotation:
  Tab / Shift-Tab   : cycle through IntentTag values (shown on HUD)
  H                 : type a hypothesis (Enter commits, Esc cancels)
  N                 : clear current hypothesis
  C                 : mark hypothesis_confirmed on the next action
  X                 : mark hypothesis_rejected on the next action
  G                 : mark goal_changed on the next action
  M                 : add a discovered_mechanic line (prompted)
  F                 : add a discovered_mistake line (prompted)
  T                 : set game_type_guess (prompted)
  O                 : set objective_guess (prompted)

Session:
  Q or Esc          : quit current episode, write EpisodeRecord, close window
  K                 : kill the current episode (mark final_state=QUIT) and
                      start a fresh one (issues RESET)

Every successful action writes one StepRecord to the steps JSONL file.
On quit or at episode terminals (WIN / GAME_OVER), an EpisodeRecord is
written to the episodes JSONL file.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Lazy pygame import: keeps the package importable in headless environments
# (e.g. for seed_cross_game_memory() during training).
def _require_pygame():
    try:
        import pygame  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pygame is required for the human recorder. "
            "Install with: pip install pygame"
        ) from e
    return pygame


from .schema import CognitiveEvent, EpisodeRecord, IntentTag, StepRecord, TraceWriter, _utc_stamp


# ARC-AGI-3 uses a 16-colour palette (values 0..15). Colours chosen to
# roughly match the official web client for familiarity.
ARC_PALETTE: List[Tuple[int, int, int]] = [
    (  0,   0,   0),  # 0  black
    ( 30, 147, 255),  # 1  blue
    (249,  60,  49),  # 2  red
    ( 79, 204,  48),  # 3  green
    (255, 220,   0),  # 4  yellow
    (153, 153, 153),  # 5  grey
    (229,  58, 163),  # 6  magenta
    (255, 133,  27),  # 7  orange
    (135, 216, 241),  # 8  light cyan
    (146,  18,  49),  # 9  maroon
    (255, 255, 255),  # 10 white
    (144,  93, 242),  # 11 purple
    (177, 242,  97),  # 12 lime
    (115,  74,  41),  # 13 brown
    ( 45, 110, 178),  # 14 dark blue
    ( 25,  25,  25),  # 15 near-black
]


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _primary_grid_from_frame(frame: Any) -> List[List[int]]:
    """Extract the 2D primary grid from FrameDataRaw.frame.

    `FrameDataRaw.frame` is a list of `np.ndarray`; `FrameData.frame` is
    a `list[list[list[int]]]`. In both cases the primary grid is the last
    element (some games stack layers; v4_1 also uses the last one).
    """
    if frame is None:
        return [[0]]
    if not frame:
        return [[0]]
    last = frame[-1]
    if isinstance(last, np.ndarray):
        return last.tolist()
    # Assume nested lists
    return [list(row) for row in last]


def _grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


# ------------------------------------------------------------------
# Recorder
# ------------------------------------------------------------------

class HumanRecorder:
    """Drives an `EnvironmentWrapper`, renders with pygame, logs traces.

    Parameters
    ----------
    env : arc_agi.EnvironmentWrapper
        Already-constructed wrapper (call `Arcade.make(...)` first).
    writer : TraceWriter
        JSONL sink; one writer per game_id / session.
    game_id : str
        Used in records. Should match what the env uses.
    cell_size : int
        Pixels per grid cell. 10 gives a 640x640 grid canvas for 64x64.
    hud_width : int
        Width of the HUD panel in pixels.
    """

    def __init__(
        self,
        env: Any,
        writer: TraceWriter,
        game_id: str,
        cell_size: int = 10,
        hud_width: int = 320,
    ) -> None:
        self.env = env
        self.writer = writer
        self.game_id = game_id
        self.cell_size = cell_size
        self.hud_width = hud_width

        self._pygame = _require_pygame()
        self._pygame.init()
        self._pygame.display.set_caption(f"ARC-AGI-3 human recorder — {game_id}")

        self.grid_side = 64 * cell_size  # nominal 64x64
        self.screen = self._pygame.display.set_mode((self.grid_side + hud_width, self.grid_side))
        self.font = self._pygame.font.SysFont("consolas", 14)
        self.font_small = self._pygame.font.SysFont("consolas", 12)
        self.font_big = self._pygame.font.SysFont("consolas", 16, bold=True)

        # Session state
        self.intent_tags = IntentTag.cycle_order()
        self.intent_idx = 0
        self.hypothesis: str = ""
        self.pending_cognitive_events: List[str] = []
        self.game_type_guess: str = ""
        self.objective_guess: str = ""
        self.discovered_mechanics: List[str] = []
        self.discovered_mistakes: List[str] = []

        # Episode state
        self.episode_id: str = ""
        self.episode_started_at: str = ""
        self.episode_start_ts: float = 0.0
        self.episode_steps: int = 0
        self.current_frame: Optional[List[List[int]]] = None
        self.current_state: str = "NOT_PLAYED"
        self.current_levels: int = 0
        self.current_available: List[int] = []

        # Message shown briefly at the bottom of the HUD
        self._flash_msg: str = ""
        self._flash_until: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main loop. Returns when the user quits."""
        self._begin_episode(issue_reset=True)
        self._render()

        clock = self._pygame.time.Clock()
        running = True
        while running:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    running = False
                    break
                if event.type == self._pygame.KEYDOWN:
                    if self._handle_keydown(event) is False:
                        running = False
                        break
                elif event.type == self._pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            self._render()
            clock.tick(30)

        self._finalize_episode(final_state_override="QUIT")
        self._pygame.quit()

    # ------------------------------------------------------------------
    # Input handlers
    # ------------------------------------------------------------------
    def _handle_keydown(self, event: Any) -> Optional[bool]:
        """Returns False to request main-loop exit."""
        pg = self._pygame
        k = event.key

        # Quit
        if k in (pg.K_q, pg.K_ESCAPE):
            return False

        # Kill and restart episode
        if k == pg.K_k:
            self._finalize_episode(final_state_override="QUIT")
            self._begin_episode(issue_reset=True)
            self._flash("episode killed, restarted")
            return None

        # Reset
        if k == pg.K_r:
            self._take_simple_action("RESET")
            return None

        # Direct action keys 1..5, 7
        keymap = {
            pg.K_1: "ACTION1", pg.K_2: "ACTION2", pg.K_3: "ACTION3",
            pg.K_4: "ACTION4", pg.K_5: "ACTION5", pg.K_7: "ACTION7",
        }
        if k in keymap:
            self._take_simple_action(keymap[k])
            return None

        # Skip (no action, just record nothing — useful for pure annotation)
        if k == pg.K_SPACE:
            self._flash("skip")
            return None

        # Intent cycle
        if k == pg.K_TAB:
            delta = -1 if (event.mod & pg.KMOD_SHIFT) else 1
            self.intent_idx = (self.intent_idx + delta) % len(self.intent_tags)
            self._flash(f"intent = {self.current_intent.value}")
            return None

        # Hypothesis
        if k == pg.K_h:
            s = self._prompt("hypothesis", self.hypothesis)
            if s is not None:
                self.hypothesis = s
                self._flash(f"hyp = {s}")
            return None
        if k == pg.K_n:
            self.hypothesis = ""
            self._flash("hyp cleared")
            return None

        # Rare cognitive event markers. They are attached to the next action,
        # so the label describes the current state before acting.
        if k == pg.K_c:
            self._toggle_cognitive_event(CognitiveEvent.HYPOTHESIS_CONFIRMED)
            return None
        if k == pg.K_x:
            self._toggle_cognitive_event(CognitiveEvent.HYPOTHESIS_REJECTED)
            return None
        if k == pg.K_g:
            self._toggle_cognitive_event(CognitiveEvent.GOAL_CHANGED)
            return None

        if k == pg.K_m:
            s = self._prompt("discovered mechanic", "")
            if s:
                self.discovered_mechanics.append(s)
                self._flash(f"+mech: {s}")
            return None
        if k == pg.K_f:
            s = self._prompt("discovered mistake", "")
            if s:
                self.discovered_mistakes.append(s)
                self._flash(f"+mistake: {s}")
            return None
        if k == pg.K_t:
            s = self._prompt("game_type_guess", self.game_type_guess)
            if s is not None:
                self.game_type_guess = s
                self._flash(f"type = {s}")
            return None
        if k == pg.K_o:
            s = self._prompt("objective_guess", self.objective_guess)
            if s is not None:
                self.objective_guess = s
                self._flash(f"obj = {s}")
            return None

        return None

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        x_px, y_px = pos
        if x_px >= self.grid_side:
            # Click in HUD area, ignore
            return
        if self.current_frame is None:
            return
        h, w = _grid_shape(self.current_frame)
        # Pygame y is vertical; grid is also (y, x). Cell size is based on 64
        # nominal side, but real grid may be smaller. Scale to real grid.
        gx = int(x_px / self.grid_side * w)
        gy = int(y_px / self.grid_side * h)
        gx = max(0, min(w - 1, gx))
        gy = max(0, min(h - 1, gy))
        self._take_action("ACTION6", {"x": gx, "y": gy})

    # ------------------------------------------------------------------
    # Action execution + logging
    # ------------------------------------------------------------------
    def _take_simple_action(self, name: str) -> None:
        self._take_action(name, None)

    def _take_action(self, name: str, args: Optional[Dict[str, Any]]) -> None:
        from arcengine import GameAction  # lazy import

        # Validate availability (except RESET which is always legal)
        if name != "RESET":
            try:
                action_id = GameAction.from_name(name).value
            except ValueError:
                self._flash(f"unknown action {name}")
                return
            if action_id not in self.current_available:
                self._flash(f"{name} not available ({self.current_available})")
                return

        grid_before = list(self.current_frame) if self.current_frame else [[0]]
        action_enum = GameAction.from_name(name)
        data: Dict[str, Any] = {"game_id": self.game_id}
        if args:
            data.update(args)

        try:
            frame_raw = self.env.step(action_enum, data=data)
        except Exception as e:  # pragma: no cover
            self._flash(f"step error: {e}")
            logger.exception("env.step failed")
            return
        if frame_raw is None:
            self._flash("env.step returned None")
            return

        grid_after = _primary_grid_from_frame(frame_raw.frame)
        state_name = frame_raw.state.name if hasattr(frame_raw.state, "name") else str(frame_raw.state)

        step = StepRecord(
            game_id=self.game_id,
            episode_id=self.episode_id,
            step=self.episode_steps,
            frame_before=grid_before,
            available_actions=list(self.current_available),
            action=name,
            action_args=args,
            frame_after=grid_after,
            game_state_after=state_name,
            levels_completed_after=int(frame_raw.levels_completed),
            intent=self.current_intent.value,
            cognitive_events=list(self.pending_cognitive_events),
            hypothesis=self.hypothesis,
            t_ms=int((time.time() - self.episode_start_ts) * 1000),
        )
        self.writer.write_step(step)
        self.pending_cognitive_events = []
        self.episode_steps += 1

        # Refresh live state
        self.current_frame = grid_after
        self.current_state = state_name
        self.current_levels = int(frame_raw.levels_completed)
        self.current_available = list(frame_raw.available_actions or [])

        # On terminal state, close the episode and start a fresh one.
        if state_name in ("WIN", "GAME_OVER"):
            self._flash(f"terminal: {state_name}")
            self._finalize_episode(final_state_override=state_name)
            # Keep the recorder open; start a new episode automatically.
            self._begin_episode(issue_reset=True)

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------
    @property
    def current_intent(self) -> IntentTag:
        return self.intent_tags[self.intent_idx]

    def _toggle_cognitive_event(self, event: CognitiveEvent) -> None:
        label = event.value
        if label in self.pending_cognitive_events:
            self.pending_cognitive_events.remove(label)
            self._flash(f"event removed: {label}")
            return

        if event == CognitiveEvent.HYPOTHESIS_CONFIRMED:
            opposite = CognitiveEvent.HYPOTHESIS_REJECTED.value
            if opposite in self.pending_cognitive_events:
                self.pending_cognitive_events.remove(opposite)
        elif event == CognitiveEvent.HYPOTHESIS_REJECTED:
            opposite = CognitiveEvent.HYPOTHESIS_CONFIRMED.value
            if opposite in self.pending_cognitive_events:
                self.pending_cognitive_events.remove(opposite)

        self.pending_cognitive_events.append(label)
        self._flash(f"event pending: {label}")

    def _begin_episode(self, issue_reset: bool) -> None:
        self.episode_id = uuid.uuid4().hex[:12]
        self.episode_started_at = datetime.now(timezone.utc).isoformat()
        self.episode_start_ts = time.time()
        self.episode_steps = 0
        # Reset per-episode annotation that should be fresh (keep sticky hypothesis).
        self.discovered_mechanics = []
        self.discovered_mistakes = []
        self.pending_cognitive_events = []

        if issue_reset:
            self._take_simple_action("RESET")
        else:
            # Pull current env state if available
            obs = self.env.observation_space
            if obs is not None:
                self.current_frame = _primary_grid_from_frame(obs.frame)
                self.current_state = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
                self.current_levels = int(obs.levels_completed)
                self.current_available = list(obs.available_actions or [])

    def _finalize_episode(self, final_state_override: Optional[str] = None) -> None:
        if not self.episode_id:
            return
        ep = EpisodeRecord(
            game_id=self.game_id,
            episode_id=self.episode_id,
            started_at=self.episode_started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            n_steps=self.episode_steps,
            final_state=final_state_override or self.current_state,
            levels_completed=self.current_levels,
            game_type_guess=self.game_type_guess,
            objective_guess=self.objective_guess,
            discovered_mechanics=list(self.discovered_mechanics),
            discovered_mistakes=list(self.discovered_mistakes),
            notes="",
        )
        self.writer.write_episode(ep)
        self.episode_id = ""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self) -> None:
        pg = self._pygame
        self.screen.fill((18, 18, 18))

        # Grid
        if self.current_frame:
            h, w = _grid_shape(self.current_frame)
            cw = self.grid_side / max(w, 1)
            ch = self.grid_side / max(h, 1)
            for y in range(h):
                row = self.current_frame[y]
                for x in range(w):
                    v = int(row[x]) if 0 <= row[x] < len(ARC_PALETTE) else 0
                    colour = ARC_PALETTE[v % len(ARC_PALETTE)]
                    rect = pg.Rect(int(x * cw), int(y * ch), int(cw) + 1, int(ch) + 1)
                    pg.draw.rect(self.screen, colour, rect)

        # HUD separator
        pg.draw.rect(
            self.screen, (40, 40, 40),
            pg.Rect(self.grid_side, 0, self.hud_width, self.grid_side),
        )

        x0 = self.grid_side + 10
        y = 10
        self._hud_line(x0, y, f"game: {self.game_id}", self.font_big, (240, 240, 240)); y += 22
        self._hud_line(x0, y, f"episode: {self.episode_id[:8]}  step {self.episode_steps}", self.font, (200, 200, 200)); y += 18
        self._hud_line(x0, y, f"state: {self.current_state}  level {self.current_levels}", self.font, (180, 220, 180)); y += 18
        self._hud_line(x0, y, f"avail: {self.current_available}", self.font_small, (180, 180, 180)); y += 20

        self._hud_line(x0, y, f"intent: {self.current_intent.value}", self.font, (255, 220, 120)); y += 18
        hyp = self.hypothesis if self.hypothesis else "(none)"
        self._hud_line(x0, y, f"hyp: {hyp[:40]}", self.font_small, (200, 200, 240)); y += 16
        events = ", ".join(self.pending_cognitive_events) if self.pending_cognitive_events else "(none)"
        self._hud_line(x0, y, f"event: {events[:40]}", self.font_small, (220, 200, 255)); y += 22

        self._hud_line(x0, y, f"type: {self.game_type_guess or '(unset)'}", self.font_small, (200, 240, 200)); y += 16
        self._hud_line(x0, y, f"obj:  {self.objective_guess or '(unset)'}", self.font_small, (200, 240, 200)); y += 24

        self._hud_line(x0, y, f"mechanics ({len(self.discovered_mechanics)}):", self.font_small, (200, 200, 200)); y += 16
        for m in self.discovered_mechanics[-4:]:
            self._hud_line(x0, y, f"  - {m[:34]}", self.font_small, (150, 200, 150)); y += 14

        self._hud_line(x0, y, f"mistakes ({len(self.discovered_mistakes)}):", self.font_small, (200, 200, 200)); y += 16
        for m in self.discovered_mistakes[-3:]:
            self._hud_line(x0, y, f"  - {m[:34]}", self.font_small, (220, 150, 150)); y += 14

        # Key hints
        y = self.grid_side - 143
        for line in [
            "1..5/7: action",
            "click: ACTION6(x,y)",
            "R: reset   K: kill-ep",
            "Tab: intent   H/N: hyp",
            "C/X: hyp +/-   G: goal",
            "M: mech   F: mistake",
            "T: type   O: obj",
            "Q/Esc: quit",
        ]:
            self._hud_line(x0, y, line, self.font_small, (140, 140, 160)); y += 13

        # Flash message
        if self._flash_msg and time.time() < self._flash_until:
            self._hud_line(x0, self.grid_side - 16, self._flash_msg, self.font_small, (255, 200, 100))

        pg.display.flip()

    def _hud_line(self, x: int, y: int, text: str, font: Any, colour: Tuple[int, int, int]) -> None:
        surf = font.render(text, True, colour)
        self.screen.blit(surf, (x, y))

    def _flash(self, msg: str, seconds: float = 1.8) -> None:
        self._flash_msg = msg
        self._flash_until = time.time() + seconds

    # ------------------------------------------------------------------
    # Inline text prompt (modal, pygame)
    # ------------------------------------------------------------------
    def _prompt(self, label: str, initial: str) -> Optional[str]:
        """Blocking text input. Returns the string on Enter, None on Esc."""
        pg = self._pygame
        buf = list(initial)
        clock = pg.time.Clock()
        while True:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_RETURN:
                        return "".join(buf)
                    if event.key == pg.K_ESCAPE:
                        return None
                    if event.key == pg.K_BACKSPACE:
                        if buf:
                            buf.pop()
                    else:
                        ch = event.unicode
                        if ch and ch.isprintable():
                            buf.append(ch)
                elif event.type == pg.QUIT:
                    return None

            # Redraw with overlay
            self._render()
            overlay = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._hud_line(30, self.grid_side // 2 - 20, f"{label}:", self.font_big, (255, 255, 255))
            shown = "".join(buf) + "_"
            self._hud_line(30, self.grid_side // 2 + 6, shown, self.font, (255, 220, 120))
            self._hud_line(30, self.grid_side // 2 + 30, "Enter=ok   Esc=cancel", self.font_small, (180, 180, 180))
            pg.display.flip()
            clock.tick(30)
