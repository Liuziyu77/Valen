"""Pure, deterministic rules. No renderer, policy or solver dependencies."""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

ACTIONS = ("up", "down", "left", "right")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class State(NamedTuple):
    player: int
    boxes: tuple

    def to_dict(self):
        return {"player": self.player, "boxes": list(self.boxes)}

    @classmethod
    def from_dict(cls, value):
        return cls(int(value["player"]), tuple(sorted(value["boxes"])))


@dataclass(frozen=True)
class Board:
    height: int
    width: int
    floors: frozenset
    goals: frozenset

    def __post_init__(self):
        if self.height < 3 or self.width < 3 or not self.goals or not self.goals <= self.floors:
            raise ValueError("Invalid board dimensions or goals")
        if any(not 0 <= p < self.height * self.width for p in self.floors):
            raise ValueError("Floor outside board")
        if any(p // self.width in (0, self.height - 1) or p % self.width in (0, self.width - 1)
               for p in self.floors):
            raise ValueError("Board must have an enclosing wall")

    def neighbor(self, position, action):
        dr, dc = DELTAS[ACTIONS.index(action)]
        r, c = divmod(position, self.width)
        r, c = r + dr, c + dc
        return r * self.width + c if 0 <= r < self.height and 0 <= c < self.width else -1

    def validate(self, state):
        if (state.player not in self.floors or state.player in state.boxes
                or len(state.boxes) != len(self.goals) or len(set(state.boxes)) != len(state.boxes)
                or not set(state.boxes) <= self.floors or tuple(sorted(state.boxes)) != state.boxes):
            raise ValueError("Invalid player or boxes")
        return state

    def solved(self, state):
        return frozenset(state.boxes) == self.goals

    def to_dict(self):
        return {"height": self.height, "width": self.width,
                "floors": sorted(self.floors), "goals": sorted(self.goals)}

    @classmethod
    def from_dict(cls, value):
        return cls(value["height"], value["width"], frozenset(value["floors"]), frozenset(value["goals"]))

    def ascii(self, state=None):
        rows = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                p = r * self.width + c
                char = "#" if p not in self.floors else "." if p in self.goals else " "
                if state is not None:
                    if p in state.boxes:
                        char = "*" if p in self.goals else "$"
                    if p == state.player:
                        char = "+" if p in self.goals else "@"
                row.append(char)
            rows.append("".join(row))
        return rows


def parse_ascii(text):
    rows = text.splitlines() if isinstance(text, str) else text
    if not rows or len(set(map(len, rows))) != 1:
        raise ValueError("ASCII board must be rectangular")
    width = len(rows[0])
    floors, goals, boxes, players = set(), set(), [], []
    for r, row in enumerate(rows):
        for c, char in enumerate(row):
            p = r * width + c
            if char not in "# .$*@+":
                raise ValueError("Unknown tile: " + char)
            if char != "#":
                floors.add(p)
            if char in ".*+":
                goals.add(p)
            if char in "$*":
                boxes.append(p)
            if char in "@+":
                players.append(p)
    if len(players) != 1:
        raise ValueError("Exactly one player required")
    board = Board(len(rows), width, frozenset(floors), frozenset(goals))
    return board, board.validate(State(players[0], tuple(sorted(boxes))))


def transition(board, state, action):
    """Invalid moves are no-ops; returns (state, moved, pushed)."""
    if action not in ACTIONS:
        raise ValueError("Unknown action: " + str(action))
    dest = board.neighbor(state.player, action)
    if dest not in board.floors:
        return state, False, False
    if dest not in state.boxes:
        return State(dest, state.boxes), True, False
    beyond = board.neighbor(dest, action)
    if beyond not in board.floors or beyond in state.boxes:
        return state, False, False
    return State(dest, tuple(sorted((set(state.boxes) - {dest}) | {beyond}))), True, True


def canonical_hash(board, state=None):
    """D4 symmetry + wall-margin invariant semantic identity (never pixel hashes)."""
    rows = board.ascii(state)
    occupied = [(r, c) for r, row in enumerate(rows) for c, char in enumerate(row) if char != "#"]
    r0, r1 = min(r for r, _ in occupied), max(r for r, _ in occupied)
    c0, c1 = min(c for _, c in occupied), max(c for _, c in occupied)
    rows = [row[c0:c1 + 1] for row in rows[r0:r1 + 1]]
    variants = []
    for _ in range(4):
        variants.append("\n".join(rows))
        variants.append("\n".join(row[::-1] for row in rows))
        rows = ["".join(row) for row in zip(*rows[::-1])]
    return hashlib.sha256(min(variants).encode()).hexdigest()


def state_hash(board, state):
    raw = json.dumps([board.to_dict(), state.to_dict()], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class SokobanEnv:
    def __init__(self, board, initial_state, max_steps=200):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.board, self.initial_state, self.max_steps = board, board.validate(initial_state), max_steps
        self.reset()

    def reset(self):
        self.state, self.steps, self.pushes = self.initial_state, 0, 0
        return self.state

    def step(self, action):
        if self.board.solved(self.state) or self.steps >= self.max_steps:
            raise RuntimeError("Episode has terminated")
        self.state, moved, pushed = transition(self.board, self.state, action)
        self.steps += 1
        self.pushes += int(pushed)
        solved = self.board.solved(self.state)
        return self.state, {"moved": moved, "pushed": pushed, "solved": solved,
                            "terminated": solved, "truncated": not solved and self.steps >= self.max_steps}

    def clone(self):
        result = SokobanEnv(self.board, self.initial_state, self.max_steps)
        result.state, result.steps, result.pushes = self.state, self.steps, self.pushes
        return result
