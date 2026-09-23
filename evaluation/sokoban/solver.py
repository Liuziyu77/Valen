"""Exact move-optimal search; a budget limit never means unsolvable."""
from collections import deque
from dataclasses import dataclass
from .env import ACTIONS, OPPOSITE, State, transition


def reverse_predecessors(board, state):
    for action in ACTIONS:
        previous_player = board.neighbor(state.player, OPPOSITE[action])
        if previous_player not in board.floors or previous_player in state.boxes:
            continue
        # Undo a walk.
        yield State(previous_player, state.boxes), action
        # Undo a push: the box ahead of the current player was on their cell.
        box = board.neighbor(state.player, action)
        if box in state.boxes:
            boxes = tuple(sorted((set(state.boxes) - {box}) | {state.player}))
            yield State(previous_player, boxes), action


def static_dead_squares(board):
    """Cells from which even an isolated box cannot reach any goal."""
    live = set(board.goals)
    queue = deque(sorted(live))
    while queue:
        box = queue.popleft()
        for action in ACTIONS:
            before = board.neighbor(box, OPPOSITE[action])
            support = board.neighbor(before, OPPOSITE[action]) if before in board.floors else -1
            if before in board.floors and support in board.floors and before not in live:
                live.add(before)
                queue.append(before)
    return board.floors - live


@dataclass
class SolveResult:
    status: str
    actions: list
    expanded: int

    @property
    def distance(self):
        return len(self.actions) if self.status == "optimal_proven" else None


def solve_bfs(board, initial, max_nodes=100000):
    """Independent forward BFS, used as oracle and correctness cross-check."""
    board.validate(initial)
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    queue, parent = deque([initial]), {initial: None}
    expanded = 0
    while queue:
        state = queue.popleft()
        if board.solved(state):
            path = []
            while parent[state] is not None:
                state, action = parent[state]
                path.append(action)
            return SolveResult("optimal_proven", path[::-1], expanded)
        if expanded >= max_nodes:
            return SolveResult("unknown", [], expanded)
        expanded += 1
        for action in ACTIONS:
            child, moved, _ = transition(board, state, action)
            if moved and child not in parent:
                parent[child] = (state, action)
                queue.append(child)
    return SolveResult("proven_unsolvable", [], expanded)


class ReverseIndex:
    """Multi-source BFS from every solved player position.

    Discovered distances are exact even if construction stops at its budget.
    FIFO discovery guarantees all depth d-1 states were discovered before any
    depth d state, so all optimal successors of every indexed state are present.
    Unseen states are unknown unless the entire graph was exhausted or a sound
    static dead-square certificate applies.
    """
    def __init__(self, board, max_states=20000, max_depth=80):
        seeds = [State(p, tuple(sorted(board.goals))) for p in sorted(board.floors - board.goals)]
        if max_states < len(seeds) or max_depth < 1:
            raise ValueError("Invalid reverse search budget")
        self.board, self.distances = board, dict.fromkeys(seeds, 0)
        self.dead_squares = static_dead_squares(board)
        self.expanded, self.complete = 0, True
        queue = deque(seeds)
        while queue:
            state = queue.popleft()
            distance = self.distances[state]
            self.expanded += 1
            for previous, _ in reverse_predecessors(board, state):
                if previous in self.distances:
                    continue
                if distance >= max_depth:
                    self.complete = False
                    continue
                if len(self.distances) >= max_states:
                    self.complete = False
                    return
                self.distances[previous] = distance + 1
                queue.append(previous)

    def optimal_actions(self, state):
        distance = self.distances.get(state)
        if distance is None or distance == 0:
            return []
        return [a for a in ACTIONS if self.distances.get(transition(self.board, state, a)[0]) == distance - 1]

    def status(self, state):
        if state in self.distances:
            return "solvable"
        if any(box in self.dead_squares for box in state.boxes) or self.complete:
            return "proven_unsolvable"
        return "unknown"

    def solution(self, state):
        if state not in self.distances:
            raise ValueError("No certified solution for state")
        actions = []
        while self.distances[state]:
            action = self.optimal_actions(state)[0]
            state = transition(self.board, state, action)[0]
            actions.append(action)
        return actions
