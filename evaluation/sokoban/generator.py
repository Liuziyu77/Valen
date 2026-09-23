"""Connected procedural rooms and certified trajectory sampling."""
import random
from .env import ACTIONS, Board, canonical_hash, transition
from .solver import ReverseIndex

SOURCES = ("optimal_trajectory", "solvable_deviation", "failure_correction")


def generate_board(seed):
    rng = random.Random(seed)
    height, width = rng.randint(6, 9), rng.randint(6, 9)
    inside = {r * width + c for r in range(1, height - 1) for c in range(1, width - 1)}
    count = rng.choices((1, 2, 3), weights=(25, 45, 30))[0]
    target = min(len(inside), rng.randint(16 + count * 2, 22 + count * 3))
    floors = {rng.choice(sorted(inside))}
    while len(floors) < target:
        boundary = sorted({q for p in floors for q in (p - width, p + width, p - 1, p + 1)
                           if q in inside and q not in floors})
        floors.add(rng.choice(boundary))
    # A solved goal needs at least one possible reverse pull to be useful.
    candidates = [p for p in sorted(floors)
                  if (p - 1 in floors and p + 1 in floors) or (p - width in floors and p + width in floors)]
    if len(candidates) < count:
        return None
    return Board(height, width, frozenset(floors), frozenset(rng.sample(candidates, count)))


def difficulty(distance):
    return "easy" if distance <= 10 else "medium" if distance <= 30 else "hard"


def choose_initial(index, rng, bucket=None):
    candidates = [s for s, d in index.distances.items() if d > 0 and (bucket is None or difficulty(d) == bucket)]
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_training(index, initial, rng, limits):
    """Every deviation/correction carries a replayable provenance prefix."""
    board = index.board
    path, states = index.solution(initial), [initial]
    for action in path:
        states.append(transition(board, states[-1], action)[0])
    candidates = {source: {} for source in SOURCES}

    # Explore actual noisy rollouts from this level's initial state. A failure
    # is an executed action with a proof of unsolvability, not a search timeout.
    # Also inspect each reference decision point for one-action failure branches.
    for i, state in enumerate(states[:-1]):
        for action in ACTIONS:
            child, moved, _ = transition(board, state, action)
            if moved and index.status(child) == "proven_unsolvable":
                candidates["failure_correction"].setdefault(state, {
                    "prefix_actions": path[:i], "failed_action": action,
                    "failure_state": child.to_dict(), "failure_type": "proven_deadlock",
                    "certificate": "static_dead_square" if any(b in index.dead_squares for b in child.boxes)
                    else "exhausted_reverse_graph"})

    for attempt in range(12):
        # Start from a replayable reference prefix, then actually deviate.
        offset = rng.randrange(len(states) - 1)
        state, prefix = states[offset], list(path[:offset])
        for _ in range(8):
            optimal = index.optimal_actions(state)
            choices = [a for a in ACTIONS if a not in optimal and transition(board, state, a)[1]]
            if not choices:
                break
            action = rng.choice(choices)
            child = transition(board, state, action)[0]
            if index.status(child) == "proven_unsolvable":
                candidates["failure_correction"].setdefault(state, {
                    "prefix_actions": list(prefix), "failed_action": action,
                    "failure_state": child.to_dict(), "failure_type": "proven_deadlock",
                    "certificate": "static_dead_square" if any(b in index.dead_squares for b in child.boxes)
                    else "exhausted_reverse_graph"})
                break
            if child not in index.distances or index.distances[child] == 0:
                break
            parent, state = state, child
            prefix.append(action)
            candidates["solvable_deviation"].setdefault(state, {
                "prefix_actions": list(prefix), "deviation_action": action,
                "parent_state": parent.to_dict(), "parent_distance": index.distances[parent]})

    for i, state in enumerate(states[:-1]):
        candidates["optimal_trajectory"][state] = {"prefix_actions": path[:i]}

    selected, seen = [], set()
    for source in reversed(SOURCES):
        items = list(candidates[source].items())
        rng.shuffle(items)
        if source == "optimal_trajectory":
            # Preserve initial states when they have not already been selected.
            items.sort(key=lambda item: item[0] != initial)
        count = 0
        for state, provenance in items:
            if count >= limits[source]:
                break
            identity = canonical_hash(board, state)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append((source, state, provenance))
            count += 1
    return selected


def generate_candidate(seed, max_states=20000, max_depth=80):
    board = generate_board(seed)
    if board is None:
        return None
    index = ReverseIndex(board, max_states=max_states, max_depth=max_depth)
    if max(index.distances.values()) < 4:
        return None
    return board, index
