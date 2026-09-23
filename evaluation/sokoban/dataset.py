"""Visual-Jev request construction. Hidden state stays outside requests."""
import hashlib
from . import VERSION
from .env import ACTIONS, canonical_hash, state_hash

RULES = {
    "en": "Legend: dark walls, light floor, green rings are goals, crossed squares are boxes, and the round face is the player. Goal rings remain visible under boxes and the player. Move one cell up, down, left or right. You can push one box, but cannot pull, pass through walls, or push two boxes together. Put every box on a goal using the fewest total moves. An invalid move consumes a step without changing the board.",
    "zh": "图例：深色墙，浅色地板，绿色圆环为目标，带叉方块为箱子，圆脸为玩家；箱子和玩家下方的目标环仍可见。每次向上下左右移动一格，可以推动一个箱子，不能拉箱、穿墙或连推两个箱子。请用最少移动步数将所有箱子推到目标。无效动作消耗一步但不改变棋盘。",
}
DIRECTION_NAMES = {"en": dict(zip(ACTIONS, ("Up", "Down", "Left", "Right"))),
                   "zh": dict(zip(ACTIONS, ("向上", "向下", "向左", "向右")))}


def action_request(image_path, language="en", action_order=ACTIONS):
    return {
        "state": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": RULES[language]},
            {"type": "image_url", "image_url": {"url": str(image_path)}},
        ]}]},
        "questions": {"next_action": {
            "type": "choice", "instructions": "Which action should you take next?" if language == "en" else "下一步选择哪个动作？",
            "criteria": {a: DIRECTION_NAMES[language][a] for a in action_order}}},
    }


def make_record(board, state, index, image_path, image_sha256, group_id,
                source, provenance, seed, number, rng, theme, tile_size):
    language = "zh" if number % 2 == 0 else "en"
    order = list(ACTIONS)
    rng.shuffle(order)
    request = action_request(image_path, language, order)
    optimal = index.optimal_actions(state)
    if not optimal:
        raise ValueError("Action training requires a certified nonterminal solvable state")
    target = {a: 1 / len(optimal) if a in optimal else 0.0 for a in order}
    targets = {"next_action": {"probabilities": target}}
    annotations = {"next_action": {"label_semantics": "optimal_action_policy",
                    "optimal_actions": optimal, "solver_status": "optimal_proven",
                    "label_method": "reverse_multisource_bfs"}}
    return {"group_id": group_id, "request": request, "targets": targets,
            "assets": [{"path": str(image_path), "sha256": image_sha256}],
            "meta": {"source": VERSION, "record_id": "sokoban-{:06d}".format(number),
                     "split": "train", "domain": "game", "environment": "sokoban",
                     "modality": "image", "language_bucket": language,
                     "objective": "minimum_moves", "sampling_source": source,
                     "state": state.to_dict(), "state_id": state_hash(board, state),
                     "canonical_state_id": canonical_hash(board, state),
                     "optimal_distance": index.distances[state], "solution_actions": index.solution(state),
                     "question_annotations": annotations, "provenance": provenance,
                     "generator_seed": seed, "renderer_version": VERSION, "solver_version": VERSION,
                     "theme": theme, "tile_size": tile_size}}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
