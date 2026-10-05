"""Reconstruct published eval maps with the original generation seeds."""
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageChops

from ._upstream.envs import make_env, ACTIONS, DELTAS

SIZES = {"maze": (5, 7, 9, 11), "frozen_lake": (5, 6, 7, 8), "sokoban": (6, 7, 8),
         "lane_racer": (12, 16, 20), "parkour_runner": (12, 16, 20)}


def stable_int(*parts):
    return int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def solution(env):
    if env.name != "maze":
        return env.solution()
    distances, position, path = env._distances(env.goal), env.player, []
    while position != env.goal:
        for action in ACTIONS:
            dr, dc = DELTAS[action]
            next_position = position[0]+dr, position[1]+dc
            if distances.get(next_position) == distances[position]-1:
                path.append(action)
                position = next_position
                break
        else:
            raise ValueError("Maze has no shortest solution")
    return tuple(path)


def make_game(record, render_size=384):
    meta = record["meta"]
    environment, seed, split = meta["environment"], meta["map_seed"], meta["split"]
    if split != "eval":
        raise ValueError("Episode evaluation requires published eval maps")
    arcade = environment in {"lane_racer", "parkour_runner"}
    parts = ("arcade-v2", environment, split, seed, "config") if arcade else (environment, split, seed, "config")
    rng = random.Random(stable_int(*parts))
    size, style = rng.choice(SIZES[environment]), rng.randrange(6)
    common = {"render_size": render_size, "style": style}
    if environment == "maze":
        env = make_env(environment, rows=(size-1)//2, cols=(size-1)//2, loop_fraction=rng.uniform(.05, .22), **common)
    elif environment == "frozen_lake":
        env = make_env(environment, size=size, hole_fraction=rng.uniform(.12, .24), slippery=False,
                       horizon=size*rng.choice((4, 5, 6)), **common)
    elif environment == "sokoban":
        boxes = 2 if size >= 7 and rng.random() < (.30 if size >= 9 else .20) else 1
        env = make_env(environment, rows=size, cols=size, boxes=boxes, search_limit=20000, **common)
    elif environment == "lane_racer":
        env = make_env(environment, rows=size, lanes=rng.choice((3, 4, 5)), obstacle_fraction=rng.uniform(.12, .26), **common)
    else:
        env = make_env(environment, length=size, hazard_fraction=rng.uniform(.30, .55), **common)
    env.reset(seed)
    if environment == "sokoban":
        # Preserve map selection at 20k, then use the released label solver budget.
        # 地图选择保持 20k 搜索预算，后续状态使用发布标签的精确搜索预算。
        env.search_limit = 1_000_000
        env._solve_cache.clear()
    return env


def eval_maps(paths):
    cases = {}
    groups = set()
    for path in map(Path, paths):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record["group_id"]
                groups.add(key)
                if key not in cases or record["meta"]["state_index"] < cases[key]["meta"]["state_index"]:
                    # Resolve images using this snapshot's root, never the working directory.
                    # 图片相对路径以数据文件目录为基准。
                    for asset in record.get("assets", []):
                        asset["path"] = str((path.parent / asset["path"]).resolve())
                    cases[key] = record
    if not cases or set(cases) != groups:
        raise ValueError("Game eval maps have missing initial states")
    return [cases[key] for key in sorted(cases)]


def validate_initial_frame(env, record):
    # Deduplication can remove initial PNGs; replay the published generator prefix.
    # 去重可能移除初始 PNG，先复现发布状态，再重置到地图起点评测。
    planned = env.solution() if env.name == "sokoban" else ()
    while env.steps < record["meta"]["episode_step"]:
        actions = env.optimal_actions()
        action = planned[env.steps] if planned else actions[stable_int(env.seed,env.steps,"trajectory") % len(actions)]
        env.step(action)
    with Image.open(record["assets"][0]["path"]) as image:
        actual, expected = env.observation().convert("RGB"), image.convert("RGB")
        if actual.size != expected.size or ImageChops.difference(actual, expected).getbbox() is not None:
            raise ValueError("Reconstructed game differs from published eval image")
    expected_prompt = record["request"]["state"]["messages"][0]["content"]
    if env.prompt() != expected_prompt:
        raise ValueError("Reconstructed game rules differ from published eval prompt")
    if env.name == "sokoban":
        env.search_limit = 20_000
        env._solve_cache.clear()
    env.reset(record["meta"]["map_seed"])
    if env.name == "sokoban":
        env.search_limit = 1_000_000
        env._solve_cache.clear()


def action_request(record, env, image_path):
    # Model receives only pixels, rules and candidates. / 不向模型提供标签或求解器信息。
    return {"state": {"messages": [{"role": "system", "content": env.prompt()},
            {"role": "user", "content": [{"type": "text", "text": env.observation_prompt()},
             {"type": "image_url", "image_url": {"url": str(image_path.resolve())}}]}]},
            "questions": record["request"]["questions"]}
