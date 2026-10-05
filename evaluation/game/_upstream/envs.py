"""Deterministic, renderable grid games with exact next-action oracles.

The same classes are used by ``generate_dataset.py`` and ``compare_agents.py``.
Every environment exposes the deliberately small API requested by the project:
``reset(seed)``, ``observation()``, and ``step(action)``.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
import hashlib
import math
import random
from typing import Iterable

from PIL import Image, ImageDraw


ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_ZH = {"up": "向上", "down": "向下", "left": "向左", "right": "向右"}

THEMES = (
    {"bg": "#f4ead5", "floor": "#fff8e8", "floor2": "#f4e2bc", "wall": "#50423a", "edge": "#291f1a", "player": "#e4572e", "goal": "#f3b61f", "accent": "#2e86ab"},
    {"bg": "#071426", "floor": "#17324d", "floor2": "#1e4262", "wall": "#7a90a8", "edge": "#d4e5f5", "player": "#ff5d8f", "goal": "#ffe66d", "accent": "#42d9c8"},
    {"bg": "#e7f2e4", "floor": "#f7fff4", "floor2": "#d7ebd3", "wall": "#386641", "edge": "#18351e", "player": "#3a86ff", "goal": "#ffbe0b", "accent": "#8338ec"},
    {"bg": "#27213c", "floor": "#4b3f72", "floor2": "#5d4e88", "wall": "#f4d35e", "edge": "#fff4bd", "player": "#ee4266", "goal": "#2de2e6", "accent": "#00bbf9"},
    {"bg": "#e8eef2", "floor": "#ffffff", "floor2": "#dbe5ea", "wall": "#334e58", "edge": "#102a43", "player": "#d64545", "goal": "#f6c85f", "accent": "#2f80ed"},
    {"bg": "#2b2118", "floor": "#725436", "floor2": "#856643", "wall": "#c7a76c", "edge": "#3b2b1a", "player": "#68d391", "goal": "#fbd38d", "accent": "#63b3ed"},
)

THEME_DESCRIPTIONS = (
    {"floor": "象牙白或浅米色", "wall": "深棕色", "player": "橙红色圆形", "goal": "金黄色星形"},
    {"floor": "深蓝色", "wall": "灰蓝色", "player": "粉红色三角形", "goal": "黄色圆环"},
    {"floor": "白色或浅绿色", "wall": "深绿色", "player": "蓝色机器人", "goal": "黄色旗帜"},
    {"floor": "深紫色", "wall": "黄色", "player": "粉红色圆形", "goal": "青色星形"},
    {"floor": "白色或浅蓝色", "wall": "深蓝灰色", "player": "红色三角形", "goal": "金黄色圆环"},
    {"floor": "棕色", "wall": "浅棕黄色", "player": "绿色机器人", "goal": "浅黄色旗帜"},
)


def _seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _add(position: tuple[int, int], action: str) -> tuple[int, int]:
    delta = DELTAS[action]
    return position[0] + delta[0], position[1] + delta[1]


def _star(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        distance = radius if index % 2 == 0 else radius * 0.42
        points.append((cx + math.cos(angle) * distance, cy + math.sin(angle) * distance))
    return points


def transition_feedback(environment: str, before: Image.Image, after: Image.Image, *, success: bool) -> str:
    """Describe only the observed transition, never hidden solver state."""
    if success:
        return "动作已执行并到达目标。"
    unchanged = before.size == after.size and before.mode == after.mode and before.tobytes() == after.tobytes()
    if unchanged and environment == "maze":
        return "撞墙：该动作未生效，玩家位置未改变。"
    if unchanged:
        return "该动作未改变当前游戏状态。"
    return "动作已执行，游戏画面和状态已更新。"


class GridEnv:
    """Small common API shared by dataset generation and live evaluation."""

    name = "grid"
    actions = ACTIONS

    def __init__(self, *, render_size: int = 512, style: int | None = None, max_steps: int | None = None):
        self.render_size = render_size
        self.requested_style = style
        self.max_steps = max_steps
        self.seed = 0
        self.steps = 0
        self.done = False
        self.success = False
        self.style_id = 0
        self.rng = random.Random(0)

    def reset(self, seed: int) -> Image.Image:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.steps = 0
        self.done = False
        self.success = False
        self.style_id = self.requested_style if self.requested_style is not None else _seed(self.name, seed, "style") % len(THEMES)
        self._reset()
        return self.observation()

    def _reset(self) -> None:
        raise NotImplementedError

    def observation(self, size: int | None = None) -> Image.Image:
        raise NotImplementedError

    def step(self, action: str) -> tuple[Image.Image, float, bool, dict]:
        if action not in self.actions:
            raise ValueError(f"Unknown action {action!r}; expected one of {self.actions}")
        if self.done:
            return self.observation(), 0.0, True, self.info()
        reward = float(self._step(action))
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps and not self.done:
            self.done = True
        return self.observation(), reward, self.done, self.info()

    def _step(self, action: str) -> float:
        raise NotImplementedError

    def optimal_actions(self) -> tuple[str, ...]:
        raise NotImplementedError

    def prompt(self) -> str:
        raise NotImplementedError

    def observation_prompt(self) -> str:
        return "目前的画面状态如下："

    def info(self) -> dict:
        return {
            "environment": self.name,
            "seed": self.seed,
            "steps": self.steps,
            "done": self.done,
            "success": self.success,
            "optimal_actions": list(self.optimal_actions()) if not self.done else [],
            "actions": list(self.actions),
            "style_id": self.style_id,
        }

    def map_metadata(self) -> dict:
        raise NotImplementedError

    def _canvas(self, rows: int, cols: int, size: int | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw, float, float, float]:
        size = int(size or self.render_size)
        theme = THEMES[self.style_id]
        margin = max(8, size // 32)
        tile = min((size - 2 * margin) / cols, (size - 2 * margin) / rows)
        width = max(1, round(cols * tile + 2 * margin))
        height = max(1, round(rows * tile + 2 * margin))
        image = Image.new("RGB", (width, height), theme["bg"])
        draw = ImageDraw.Draw(image)
        ox = (width - cols * tile) / 2
        oy = (height - rows * tile) / 2
        # Three genuinely different background motifs, not just palette swaps.
        motif = self.style_id % 3
        if motif == 1:
            for y in range(0, height, max(10, size // 30)):
                draw.line((0, y, width, y), fill=theme["floor2"], width=1)
        elif motif == 2:
            gap = max(12, size // 24)
            for y in range(gap // 2, height, gap):
                for x in range((y // gap % 2) * gap // 2, width, gap):
                    draw.ellipse((x, y, x + 2, y + 2), fill=theme["floor2"])
        return image, draw, tile, ox, oy

    def _tile_box(self, row: int, col: int, tile: float, ox: float, oy: float, inset: float = 0.0) -> tuple[float, float, float, float]:
        return (ox + col * tile + inset, oy + row * tile + inset, ox + (col + 1) * tile - inset, oy + (row + 1) * tile - inset)

    def _draw_player(self, draw: ImageDraw.ImageDraw, row: int, col: int, tile: float, ox: float, oy: float) -> None:
        theme = THEMES[self.style_id]
        x0, y0, x1, y1 = self._tile_box(row, col, tile, ox, oy, tile * 0.20)
        variant = self.style_id % 3
        if variant == 0:
            draw.ellipse((x0, y0, x1, y1), fill=theme["player"], outline=theme["edge"], width=max(1, int(tile * 0.06)))
            eye = tile * 0.07
            draw.ellipse((x0 + tile * 0.25, y0 + tile * 0.22, x0 + tile * 0.25 + eye, y0 + tile * 0.22 + eye), fill="white")
            draw.ellipse((x1 - tile * 0.32, y0 + tile * 0.22, x1 - tile * 0.32 + eye, y0 + tile * 0.22 + eye), fill="white")
        elif variant == 1:
            draw.polygon(((x0 + x1) / 2, y0, x1, y1, (x0 + x1) / 2, y1 - tile * 0.18, x0, y1), fill=theme["player"], outline=theme["edge"])
        else:
            radius = tile * 0.12
            draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=theme["player"], outline=theme["edge"], width=max(1, int(tile * 0.05)))
            draw.line((x0 + tile * 0.2, y0 - tile * 0.10, x1 - tile * 0.2, y0 - tile * 0.10), fill=theme["accent"], width=max(2, int(tile * 0.05)))

    def _draw_goal(self, draw: ImageDraw.ImageDraw, row: int, col: int, tile: float, ox: float, oy: float) -> None:
        theme = THEMES[self.style_id]
        cx, cy = ox + (col + 0.5) * tile, oy + (row + 0.5) * tile
        variant = self.style_id % 3
        if variant == 0:
            draw.polygon(_star(cx, cy, tile * 0.34), fill=theme["goal"], outline=theme["edge"])
        elif variant == 1:
            draw.ellipse((cx - tile * 0.32, cy - tile * 0.32, cx + tile * 0.32, cy + tile * 0.32), outline=theme["goal"], width=max(2, int(tile * 0.11)))
            draw.ellipse((cx - tile * 0.10, cy - tile * 0.10, cx + tile * 0.10, cy + tile * 0.10), fill=theme["goal"])
        else:
            pole_x = cx - tile * 0.22
            draw.line((pole_x, cy - tile * 0.34, pole_x, cy + tile * 0.34), fill=theme["edge"], width=max(2, int(tile * 0.06)))
            draw.polygon(((pole_x, cy - tile * 0.32), (cx + tile * 0.30, cy - tile * 0.17), (pole_x, cy)), fill=theme["goal"], outline=theme["edge"])


class MazeEnv(GridEnv):
    name = "maze"

    def __init__(self, *, rows: int = 7, cols: int | None = None, loop_fraction: float = 0.12, **kwargs):
        super().__init__(**kwargs)
        self.logical_rows = int(rows)
        self.logical_cols = int(cols or rows)
        self.loop_fraction = float(loop_fraction)

    def _reset(self) -> None:
        rows, cols = self.logical_rows, self.logical_cols
        self.grid_rows, self.grid_cols = rows * 2 + 1, cols * 2 + 1
        self.passable: set[tuple[int, int]] = {(1, 1)}
        visited = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            cell = stack[-1]
            neighbors = []
            for dr, dc in DELTAS.values():
                nxt = cell[0] + dr, cell[1] + dc
                if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in visited:
                    neighbors.append(nxt)
            if not neighbors:
                stack.pop()
                continue
            nxt = self.rng.choice(neighbors)
            visited.add(nxt)
            a = (cell[0] * 2 + 1, cell[1] * 2 + 1)
            b = (nxt[0] * 2 + 1, nxt[1] * 2 + 1)
            self.passable.update((a, b, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)))
            stack.append(nxt)
        walls = []
        for row in range(1, self.grid_rows - 1):
            for col in range(1, self.grid_cols - 1):
                if (row, col) in self.passable or row % 2 == col % 2:
                    continue
                if any(_add((row, col), action) in self.passable for action in ACTIONS):
                    walls.append((row, col))
        self.rng.shuffle(walls)
        for wall in walls[: round(len(walls) * self.loop_fraction)]:
            self.passable.add(wall)
        cells = [(r * 2 + 1, c * 2 + 1) for r in range(rows) for c in range(cols)]
        self.start = self.rng.choice(cells)
        distances = self._distances(self.start)
        farthest = sorted(cells, key=lambda p: (distances.get(p, -1), p), reverse=True)
        self.goal = self.rng.choice(farthest[: max(1, len(farthest) // 8)])
        self.player = self.start
        if self.max_steps is None:
            self.max_steps = max(40, rows * cols * 4)

    def _distances(self, origin: tuple[int, int]) -> dict[tuple[int, int], int]:
        distances = {origin: 0}
        queue = deque([origin])
        while queue:
            current = queue.popleft()
            for action in ACTIONS:
                nxt = _add(current, action)
                if nxt in self.passable and nxt not in distances:
                    distances[nxt] = distances[current] + 1
                    queue.append(nxt)
        return distances

    def _step(self, action: str) -> float:
        nxt = _add(self.player, action)
        if nxt in self.passable:
            self.player = nxt
        if self.player == self.goal:
            self.done = self.success = True
            return 1.0
        return -0.01

    def optimal_actions(self) -> tuple[str, ...]:
        if self.player == self.goal:
            return ()
        distances = self._distances(self.goal)
        legal = [(distances.get(_add(self.player, action), 10**9), action) for action in ACTIONS if _add(self.player, action) in self.passable]
        best = min(distance for distance, _ in legal)
        return tuple(action for distance, action in legal if distance == best)

    def prompt(self) -> str:
        visual = THEME_DESCRIPTIONS[self.style_id]
        return (
            f"你正在玩一个画面中共 {self.grid_rows} 行、{self.grid_cols} 列的俯视网格迷宫。玩家是带眼睛的圆形、三角形或机器人图标，目标是星形、圆环或旗帜图标。"
            f"本局中，{visual['floor']}格是可走通路，{visual['wall']}格是墙，玩家是{visual['player']}图标，目标是{visual['goal']}图标。"
            "玩家向上向下向左向右移动分别对应 up/down/left/right，每步只移动到紧邻的一格，不能进入墙。"
            "请沿通路规划到目标的最短路线并选择下一步动作。"
        )

    def observation_prompt(self) -> str:
        return "目前的路径状态如下："

    def map_metadata(self) -> dict:
        return {"logical_size": [self.logical_rows, self.logical_cols], "render_grid_size": [self.grid_rows, self.grid_cols], "start": list(self.start), "goal": list(self.goal), "loop_fraction": self.loop_fraction}

    def observation(self, size: int | None = None) -> Image.Image:
        image, draw, tile, ox, oy = self._canvas(self.grid_rows, self.grid_cols, size)
        theme = THEMES[self.style_id]
        wall_variant = self.style_id % 3
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                box = self._tile_box(row, col, tile, ox, oy)
                if (row, col) in self.passable:
                    color = theme["floor"] if (row + col) % 2 == 0 else theme["floor2"]
                    draw.rectangle(box, fill=color)
                elif wall_variant == 0:
                    draw.rectangle(box, fill=theme["wall"], outline=theme["edge"], width=max(1, int(tile * 0.05)))
                elif wall_variant == 1:
                    inset = tile * 0.05
                    draw.rounded_rectangle(self._tile_box(row, col, tile, ox, oy, inset), radius=tile * 0.18, fill=theme["wall"], outline=theme["edge"], width=max(1, int(tile * 0.05)))
                else:
                    draw.rectangle(box, fill=theme["wall"], outline=theme["edge"])
                    draw.line((box[0], (box[1] + box[3]) / 2, box[2], (box[1] + box[3]) / 2), fill=theme["edge"], width=1)
                    if (row + col) % 2:
                        draw.line(((box[0] + box[2]) / 2, box[1], (box[0] + box[2]) / 2, (box[1] + box[3]) / 2), fill=theme["edge"], width=1)
        self._draw_goal(draw, *self.goal, tile, ox, oy)
        self._draw_player(draw, *self.player, tile, ox, oy)
        return image


class FrozenLakeEnv(GridEnv):
    name = "frozen_lake"

    def __init__(self, *, size: int = 7, hole_fraction: float = 0.20, slippery: bool = True, horizon: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.size = int(size)
        self.hole_fraction = float(hole_fraction)
        self.slippery = bool(slippery)
        self.horizon = int(horizon or self.size * 5)
        if self.max_steps is None:
            self.max_steps = self.horizon

    def _reset(self) -> None:
        n = self.size
        # Sample both endpoints across the full board instead of baking in the
        # usual top-left -> bottom-right layout.  A modest distance floor keeps
        # episodes useful while retaining edge, corner, and interior starts.
        cells = [(row, col) for row in range(n) for col in range(n)]
        self.start = self.rng.choice(cells)
        goal_candidates = [
            cell
            for cell in cells
            if cell != self.start
            and abs(cell[0] - self.start[0]) + abs(cell[1] - self.start[1]) >= max(2, n // 2)
        ]
        self.goal = self.rng.choice(goal_candidates)
        safe = {self.start, self.goal}
        current = self.start
        while current != self.goal:
            choices = []
            if current[0] < self.goal[0]:
                choices.append((current[0] + 1, current[1]))
            elif current[0] > self.goal[0]:
                choices.append((current[0] - 1, current[1]))
            if current[1] < self.goal[1]:
                choices.append((current[0], current[1] + 1))
            elif current[1] > self.goal[1]:
                choices.append((current[0], current[1] - 1))
            current = self.rng.choice(choices)
            safe.add(current)
            # Widen the guaranteed corridor so slippery maps remain reasonable.
            for action in ACTIONS:
                neighbor = _add(current, action)
                if 0 <= neighbor[0] < n and 0 <= neighbor[1] < n and self.rng.random() < 0.28:
                    safe.add(neighbor)
        candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in safe]
        self.rng.shuffle(candidates)
        self.holes = set(candidates[: round(n * n * self.hole_fraction)])
        self.player = self.start
        self._value.cache_clear()
        self._policy_metrics.cache_clear()

    def _move(self, position: tuple[int, int], action: str) -> tuple[int, int]:
        nxt = _add(position, action)
        return (min(self.size - 1, max(0, nxt[0])), min(self.size - 1, max(0, nxt[1])))

    def _outcomes(self, position: tuple[int, int], action: str) -> tuple[tuple[float, tuple[int, int]], ...]:
        if not self.slippery:
            return ((1.0, self._move(position, action)),)
        index = ACTIONS.index(action)
        # Perpendicular slip plus the intended direction, matching FrozenLake semantics.
        if action in {"up", "down"}:
            actual = ("left", action, "right")
        else:
            actual = ("up", action, "down")
        merged: dict[tuple[int, int], float] = {}
        for value in actual:
            nxt = self._move(position, value)
            merged[nxt] = merged.get(nxt, 0.0) + 1 / 3
        return tuple((probability, position) for position, probability in sorted(merged.items()))

    @lru_cache(maxsize=None)
    def _value(self, position: tuple[int, int], remaining: int) -> float:
        return self._policy_metrics(position, remaining)[0]

    @lru_cache(maxsize=None)
    def _policy_metrics(self, position: tuple[int, int], remaining: int) -> tuple[float, float]:
        """Return success probability and success-weighted steps.

        The probability is the primary objective. Among policies with equal
        success probability, fewer expected steps to a successful arrival is
        preferred. This prevents deterministic maps from treating arbitrary
        detours and boundary no-ops as equally optimal merely because all of
        them can still reach the goal before a generous horizon.
        """
        if position == self.goal:
            return 1.0, 0.0
        if position in self.holes or remaining <= 0:
            return 0.0, 0.0
        candidates = []
        for action in ACTIONS:
            probability_sum = 0.0
            step_mass = 0.0
            for transition_probability, nxt in self._outcomes(position, action):
                success_probability, future_step_mass = self._policy_metrics(nxt, remaining - 1)
                probability_sum += transition_probability * success_probability
                step_mass += transition_probability * (future_step_mass + success_probability)
            candidates.append((probability_sum, step_mass))
        best_probability = max(item[0] for item in candidates)
        best_step_mass = min(
            item[1]
            for item in candidates
            if math.isclose(item[0], best_probability, rel_tol=1e-10, abs_tol=1e-12)
        )
        return best_probability, best_step_mass

    def action_metrics(self) -> dict[str, tuple[float, float]]:
        remaining = max(0, self.horizon - self.steps)
        result = {}
        for action in ACTIONS:
            probability_sum = 0.0
            step_mass = 0.0
            for transition_probability, nxt in self._outcomes(self.player, action):
                success_probability, future_step_mass = self._policy_metrics(nxt, remaining - 1)
                probability_sum += transition_probability * success_probability
                step_mass += transition_probability * (future_step_mass + success_probability)
            result[action] = (probability_sum, step_mass)
        return result

    def action_values(self) -> dict[str, float]:
        return {action: metrics[0] for action, metrics in self.action_metrics().items()}

    def _shortest_distances(self) -> dict[tuple[int, int], int]:
        distances = {self.goal: 0}
        queue = deque([self.goal])
        while queue:
            position = queue.popleft()
            for action in ACTIONS:
                neighbor = _add(position, action)
                if (
                    0 <= neighbor[0] < self.size
                    and 0 <= neighbor[1] < self.size
                    and neighbor not in self.holes
                    and neighbor not in distances
                ):
                    distances[neighbor] = distances[position] + 1
                    queue.append(neighbor)
        return distances

    def solution(self) -> tuple[str, ...]:
        """Return one exact shortest path for deterministic FrozenLake."""
        if self.slippery:
            return ()
        distances = self._shortest_distances()
        position = self.player
        path = []
        while position != self.goal:
            distance = distances.get(position)
            if distance is None:
                return ()
            choices = [
                action
                for action in ACTIONS
                if distances.get(self._move(position, action)) == distance - 1
            ]
            if not choices:
                return ()
            action = choices[0]
            path.append(action)
            position = self._move(position, action)
        return tuple(path)

    def optimal_actions(self) -> tuple[str, ...]:
        if self.player == self.goal or self.player in self.holes:
            return ()
        if not self.slippery:
            distances = self._shortest_distances()
            distance = distances.get(self.player)
            if distance is None:
                return ()
            return tuple(
                action
                for action in ACTIONS
                if distances.get(self._move(self.player, action)) == distance - 1
            )
        metrics = self.action_metrics()
        best_probability = max(item[0] for item in metrics.values())
        probability_ties = [
            action
            for action in ACTIONS
            if math.isclose(metrics[action][0], best_probability, rel_tol=1e-10, abs_tol=1e-12)
        ]
        best_step_mass = min(metrics[action][1] for action in probability_ties)
        return tuple(
            action
            for action in probability_ties
            if math.isclose(metrics[action][1], best_step_mass, rel_tol=1e-10, abs_tol=1e-12)
        )

    def _step(self, action: str) -> float:
        outcomes = self._outcomes(self.player, action)
        draw = self.rng.random()
        cumulative = 0.0
        for probability, nxt in outcomes:
            cumulative += probability
            if draw <= cumulative:
                self.player = nxt
                break
        if self.player in self.holes:
            self.done = True
        elif self.player == self.goal:
            self.done = self.success = True
            return 1.0
        return 0.0

    def prompt(self) -> str:
        slip = "选择某方向后，实际方向在该方向及两个垂直方向中等概率出现" if self.slippery else "动作不会打滑"
        return (
            f"你正在玩 FrozenLake。识别玩家、冰洞和终点；{slip}。"
            "请选择在剩余步数内使到达终点概率最大的下一步动作。"
        )

    def observation_prompt(self) -> str:
        return f"本局还剩 {max(0, self.horizon - self.steps)} 步。目前的画面状态如下："

    def info(self) -> dict:
        info = super().info()
        if not self.done:
            info["action_values"] = self.action_values()
        info["remaining_steps"] = max(0, self.horizon - self.steps)
        return info

    def map_metadata(self) -> dict:
        return {"size": self.size, "start": list(self.start), "goal": list(self.goal), "holes": [list(x) for x in sorted(self.holes)], "slippery": self.slippery, "horizon": self.horizon}

    def observation(self, size: int | None = None) -> Image.Image:
        image, draw, tile, ox, oy = self._canvas(self.size, self.size, size)
        theme = THEMES[self.style_id]
        for row in range(self.size):
            for col in range(self.size):
                box = self._tile_box(row, col, tile, ox, oy, tile * 0.035)
                color = theme["floor"] if (row + col) % 2 == 0 else theme["floor2"]
                if self.style_id % 3 == 1:
                    draw.rounded_rectangle(box, radius=tile * 0.18, fill=color, outline=theme["accent"], width=max(1, int(tile * 0.025)))
                else:
                    draw.rectangle(box, fill=color, outline=theme["accent"], width=max(1, int(tile * 0.025)))
                if (row, col) in self.holes:
                    cx, cy = ox + (col + 0.5) * tile, oy + (row + 0.5) * tile
                    if self.style_id % 3 == 0:
                        draw.ellipse((cx - tile * 0.31, cy - tile * 0.24, cx + tile * 0.31, cy + tile * 0.24), fill=theme["edge"])
                    elif self.style_id % 3 == 1:
                        draw.polygon(((cx, cy - tile * 0.34), (cx + tile * 0.12, cy - tile * 0.08), (cx + tile * 0.32, cy), (cx + tile * 0.10, cy + tile * 0.09), (cx, cy + tile * 0.34), (cx - tile * 0.10, cy + tile * 0.09), (cx - tile * 0.32, cy), (cx - tile * 0.12, cy - tile * 0.08)), fill=theme["edge"])
                    else:
                        for offset in (-0.18, 0, 0.18):
                            draw.arc((cx - tile * 0.30, cy + offset * tile - tile * 0.10, cx + tile * 0.30, cy + offset * tile + tile * 0.16), 0, 180, fill=theme["edge"], width=max(1, int(tile * 0.05)))
        self._draw_goal(draw, *self.goal, tile, ox, oy)
        if self.player not in self.holes:
            self._draw_player(draw, *self.player, tile, ox, oy)
        return image


SokobanState = tuple[tuple[int, int], frozenset[tuple[int, int]]]


class SokobanEnv(GridEnv):
    name = "sokoban"

    def __init__(self, *, rows: int = 7, cols: int | None = None, boxes: int | None = None, search_limit: int = 80_000, **kwargs):
        super().__init__(**kwargs)
        self.rows = int(rows)
        self.cols = int(cols or rows)
        self.requested_boxes = boxes
        self.search_limit = int(search_limit)
        if self.max_steps is None:
            self.max_steps = self.rows * self.cols * 4

    def _reset(self) -> None:
        self._solve_cache: dict[SokobanState, tuple[int, tuple[str, ...], tuple[str, ...]] | None] = {}
        cells = [(row, col) for row in range(1, self.rows - 1) for col in range(1, self.cols - 1)]
        box_count = self.requested_boxes or (2 if min(self.rows, self.cols) >= 8 and self.rng.random() < 0.25 else 1)
        for attempt in range(500):
            walls = {(row, col) for row in range(self.rows) for col in range(self.cols) if row in {0, self.rows - 1} or col in {0, self.cols - 1}}
            candidates = cells[:]
            self.rng.shuffle(candidates)
            interior_count = round(len(cells) * self.rng.uniform(0.02, 0.12))
            walls.update(candidates[:interior_count])
            free = [cell for cell in cells if cell not in walls]
            if len(free) < box_count * 2 + 1:
                continue
            chosen = self.rng.sample(free, box_count * 2 + 1)
            goals = frozenset(chosen[:box_count])
            boxes = frozenset(chosen[box_count : box_count * 2])
            player = chosen[-1]
            self.walls, self.goals = walls, goals
            # Solver results depend on walls and goals, not only on the
            # (player, boxes) state key.  Every map-generation attempt must
            # therefore start with a fresh cache.
            self._solve_cache.clear()
            result = self._solve((player, boxes))
            if result is None or not (3 <= result[0] <= min(100, self.rows * self.cols * 2)):
                continue
            self.start_player, self.start_boxes = player, boxes
            self.player, self.boxes = player, boxes
            self.initial_solution_length = result[0]
            return
        raise RuntimeError(f"Could not generate solvable Sokoban map for seed {self.seed}")

    def _transition(self, state: SokobanState, action: str) -> SokobanState | None:
        player, boxes = state
        nxt = _add(player, action)
        if nxt in self.walls:
            return None
        if nxt in boxes:
            beyond = _add(nxt, action)
            if beyond in self.walls or beyond in boxes:
                return None
            boxes = frozenset((boxes - {nxt}) | {beyond})
        return nxt, boxes

    def _deadlock(self, boxes: frozenset[tuple[int, int]]) -> bool:
        for box in boxes - self.goals:
            up, down = _add(box, "up") in self.walls, _add(box, "down") in self.walls
            left, right = _add(box, "left") in self.walls, _add(box, "right") in self.walls
            if (up or down) and (left or right):
                return True
        return False

    def _solve(self, initial: SokobanState) -> tuple[int, tuple[str, ...], tuple[str, ...]] | None:
        if initial in self._solve_cache:
            return self._solve_cache[initial]
        if initial[1] == self.goals:
            result = (0, (), ())
            self._solve_cache[initial] = result
            return result
        queue = deque([(initial, (), None)])
        seen: set[tuple[SokobanState, str | None]] = {(initial, None)}
        best_depth: int | None = None
        best_first: set[str] = set()
        best_path: tuple[str, ...] | None = None
        expanded = 0
        while queue:
            state, path, first = queue.popleft()
            if best_depth is not None and len(path) >= best_depth:
                continue
            for action in ACTIONS:
                nxt = self._transition(state, action)
                if nxt is None or self._deadlock(nxt[1]):
                    continue
                nxt_first = first or action
                nxt_path = path + (action,)
                if nxt[1] == self.goals:
                    if best_depth is None:
                        best_depth, best_path = len(nxt_path), nxt_path
                    if len(nxt_path) == best_depth:
                        best_first.add(nxt_first)
                        if best_path is None or nxt_path < best_path:
                            best_path = nxt_path
                    continue
                key = (nxt, nxt_first)
                if key not in seen:
                    seen.add(key)
                    queue.append((nxt, nxt_path, nxt_first))
                    expanded += 1
                    if expanded >= self.search_limit:
                        self._solve_cache[initial] = None
                        return None
        if best_depth is None or best_path is None:
            self._solve_cache[initial] = None
            return None
        result = (best_depth, tuple(action for action in ACTIONS if action in best_first), best_path)
        self._solve_cache[initial] = result
        return result

    def optimal_actions(self) -> tuple[str, ...]:
        result = self._solve((self.player, self.boxes))
        return result[1] if result else ()

    def solution(self) -> tuple[str, ...]:
        result = self._solve((self.player, self.boxes))
        return result[2] if result else ()

    def _step(self, action: str) -> float:
        nxt = self._transition((self.player, self.boxes), action)
        if nxt is not None:
            self.player, self.boxes = nxt
        if self.boxes == self.goals:
            self.done = self.success = True
            return 1.0
        return -0.01

    def prompt(self) -> str:
        return (
            "你正在玩 Sokoban。每步可向上下左右移动一格；玩家可以推动一个箱子，但不能拉箱子、穿墙或同时推两个箱子。"
            "请将所有箱子推到目标，并选择最少移动步数解中的下一步动作。"
        )

    def info(self) -> dict:
        info = super().info()
        if not self.done:
            solution = self.solution()
            info["remaining_optimal_moves"] = len(solution)
        return info

    def map_metadata(self) -> dict:
        return {"size": [self.rows, self.cols], "walls": [list(x) for x in sorted(self.walls)], "goals": [list(x) for x in sorted(self.goals)], "start_player": list(self.start_player), "start_boxes": [list(x) for x in sorted(self.start_boxes)], "initial_solution_length": self.initial_solution_length, "objective": "minimum_moves"}

    def observation(self, size: int | None = None) -> Image.Image:
        image, draw, tile, ox, oy = self._canvas(self.rows, self.cols, size)
        theme = THEMES[self.style_id]
        for row in range(self.rows):
            for col in range(self.cols):
                box = self._tile_box(row, col, tile, ox, oy, tile * 0.025)
                if (row, col) in self.walls:
                    if self.style_id % 3 == 1:
                        draw.rounded_rectangle(box, radius=tile * 0.12, fill=theme["wall"], outline=theme["edge"], width=max(1, int(tile * 0.05)))
                    else:
                        draw.rectangle(box, fill=theme["wall"], outline=theme["edge"], width=max(1, int(tile * 0.05)))
                    if self.style_id % 3 == 2:
                        draw.line((box[0], (box[1] + box[3]) / 2, box[2], (box[1] + box[3]) / 2), fill=theme["edge"], width=1)
                else:
                    draw.rectangle(box, fill=theme["floor"] if (row + col) % 2 == 0 else theme["floor2"])
        for goal in self.goals:
            self._draw_goal(draw, *goal, tile, ox, oy)
        for row, col in self.boxes:
            box = self._tile_box(row, col, tile, ox, oy, tile * 0.15)
            on_goal = (row, col) in self.goals
            fill = theme["goal"] if on_goal else theme["accent"]
            if self.style_id % 3 == 0:
                draw.rectangle(box, fill=fill, outline=theme["edge"], width=max(2, int(tile * 0.06)))
                draw.line((box[0], box[1], box[2], box[3]), fill=theme["edge"], width=max(1, int(tile * 0.035)))
                draw.line((box[2], box[1], box[0], box[3]), fill=theme["edge"], width=max(1, int(tile * 0.035)))
            elif self.style_id % 3 == 1:
                draw.rounded_rectangle(box, radius=tile * 0.18, fill=fill, outline=theme["edge"], width=max(2, int(tile * 0.06)))
            else:
                draw.polygon(_star((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, tile * 0.36), fill=fill, outline=theme["edge"])
        self._draw_player(draw, *self.player, tile, ox, oy)
        return image


def make_env(name: str, **kwargs) -> GridEnv:
    normalized = name.lower().replace("-", "_")
    if normalized == "maze":
        return MazeEnv(**kwargs)
    if normalized in {"frozen_lake", "frozenlake"}:
        return FrozenLakeEnv(**kwargs)
    if normalized == "sokoban":
        return SokobanEnv(**kwargs)
    if normalized in {"lane_racer", "laneracer", "racer"}:
        from .arcade_envs import LaneRacerEnv

        return LaneRacerEnv(**kwargs)
    if normalized in {"parkour_runner", "parkour", "runner"}:
        from .arcade_envs import ParkourRunnerEnv

        return ParkourRunnerEnv(**kwargs)
    raise ValueError(f"Unknown environment {name!r}")
