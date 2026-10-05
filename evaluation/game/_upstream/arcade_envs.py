"""Procedural racing and parkour environments with exact shortest-path oracles."""

from __future__ import annotations

from collections import deque
import math

from PIL import Image, ImageDraw

from .envs import GridEnv, THEMES, THEME_DESCRIPTIONS


RACER_ACTIONS = ("left", "right", "accelerate", "brake")
RUNNER_ACTIONS = ("run", "jump", "duck", "dash")
PLAYER_COLORS = ("橙红色", "粉红色", "蓝色", "粉红色", "红色", "绿色")


def _longest_action_run(actions: tuple[str, ...]) -> int:
    longest = current = 0
    previous = None
    for action in actions:
        current = current + 1 if action == previous else 1
        longest = max(longest, current)
        previous = action
    return longest


class LaneRacerEnv(GridEnv):
    """Top-down lane race with steering, two speeds, and collision avoidance."""

    name = "lane_racer"
    actions = RACER_ACTIONS

    def __init__(
        self,
        *,
        rows: int = 16,
        lanes: int = 4,
        obstacle_fraction: float = 0.18,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rows = int(rows)
        self.lanes = int(lanes)
        self.obstacle_fraction = float(obstacle_fraction)
        if self.rows < 8 or not 3 <= self.lanes <= 6:
            raise ValueError("LaneRacer requires rows >= 8 and 3 <= lanes <= 6")
        if self.max_steps is None:
            self.max_steps = self.rows * 3

    def _reset(self) -> None:
        cells = [(row, lane) for row in range(1, self.rows - 1) for lane in range(self.lanes)]
        for _ in range(300):
            self.start_lane = self.rng.randrange(self.lanes)
            count = round(len(cells) * self.obstacle_fraction)
            candidates = cells[:]
            self.rng.shuffle(candidates)
            obstacles = set(candidates[:count])
            # Never close a complete road row; this also keeps frames visually legible.
            for row in range(1, self.rows - 1):
                occupied = sorted(lane for r, lane in obstacles if r == row)
                if len(occupied) == self.lanes:
                    obstacles.remove((row, self.rng.choice(occupied)))
            self.obstacles = obstacles
            self.player_row = self.rows - 1
            self.player_lane = self.start_lane
            self.speed = 1
            result = self._solve((self.player_row, self.player_lane, self.speed))
            if result is not None:
                path = result[2]
                # Reject visually busy but strategically trivial roads.  A
                # retained shortest solution must use speed and steering, and
                # must not collapse into a long repetition of one action.
                if "accelerate" not in path or not ({"left", "right"} & set(path)):
                    continue
                if _longest_action_run(path) > 3:
                    continue
                self.initial_solution_length = result[0]
                return
        raise RuntimeError(f"Could not generate solvable LaneRacer map for seed {self.seed}")

    def _transition(self, state: tuple[int, int, int], action: str) -> tuple[int, int, int] | None:
        row, lane, speed = state
        if action == "left":
            lane -= 1
        elif action == "right":
            lane += 1
        elif action == "accelerate":
            speed = 2
        elif action == "brake":
            speed = 1
        if not 0 <= lane < self.lanes:
            return None
        destination = max(0, row - speed)
        if any((crossed, lane) in self.obstacles for crossed in range(row - 1, destination - 1, -1)):
            return None
        return destination, lane, speed

    def _solve(self, initial: tuple[int, int, int]) -> tuple[int, tuple[str, ...], tuple[str, ...]] | None:
        queue = deque([(initial, ())])
        # Keep reachability separate for each first action.  Two equally short
        # branches may merge into the same later state; a state-only ``seen``
        # set would then silently drop one valid optimal first action.
        seen: set[tuple[tuple[int, int, int], str | None]] = {(initial, None)}
        best_depth = None
        first_actions = set()
        best_path = None
        while queue:
            state, path = queue.popleft()
            if best_depth is not None and len(path) >= best_depth:
                continue
            for action in self.actions:
                nxt = self._transition(state, action)
                if nxt is None:
                    continue
                candidate = path + (action,)
                if nxt[0] == 0:
                    if best_depth is None:
                        best_depth, best_path = len(candidate), candidate
                    if len(candidate) == best_depth:
                        first_actions.add(candidate[0])
                        if best_path is None or candidate < best_path:
                            best_path = candidate
                    continue
                key = (nxt, candidate[0])
                if key not in seen:
                    seen.add(key)
                    queue.append((nxt, candidate))
        if best_depth is None or best_path is None:
            return None
        return best_depth, tuple(action for action in self.actions if action in first_actions), best_path

    def optimal_actions(self) -> tuple[str, ...]:
        result = self._solve((self.player_row, self.player_lane, self.speed))
        return result[1] if result else ()

    def solution(self) -> tuple[str, ...]:
        result = self._solve((self.player_row, self.player_lane, self.speed))
        return result[2] if result else ()

    def _step(self, action: str) -> float:
        nxt = self._transition((self.player_row, self.player_lane, self.speed), action)
        if nxt is None:
            return -0.05
        self.player_row, self.player_lane, self.speed = nxt
        if self.player_row == 0:
            self.done = self.success = True
            return 1.0
        return -0.01

    def prompt(self) -> str:
        visual = THEME_DESCRIPTIONS[self.style_id]
        player_color = PLAYER_COLORS[self.style_id]
        return (
            f"你正在玩一个俯视赛车游戏，赛道共 {self.rows} 行、{self.lanes} 条车道，终点是最上方的黑白格线。"
            f"玩家是靠近画面下方的{player_color}赛车，车尾三角形数量表示当前速度；"
            f"赛道中的{visual['wall']}圆角块是其他车辆或路障，不能碰撞。画面上方是前进方向。"
            "left/right 会先向相邻车道变道并按当前速度前进；accelerate 将速度设为 2 格并前进；"
            "brake 将速度设为 1 格并前进。高速前进时经过的每一格都不能有障碍。"
            "请选择安全到达终点所需步数最少的下一步动作。"
        )

    def observation_prompt(self) -> str:
        return f"当前速度为 {self.speed} 格/步。目前的赛道状态如下："

    def map_metadata(self) -> dict:
        return {
            "rows": self.rows,
            "lanes": self.lanes,
            "start_lane": self.start_lane,
            "obstacles": [list(value) for value in sorted(self.obstacles)],
            "obstacle_fraction": self.obstacle_fraction,
            "initial_solution_length": self.initial_solution_length,
        }

    def observation(self, size: int | None = None) -> Image.Image:
        size = int(size or self.render_size)
        theme = THEMES[self.style_id]
        margin = max(10, size // 30)
        lane_width = (size - 2 * margin) / self.lanes
        row_height = (size - 2 * margin) / self.rows
        image = Image.new("RGB", (size, size), theme["bg"])
        draw = ImageDraw.Draw(image)
        road_box = (margin, margin, size - margin, size - margin)
        draw.rounded_rectangle(road_box, radius=max(4, size // 35), fill=theme["floor"], outline=theme["edge"], width=max(2, size // 100))
        for lane in range(1, self.lanes):
            x = margin + lane * lane_width
            for row in range(0, self.rows, 2):
                y0 = margin + row * row_height
                draw.line((x, y0, x, min(size - margin, y0 + row_height)), fill=theme["accent"], width=max(1, size // 160))
        tile = max(2, int(lane_width / 5))
        for lane in range(self.lanes):
            for index in range(6):
                x0 = margin + lane * lane_width + index * lane_width / 6
                x1 = margin + lane * lane_width + (index + 1) * lane_width / 6
                color = theme["edge"] if (lane + index) % 2 == 0 else theme["goal"]
                draw.rectangle((x0, margin, x1, margin + max(4, row_height * 0.75)), fill=color)
        for row, lane in self.obstacles:
            cx = margin + (lane + 0.5) * lane_width
            cy = margin + (row + 0.5) * row_height
            half_w, half_h = lane_width * 0.27, max(row_height * 0.38, 4)
            draw.rounded_rectangle((cx - half_w, cy - half_h, cx + half_w, cy + half_h), radius=max(2, tile), fill=theme["wall"], outline=theme["edge"], width=max(1, size // 180))
            draw.line((cx - half_w * 0.65, cy, cx + half_w * 0.65, cy), fill=theme["goal"], width=max(1, size // 150))
        cx = margin + (self.player_lane + 0.5) * lane_width
        cy = margin + (self.player_row + 0.5) * row_height
        half_w, half_h = lane_width * 0.28, max(row_height * 0.42, 5)
        draw.rounded_rectangle((cx - half_w, cy - half_h, cx + half_w, cy + half_h), radius=max(3, tile), fill=theme["player"], outline=theme["edge"], width=max(2, size // 120))
        wheel = max(2, size // 100)
        draw.ellipse((cx - wheel, cy - wheel, cx + wheel, cy + wheel), fill="white")
        for level in range(self.speed):
            y = cy + half_h + (level + 1) * max(3, row_height * 0.22)
            draw.polygon(((cx, y + 3), (cx - 5, y - 3), (cx + 5, y - 3)), fill=theme["accent"])
        return image


class ParkourRunnerEnv(GridEnv):
    """Side-view auto-runner with exact discrete obstacle semantics."""

    name = "parkour_runner"
    actions = RUNNER_ACTIONS

    def __init__(self, *, length: int = 16, hazard_fraction: float = 0.42, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.hazard_fraction = float(hazard_fraction)
        if self.length < 8:
            raise ValueError("ParkourRunner requires length >= 8")
        if self.max_steps is None:
            self.max_steps = self.length * 3

    def _reset(self) -> None:
        hazard_types = ("gap", "hurdle", "overhead")
        for _ in range(300):
            self.hazards = {column: "open" for column in range(self.length)}
            for column in range(1, self.length - 1):
                if self.rng.random() < self.hazard_fraction:
                    self.hazards[column] = self.rng.choice(hazard_types)
            self.position = 0
            result = self._solve(0)
            if result is not None and result[0] >= 5:
                path = result[2]
                # Every retained course must exercise both obstacle-specific
                # actions and at least one locomotion action.  Limiting runs of
                # identical actions removes empty "dash to the finish" maps.
                if not {"jump", "duck"}.issubset(path) or len(set(path)) < 3:
                    continue
                if _longest_action_run(path) > 2:
                    continue
                self.initial_solution_length = result[0]
                return
        raise RuntimeError(f"Could not generate solvable ParkourRunner map for seed {self.seed}")

    def _transition(self, position: int, action: str) -> int | None:
        if action in {"run", "duck"}:
            destination = min(self.length - 1, position + 1)
            hazard = self.hazards[destination]
            allowed = (action == "run" and hazard == "open") or (
                action == "duck" and hazard == "overhead"
            )
            return destination if allowed else None
        destination = min(self.length - 1, position + 2)
        middle = min(self.length - 1, position + 1)
        if action == "jump":
            return (
                destination
                if self.hazards[middle] in {"gap", "hurdle"} and self.hazards[destination] == "open"
                else None
            )
        if action == "dash":
            return destination if self.hazards[middle] == self.hazards[destination] == "open" else None
        return None

    def _solve(self, initial: int) -> tuple[int, tuple[str, ...], tuple[str, ...]] | None:
        queue = deque([(initial, ())])
        best_depth = None
        best_path = None
        first_actions = set()
        # Preserve the first-action branch when paths merge so the returned
        # policy contains every shortest first action, not an arbitrary subset.
        seen: set[tuple[int, str | None]] = {(initial, None)}
        while queue:
            position, path = queue.popleft()
            if best_depth is not None and len(path) >= best_depth:
                continue
            for action in self.actions:
                nxt = self._transition(position, action)
                if nxt is None or nxt == position:
                    continue
                candidate = path + (action,)
                if nxt == self.length - 1:
                    if best_depth is None:
                        best_depth, best_path = len(candidate), candidate
                    if len(candidate) == best_depth:
                        first_actions.add(candidate[0])
                        if best_path is None or candidate < best_path:
                            best_path = candidate
                    continue
                key = (nxt, candidate[0])
                if key not in seen:
                    seen.add(key)
                    queue.append((nxt, candidate))
        if best_depth is None or best_path is None:
            return None
        return best_depth, tuple(action for action in self.actions if action in first_actions), best_path

    def optimal_actions(self) -> tuple[str, ...]:
        result = self._solve(self.position)
        return result[1] if result else ()

    def solution(self) -> tuple[str, ...]:
        result = self._solve(self.position)
        return result[2] if result else ()

    def _step(self, action: str) -> float:
        nxt = self._transition(self.position, action)
        if nxt is None:
            return -0.05
        self.position = nxt
        if self.position == self.length - 1:
            self.done = self.success = True
            return 1.0
        return -0.01

    def prompt(self) -> str:
        visual = THEME_DESCRIPTIONS[self.style_id]
        player_color = PLAYER_COLORS[self.style_id]
        return (
            f"你正在玩一个横向跑酷游戏，跑道从左到右共 {self.length} 段，玩家要到达最右侧旗帜。"
            f"玩家是{player_color}圆形图标；{visual['floor']}横条是可落脚地面；没有横条的位置是缺口；"
            f"{visual['wall']}矮方块是路障，悬在玩家头顶高度的{visual['wall']}长条是横杆。"
            "地面缺口和矮路障需要用 jump 向右跨两段，落脚段必须是普通地面；悬空横杆需要用 duck 向右通过一段；"
            "run 向右跑一段且只能进入普通地面；dash 向右冲两段且经过和落脚处都必须是普通地面。"
            "撞到障碍或跳入缺口时动作不生效。请选择到达终点所需步数最少的下一步动作。"
        )

    def observation_prompt(self) -> str:
        return "目前的跑酷赛道状态如下："

    def map_metadata(self) -> dict:
        return {
            "length": self.length,
            "hazards": {str(column): value for column, value in self.hazards.items() if value != "open"},
            "hazard_fraction": self.hazard_fraction,
            "initial_solution_length": self.initial_solution_length,
        }

    def observation(self, size: int | None = None) -> Image.Image:
        size = int(size or self.render_size)
        theme = THEMES[self.style_id]
        margin = max(10, size // 30)
        width = size
        height = max(220, round(size * 0.62))
        image = Image.new("RGB", (width, height), theme["bg"])
        draw = ImageDraw.Draw(image)
        cell = (width - 2 * margin) / self.length
        ground_y = height * 0.72
        platform_h = max(8, height * 0.08)
        for column in range(self.length):
            x0 = margin + column * cell
            x1 = margin + (column + 1) * cell
            hazard = self.hazards[column]
            if hazard != "gap":
                draw.rectangle((x0, ground_y, x1, ground_y + platform_h), fill=theme["floor"], outline=theme["edge"], width=1)
            if hazard == "hurdle":
                draw.rectangle((x0 + cell * 0.18, ground_y - cell * 0.65, x1 - cell * 0.18, ground_y), fill=theme["wall"], outline=theme["edge"], width=1)
            elif hazard == "overhead":
                draw.rounded_rectangle((x0 + cell * 0.08, ground_y - cell * 1.45, x1 - cell * 0.08, ground_y - cell * 0.75), radius=max(2, int(cell * 0.12)), fill=theme["wall"], outline=theme["edge"], width=1)
            elif hazard == "gap":
                draw.line((x0, ground_y + platform_h, x1, ground_y + platform_h), fill=theme["accent"], width=max(2, int(cell * 0.08)))
        goal_x = margin + (self.length - 0.5) * cell
        pole_y = ground_y - cell * 1.6
        draw.line((goal_x, pole_y, goal_x, ground_y), fill=theme["edge"], width=max(2, int(cell * 0.10)))
        draw.polygon(((goal_x, pole_y), (goal_x - cell * 0.75, pole_y + cell * 0.25), (goal_x, pole_y + cell * 0.55)), fill=theme["goal"], outline=theme["edge"])
        player_x = margin + (self.position + 0.5) * cell
        radius = max(5, cell * 0.30)
        player_y = ground_y - radius
        draw.ellipse((player_x - radius, player_y - radius, player_x + radius, player_y + radius), fill=theme["player"], outline=theme["edge"], width=max(1, int(cell * 0.08)))
        eye = max(1, radius * 0.18)
        draw.ellipse((player_x + radius * 0.15, player_y - radius * 0.25, player_x + radius * 0.15 + eye, player_y - radius * 0.25 + eye), fill="white")
        return image
