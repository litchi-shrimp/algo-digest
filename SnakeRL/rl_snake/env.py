"""Snake RL Environment with moving obstacles.

Game mechanics (obstacle spawn/move) follow snake_game/index_ai.html.
Observation is a 47-dimensional flat vector for a fully-connected Double DQN.

Reward v2:
    terminal:  food +1.0 / death -1.0
    dense:     step_cost + PBRS(food) + bounded danger, clipped to [-0.10, 0.10]
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    # Relative-turn actions. Reversing is impossible by construction.
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    ACTION_NAMES = ["STRAIGHT", "LEFT", "RIGHT"]

    DIRECTION_VEC = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    LEFT_TURN = {"up": "left", "left": "down", "down": "right", "right": "up"}
    RIGHT_TURN = {"up": "right", "right": "down", "down": "left", "left": "up"}

    # Reward v2 constants. All terminal rewards and dense rewards are in a
    # small, comparable range to keep TD targets stable.
    REWARD_FOOD = 1.0
    REWARD_DEATH = -1.0
    REWARD_STEP = -0.005
    GAMMA = 0.99
    POTENTIAL_MAX_DIST = 58.0
    DENSE_CLIP = 0.10
    DANGER_CAP = -0.10
    RAY_SATURATION = 8.0
    INITIAL_LENGTH = 3

    # One nearest-distance value per danger source. The sum is capped at
    # DANGER_CAP so multiple nearby threats cannot dominate a step.
    DANGER_BODY = {1: -0.05, 2: -0.02}
    DANGER_OBSTACLE = {1: -0.05, 2: -0.02}
    DANGER_WALL = {1: -0.03, 2: -0.01}

    def __init__(
        self,
        tile_count: int = 30,
        render_mode: Optional[str] = None,
        max_steps: int = 5000,
    ):
        super().__init__()

        self.tile_count = tile_count
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 2 head orientation + 4 wall rays + 8 body rays + 8 obstacle rays
        # + 3 food features + 2 tail features + 1 length
        # + 2 * 8 obstacle dynamics + 3 action-collision masks.
        self.obs_dim = 47
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.snake: list[dict] = []
        self.direction = "right"
        self.food: dict = {}
        self.obstacles: list[dict] = []
        self.score = 0
        self.step_count = 0
        self.done = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Same initial state as setupGame() in index_ai.html.
        self.snake = [
            {"x": 5, "y": 10},
            {"x": 4, "y": 10},
            {"x": 3, "y": 10},
        ]
        self.direction = "right"
        self.score = 0
        self.step_count = 0
        self.done = False

        self._spawn_obstacles()
        self._place_food()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode has ended - call reset().")

        self.step_count += 1

        # Update heading from a relative turn.
        if action == self.LEFT:
            self.direction = self.LEFT_TURN[self.direction]
        elif action == self.RIGHT:
            self.direction = self.RIGHT_TURN[self.direction]

        old_head = self.snake[0].copy()
        phi_before = self._potential(old_head, self.food)

        head = old_head.copy()
        dx, dy = self.DIRECTION_VEC[self.direction]
        head["x"] += dx
        head["y"] += dy

        # Terminal checks happen before body update, exactly like env v1.
        if (
            head["x"] < 0
            or head["x"] >= self.tile_count
            or head["y"] < 0
            or head["y"] >= self.tile_count
        ):
            self.done = True
            return self._get_obs(), self.REWARD_DEATH, True, False, self._get_info()

        if any(head["x"] == s["x"] and head["y"] == s["y"] for s in self.snake):
            self.done = True
            return self._get_obs(), self.REWARD_DEATH, True, False, self._get_info()

        if self._check_obstacle_collision(head):
            self.done = True
            return self._get_obs(), self.REWARD_DEATH, True, False, self._get_info()

        # Update body and possibly eat food.
        ate = head["x"] == self.food["x"] and head["y"] == self.food["y"]
        if ate:
            self.snake.insert(0, head)
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
            self.snake.insert(0, head)

        # Obstacles move after the snake body update, exactly like the HTML.
        self._move_obstacles()

        # v2 PBRS uses the current food position. When food was eaten, the
        # new food position defines the next state potential.
        phi_after = self._potential(head, self.food)
        shaping = self.GAMMA * phi_after - phi_before
        danger = self._danger_penalty(head)

        dense = self.REWARD_STEP + shaping + danger
        dense = float(np.clip(dense, -self.DENSE_CLIP, self.DENSE_CLIP))

        reward = dense
        if ate:
            reward += self.REWARD_FOOD

        truncated = self.step_count >= self.max_steps
        if truncated:
            self.done = True

        return self._get_obs(), reward, self.done, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Food
    # ------------------------------------------------------------------
    def _place_food(self):
        x = self.np_random.integers(1, self.tile_count - 1)
        y = self.np_random.integers(1, self.tile_count - 1)

        while any(s["x"] == x and s["y"] == y for s in self.snake):
            x = self.np_random.integers(1, self.tile_count - 1)
            y = self.np_random.integers(1, self.tile_count - 1)

        self.food = {"x": int(x), "y": int(y)}

    # ------------------------------------------------------------------
    # Obstacles -- direct translation of the HTML game logic.
    # ------------------------------------------------------------------
    def _spawn_obstacles(self):
        self.obstacles = [self._create_obstacle(), self._create_obstacle()]

    def _create_obstacle(self) -> dict:
        tc = self.tile_count

        area_min_x = math.floor(tc * 0.15 + self.np_random.random() * tc * 0.7)
        area_min_y = math.floor(tc * 0.15 + self.np_random.random() * tc * 0.7)
        area_min_x = max(2, min(area_min_x, tc - 6))
        area_min_y = max(2, min(area_min_y, tc - 6))

        shape = []
        for y in range(4):
            for x in range(4):
                if self.np_random.random() < 0.65:
                    shape.append({"x": x, "y": y})
        if len(shape) == 0:
            shape.append({"x": 1, "y": 1})

        d = 1 if self.np_random.random() < 0.5 else -1
        return {
            "x": area_min_x,
            "y": area_min_y,
            "vx": d,
            "vy": -d,
            "shape": shape,
            "bounds": {
                "min_x": max(1, area_min_x - 6),
                "max_x": min(tc - 5, area_min_x + 6),
                "min_y": max(1, area_min_y - 6),
                "max_y": min(tc - 5, area_min_y + 6),
            },
        }

    def _move_obstacles(self):
        for o in self.obstacles:
            if self.np_random.random() < 0.1:
                o["vx"] += -1 if self.np_random.random() < 0.5 else 1
                o["vy"] += -1 if self.np_random.random() < 0.5 else 1
                o["vx"] = max(-1, min(1, o["vx"]))
                o["vy"] = max(-1, min(1, o["vy"]))
                if o["vx"] == 0 and o["vy"] == 0:
                    o["vx"] = 1

            o["x"] += o["vx"]
            o["y"] += o["vy"]

            b = o["bounds"]
            if o["x"] < b["min_x"] or o["x"] > b["max_x"]:
                o["vx"] *= -1
                o["x"] += o["vx"]
            if o["y"] < b["min_y"] or o["y"] > b["max_y"]:
                o["vy"] *= -1
                o["y"] += o["vy"]

    def _check_obstacle_collision(self, point: dict) -> bool:
        for o in self.obstacles:
            for cell in o["shape"]:
                cx = o["x"] + cell["x"]
                cy = o["y"] + cell["y"]
                if point["x"] == cx and point["y"] == cy:
                    return True
        return False

    def _obstacle_cells(self) -> set[tuple[int, int]]:
        cells = set()
        for o in self.obstacles:
            for cell in o["shape"]:
                x = o["x"] + cell["x"]
                y = o["y"] + cell["y"]
                if 0 <= x < self.tile_count and 0 <= y < self.tile_count:
                    cells.add((x, y))
        return cells

    def _obstacle_nearest_distance(self, hx: int, hy: int, o: dict) -> int:
        return min(
            max(abs((o["x"] + cell["x"]) - hx), abs((o["y"] + cell["y"]) - hy))
            for cell in o["shape"]
        )

    # ------------------------------------------------------------------
    # Observation features
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        tc = self.tile_count
        head = self.snake[0]
        hx, hy = head["x"], head["y"]
        fx, fy = self.food["x"], self.food["y"]

        u, r = self._heading_vectors()
        features: list[float] = []

        # Head orientation. Kept only because obstacle bounce bounds are
        # axis-aligned in the global frame.
        features.append(float(u[0]))
        features.append(float(u[1]))

        # Four body-frame wall rays: F, R, B, L.
        for dx, dy in self._body_directions(wall_only=True):
            features.append(self._norm_ray(self._ray_wall_distance(hx, hy, dx, dy)))

        # Eight body-frame body rays.
        body_set = {(s["x"], s["y"]) for s in self.snake[1:]}
        for dx, dy in self._body_directions():
            d = self._ray_entity_distance(hx, hy, dx, dy, body_set)
            features.append(self._norm_ray(d))

        # Eight body-frame obstacle rays.
        obstacle_set = self._obstacle_cells()
        for dx, dy in self._body_directions():
            d = self._ray_entity_distance(hx, hy, dx, dy, obstacle_set)
            features.append(self._norm_ray(d))

        # Food relative vector and distance in body frame.
        food_f, food_r = self._project(fx - hx, fy - hy, u, r)
        features.append(food_f / (tc - 1))
        features.append(food_r / (tc - 1))
        features.append(min(self._chebyshev(hx, hy, fx, fy), tc - 1) / (tc - 1))

        # Tail relative vector in body frame.
        tail = self.snake[-1]
        tail_f, tail_r = self._project(tail["x"] - hx, tail["y"] - hy, u, r)
        features.append(tail_f / (tc - 1))
        features.append(tail_r / (tc - 1))

        # Snake length progress.
        features.append((len(self.snake) - self.INITIAL_LENGTH) / (tc * tc - self.INITIAL_LENGTH))

        # Two obstacle objects, sorted by nearest obstacle-cell distance.
        sorted_obstacles = sorted(
            self.obstacles,
            key=lambda o: self._obstacle_nearest_distance(hx, hy, o),
        )
        for o in sorted_obstacles:
            anchor_f, anchor_r = self._project(o["x"] - hx, o["y"] - hy, u, r)
            features.append(anchor_f / (tc - 1))
            features.append(anchor_r / (tc - 1))

            v_f, v_r = self._project(o["vx"], o["vy"], u, r)
            features.append(v_f / math.sqrt(2.0))
            features.append(v_r / math.sqrt(2.0))

            b = o["bounds"]
            features.append((o["x"] - b["min_x"]) / tc)
            features.append((b["max_x"] - o["x"]) / tc)
            features.append((o["y"] - b["min_y"]) / tc)
            features.append((b["max_y"] - o["y"]) / tc)

        # Immediate next-step collision masks for the three actions.
        features.extend(self._action_collision_mask())

        return np.asarray(features, dtype=np.float32)

    def _heading_vectors(self):
        dx, dy = self.DIRECTION_VEC[self.direction]
        u = np.asarray([dx, dy], dtype=np.float32)
        # Screen y axis points down; this is the "right turn" side vector.
        r = np.asarray([-dy, dx], dtype=np.float32)
        return u, r

    def _body_directions(self, wall_only: bool = False):
        u, r = self._heading_vectors()
        u = u.astype(int)
        r = r.astype(int)
        if wall_only:
            return [(u[0], u[1]), (r[0], r[1]), (-u[0], -u[1]), (-r[0], -r[1])]

        # F, FR, R, BR, B, BL, L, FL.
        return [
            (u[0], u[1]),
            (u[0] + r[0], u[1] + r[1]),
            (r[0], r[1]),
            (r[0] - u[0], r[1] - u[1]),
            (-u[0], -u[1]),
            (-u[0] - r[0], -u[1] - r[1]),
            (-r[0], -r[1]),
            (u[0] - r[0], u[1] - r[1]),
        ]

    @staticmethod
    def _project(x: int, y: int, u: np.ndarray, r: np.ndarray):
        return float(x * u[0] + y * u[1]), float(x * r[0] + y * r[1])

    @staticmethod
    def _chebyshev(hx: int, hy: int, x: int, y: int) -> int:
        return max(abs(x - hx), abs(y - hy))

    def _norm_ray(self, d: Optional[int]) -> float:
        if d is None:
            return 1.0
        return min(d, self.RAY_SATURATION) / self.RAY_SATURATION

    def _ray_wall_distance(self, hx: int, hy: int, dx: int, dy: int) -> int:
        steps = 0
        cx, cy = hx, hy
        while True:
            cx += dx
            cy += dy
            if 0 <= cx < self.tile_count and 0 <= cy < self.tile_count:
                steps += 1
            else:
                return steps

    def _ray_entity_distance(
        self, hx: int, hy: int, dx: int, dy: int, cells: set[tuple[int, int]]
    ) -> Optional[int]:
        tc = self.tile_count
        steps = 0
        cx, cy = hx, hy
        while True:
            cx += dx
            cy += dy
            steps += 1
            if (cx, cy) in cells:
                return steps
            if cx < 0 or cx >= tc or cy < 0 or cy >= tc:
                return None

    def _action_collision_mask(self) -> list[float]:
        hx, hy = self.snake[0]["x"], self.snake[0]["y"]
        obstacle_set = self._obstacle_cells()
        snake_set = {(s["x"], s["y"]) for s in self.snake}

        masks = []
        for action in (self.STRAIGHT, self.LEFT, self.RIGHT):
            if action == self.STRAIGHT:
                direction = self.direction
            elif action == self.LEFT:
                direction = self.LEFT_TURN[self.direction]
            else:
                direction = self.RIGHT_TURN[self.direction]

            dx, dy = self.DIRECTION_VEC[direction]
            nx, ny = hx + dx, hy + dy
            unsafe = (
                nx < 0
                or nx >= self.tile_count
                or ny < 0
                or ny >= self.tile_count
                or (nx, ny) in snake_set
                or (nx, ny) in obstacle_set
            )
            masks.append(1.0 if unsafe else 0.0)
        return masks

    def _get_greedy_action(self) -> int:
        """Relative turn that moves greedily toward the food if possible."""
        hx, hy = self.snake[0]["x"], self.snake[0]["y"]
        fx, fy = self.food["x"], self.food["y"]
        dx, dy = fx - hx, fy - hy

        if dx == 0 and dy == 0:
            return self.STRAIGHT

        if abs(dx) >= abs(dy):
            desired = "right" if dx > 0 else "left"
        else:
            desired = "down" if dy > 0 else "up"

        if desired == self.direction:
            return self.STRAIGHT
        if self.LEFT_TURN[self.direction] == desired:
            return self.LEFT
        if self.RIGHT_TURN[self.direction] == desired:
            return self.RIGHT

        # The food is directly behind. Reversal is impossible with relative
        # turns, so pick one of the two legal turns at random.
        return int(self.np_random.integers(1, 3))

    def _get_info(self) -> dict:
        return {
            "score": self.score,
            "length": len(self.snake),
            "step": self.step_count,
            "greedy_action": self._get_greedy_action(),
            "collision_mask": self._action_collision_mask(),
        }

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan(a, b) -> int:
        return abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])

    def _potential(self, head: dict, food: dict) -> float:
        dist = min(self._manhattan(head, food), self.POTENTIAL_MAX_DIST)
        return -dist / self.POTENTIAL_MAX_DIST

    def _danger_penalty(self, head: dict) -> float:
        hx, hy = head["x"], head["y"]
        tc = self.tile_count

        body_set = {(s["x"], s["y"]) for s in self.snake[1:]}
        obstacle_set = self._obstacle_cells()

        penalty = 0.0

        if body_set:
            body_dist = min(self._chebyshev(hx, hy, x, y) for x, y in body_set)
            penalty += self.DANGER_BODY.get(body_dist, 0.0)

        if obstacle_set:
            obs_dist = min(self._chebyshev(hx, hy, x, y) for x, y in obstacle_set)
            penalty += self.DANGER_OBSTACLE.get(obs_dist, 0.0)

        wall_dist = min(hx, hy, tc - 1 - hx, tc - 1 - hy)
        penalty += self.DANGER_WALL.get(wall_dist, 0.0)

        return max(penalty, self.DANGER_CAP)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        grid = [["." for _ in range(self.tile_count)] for _ in range(self.tile_count)]

        for i in range(self.tile_count):
            grid[0][i] = grid[-1][i] = "-"
            grid[i][0] = grid[i][-1] = "|"
        grid[0][0] = grid[0][-1] = grid[-1][0] = grid[-1][-1] = "+"

        for o in self.obstacles:
            for cell in o["shape"]:
                cx, cy = o["x"] + cell["x"], o["y"] + cell["y"]
                if 0 <= cx < self.tile_count and 0 <= cy < self.tile_count:
                    if grid[cy][cx] == ".":
                        grid[cy][cx] = "#"

        grid[self.food["y"]][self.food["x"]] = "o"
        for i, seg in enumerate(self.snake):
            if 0 <= seg["y"] < self.tile_count and 0 <= seg["x"] < self.tile_count:
                grid[seg["y"]][seg["x"]] = "O" if i == 0 else "o"

        lines = ["".join(row) for row in grid]
        lines.append(
            f"Score: {self.score} | Length: {len(self.snake)} | "
            f"Dir: {self.direction} | Step: {self.step_count}"
        )
        return "\n".join(lines)
