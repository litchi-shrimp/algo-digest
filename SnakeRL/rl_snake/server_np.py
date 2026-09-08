"""Torch-free Flask server for the trained Snake DQN (NumPy inference).

Same HTTP API as server.py, but loads model_weights.npz and runs inference in
pure NumPy so it only needs numpy + flask + gymnasium (~80MB RSS) instead of
PyTorch (~500MB).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
GAME_DIR = BASE_DIR.parent / "snake_game"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from env import SnakeEnv
from model_np import DuelingDQNNumpy

INPUT_DIM = 47
N_ACTIONS = 3

app = Flask(__name__, static_folder=None)

MODEL = {
    "net": None,
    "checkpoint": None,
    "device": "cpu",
}

DIRECTION_TO_VEC = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
LEFT_TURN = {"up": "left", "left": "down", "down": "right", "right": "up"}
RIGHT_TURN = {"up": "right", "right": "down", "down": "left", "left": "up"}


def steps_trained_from_name(checkpoint_path: Path) -> int:
    m = re.search(r"(\d+)\.(?:pt|npz)$", checkpoint_path.name)
    return int(m.group(1)) if m else 0


def load_model(checkpoint_path: Path) -> None:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net = DuelingDQNNumpy()
    net.load(str(checkpoint_path))

    MODEL["net"] = net
    MODEL["checkpoint"] = checkpoint_path
    MODEL["device"] = "cpu"


# ---------------------------------------------------------------------------
# State reconstruction (identical to server.py)
# ---------------------------------------------------------------------------
def _as_int(value) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(value)


def state_to_observation(payload: dict) -> np.ndarray:
    """Reconstruct the exact SnakeEnv state and reuse its 47-dim mapping."""
    env = SnakeEnv(tile_count=30)

    snake = payload.get("snake")
    if not isinstance(snake, list) or not snake:
        raise ValueError("payload.snake must be a non-empty list")
    env.snake = [{"x": _as_int(s["x"]), "y": _as_int(s["y"])} for s in snake]

    direction = payload.get("direction", "right")
    if direction not in DIRECTION_TO_VEC:
        raise ValueError(f"Unsupported direction: {direction}")
    env.direction = direction

    food = payload.get("food", {})
    env.food = {"x": _as_int(food.get("x", 0)), "y": _as_int(food.get("y", 0))}

    obstacles = payload.get("obstacles", [])
    parsed_obstacles = []
    for o in obstacles:
        shape = o.get("shape")
        if not isinstance(shape, list):
            shape = []
        cells = [{"x": _as_int(c.get("x", 0)), "y": _as_int(c.get("y", 0))} for c in shape]
        if not cells:
            cells = [{"x": 0, "y": 0}]

        parsed_obstacles.append(
            {
                "x": _as_int(o.get("x", 0)),
                "y": _as_int(o.get("y", 0)),
                "vx": _as_int(o.get("vx", 1)),
                "vy": _as_int(o.get("vy", -1)),
                "shape": cells,
                "bounds": {
                    "min_x": _as_int(o.get("bounds", {}).get("minX", 1)),
                    "max_x": _as_int(o.get("bounds", {}).get("maxX", 29)),
                    "min_y": _as_int(o.get("bounds", {}).get("minY", 1)),
                    "max_y": _as_int(o.get("bounds", {}).get("maxY", 29)),
                },
            }
        )
    if len(parsed_obstacles) < 2:
        raise ValueError("payload.obstacles must contain at least 2 obstacles")
    head = env.snake[0]
    parsed_obstacles.sort(
        key=lambda o: env._obstacle_nearest_distance(head["x"], head["y"], o)
    )
    env.obstacles = parsed_obstacles[:2]

    obs = env._get_obs().astype(np.float32)
    if obs.shape != (INPUT_DIM,):
        raise ValueError(f"Expected observation shape ({INPUT_DIM},), got {obs.shape}")
    return obs


def relative_action_to_direction(direction: str, action: int) -> str:
    if action == 0:
        return direction
    if action == 1:
        return LEFT_TURN[direction]
    if action == 2:
        return RIGHT_TURN[direction]
    raise ValueError(f"Unsupported relative action: {action}")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(GAME_DIR, "index_rl.html")


@app.route("/index_rl.html", methods=["GET"])
def index_rl():
    return send_from_directory(GAME_DIR, "index_rl.html")


@app.route("/index_ai.html", methods=["GET"])
def index_ai():
    return send_from_directory(GAME_DIR, "index_rl.html")


@app.route("/<path:filename>", methods=["GET"])
def static_files(filename: str):
    return send_from_directory(GAME_DIR, filename)


@app.route("/api/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)
    checkpoint = MODEL["checkpoint"]
    return jsonify(
        {
            "status": "ok" if MODEL["net"] is not None else "model_not_loaded",
            "model": "DuelingDQN",
            "input_dim": INPUT_DIM,
            "n_actions": N_ACTIONS,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "steps_trained": steps_trained_from_name(checkpoint) if checkpoint else 0,
            "device": MODEL["device"],
        }
    )


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    if MODEL["net"] is None:
        return jsonify({"error": "model_not_loaded"}), 503

    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid_json_body"}), 400

    try:
        obs = state_to_observation(payload)
    except Exception as exc:  # noqa: BLE001 - return useful client error
        return jsonify({"error": "bad_state", "detail": str(exc)}), 400

    q = MODEL["net"].forward(obs)

    action = int(np.argmax(q))
    direction = relative_action_to_direction(payload.get("direction", "right"), action)

    return jsonify(
        {
            "action": action,
            "direction": direction,
            "q_values": [round(float(v), 4) for v in q],
        }
    )


@app.route("/api/death_snapshot", methods=["POST", "OPTIONS"])
def death_snapshot():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(force=True, silent=True)
    out_path = BASE_DIR / "death_snapshots.jsonl"
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"ts": datetime.now(timezone.utc).isoformat(), "payload": payload or {}},
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        return jsonify({"status": "ignored"})
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the trained Snake DQN model (NumPy)")
    parser.add_argument("--checkpoint", default=None, help="Path to a .npz checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (BASE_DIR / "model_weights.npz").resolve()
    )
    load_model(checkpoint)
    print(f"[server] loaded checkpoint: {checkpoint}", flush=True)
    print(f"[server] device: {MODEL['device']}", flush=True)
    print(f"[server] open: http://{args.host}:{args.port}/", flush=True)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
