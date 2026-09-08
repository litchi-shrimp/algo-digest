"""Training loop for the fully-connected Double DQN snake agent."""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import DQNAgent, ReplayBuffer
from env import SnakeEnv

# Hyperparameters.
DEFAULT_EPISODES = 5000
TILE_COUNT = 30
BATCH_SIZE = 64
LR = 1e-4
GAMMA = 0.99

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 150_000

BUFFER_SIZE = 100_000
LEARN_START = 1_000
TARGET_UPDATE = 1_000
LOG_EVERY = 1
PLOT_EVERY = 50
CHECKPOINT_EVERY = 100

GREEDY_RATIO = 0.5
WINDOW = 50
BEST_WINDOW = 50

DEFAULT_CHECKPOINT_DIR = "checkpoints_v2"
DEFAULT_PLOT_DIR = "plots_v2"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def moving_average(x, w: int):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) >= w:
        csum = np.cumsum(np.insert(x, 0, 0.0))
        out[w - 1:] = (csum[w:] - csum[:-w]) / w
    return out


def plot_metrics(metrics: dict, path: str, window: int):
    rewards = metrics["rewards"]
    if not rewards:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ep = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(ep, rewards, alpha=0.3, lw=0.7, label="raw")
    ax.plot(ep, moving_average(rewards, window), lw=1.5, label=f"{window}-ep mean")
    ax.set_title("Episode Reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, metrics["scores"], alpha=0.3, lw=0.7, label="raw")
    ax.plot(ep, moving_average(metrics["scores"], window), lw=1.5, label=f"{window}-ep mean")
    ax.set_title("Score (food eaten)")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(ep, moving_average(metrics["qs"], window), lw=1.5, color="tab:orange")
    ax.set_title("Avg Max Q")
    ax.set_xlabel("episode")
    ax.set_ylabel("Q")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, moving_average(metrics["losses"], window), lw=1.5, color="tab:red")
    ax.set_title("TD Loss (Huber)")
    ax.set_xlabel("episode")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, moving_average(metrics["lengths"], window), lw=1.5, color="tab:green")
    ax.set_title("Episode Length")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(ep, metrics["epsilons"], lw=1.5, color="tab:purple")
    ax.set_title("Epsilon")
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Snake Double DQN Training (window={window})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--greedy-ratio", type=float, default=GREEDY_RATIO)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    args = parser.parse_args()

    plot_path = os.path.join(args.plot_dir, "training_curves.png")

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = SnakeEnv(tile_count=TILE_COUNT)
    agent = DQNAgent(
        input_dim=env.obs_dim,
        n_actions=env.action_space.n,
        lr=LR,
        gamma=GAMMA,
        device=device,
    )
    buffer = ReplayBuffer(BUFFER_SIZE)

    global_step = 0
    best_mean_score = float("-inf")
    metrics = {
        "rewards": [],
        "scores": [],
        "lengths": [],
        "qs": [],
        "losses": [],
        "epsilons": [],
    }

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        ep_losses = []
        ep_qs = []

        while True:
            epsilon = max(
                EPS_END,
                EPS_START - (EPS_START - EPS_END) * global_step / EPS_DECAY_STEPS,
            )
            action = agent.act(
                obs,
                epsilon,
                greedy_action=info["greedy_action"],
                greedy_ratio=args.greedy_ratio,
            )
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Truncation is not a true terminal state for the Bellman target.
            buffer.push(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_reward += reward
            global_step += 1

            if len(buffer) >= LEARN_START:
                loss, avg_q = agent.update(buffer.sample(BATCH_SIZE, device))
                ep_losses.append(loss)
                ep_qs.append(avg_q)

            if global_step % TARGET_UPDATE == 0:
                agent.sync_target()

            if terminated or truncated:
                break

        metrics["rewards"].append(episode_reward)
        metrics["scores"].append(info["score"])
        metrics["lengths"].append(info["step"])
        metrics["qs"].append(np.mean(ep_qs) if ep_qs else 0.0)
        metrics["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)
        metrics["epsilons"].append(epsilon)

        mean_score = np.mean(metrics["scores"][-BEST_WINDOW:])
        mean_reward = np.mean(metrics["rewards"][-100:])
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(args.checkpoint_dir, "best.pt"),
            )

        if episode % LOG_EVERY == 0:
            print(
                f"Ep {episode:5d} | steps {global_step:8d} | "
                f"ep_step {info['step']:4d} | ep_reward {episode_reward:8.2f} | "
                f"score {info['score']:3d} | reward(100ep) {mean_reward:7.2f} | "
                f"eps {epsilon:.3f} | best50 {best_mean_score:6.2f}"
            )

        if episode % PLOT_EVERY == 0:
            plot_metrics(metrics, plot_path, WINDOW)

        if episode % CHECKPOINT_EVERY == 0:
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(args.checkpoint_dir, f"checkpoint_{episode:06d}.pt"),
            )

    plot_metrics(metrics, plot_path, WINDOW)
    torch.save(agent.q_net.state_dict(), os.path.join(args.checkpoint_dir, "final.pt"))
    print("Training finished.")


if __name__ == "__main__":
    main()
