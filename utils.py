"""
utils.py — Shared utilities for the RL benchmark.

Provides convergence statistics, Q-table heatmaps, policy arrow visualisation,
unbiased greedy evaluation, navigation GIF rendering, and the smoothing helper
used by all three agents and compare_algorithms.py.

Outputs:
    results/qtable_heatmap.png
    results/policy_arrows.png
    results/navigation_path.gif

Usage:
    import utils  (imported by other modules, not run directly)
"""

import os
import pickle
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio
import gymnasium as gym


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Convergence & Statistics ──────────────────────────────────────────────────

def compute_stats(rewards: list, window: int = 50) -> dict:
    """
    Compute mean, std, and convergence episode for a reward history.

    Convergence is defined as the first episode where the rolling mean
    exceeds 0.8 — a threshold that works well for FrozenLake.
    """
    arr = np.array(rewards)
    rolling = np.convolve(arr, np.ones(window) / window, mode='valid')

    convergence_ep = None
    for i, val in enumerate(rolling):
        if val >= 0.8:
            convergence_ep = i + window
            break

    return {
        "mean_reward":        float(np.mean(arr)),
        "std_reward":         float(np.std(arr)),
        "max_reward":         float(np.max(arr)),
        "convergence_episode": convergence_ep,
        "final_100_mean":     float(np.mean(arr[-100:])) if len(arr) >= 100 else float(np.mean(arr)),
    }


def print_stats(label: str, stats: dict) -> None:
    pad = 28
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  {'Mean reward':<{pad}} {stats['mean_reward']:.4f}")
    print(f"  {'Std reward':<{pad}} {stats['std_reward']:.4f}")
    print(f"  {'Final-100 mean':<{pad}} {stats['final_100_mean']:.4f}")
    conv = stats['convergence_episode']
    print(f"  {'Convergence episode':<{pad}} {conv if conv else 'Not reached'}")
    print(f"{'─' * 50}")


def save_results(data: dict, filename: str) -> None:
    """Pickle results dict so compare_algorithms can load without re-training."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_results(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Unbiased Policy Evaluation ────────────────────────────────────────────────

def evaluate_greedy_policy(q_table: np.ndarray, n_episodes: int = 500,
                           env_name: str = "FrozenLake-v1") -> dict:
    """
    Run n_episodes with the fully greedy policy (epsilon=0) and return stats.

    Training rewards are always biased because the agent is still exploring.
    This gives the real success rate of the learned policy — the number
    that should go in the results table.
    """
    env = gym.make(env_name, is_slippery=True)
    wins       = 0
    step_counts = []
    MAX_EVAL_STEPS = 200

    for _ in range(n_episodes):
        state, _ = env.reset()
        for step in range(MAX_EVAL_STEPS):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                if reward > 0:
                    wins += 1
                step_counts.append(step + 1)
                break
        else:
            # episode timed out — still record so mean_steps is unbiased
            step_counts.append(MAX_EVAL_STEPS)

    env.close()
    return {
        "success_rate":    wins / n_episodes,
        "mean_steps":      float(np.mean(step_counts)),
        "n_eval_episodes": n_episodes,
    }


# ── Q-Table Heatmap ───────────────────────────────────────────────────────────

def plot_qtable_heatmap(q_table: np.ndarray, grid_size: int = 4,
                        save_path: Optional[str] = None) -> None:
    """
    Render the learned Q-table as four per-action heatmaps over the grid.

    The colour intensity shows which states have high expected value for each
    action — useful for spotting whether the agent has learned to avoid holes.
    """
    action_labels = ['← Left', '↓ Down', '→ Right', '↑ Up']
    cmap = LinearSegmentedColormap.from_list(
        "rl_map", ["#1a1a2e", "#16213e", "#0f3460", "#e94560"]
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle('Q-Table: Action Values Across States', fontsize=13,
                 color='white', fontweight='bold', y=1.02)

    for action_idx, ax in enumerate(axes):
        grid_vals = q_table[:, action_idx].reshape(grid_size, grid_size)
        im = ax.imshow(grid_vals, cmap=cmap, aspect='auto')
        ax.set_title(action_labels[action_idx], color='white', fontsize=11)
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

        for row in range(grid_size):
            for col in range(grid_size):
                val = grid_vals[row, col]
                # switch text colour so it stays readable on both dark and bright cells
                text_col = 'white' if val < grid_vals.max() * 0.6 else '#0d0d0d'
                ax.text(col, row, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=text_col)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

    plt.tight_layout()
    out = save_path or os.path.join(RESULTS_DIR, "qtable_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  Q-table heatmap saved → {out}")


# ── Policy Arrow Visualisation ────────────────────────────────────────────────

def plot_policy_arrows(q_table: np.ndarray, grid_size: int = 4,
                       save_path: Optional[str] = None) -> None:
    """
    Draw the greedy policy as directional arrows on the FrozenLake grid.

    Much more intuitive than raw Q-value numbers — you can immediately see
    whether the agent has learned to route around the holes.
    """
    frozen_map = [
        ['S', 'F', 'F', 'F'],
        ['F', 'H', 'F', 'H'],
        ['F', 'F', 'F', 'H'],
        ['H', 'F', 'F', 'G'],
    ]
    # (dx, dy) offsets for arrows in axes coords; y is flipped because imshow
    action_arrows = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}
    tile_colours  = {'S': '#0f3460', 'F': '#16213e', 'H': '#7f1d1d', 'G': '#14532d'}

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Learned Policy: Greedy Action per State',
                 color='white', fontsize=12, pad=10)

    for row in range(grid_size):
        for col in range(grid_size):
            tile = frozen_map[row][col]
            rect = plt.Rectangle(
                (col - 0.5, (grid_size - 1 - row) - 0.5),
                1, 1, color=tile_colours[tile], linewidth=1, edgecolor='#333333'
            )
            ax.add_patch(rect)

            state_idx = row * grid_size + col
            if tile in ('H', 'G'):
                symbol = '✕' if tile == 'H' else '★'
                ax.text(col, grid_size - 1 - row, symbol,
                        ha='center', va='center', fontsize=18, color='white')
                continue

            best_action = int(np.argmax(q_table[state_idx]))
            dx, dy = action_arrows[best_action]
            cx, cy = col, grid_size - 1 - row
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                arrowprops=dict(arrowstyle="->", color='#38bdf8',
                                lw=2.0, mutation_scale=18)
            )
            if tile == 'S':
                ax.text(col - 0.35, grid_size - 1 - row + 0.35, 'S',
                        fontsize=8, color='#aaaaaa')

    plt.tight_layout()
    out = save_path or os.path.join(RESULTS_DIR, "policy_arrows.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  Policy arrow map saved → {out}")


# ── Navigation GIF ────────────────────────────────────────────────────────────

def render_navigation_gif(env_name: str = "FrozenLake-v1",
                          policy: Optional[np.ndarray] = None,
                          n_steps: int = 30,
                          save_path: Optional[str] = None) -> None:
    """
    Run one greedy episode and save a frame-by-frame GIF of the agent's path.

    Uses gymnasium's rgb_array render mode — no display or GUI needed.
    If policy is None, falls back to random actions (useful as a baseline).
    """
    env = gym.make(env_name, is_slippery=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    frames = []

    for _ in range(n_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = int(np.argmax(policy[obs])) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            final_frame = env.render()
            if final_frame is not None:
                frames.append(final_frame)
            break

    env.close()

    if not frames:
        print("  Warning: no frames captured — skipping GIF")
        return

    out = save_path or os.path.join(RESULTS_DIR, "navigation_path.gif")
    # 0.4s per frame looks right for a 4x4 grid walkthrough
    imageio.mimsave(out, frames, duration=0.4, loop=0)
    print(f"  Navigation GIF saved → {out}  ({len(frames)} frames)")


# ── Smoothing Helper ──────────────────────────────────────────────────────────

def smooth_curve(values: list, window: int = 50) -> np.ndarray:
    """Rolling mean with edge padding so the output length matches the input."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel  = np.ones(window) / window
    padded  = np.pad(values, (window // 2, window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(values)]
