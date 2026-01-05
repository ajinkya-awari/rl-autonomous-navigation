"""
q_learning.py — Tabular Q-Learning on FrozenLake-v1 (pure NumPy, no DL frameworks).

Implements the classic Bellman update from scratch and tracks reward history,
step counts, and Q-table convergence across training episodes.

Outputs:
    results/q_learning_rewards.png
    results/qtable_heatmap.png
    results/policy_arrows.png
    results/ql_results.pkl

Usage:
    python q_learning.py
"""

import os
import time
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym

from utils import (compute_stats, print_stats, save_results,
                   plot_qtable_heatmap, plot_policy_arrows,
                   evaluate_greedy_policy, smooth_curve)

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_EPISODES    = 10_000
MAX_STEPS     = 100
ALPHA         = 0.8       # higher than typical — FrozenLake's sparse reward needs it
GAMMA         = 0.95      # discount; longer horizon matters since reward is only at goal
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.001     # exponential — hits ~0.01 around episode 4600

RESULTS_DIR   = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ── Q-Table Initialisation ────────────────────────────────────────────────────

def build_qtable(n_states: int, n_actions: int) -> np.ndarray:
    # small random init breaks symmetry; converges faster than all-zeros
    return np.random.uniform(low=0.0, high=0.01, size=(n_states, n_actions))


# ── Epsilon-Greedy Policy ─────────────────────────────────────────────────────

def select_action(q_table: np.ndarray, state: int,
                  epsilon: float, n_actions: int) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    # tie-break randomly so the agent doesn't get trapped by symmetric Q-values
    return int(np.random.choice(
        np.where(q_table[state] == q_table[state].max())[0]
    ))


# ── Single Episode ────────────────────────────────────────────────────────────

def run_episode(env: gym.Env, q_table: np.ndarray, epsilon: float) -> tuple:
    """Run one episode and return (total_reward, steps_taken, updated q_table)."""
    state, _ = env.reset()
    total_reward = 0.0
    n_actions = env.action_space.n

    for step in range(MAX_STEPS):
        action = select_action(q_table, state, epsilon, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # ── Bellman optimality update ─────────────────────────────────────────
        # Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
        # space complexity: O(|S||A|) for the table; O(1) per update step
        best_next = 0.0 if (terminated or truncated) else np.max(q_table[next_state])
        td_error  = (reward + GAMMA * best_next) - q_table[state, action]
        q_table[state, action] += ALPHA * td_error

        state        = next_state
        total_reward += reward

        if terminated or truncated:
            return total_reward, step + 1, q_table

    return total_reward, MAX_STEPS, q_table


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(verbose: bool = True) -> dict:
    """
    Train the Q-Learning agent over N_EPISODES episodes.

    Returns a results dict containing reward history, Q-table, and stats
    so compare_algorithms.py can load it without re-training.
    """
    env      = gym.make("FrozenLake-v1", is_slippery=True)
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    q_table = build_qtable(n_states, n_actions)

    episode_rewards = []
    episode_steps   = []
    qtable_deltas   = []  # mean absolute change per episode — convergence proxy
    epsilon         = EPSILON_START

    print(f"\nTraining Q-Learning agent  ({N_EPISODES:,} episodes) ...")
    t_start = time.time()

    for ep in range(N_EPISODES):
        q_before = q_table.copy()
        reward, steps, q_table = run_episode(env, q_table, epsilon)

        episode_rewards.append(reward)
        episode_steps.append(steps)
        qtable_deltas.append(float(np.mean(np.abs(q_table - q_before))))

        epsilon = max(EPSILON_END, EPSILON_START * np.exp(-EPSILON_DECAY * ep))

        if verbose and (ep + 1) % 1000 == 0:
            win_rate = np.mean(episode_rewards[-500:])
            print(f"  ep {ep+1:>6,} | ε={epsilon:.3f} | win_rate(500)={win_rate:.3f}")

    elapsed = time.time() - t_start
    print(f"  Training complete in {elapsed:.1f}s")
    env.close()

    stats = compute_stats(episode_rewards)
    print_stats("Q-Learning", stats)

    results = {
        "rewards":      episode_rewards,
        "steps":        episode_steps,
        "q_table":      q_table,
        "qtable_delta": qtable_deltas,
        "stats":        stats,
    }
    save_results(results, "ql_results.pkl")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve(rewards: list) -> None:
    smooth = smooth_curve(rewards, window=200)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#111111')

    ax.plot(rewards, color='#334155', linewidth=0.4, alpha=0.6, label='Raw')
    ax.plot(smooth,  color='#38bdf8', linewidth=2.0, label='Smoothed (200-ep)')

    ax.set_xlabel("Episode", color='#aaaaaa')
    ax.set_ylabel("Reward",  color='#aaaaaa')
    ax.set_title("Q-Learning: Learning Curve on FrozenLake-v1",
                 color='white', fontsize=13)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "q_learning_rewards.png")
    plt.savefig(out, dpi=150, facecolor='#0d0d0d')
    plt.close()
    print(f"  Learning curve saved → {out}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results  = train(verbose=True)
    plot_learning_curve(results["rewards"])
    plot_qtable_heatmap(results["q_table"], grid_size=4)
    plot_policy_arrows(results["q_table"],  grid_size=4)

    eval_stats = evaluate_greedy_policy(results["q_table"], n_episodes=500)
    print(f"\n  Greedy eval (500 ep): "
          f"success_rate={eval_stats['success_rate']:.3f}  "
          f"mean_steps={eval_stats['mean_steps']:.1f}")
    print("\nQ-Learning done. Results saved to results/")
