"""
ppo_agent.py — Proximal Policy Optimization (PPO) on FrozenLake-v1.

Uses stable-baselines3's PPO with a custom callback for reward tracking.
The clipped surrogate objective prevents destructively large policy updates,
which is exactly what you need on a sparse-reward environment like FrozenLake.

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
           https://arxiv.org/abs/1707.06347

Outputs:
    results/ppo_rewards.png
    results/ppo_model.zip
    results/ppo_results.pkl

Usage:
    python ppo_agent.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

from utils import compute_stats, print_stats, save_results, smooth_curve

# ── Hyperparameters ───────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 100_000
N_ENVS          = 4       # parallel envs — more diverse experience per update
RANDOM_SEED     = 42

# n_steps=512, batch_size=64: n_steps * N_ENVS = 2048, divisible by 64 ✓
# clip_range=0.2 is the Schulman et al. recommended default
# ent_coef=0.01 — small entropy bonus to escape the "stay still" local optimum
PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 512,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 0,
    seed          = RANDOM_SEED,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Reward Tracking Callback ──────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """
    Collects episode rewards during training across all parallel envs.

    SB3 callbacks are the cleanest way to get fine-grained training data
    without monkey-patching the training loop internals.
    """

    def __init__(self):
        super().__init__()
        self.episode_rewards   = []
        self.episode_timesteps = []
        self._current_rewards  = None

    def _on_training_start(self) -> None:
        self._current_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        self._current_rewards += self.locals["rewards"]

        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_rewards.append(float(self._current_rewards[i]))
                self.episode_timesteps.append(self.num_timesteps)
                self._current_rewards[i] = 0.0

        return True  # returning False stops training early — don't do that


# ── Training ──────────────────────────────────────────────────────────────────

def train(verbose: bool = True) -> dict:
    """
    Train PPO for TOTAL_TIMESTEPS across N_ENVS parallel environments.

    Runs a separate 500-episode greedy evaluation after training for an
    unbiased success rate — the training callback rewards include entropy
    exploration steps and shouldn't be reported as the final metric.
    """
    vec_env  = make_vec_env("FrozenLake-v1", n_envs=N_ENVS, seed=RANDOM_SEED,
                            env_kwargs={"is_slippery": True})
    model    = PPO("MlpPolicy", vec_env, **PPO_KWARGS)
    callback = RewardLogger()

    print(f"\nTraining PPO agent  ({TOTAL_TIMESTEPS:,} timesteps, {N_ENVS} parallel envs) ...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=10,
        progress_bar=False,
    )

    model_path = os.path.join(RESULTS_DIR, "ppo_model.zip")
    model.save(model_path)
    print(f"  Model saved → {model_path}")

    # fresh single env for unbiased evaluation — vec_env has its own internal state
    eval_env  = gym.make("FrozenLake-v1", is_slippery=True)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=500,
                                    deterministic=True)
    eval_env.close()
    vec_env.close()

    if verbose:
        print(f"  Greedy eval (500 ep): mean={mean_r:.4f}  std={std_r:.4f}")

    episode_rewards = callback.episode_rewards
    stats = compute_stats(episode_rewards)
    stats["eval_mean"] = float(mean_r)
    stats["eval_std"]  = float(std_r)
    print_stats("PPO", stats)

    results = {
        "rewards":   episode_rewards,
        "timesteps": callback.episode_timesteps,
        "stats":     stats,
        "eval_mean": float(mean_r),
        "eval_std":  float(std_r),
    }
    save_results(results, "ppo_results.pkl")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve(rewards: list) -> None:
    smooth = smooth_curve(rewards, window=200)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#111111')

    ax.plot(rewards, color='#334155', linewidth=0.4, alpha=0.5)
    ax.plot(smooth,  color='#4ade80', linewidth=2.0, label='Smoothed (200-ep)')

    ax.set_xlabel("Episode", color='#aaaaaa')
    ax.set_ylabel("Reward",  color='#aaaaaa')
    ax.set_title("PPO: Learning Curve on FrozenLake-v1", color='white', fontsize=13)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ppo_rewards.png")
    plt.savefig(out, dpi=150, facecolor='#0d0d0d')
    plt.close()
    print(f"  Learning curve saved → {out}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = train(verbose=True)
    plot_learning_curve(results["rewards"])
    print("\nPPO done. Results saved to results/")
