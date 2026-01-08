"""
dqn_agent.py — Double Deep Q-Network (DDQN) on FrozenLake-v1 using PyTorch.

Implements experience replay, target network, and epsilon-greedy exploration
following Mnih et al. (2015), with the Double DQN correction from van Hasselt
et al. (2016) to mitigate Q-value overestimation on sparse-reward environments.

Outputs:
    results/dqn_rewards.png
    results/dqn_model.pth
    results/dqn_results.pkl

Usage:
    python dqn_agent.py
"""

import os
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym

from utils import compute_stats, print_stats, save_results, smooth_curve

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_EPISODES      = 5_000
MAX_STEPS       = 200
GAMMA           = 0.99
LR              = 1e-3
BATCH_SIZE      = 64
BUFFER_CAPACITY = 10_000
TARGET_UPDATE   = 100       # hard sync every N gradient steps
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.995     # multiplicative per episode

# FrozenLake is 16-state 4-action — a small MLP is intentional here
HIDDEN_DIM = 64

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: float) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # np.array() first avoids slow element-by-element tensor construction
        return (
            torch.as_tensor(np.array(states,      dtype=np.float32)).to(DEVICE),
            torch.as_tensor(np.array(actions,     dtype=np.int64)).to(DEVICE),
            torch.as_tensor(np.array(rewards,     dtype=np.float32)).to(DEVICE),
            torch.as_tensor(np.array(next_states, dtype=np.float32)).to(DEVICE),
            torch.as_tensor(np.array(dones,       dtype=np.float32)).to(DEVICE),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Fully-connected Q-network mapping one-hot state → action values.

    Intentionally small — FrozenLake has 16 states and a deeper net would
    just overfit early episodes.
    TODO: try dueling architecture (Wang et al., 2016) for smoother value estimates.
    """

    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── State Encoding ────────────────────────────────────────────────────────────

def encode_state(state: int, n_states: int) -> np.ndarray:
    # one-hot is the standard way to feed discrete states into a network
    vec = np.zeros(n_states, dtype=np.float32)
    vec[state] = 1.0
    return vec


# ── DDQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN agent with experience replay and target network.

    Space:    O(buffer_capacity) for replay buffer.
    Per-step: O(batch_size) for gradient update.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states  = n_states
        self.n_actions = n_actions

        self.policy_net = QNetwork(n_states, n_actions).to(DEVICE)
        self.target_net = QNetwork(n_states, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is never directly trained

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay     = ReplayBuffer(BUFFER_CAPACITY)
        self.epsilon    = EPSILON_START
        self.step_count = 0

    def select_action(self, state_vec: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            # torch.as_tensor respects DEVICE from the start, avoids CPU→GPU copy
            state_t = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return int(self.policy_net(state_t).argmax().item())

    def update(self) -> float:
        """
        One Double DQN gradient step. Returns the loss value.

        Standard DQN picks the greedy action AND evaluates it with the target net,
        causing systematic Q-value overestimation (van Hasselt et al., 2016).
        Double DQN fix: policy net selects the action, target net evaluates it.
        Two lines of change, meaningful improvement on sparse-reward environments.
        """
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)

        q_pred = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: a* = argmax_a Q_policy(s', a); value = Q_target(s', a*)
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            q_next       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            td_target    = rewards + GAMMA * q_next * (1.0 - dones)

        loss = nn.MSELoss()(q_pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping prevents loss spikes on early sparse-reward episodes
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.detach().item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(verbose: bool = True) -> dict:
    """
    Train DDQN for N_EPISODES episodes and return a results dict.

    Saves trained weights to results/dqn_model.pth. The agent object is NOT
    pickled — only serialise the lightweight metrics to keep the pkl small.
    """
    env       = gym.make("FrozenLake-v1", is_slippery=True)
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(n_states, n_actions)
    print(f"\nTraining Double DQN on {DEVICE}  ({N_EPISODES:,} episodes) ...")

    episode_rewards = []
    episode_losses  = []
    t_start = time.time()

    for ep in range(N_EPISODES):
        state, _  = env.reset()
        state_vec = encode_state(state, n_states)
        total_reward = 0.0
        ep_losses    = []

        for _ in range(MAX_STEPS):
            action   = agent.select_action(state_vec)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done     = terminated or truncated
            next_vec = encode_state(next_obs, n_states)

            agent.replay.push(state_vec, action, reward, next_vec, float(done))
            loss_val = agent.update()
            if loss_val > 0:
                ep_losses.append(loss_val)

            state_vec    = next_vec
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        episode_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        agent.decay_epsilon()

        if verbose and (ep + 1) % 500 == 0:
            win_rate = np.mean(episode_rewards[-200:])
            print(f"  ep {ep+1:>5,} | ε={agent.epsilon:.3f} | "
                  f"win_rate(200)={win_rate:.3f} | loss={episode_losses[-1]:.4f}")

    elapsed = time.time() - t_start
    print(f"  Training complete in {elapsed:.1f}s")

    model_path = os.path.join(RESULTS_DIR, "dqn_model.pth")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"  Model saved → {model_path}")
    env.close()

    stats = compute_stats(episode_rewards)
    print_stats("Double DQN", stats)

    # don't pickle the agent — the buffer alone is ~5 MB and nothing downstream needs it
    results = {
        "rewards": episode_rewards,
        "losses":  episode_losses,
        "stats":   stats,
    }
    save_results(results, "dqn_results.pkl")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve(rewards: list, losses: list) -> None:
    smooth_r = smooth_curve(rewards, window=100)
    smooth_l = smooth_curve(losses,  window=100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('#0d0d0d')

    for ax in (ax1, ax2):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    ax1.plot(rewards,  color='#334155', linewidth=0.4, alpha=0.5)
    ax1.plot(smooth_r, color='#f97316', linewidth=2.0, label='Smoothed (100-ep)')
    ax1.set_ylabel("Reward", color='#aaaaaa')
    ax1.set_title("Double DQN: Training Progress", color='white', fontsize=13)
    ax1.legend(facecolor='#1a1a2e', labelcolor='white')

    ax2.plot(losses,   color='#334155', linewidth=0.4, alpha=0.5)
    ax2.plot(smooth_l, color='#a78bfa', linewidth=2.0, label='Smoothed loss')
    ax2.set_ylabel("Loss (MSE)", color='#aaaaaa')
    ax2.set_xlabel("Episode",    color='#aaaaaa')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dqn_rewards.png")
    plt.savefig(out, dpi=150, facecolor='#0d0d0d')
    plt.close()
    print(f"  Learning curve saved → {out}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = train(verbose=True)
    plot_learning_curve(results["rewards"], results["losses"])
    print("\nDouble DQN done. Results saved to results/")
