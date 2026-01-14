"""
compare_algorithms.py — Comparative analysis of Q-Learning, Double DQN, and PPO.

Loads pre-saved result pickles from each agent and generates four publication-
quality comparison figures plus a summary CSV. Run this after all three agents
have been trained, or after main.py / run_all.py.

Outputs:
    results/compare_learning_curves.png
    results/compare_sample_efficiency.png
    results/compare_final_performance.png
    results/compare_convergence.png
    results/comparison_results.csv

Usage:
    python compare_algorithms.py
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_results, smooth_curve

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour Palette ────────────────────────────────────────────────────────────

COLOURS = {
    "Q-Learning":  "#38bdf8",  # sky blue
    "Double DQN":  "#f97316",  # orange
    "PPO":         "#4ade80",  # green
}
BG    = '#0d0d0d'
PANEL = '#111111'
MUTED = '#334155'
TICK  = '#aaaaaa'


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    if title:
        ax.set_title(title, color='white', fontsize=12, pad=8)


# ── 1. Learning Curves ────────────────────────────────────────────────────────

def plot_learning_curves(ql_r: list, dqn_r: list, ppo_r: list) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, "Algorithm Comparison: Learning Curves (FrozenLake-v1)")
    # episode counts differ per algorithm — Q-Learning:10k, DQN:5k, PPO:variable
    ax.text(0.99, 0.02, "Note: episode scales differ per algorithm",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#555555', style='italic')

    for label, rewards, window in [("Q-Learning", ql_r, 300),
                                    ("Double DQN", dqn_r, 200),
                                    ("PPO",        ppo_r, 300)]:
        sm = smooth_curve(rewards, window=window)
        ax.plot(rewards, color=MUTED,          linewidth=0.3, alpha=0.4)
        ax.plot(sm,      color=COLOURS[label], linewidth=2.2, label=label)

    ax.axhline(0.8, color='#e94560', linewidth=1.2, linestyle='--', label='80% target')
    ax.set_xlabel("Episode", color=TICK)
    ax.set_ylabel("Reward",  color=TICK)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "compare_learning_curves.png")
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved → {out}")


# ── 2. Sample Efficiency ──────────────────────────────────────────────────────

def plot_sample_efficiency(ql_r: list, dqn_r: list, ppo_r: list) -> dict:
    """
    Bar chart of episodes-to-80%-success-rate per algorithm.

    Uses a 100-episode rolling window; the first crossing above 0.8 is the
    sample efficiency point. Fallback uses each algorithm's own episode count
    so the bars don't get a misleading height from a different algorithm.
    """
    TARGET = 0.8
    WINDOW = 100

    def first_crossing(rewards: list) -> int | None:
        rolling = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode='valid')
        hits = np.where(rolling >= TARGET)[0]
        return int(hits[0] + WINDOW) if len(hits) > 0 else None

    algo_data = {
        "Q-Learning": (ql_r,  first_crossing(ql_r)),
        "Double DQN": (dqn_r, first_crossing(dqn_r)),
        "PPO":        (ppo_r, first_crossing(ppo_r)),
    }

    labels  = list(algo_data.keys())
    # fallback per algorithm — use that algorithm's own episode count, not Q-Learning's
    values  = [
        v if v is not None else len(rewards)
        for rewards, v in algo_data.values()
    ]
    reached = {label: algo_data[label][1] for label in labels}

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, f"Sample Efficiency (episodes to reach {int(TARGET*100)}% success rate)")

    bars = ax.bar(labels, values,
                  color=[COLOURS[k] for k in labels],
                  width=0.5, edgecolor='#1a1a2e', linewidth=0.8)

    for bar, val, label in zip(bars, values, labels):
        display = str(val) if reached[label] is not None else "Not reached"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                display, ha='center', color='white', fontsize=11)

    ax.set_ylabel("Episodes", color=TICK)
    ax.set_ylim(0, max(values) * 1.25 + 200)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "compare_sample_efficiency.png")
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved → {out}")

    return {label: reached[label] for label in labels}


# ── 3. Final Performance Bar Chart ───────────────────────────────────────────

def plot_final_performance(ql_stats: dict, dqn_stats: dict, ppo_stats: dict) -> None:
    labels = ["Q-Learning", "Double DQN", "PPO"]
    means  = [
        ql_stats["final_100_mean"],
        dqn_stats["final_100_mean"],
        ppo_stats.get("eval_mean", ppo_stats["final_100_mean"]),
    ]
    stds   = [
        ql_stats["std_reward"],
        dqn_stats["std_reward"],
        ppo_stats.get("eval_std", ppo_stats["std_reward"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, "Final Performance: Mean Reward ± Std")

    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=[COLOURS[l] for l in labels],
                  error_kw=dict(ecolor='white', linewidth=1.5),
                  width=0.5, edgecolor='#1a1a2e', linewidth=0.8)

    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean_val + 0.02,
                f'{mean_val:.3f}', ha='center', color='white', fontsize=11)

    ax.set_ylabel("Mean Reward", color=TICK)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color='#e94560', linewidth=1.0, linestyle='--')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "compare_final_performance.png")
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved → {out}")


# ── 4. Convergence Speed ──────────────────────────────────────────────────────

def plot_convergence_speed(ql_r: list, dqn_r: list, ppo_r: list) -> None:
    """
    Rolling mean over episode index for all three algorithms.

    Plotting on a shared episode axis makes the convergence rate differences
    more interpretable than bar charts alone.
    """
    WINDOW  = 200
    max_len = max(len(ql_r), len(dqn_r), len(ppo_r))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, "Convergence Speed: Rolling Mean Reward (window=200)")

    for label, rewards in [("Q-Learning", ql_r), ("Double DQN", dqn_r), ("PPO", ppo_r)]:
        sm = smooth_curve(rewards, window=WINDOW)
        ax.plot(range(len(sm)), sm, color=COLOURS[label], linewidth=2.2, label=label)

    ax.axhline(0.8, color='#e94560', linewidth=1.2, linestyle='--',
               alpha=0.7, label='0.8 threshold')
    ax.set_xlabel("Episode",        color=TICK)
    ax.set_ylabel("Smoothed Reward", color=TICK)
    ax.set_xlim(0, max_len)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "compare_convergence.png")
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved → {out}")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_csv(ql_stats: dict, dqn_stats: dict, ppo_stats: dict,
             efficiency: dict) -> None:
    header = ["Algorithm", "Mean Reward", "Std", "Final-100 Mean",
              "Convergence Episode", "Sample Efficiency (ep)"]
    rows = [
        ["Q-Learning",
         f"{ql_stats['mean_reward']:.4f}",
         f"{ql_stats['std_reward']:.4f}",
         f"{ql_stats['final_100_mean']:.4f}",
         ql_stats['convergence_episode'] or "N/A",
         efficiency.get("Q-Learning") or "N/A"],
        ["Double DQN",
         f"{dqn_stats['mean_reward']:.4f}",
         f"{dqn_stats['std_reward']:.4f}",
         f"{dqn_stats['final_100_mean']:.4f}",
         dqn_stats['convergence_episode'] or "N/A",
         efficiency.get("Double DQN") or "N/A"],
        ["PPO",
         f"{ppo_stats['mean_reward']:.4f}",
         f"{ppo_stats['std_reward']:.4f}",
         f"{ppo_stats.get('eval_mean', ppo_stats['final_100_mean']):.4f}",
         ppo_stats['convergence_episode'] or "N/A",
         efficiency.get("PPO") or "N/A"],
    ]
    out = os.path.join(RESULTS_DIR, "comparison_results.csv")
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Summary CSV saved → {out}")


# ── Print Table ───────────────────────────────────────────────────────────────

def print_summary(ql_stats: dict, dqn_stats: dict, ppo_stats: dict) -> None:
    print(f"\n{'─'*62}")
    print(f"  {'Algorithm':<14} {'Mean Reward':>12} {'Std':>8} {'Final-100':>10}")
    print(f"{'─'*62}")
    for label, stats in [("Q-Learning", ql_stats), ("Double DQN", dqn_stats), ("PPO", ppo_stats)]:
        f100 = stats.get("eval_mean", stats["final_100_mean"])
        print(f"  {label:<14} {stats['mean_reward']:>12.4f} "
              f"{stats['std_reward']:>8.4f} {f100:>10.4f}")
    print(f"{'─'*62}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading results ...")
    ql  = load_results("ql_results.pkl")
    dqn = load_results("dqn_results.pkl")
    ppo = load_results("ppo_results.pkl")

    ql_r, dqn_r, ppo_r = ql["rewards"], dqn["rewards"], ppo["rewards"]

    print("\nGenerating comparison plots ...")
    plot_learning_curves(ql_r, dqn_r, ppo_r)
    efficiency = plot_sample_efficiency(ql_r, dqn_r, ppo_r)
    plot_final_performance(ql["stats"], dqn["stats"], ppo["stats"])
    plot_convergence_speed(ql_r, dqn_r, ppo_r)
    save_csv(ql["stats"], dqn["stats"], ppo["stats"], efficiency)
    print_summary(ql["stats"], dqn["stats"], ppo["stats"])

    print("\nAll comparison outputs saved to results/")
