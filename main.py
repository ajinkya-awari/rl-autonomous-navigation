"""
main.py — Orchestrates training and evaluation of all three RL agents.

Runs Q-Learning → Double DQN → PPO → comparison plots in sequence and
prints a final timing summary. Equivalent to run_all.py but single-process.

Usage:
    python main.py
"""

import time

from q_learning         import train as train_ql,  plot_learning_curve as plot_ql
from dqn_agent          import train as train_dqn, plot_learning_curve as plot_dqn
from ppo_agent          import train as train_ppo, plot_learning_curve as plot_ppo
from compare_algorithms import (
    plot_learning_curves, plot_sample_efficiency,
    plot_final_performance, plot_convergence_speed,
    save_csv, print_summary, load_results,
)
from utils import (plot_qtable_heatmap, plot_policy_arrows,
                   evaluate_greedy_policy, render_navigation_gif)

SEPARATOR = "═" * 62


def banner(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(f"{SEPARATOR}")


if __name__ == "__main__":
    t_total = time.time()

    # ── Q-Learning ─────────────────────────────────────────────────────────
    banner("Phase 1 / 3 — Q-Learning (tabular, pure NumPy)")
    t0     = time.time()
    ql_res = train_ql(verbose=True)
    plot_ql(ql_res["rewards"])
    plot_qtable_heatmap(ql_res["q_table"], grid_size=4)
    plot_policy_arrows(ql_res["q_table"],  grid_size=4)
    ql_eval = evaluate_greedy_policy(ql_res["q_table"], n_episodes=500)
    print(f"  Greedy eval (500 ep): success_rate={ql_eval['success_rate']:.3f}  "
          f"mean_steps={ql_eval['mean_steps']:.1f}")
    print(f"  Phase 1 done in {time.time()-t0:.1f}s")

    # ── Double DQN ─────────────────────────────────────────────────────────
    banner("Phase 2 / 3 — Double DQN (PyTorch)")
    t0      = time.time()
    dqn_res = train_dqn(verbose=True)
    plot_dqn(dqn_res["rewards"], dqn_res["losses"])
    print(f"  Phase 2 done in {time.time()-t0:.1f}s")

    # ── PPO ────────────────────────────────────────────────────────────────
    banner("Phase 3 / 3 — PPO (stable-baselines3)")
    t0      = time.time()
    ppo_res = train_ppo(verbose=True)
    plot_ppo(ppo_res["rewards"])
    print(f"  Phase 3 done in {time.time()-t0:.1f}s")

    # ── Comparison plots ───────────────────────────────────────────────────
    banner("Comparison — Generating all plots & CSV")
    ql  = load_results("ql_results.pkl")
    dqn = load_results("dqn_results.pkl")
    ppo = load_results("ppo_results.pkl")

    ql_r, dqn_r, ppo_r = ql["rewards"], dqn["rewards"], ppo["rewards"]
    plot_learning_curves(ql_r, dqn_r, ppo_r)
    eff = plot_sample_efficiency(ql_r, dqn_r, ppo_r)
    plot_final_performance(ql["stats"], dqn["stats"], ppo["stats"])
    plot_convergence_speed(ql_r, dqn_r, ppo_r)
    save_csv(ql["stats"], dqn["stats"], ppo["stats"], eff)

    # ── Navigation GIF ─────────────────────────────────────────────────────
    banner("Bonus — Rendering navigation GIF")
    render_navigation_gif(policy=ql_res["q_table"])

    # ── Final Summary ──────────────────────────────────────────────────────
    banner("Summary")
    print_summary(ql["stats"], dqn["stats"], ppo["stats"])
    print(f"\n  Total runtime: {(time.time()-t_total)/60:.1f} min")
    print(f"  All outputs saved to results/\n")
