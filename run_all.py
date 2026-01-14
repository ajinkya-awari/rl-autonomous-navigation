"""
run_all.py — Runs each training script as a subprocess in order.

Useful if you want to inspect outputs between steps or restart from
a specific phase without re-running everything. For a single in-process
run, use main.py instead.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time

SCRIPTS = [
    ("Q-Learning",             "q_learning.py"),
    ("Double DQN",             "dqn_agent.py"),
    ("PPO",                    "ppo_agent.py"),
    ("Algorithm Comparison",   "compare_algorithms.py"),
]


def run_script(label: str, script: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  Running: {label}  ({script})")
    print(f"{'─'*55}")
    t0  = time.time()
    ret = subprocess.run([sys.executable, script], check=False)
    if ret.returncode != 0:
        print(f"  ✗  {script} exited with code {ret.returncode} — stopping.")
        sys.exit(ret.returncode)
    print(f"  ✓  {label} complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    t_start = time.time()
    for label, script in SCRIPTS:
        run_script(label, script)

    total = (time.time() - t_start) / 60
    print(f"\n{'═'*55}")
    print(f"  All scripts complete.  Total: {total:.1f} min")
    print(f"  Results in:  results\\")
    print(f"{'═'*55}\n")
