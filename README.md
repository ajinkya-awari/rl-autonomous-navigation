# Behavioural Analysis of Reinforcement Learning Algorithms Under Stochastic Navigation Conditions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch)
![stable-baselines3](https://img.shields.io/badge/stable--baselines3-2.3.2-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Research Snapshot

This repository presents a systematic empirical comparison of three reinforcement learning algorithms  **Tabular Q-Learning**, **Double Deep Q-Network (DDQN)**, and **Proximal Policy Optimization (PPO)**  applied to the stochastic FrozenLake-v1 navigation task. We analyse convergence behaviour, sample efficiency, and final policy quality across 10,000 training episodes for Q-Learning, 5,000 for Double DQN, and 100,000 timesteps for PPO. All algorithms are evaluated using an unbiased 500-episode greedy policy evaluation after training, separating exploration-noise-corrupted training rewards from true policy performance. The framework is modular: replace FrozenLake with any Gymnasium-compatible environment and all analysis pipelines run unchanged.

---

## Project Overview

The environment is FrozenLake-v1 from the Gymnasium library: a 4×4 grid where an agent navigates from a start tile to a goal tile while avoiding holes on a slippery surface. Stochastic transitions (1/3 probability of lateral slip regardless of intended direction) make this a genuinely non-trivial credit assignment problem. The reward signal is sparse: +1 only on reaching the goal, 0 everywhere else.

Three algorithms were implemented and evaluated:

- **Tabular Q-Learning** — no function approximation; the full state-action value table is updated directly via the Bellman equation
- **Double DQN** — a neural network approximates Q-values; experience replay and a target network stabilise training; the Double DQN correction (van Hasselt et al., 2016) addresses Q-value overestimation
- **PPO** — a policy-gradient method that directly optimises a clipped surrogate objective across 4 parallel environments

The agent must navigate from start (S) to goal (G) while avoiding holes (H) on a slippery surface. The stochastic transition model makes this a non-trivial credit assignment problem  the agent cannot simply memorise a deterministic path.

### Objective

Find policy $\pi^*$ maximising the expected discounted return:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

---

## Research Questions

1. **Convergence dynamics:** Do value-based and policy-gradient methods converge at different rates, and does convergence speed predict final policy quality?

2. **Overestimation in sparse-reward settings:** Does the Q-value overestimation bias documented in standard DQN produce measurable degradation on a sparse-reward task, and does the Double DQN correction recover meaningful performance?

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where $\alpha = 0.8$ and $\gamma = 0.95$. Exploration uses $\varepsilon$-greedy with exponential decay: $\varepsilon_t = \varepsilon_0 \cdot e^{-\lambda t}$.

**Complexity:**
- Space: $O(|\mathcal{S}||\mathcal{A}|) = O(64)$
- Per-update: $O(1)$  a single table lookup and scalar write

---

## Methodology

### Environment and MDP Formulation

**Experience Replay** - transitions $(s, a, r, s', d)$ are stored in a circular buffer of capacity $N = 10{,}000$ and sampled uniformly. This breaks temporal correlations that destabilise gradient updates.

**Target Network** - a frozen copy $\hat{\theta}$ is used for TD targets, synced every 100 gradient steps. Without this, the network chases a non-stationary target and training diverges.

### Algorithm Implementations

**Tabular Q-Learning** updates the value table via the Bellman optimality equation:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

with $\alpha = 0.8$ and exponential $\varepsilon$-decay. Q-table initialised with small random values (uniform 0–0.01) to break action-selection symmetry.

**Double DQN** approximates $Q(s,a;\theta)$ with a two-layer MLP (16→64→64→4). The Double DQN correction decouples action selection from action evaluation:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \cdot Q\!\left(s',\arg\max_{a'} Q(s',a';\theta);\,\hat{\theta}\right) - Q(s,a;\theta)\right)^2\right]$$

Target network synced every 100 gradient steps; replay buffer capacity 10,000; gradient clipping at max-norm 1.0.

**PPO** optimises a clipped surrogate objective with clip parameter $\varepsilon = 0.2$:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

Trained across 4 parallel environments for 100,000 timesteps. Entropy coefficient 0.01 to discourage premature policy collapse.

### Evaluation Protocol

A deliberate methodological decision was made to report two separate metrics for each algorithm:

- **Training reward** — mean reward across all training episodes, corrupted by exploration
- **Greedy success rate** — 500-episode evaluation with $\varepsilon = 0$ (or deterministic policy for PPO) after training completes

This separation is important: training rewards reflect the exploration-exploitation tradeoff during learning, while greedy success rate reflects the quality of the final learned policy.

---

## Experimental Findings

**Convergence speed vs. final policy quality are decoupled.** DDQN converged fastest (rolling mean crossed 0.8 at episode 1,278), yet PPO produced the strongest greedy policy (73.8% success rate). Q-Learning never reached the 0.8 rolling-mean threshold across 10,000 episodes, though its greedy policy achieved 56.6% — higher than its training mean suggests.

**On-policy mean rewards understate final policy quality.** PPO's training mean reward (0.1676) is the lowest of the three algorithms — a consequence of heavy early exploration across 4 parallel environments pulling the aggregate down. Its greedy evaluation (73.8%) is the highest. This discrepancy illustrates exactly why training rewards should not be the primary evaluation metric for on-policy algorithms.

where $r_t(\theta) = \pi_\theta(a_t|s_t)\,/\,\pi_{\theta_{\text{old}}}(a_t|s_t)$ is the probability ratio and $\hat{A}_t$ is the advantage estimate. The clip parameter $\varepsilon = 0.2$ prevents the policy from changing too drastically in a single update, critical on sparse-reward tasks where a few lucky rollouts would otherwise dominate the gradient.

**Stochasticity imposes a ceiling on tabular methods.** Q-Learning's greedy policy (56.6%) is substantially below DDQN and PPO despite 10,000 training episodes. The 1/3 slip probability means the greedy policy cannot reliably execute any planned path — the tabular representation cannot express uncertainty over outcomes, only expected values.

---

## Results


| Algorithm | Mean Reward | Final-100 Mean | Greedy Success Rate | Convergence Episode |
|-----------|-------------|----------------|---------------------|---------------------|
| Q-Learning | 0.3377 | 0.4900 | 56.6% | Not reached |
| Double DQN | 0.6092 | 0.6200 | - | Episode 1278 |
| PPO | 0.1676 | 0.6300 | 73.8% | Not reached |

*Greedy success rate: 500-episode evaluation with ε=0 after training. Full numeric results in `results/comparison_results.csv`.*

### Visualisations

**Learning Curves**
![Learning Curves](results/compare_learning_curves.png)

**Convergence Speed**
![Convergence](results/compare_convergence.png)

**Final Performance**
![Final Performance](results/compare_final_performance.png)

**Sample Efficiency**
![Sample Efficiency](results/compare_sample_efficiency.png)

**Learned Policy (Greedy Actions)**
![Policy Arrows](results/policy_arrows.png)

**Q-Table Heatmap**
![Q-Table Heatmap](results/qtable_heatmap.png)

**Navigation GIF (Q-Learning greedy policy)**
![Navigation](results/navigation_path.gif)

---

## Connection to Applied Research

This work extends my applied research at **The Leadership 30**, where I lead data-driven analysis of climate and disaster-response indicators across Maharashtra. A recurring challenge is coordinating physical delivery of resources to flood-isolated communities where road access is compromised. Autonomous navigation agents  trained with algorithms like those benchmarked here  form the decision-making core for drone-based emergency delivery systems. The stochastic MDP formulation studied here mirrors real operational uncertainty: sensor noise, wind disturbance, and partial observability in field deployments.

---

## Repository Structure

```
rl-autonomous-navigation/
│
├── q_learning.py           ← Tabular Q-Learning from scratch (pure NumPy)
│                             Bellman update, epsilon-greedy, Q-table delta tracking
│
├── dqn_agent.py            ← Double DQN (PyTorch)
│                             Experience replay, target network, DDQN correction
│
├── ppo_agent.py            ← PPO (stable-baselines3)
│                             4 parallel envs, custom RewardLogger callback
│
├── compare_algorithms.py   ← Comparison plots and summary CSV
│                             Learning curves, sample efficiency, convergence speed
│
├── utils.py                ← Shared utilities
│                             evaluate_greedy_policy, plot_policy_arrows,
│                             plot_qtable_heatmap, render_navigation_gif,
│                             compute_stats, smooth_curve
│
├── main.py                 ← Single-process orchestrator (recommended entry point)
├── run_all.py              ← Subprocess-based sequential runner
├── requirements.txt
└── results/                ← Generated at runtime by main.py
    ├── *.png               ← 9 plots
    ├── navigation_path.gif
    ├── comparison_results.csv
    ├── dqn_model.pth
    ├── ppo_model.zip
    └── *.pkl               ← Serialised results for each algorithm
```

---

## Research Context

This project was built to investigate a specific question about learning dynamics rather than to maximise benchmark performance: whether the structural differences between tabular, value-approximation, and policy-gradient methods produce meaningfully different behaviours when the environment is genuinely stochastic and the reward signal is sparse.

The choice of FrozenLake-v1 was deliberate. Its stochastic transitions stress-test the credit assignment mechanism of each algorithm in a way that deterministic environments cannot. Its small state space (16 states) allows the tabular method to be included without the comparison becoming unfair — Q-Learning operates with full expressivity here, yet still falls short of the approximation-based methods on greedy evaluation.

The broader motivation is applied: at The Leadership 30, I work on data-driven flood response for communities in Maharashtra where physical access is compromised during monsoon season. Autonomous navigation agents form the decision-making core of drone-based delivery systems in such settings — environments that are inherently stochastic, partially observable, and sparse in feedback. Understanding how RL algorithms behave under these conditions, not just which achieves the highest score, is the practical question this investigation addresses.

---

## Setup and Usage

```bash
git clone https://github.com/ajinkya-awari/rl-autonomous-navigation.git
cd rl-autonomous-navigation

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python main.py
```

All outputs are written to `results/`. Runtime approximately 5–10 minutes on CPU.

---

## References

1. Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.
2. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
3. van Hasselt, H., Guez, A. & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
4. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
5. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
6. Raffin, A. et al. (2021). Stable-Baselines3. *JMLR*, 22(268), 1–8.

---

## Citation

```bibtex
@misc{awari2025rl_navigation,
  author       = {Awari, Ajinkya},
  title        = {Behavioural Analysis of RL Algorithms Under Stochastic Navigation Conditions},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ajinkya-awari/rl-autonomous-navigation}},
}
```

---

## License

MIT - see [LICENSE](LICENSE).
