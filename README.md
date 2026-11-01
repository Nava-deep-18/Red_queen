# 👑 Red Queen AI Arena: Multi-Agent Reinforcement Learning & Game Theory

> **A Multi-Agent Reinforcement Learning (MARL) testbed investigating non-stationary co-adaptation, catastrophic forgetting, and Nash equilibrium convergence in asymmetric bottleneck environments.**

**GitHub Repository:** [https://github.com/Nava-deep-18/Red_queen](https://github.com/Nava-deep-18/Red_queen)

---

## 👥 Authors & Project Team

**Department of Mathematics and Computing (MnC)**  
**Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, Gujarat, India**

* **Navadeep G.** — [nindra.navadeep11@gmail.com](mailto:nindra.navadeep11@gmail.com) | GitHub: [@Nava-deep-18](https://github.com/Nava-deep-18)
* **Brij Radadiya** — [radadiyabrij43@gmail.com](mailto:radadiyabrij43@gmail.com) | GitHub: [@RadadiyaBrij](https://github.com/RadadiyaBrij)
* **Shrey Marakana** — [marakanashrey2426@gmail.com](mailto:marakanashrey2426@gmail.com) | GitHub: [@SHREY-RK](https://github.com/SHREY-RK)
* **Shree Pastagia** — [shreepastagia126219@gmail.com](mailto:shreepastagia126219@gmail.com) | GitHub: [@Shree-Pastagia](https://github.com/Shree-Pastagia)

---

## 📦 Dataset Resources

**No external dataset is required or used in this project.**  
The environment is an entirely self-contained, procedurally simulated two-player discrete zero-sum Markov decision process (`ArenaEnv`) built on the **Farama PettingZoo** `ParallelEnv` standard. State observations, obstacle geometries, river crossing bottlenecks, and terminal rewards are synthesized dynamically at runtime according to exact game-theoretic specifications.

---

## 💻 Implementation Environment Setup

### System Prerequisites
* **Operating System:** Cross-platform (tested on Windows 11, Ubuntu 22.04 LTS, macOS Sonoma)
* **Python Runtime:** Python 3.10, 3.11, 3.12, or 3.13
* **Hardware:** Minimal CPU footprint (standard multi-core laptop CPU sufficient for both training and evaluation; no dedicated GPU required, though PyTorch CUDA is supported if available).

### Installation Instructions
```bash
# 1. Clone the repository
git clone https://github.com/Nava-deep-18/Red_queen.git
cd Red_queen

# 2. Set up a virtual environment
# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Core Dependencies (`requirements.txt`)
* `torch`: Neural actor-critic policy architectures & PPO backpropagation.
* `pettingzoo>=1.24.0` & `gymnasium`: Multi-agent parallel environment API standards.
* `numpy`: Coordinate vector mathematics & matrix game fictitious play.
* `matplotlib`: Publication-grade empirical charts and diagnostic plots.
* `streamlit`: Interactive web-based analytics dashboard and 2D replay player.
* `plotly`: High-dimensional trajectory visualizations.

---

## 🔬 1. The Research Problem

### Theoretical Background: The Red Queen Effect in MARL
Named after Lewis Carroll’s Red Queen in *Through the Looking-Glass* (*"Now, here, you see, it takes all the running you can do, to keep in the same place"*), Leigh Van Valen formulated the evolutionary hypothesis that competing organisms must continually adapt simply to maintain relative fitness parity.

In **Multi-Agent Reinforcement Learning (MARL)**, simultaneous self-play induces severe **environmental non-stationarity**. Because both the learning agent and its adversary update their policies concurrently, the transition and reward distribution from either agent's perspective shifts continually:
$$\mathcal{P}(s_{t+1} \mid s_t, a_t, \pi_{\text{opponent}}^{(t)}) \ne \mathcal{P}(s_{t+1} \mid s_t, a_t, \pi_{\text{opponent}}^{(t+k)})$$

In environments with intransitive strategy topologies (e.g., spatial Rock-Paper-Scissors), naive self-play triggers closed policy cycling and **catastrophic forgetting**: an agent specializes against its immediate opponent, discarding previously mastered behaviors and leaving itself vulnerable to past strategies.

### The Three Bridges Spatial Arena
To study this phenomenon empirically without unnecessary visual complexity, we design an $11 \times 7$ discrete grid-world arena featuring an impassable river divided by **Three Bridges** (North $Y=1$, Center $Y=3$, South $Y=5$) and four impassable terrain pillars:

```
  Y=0  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
  Y=1  [   ] [   ] [   ] [   ] [   ] [ 🌉 ] [   ] [   ] [   ] [   ] [   ]  <-- North Bridge
  Y=2  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
  Y=3  [   ] [   ] [   ] [   ] [   ] [ 🌉 ] [   ] [   ] [   ] [ 🎯 ] [   ]  <-- Center Bridge & Citadel
  Y=4  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
  Y=5  [   ] [   ] [   ] [   ] [   ] [ 🌉 ] [   ] [   ] [   ] [   ] [   ]  <-- South Bridge
  Y=6  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
       X=0   X=1   X=2   X=3   X=4    X=5    X=6   X=7   X=8   X=9   X=10
       <------ Infiltration Basin ----> River <------- Defense Basin ------->
```

- **Why Three Bridges?**
  - **1 Bridge:** Trivial bottleneck; defender parks at the single chokepoint.
  - **2 Bridges:** Binary matching penny game.
  - **3 Bridges:** Creates an irreducible spatial **Rock-Paper-Scissors** dilemma. With 1 defender occupying only 1 cell and moving at speed parity ($1\text{ cell/step}$), the defender *cannot physically guard all three bridges simultaneously*. Pure strategies are strictly exploitable.

---

## 🛠️ 2. Methodology & Architecture

### Phase 1: Environment & Mathematical Formulation ([`src/environment.py`](src/environment.py))
* **State Vector:** 6-dimensional continuous normalized observation $S = [\frac{x_A}{10}, \frac{y_A}{6}, \frac{x_D}{10}, \frac{y_D}{6}, \frac{x_T}{10}, \frac{y_T}{6}] \in [0, 1]^6$.
* **Action Space:** Discrete 5-action space for each agent: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`.
* **Zero-Sum Potential-Based Reward Shaping (PBRS):** Implements discounted shaping satisfying Ng et al. (1999) under discount $\gamma = 0.99$:
  $$F_A(s, s') = +F_{\text{tgt}}(s, s') - F_{\text{rel}}(s, s'), \quad F_D(s, s') = -F_A(s, s')$$
  guaranteeing $R_A(s, a) + R_D(s, a) \equiv 0.000000$ identically at every timestep, preserving the underlying Nash equilibrium (Devlin & Kudenko, 2011).

### Phase 2: Agent Architecture ([`src/agents/ppo.py`](src/agents/ppo.py))
* **Proximal Policy Optimization (PPO):** Actor-Critic network with two hidden layers $[64, 64]$ and $\tanh$ activations.
* **Clipped Surrogate Objective:** $\epsilon = 0.2$, minibatch size 64, 4 optimization epochs per update.
* **Numerical Safeguards:** Advantage standardization with single-sample NaN guards and gradient norm clipping ($\tau_{\text{clip}} = 0.5$).

### Phase 3: Single-Agent Baseline ([`src/train_phase3.py`](src/train_phase3.py))
* Attacker learns basic pathfinding and chokepoint navigation against a passive STAY defender over 1,200 episodes.

### Phase 4: Naive Self-Play ([`src/train_phase4.py`](src/train_phase4.py))
* Concurrent PPO updates for Attacker and Defender over 1,500 episodes without historical memory.
* Induces policy cycling and route collapse (asymmetric North bias, South corridor neglect, unforced timeouts).

### Phase 6: Dual-Archive Co-Evolution ([`src/train_phase6.py`](src/train_phase6.py))
* Maintains dual historical snapshot reservoirs $\mathcal{H}_A$ and $\mathcal{H}_D$, appending frozen checkpoints every 100 episodes after a 300-episode warmup.
* **Tripartite Opponent Sampling Split:**
  - **60% Current vs. Current:** Both active agents co-evolve against contemporary strategies.
  - **20% Current Attacker vs. Sampled Historical Defender:** Attacker trains against past defensive strategies; defender weights remain frozen.
  - **20% Sampled Historical Attacker vs. Current Defender:** Defender trains against past offensive routes; attacker weights remain frozen.

---

## 📊 3. Empirical Results & Evidence

### A. Shannon Route Entropy (Catastrophic Forgetting)
Evaluated across 150 paired pseudo-random seeds against an uncontested open-river baseline:

| Architecture | North ($Y=1$) | Center ($Y=3$) | South ($Y=5$) | Timeouts | Shannon Entropy | Diversity (% of Max) | Behavioral Profile |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Phase 3 (Fixed Baseline)** | 21.3% | 43.3% | 35.3% | 0.0% | 1.529 bits | 96.4% | Baseline navigation |
| **Phase 4 (Naive Self-Play)** | 46.0% | 37.9% | 16.1% | 17.3% | 1.470 bits | 92.8% | **Asymmetric North bias, 17.3% timeouts** |
| **Phase 6 (Dual Archive)** | 26.7% | 34.0% | 39.3% | 0.0% | **1.567 bits** | **98.9%** | **Uniform multi-route coverage** |

* Theoretical Maximum Entropy: $H_{\max} = \log_2(3) \approx 1.585$ bits.
* Naive self-play suffers 17.3% unforced navigational timeouts and severe South bridge neglect (16.1%).
* Dual-Archive co-evolution restores multi-bridge traversal to **98.9% of maximum entropy** with **0.0% timeouts**.

### B. Empirical Game-Theoretic Analysis (EGTA)
Evaluated against the spatial dilemma archetype suite $\mathcal{P}_{\text{def}}$ (North Camper, Center Camper, South Camper, River Patroller, Learned Defenders):

| Attacker Policy | Security Level ($v_{\text{sec}}$) | Worst-Case Win Rate | Meta Exploitability ($\delta$) | Average Win Margin |
|---|:---:|:---:|:---:|:---:|
| **Phase 4 (Naive Self-Play)** | -0.207 | 39.3% | 0.160 | +0.190 |
| **Phase 6 (Dual Archive)** | **-0.187** | 31.3% | **0.140** | **+0.184** |

* **Exploitability Reduction:** Phase 6 reduces empirical exploitability $\delta_{\text{meta}} = v^* - v_{\text{sec}}$ from 0.160 down to **0.140** (a 12.5% reduction).
* **Security Level:** Worst-case net margin improves from $-0.207$ to **$-0.187$**.
* **Patroller Win Rate:** Phase 4 achieves 76.0%, whereas Phase 6 achieves 81.3%.
* **Pooled Win Rate Test:** Paired difference across all seeds is $\Delta = -0.6\%$ ($p = 0.771$). This confirms the key game-theoretic insight: **raw win rate is uninformative in MARL**, while Shannon route entropy and empirical minimax exploitability cleanly isolate strategic robustness.

---

## 📂 4. Repository Structure

```
Red_queen/
├── dashboard_3d.html          # WebGL / Three.js 3D Tactical Arena replay viewer
├── requirements.txt           # Python dependency requirements
├── README.md                  # Comprehensive documentation & setup guide
│
├── docs/
│   ├── research_paper_ieee.tex    # Full IEEE conference publication source (LaTeX)
│   ├── phase1_game_theory_spec.md # Formal mathematical game theory specification
│   ├── empirical_evidence_diversity_and_nash.md # Standardized empirical test report
│   ├── scientific_audit_report.md # Multi-stage rigorous architectural audit
│   ├── TO_DO_NEXT.md              # Planned algorithmic and experimental extensions
│   ├── bridge_diversity_chart.png # Shannon entropy & crossing distribution plot
│   ├── nash_exploitability_chart.png # Exploitability & security level plot
│   ├── phase3_training_curve.png  # Phase 3 learning curve
│   ├── phase4_marl_curve.png      # Phase 4 self-play learning curve
│   ├── phase6_archive_curve.png   # Phase 6 dual-archive 2-panel diagnostic curve
│   └── robustness_chart.png       # 4-archetype win rate comparison chart
│
├── models/                    # Serialized neural network weights (.pth) [gitignored]
│   ├── phase3_attacker.pth
│   ├── phase4_attacker.pth
│   ├── phase6_attacker.pth
│   ├── phase4_defender.pth
│   └── phase6_defender.pth
│
├── replays/                   # JSON trajectory logs for 2D/3D replay viewers
│   ├── fixed_opponent.json
│   ├── red_queen.json
│   └── red_queen_history.json
│
└── src/
    ├── environment.py             # Three Bridges MARL Farama PettingZoo environment
    ├── replay_logger.py           # Trajectory serialization module
    ├── evaluate_bridge_diversity.py # Empirical Shannon entropy & EGTA exploitability evaluator
    ├── evaluate_robustness.py     # Paired-seed evaluation against archetypes
    ├── dashboard.py               # Streamlit interactive analytics dashboard
    │
    ├── agents/
    │   ├── ppo.py                 # Actor-Critic PPO implementation with action masking
    │   └── dqn.py                 # Modular Deep Q-Network baseline agent
    │
    ├── train_phase3.py            # Baseline training: Attacker vs Scripted STAY Defender
    ├── train_phase4.py            # Naive MARL self-play training
    └── train_phase6.py            # Dual-Archive co-evolutionary MARL training
```

> **Note on `models/`:** Pretrained checkpoint weights (`.pth`) are gitignored by default to avoid bloating repository history. Running the training scripts below generates checkpoints in `models/`, or existing weights can be placed directly into the `models/` folder.

---

## 🚀 5. Quickstart & Execution Guide

### 1. Training Agents
```bash
# Phase 3: Train baseline pathfinding against static defender (1,200 episodes)
python src/train_phase3.py

# Phase 4: Train simultaneous naive self-play (1,500 episodes)
python src/train_phase4.py

# Phase 6: Train Dual-Archive co-evolutionary PPO (1,500 episodes)
python src/train_phase6.py
```

### 2. Evaluating Policies & Generating Empirical Evidence
```bash
# Run comprehensive Shannon Entropy & Minimax Exploitability benchmark (150 seeds)
python src/evaluate_bridge_diversity.py

# Run paired-seed robustness evaluation against 4 fixed defender archetypes
python src/evaluate_robustness.py
```

### 3. Launching Analytics Dashboards & Replay Visualizers
```bash
# Launch interactive Streamlit dashboard
streamlit run src/dashboard.py

# View 3D WebGL tactical arena
# Open dashboard_3d.html in Google Chrome, Firefox, or Edge
python -m http.server 8000
# Navigate to http://localhost:8000/dashboard_3d.html
```

---

## 📚 Citation & Academic Submission

If utilizing this codebase or experimental framework, please refer to:
* **Paper Title:** *Mitigating Catastrophic Forgetting and Exploitability in Competitive Multi-Agent Reinforcement Learning via Dual-Archive Co-Evolution*
* **LaTeX Source:** [`docs/research_paper_ieee.tex`](docs/research_paper_ieee.tex)
* **Conference Format:** IEEE Conference Template standard (Overleaf-ready).
