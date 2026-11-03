# 👑 Red Queen AI Arena: Multi-Agent Reinforcement Learning & Game Theory

> **A Multi-Agent Reinforcement Learning (MARL) testbed investigating non-stationary co-adaptation, catastrophic forgetting, and Nash equilibrium convergence in asymmetric bottleneck environments.**

**GitHub Repository:** [https://github.com/Nava-deep-18/Red_queen](https://github.com/Nava-deep-18/Red_queen)

---

## 👥 Authors & Project Team

**Department of Mathematics**  
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

## 🎯 Multiple Targets Environment

### 1. Overview
* **Original Single-Target Environment (`ArenaEnv`):** The baseline Three Bridges arena tasks the Attacker with breaching a single static chokepoint target (the Citadel) positioned at coordinates $(X=10, Y=3)$ on the Eastern boundary. Because the target is unique and situated directly in line with the Center Bridge ($Y=3$), the Defender can establish a stationary or localized chokepoint blockade, leaving little ambiguity regarding the Attacker's ultimate destination.
* **Multiple Targets Extension (`MultiTargetArenaEnv`):** The extended environment introduces configurable target frontiers with $K$ simultaneous target attractors on the Eastern front ($X=10$), supporting configurations $K \in \{1, 2, 3, 5\}$. In particular, the primary multi-target benchmark focuses on the **$K=2$ Flank Perimeter** setup with targets at $(10, 1)$ (North Flank) and $(10, 5)$ (South Flank), as well as $K=3$ (bridge-aligned perimeter) and $K=5$ (saturated frontier).
* **Why Multi-Target Increases Complexity:** Multi-target scaling fundamentally alters the strategic game:
  - **Spatial Defensive Dilemma:** A single defender moving at speed parity ($1\text{ cell/step}$) cannot simultaneously cover multiple disjoint target locations separated by up to 4 vertical grid cells. Defending one flank corridor inherently leaves the opposite river exit vulnerable.
  - **Attacker Feints & Opportunism:** The Attacker needs to breach **only one** active target to win. This rewards dynamic mixed strategies: committing towards one corridor to draw defensive coverage, then pivoting across the river basin to exploit the unprotected target.
  - **Zero-Sum Equilibrium Breakdown:** As target count scales, defensive interception rates collapse, providing an empirical benchmark for perimeter defense and multi-goal spatial decision-making.

---

### 2. Environment Changes
The environment changes are implemented in [`src/multi_target_environment.py`](src/multi_target_environment.py):

* **Target Representation & Configuration:**
  - Standard target coordinates are indexed by target count $K$ via `PRESET_TARGETS`:
    - $K=1$: `[(10, 3)]` (Central Citadel)
    - $K=2$: `[(10, 1), (10, 5)]` (North and South Flank Corridors)
    - $K=3$: `[(10, 1), (10, 3), (10, 5)]` (Aligned with all 3 bridge exits)
    - $K=5$: `[(10, 0), (10, 1), (10, 3), (10, 5), (10, 6)]` (All obstacle-free tiles on row $X=10$)
  - Maximum target capacity is fixed at `MAX_TARGETS = 5`.
* **State Representation (19-Dimensional Box):**
  - To maintain fixed neural network layer dimensions regardless of active target count, observations are projected into a **19-dimensional continuous normalized observation vector** $S \in [0, 1]^{19}$:
    $$S = \big[ \bar{x}_A,\, \bar{y}_A,\, \bar{x}_D,\, \bar{y}_D,\, (\bar{x}_{t_i},\, \bar{y}_{t_i},\, a_{t_i})_{i=0}^4 \big]$$
  - Coordinates are min-max normalized by grid dimensions: $\bar{x} = \frac{x}{10}$, $\bar{y} = \frac{y}{6}$.
  - Each target slot includes an **Active Indicator** $a_{t_i} \in \{0.0, 1.0\}$. Inactive slots ($i \ge K$) are zero-padded: $(0.0, 0.0, 0.0)$.
  - For single-target backward compatibility, `get_state_single(target_idx)` extracts a 6D projection $[\bar{x}_A, \bar{y}_A, \bar{x}_D, \bar{y}_D, \bar{x}_t, \bar{y}_t]$.
* **Attacker Interaction:**
  - The Attacker moves via discrete actions (`UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`).
  - Breaching **ANY** active target $t \in \mathcal{T}$ triggers immediate termination and victory.
* **Defender Interaction:**
  - The Defender attempts to intercept the Attacker via either:
    1. **Same-tile collision:** Both agents occupy identical coordinates at step conclusion.
    2. **Crossing-swap:** Agents cross paths across adjacent tiles in a single simultaneous step.
* **Terminal Priority Axiom:**
  - If the Attacker reaches an active target on the exact step it is intercepted, **the Target Breach takes strict precedence**:
    $$\text{Terminal Outcome} = \begin{cases} 
    \text{Attacker Win } (+10.0, -10.0) & \text{if } p_A \in \mathcal{T} \\
    \text{Defender Intercept } (-10.0, +10.0) & \text{if } \text{Caught} \land p_A \notin \mathcal{T} \\
    \text{Timeout } (0.0, 0.0) & \text{if } t \ge T_{\max} = 75 \text{ steps}
    \end{cases}$$
* **Zero-Sum Multi-Target Potential Shaping:**
  - Generalizes Ng et al. (1999) potential shaping to multiple targets using the bridge-aware shortest route to the **nearest active target**:
    $$D_{\text{multi}}(p) = \min_{t \in \mathcal{T}} D_{\text{route}}(p, t)$$
  - With bridge-routing distance:
    $$D_{\text{route}}(p, t) = \begin{cases}
    \min\limits_{b_y \in \{1, 3, 5\}} \Big(|x - 4| + |y - b_y| + (t_x - 4) + |b_y - t_y|\Big) & \text{if } x \le 4 \\
    |x - t_x| + |y - t_y| & \text{if } x > 4
    \end{cases}$$
  - Symmetrically applied ($+F$ Attacker, $-F$ Defender) with $\gamma = 0.99$, guaranteeing $R_A(s, a) + R_D(s, a) \equiv 0.000000$ at every step.

---

### 3. Single Target vs Multiple Targets

| Feature | Single-Target Baseline (`ArenaEnv`) | Multiple Targets Extension (`MultiTargetArenaEnv`) |
|---|---|---|
| **Primary Code File** | [`src/environment.py`](src/environment.py) | [`src/multi_target_environment.py`](src/multi_target_environment.py) |
| **Target Count ($K$)** | $K = 1$ (fixed) | Configurable $K \in \{1, 2, 3, 5\}$ |
| **Target Locations** | Single Citadel at $(10, 3)$ | $K=2$: $(10, 1)$ and $(10, 5)$; $K=3$: $(10, 1), (10, 3), (10, 5)$; $K=5$: Full Eastern front |
| **Observation Dimension** | 6D vector: $[x_A, y_A, x_D, y_D, x_T, y_T]$ | 19D vector: $[x_A, y_A, x_D, y_D, (x_{t_i}, y_{t_i}, a_{t_i})_{i=0}^4]$ |
| **Target Identification** | Implicit (only one coordinate) | Explicit per-slot activation flag ($a_{t_i} \in \{0.0, 1.0\}$) |
| **Attacker Goal** | Reach $(10, 3)$ | Reach **any** active target $t \in \mathcal{T}$ |
| **Defensive Coverage** | Chokepoint anchor at Center Bridge / Citadel | Split perimeter dilemma across disjoint flanks |
| **Reward Shaping Potential** | Distance to single target $(10, 3)$ | Distance to nearest active target $\min_{t \in \mathcal{T}} D_{\text{route}}(p, t)$ |
| **Zero-Sum Guarantee** | $R_A + R_D \equiv 0.0$ | $R_A + R_D \equiv 0.0$ |
| **Maximum Steps ($T_{\max}$)** | 75 steps | 75 steps |
| **Action Space** | Discrete 5 (`UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`) | Discrete 5 (`UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`) |

---

### 4. Training and Evaluation

#### Dedicated Scripts:
* **Training Script:** [`src/train_multi_target.py`](src/train_multi_target.py)
* **Evaluation Script:** [`src/evaluate_multi_target.py`](src/evaluate_multi_target.py)
* **Cross-Comparison Script:** [`src/compare_single_vs_multi_target.py`](src/compare_single_vs_multi_target.py)
* **Replay Generator:** [`src/generate_k2_replays.py`](src/generate_k2_replays.py)

#### Algorithms & Neural Architecture:
* **Algorithm:** Decentralized Proximal Policy Optimization (PPO) in simultaneous self-play.
* **Network Structure:** Actor-Critic networks with two hidden layers $[64, 64]$ and $\tanh$ activations, accepting 19-dimensional observation inputs.
* **Hyperparameters:** Discount factor $\gamma = 0.99$, PPO clip $\epsilon = 0.2$, learning rates $\alpha_{\text{actor}} = 0.001$, $\alpha_{\text{critic}} = 0.002$, batch updates every 20 complete episodes over 1,500 total training episodes per configuration.

#### Execution Commands:
```bash
# 1. Train PPO policies on Multi-Target Arena (default K=2, 1,500 episodes)
python src/train_multi_target.py --k 2 --episodes 1500 --update-every 20

# Train other target configurations
python src/train_multi_target.py --k 1 --episodes 1500
python src/train_multi_target.py --k 3 --episodes 1500
python src/train_multi_target.py --k 5 --episodes 1500

# 2. Run deterministic scientific evaluation suite (500 paired episodes per archetype)
python src/evaluate_multi_target.py --k 2 --episodes 500

# 3. Run comprehensive Single-Target vs Multi-Target comparison benchmark
python src/compare_single_vs_multi_target.py --episodes 500

# 4. Generate K=2 3D WebGL trajectory replays (for Dual-Archive, Red Queen, and Fixed baseline)
python src/generate_k2_replays.py
```

#### Results & Artifact Locations:
* Checkpoints: `models/multi_target/` (e.g., `ppo_k2_attacker.pth`, `ppo_k2_defender.pth`) and `models/multi_target_k2/`
* Replays: `replays/multi_target_k2/` (`k2_dual_archive.json`, `k2_red_queen.json`, `k2_fixed_opponent.json`)
* Performance Metrics & Plots: `results/multi_target/` and `results/multi_target_k2/` (`comparison_metrics.json`, `learning_curve_k2.png`, `k2_robustness_archetype.png`, `k1_vs_k2_comparison.png`)

---

### 5. Dashboard Visualization

The interactive Streamlit dashboard ([`src/dashboard.py`](src/dashboard.py)) features a top-level mode switcher:

1. **Top-Level Mode Selector:**
   - **`⚔️ Single Target (Citadel)`**: Displays the original 7 single-target analysis pages (3D Arena, Game Theory, Baseline PPO, Naive Self-Play, Dual-Archive, Robustness EGTA, Applications).
   - **`🎯 Multi-Target (K=2 Flank Perimeter)`**: Dedicated sub-dashboard focused on the $K=2$ multi-target system.

2. **Multi-Target K=2 Sub-Pages:**
   - **`🎮 3D Arena (K=2)`:** Embedded WebGL / Three.js tactical arena ([`dashboard_3d_k2.html`](dashboard_3d_k2.html)) rendering **two animated yellow gem targets** with dedicated point lights at $(10, 1)$ and $(10, 5)$, with mode switching between Dual-Archive, Naive Self-Play, and Fixed Baseline.
   - **`📈 Self-Play Training`:** Training dynamics and learning curves for $K=2$ dual-flank self-play (Phase 4 equivalent).
   - **`🛡️ Dual-Archive K=2`:** Analysis of Dual-Archive co-evolutionary regularized training.
   - **`📊 Robustness (K=2)`:** Empirical robustness breakdown against 4 key defender archetypes (North Camper, Center Camper, South Camper, River Patroller) with 95% Wilson Score confidence intervals.
   - **`⚖️ vs Single Target`:** Side-by-side comparative visualization and metric shifts ($K=1$ vs $K=2$).

To launch the dashboard:
```bash
streamlit run src/dashboard.py
```

---

### 6. Research Significance
Extending the Red Queen AI Arena to multiple targets deepens the empirical study of competitive co-evolution in several ways:
* **Overcoming Chokepoint Overfitting:** In single-target environments, defenders easily overfit to a single chokepoint. Multiple targets force both agents to formulate mixed spatial strategies over distributed attractors.
* **Target Saturation & Defender Capacity Limits:** Demonstrates that as $K$ increases, a single defender suffers acute capacity collapse (attacker win rate increases from ~54% at $K=1$ to >71% at $K=5$), highlighting the mathematical limits of solitary defenders in asymmetric security games.
* **Zero-Shot Transfer Failure:** Pretrained single-target policies (6D) fail when transferred zero-shot to multi-target perimeters (win rates dropping from 54% to 24–39%), showing that single-goal policies lack spatial representations for opportunistic flank pivoting.
* **Theoretical Payoff Invariance:** Validates that potential-based zero-sum reward shaping generalizes cleanly to complex multi-goal topologies without altering underlying Nash equilibria.

---

## 📚 Citation & Academic Submission

If utilizing this codebase or experimental framework, please refer to:
* **Paper Title:** *Mitigating Catastrophic Forgetting and Exploitability in Competitive Multi-Agent Reinforcement Learning via Dual-Archive Co-Evolution*
* **LaTeX Source:** [`docs/research_paper_ieee.tex`](docs/research_paper_ieee.tex)
* **Conference Format:** IEEE Conference Template standard (Overleaf-ready).
