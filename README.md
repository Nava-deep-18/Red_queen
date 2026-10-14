# 👑 Red Queen AI Arena: Multi-Agent Reinforcement Learning & Game Theory

> **A Multi-Agent Reinforcement Learning (MARL) testbed investigating non-stationary co-adaptation, catastrophic forgetting, and Nash equilibrium convergence in asymmetric bottleneck environments.**

---

## 🔬 1. The Research Problem: What We Are Investigating

### Theoretical Background: The Red Queen Hypothesis
Named after Lewis Carroll’s Red Queen in *Through the Looking-Glass* (*"Now, here, you see, it takes all the running you can do, to keep in the same place"*), Leigh Van Valen formulated the evolutionary hypothesis that organisms must constantly adapt and evolve simply to survive against ever-evolving competing organisms.

In **Multi-Agent Reinforcement Learning (MARL)**, simultaneous self-play introduces severe **environmental non-stationarity**. Because both the learning agent and its opponent are updating their policies concurrently, the environment from either agent's perspective is constantly changing:
$$\mathcal{P}(s_{t+1} \mid s_t, a_t, \pi_{\text{opponent}}^{(t)}) \ne \mathcal{P}(s_{t+1} \mid s_t, a_t, \pi_{\text{opponent}}^{(t+k)})$$

### The Three Bridges Spatial Dilemma
To study this phenomenon empirically without unnecessary state-space complexity, the project designs an $11 \times 7$ discrete grid-world arena featuring an impassable river divided by **Three Bridges** (North $Y=1$, Center $Y=3$, South $Y=5$) and four impassable obstacle pillars:

```
  Y=0  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
  Y=1  [   ] [   ] [   ] [   ] [   ] [ 🌉 ] [   ] [   ] [   ] [   ] [   ]  <-- North Bridge
  Y=2  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
  Y=3  [   ] [   ] [   ] [   ] [   ] [ 🌉 ] [   ] [   ] [   ] [ 🎯 ] [   ]  <-- Center Bridge & Target
  Y=4  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
  Y=5  [   ] [   ] [   ] [   ] [   ] [ 🌉 ] [   ] [   ] [   ] [   ] [   ]  <-- South Bridge
  Y=6  [   ] [   ] [   ] [   ] [   ] [ 🔒 ] [   ] [   ] [   ] [   ] [   ]
       X=0   X=1   X=2   X=3   X=4    X=5    X=6   X=7   X=8   X=9   X=10
       <------ Left Basin ------>    River   <------ Right Basin ------>
```

- **Why Three Bridges?**
  - **1 Bridge:** Trivial pathfinding obstacle; defender camps the single chokepoint.
  - **2 Bridges:** Trivial binary coin-flip.
  - **3 Bridges:** Creates an intractable spatial **Rock-Paper-Scissors** dilemma. With 1 defender occupying only 1 cell and moving at the same speed as the attacker, the defender *cannot physically block all three bridges simultaneously*.

### Core Research Questions
1. **The Non-Stationary Cycle (Red Queen Dynamics):**
   In simultaneous self-play, do agents cycle through counter-strategies indefinitely (North $\rightarrow$ South $\rightarrow$ Center $\rightarrow$ North) rather than converging to a stable solution?
2. **Catastrophic Forgetting:**
   When an attacker policy learns to exploit a defender camping at North by routing South, does its neural network catastrophically overwrite and forget how to bypass the Center or North routes when the defender subsequently shifts South?
3. **Double History Archive & Nash Equilibrium Convergence:**
   Can a dual-archive mechanism (storing historical policies of both Attacker and Defender and training against a mixture of past snapshots, analogous to Policy-Space Response Oracles / fictitious play) prevent catastrophic forgetting and force both agents to converge toward an unexploitable mixed-strategy Nash equilibrium?
4. **Generalization vs. Overfitting in Self-Play:**
   Does training against an opponent archive produce an agent that generalizes better against unseen, hand-crafted behavioral archetypes compared to naive self-play?

---

## 🛠️ 2. What We Have Done Yet: Current Implementation & Findings

### Phase 1: Environment & Mathematical Formulation
- **File:** [`src/environment.py`](file:///c:/Red_queen/src/environment.py)
- **Multi-Agent Grid-World (`ArenaEnv`):** Multi-agent parallel interface (following PettingZoo Parallel API semantics) on an $11 \times 7$ grid.
- **State Representation:** 6-dimensional normalized vector $S = [\frac{a_x}{10}, \frac{a_y}{6}, \frac{d_x}{10}, \frac{d_y}{6}, \frac{t_x}{10}, \frac{t_y}{6}]$.
- **Action Space:** Discrete 5-action space for each agent: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`.
- **Termination Dynamics:** Attacker wins by securing the Citadel at $(10, 3)$; Defender wins by intercepting the Attacker before reaching Citadel; episode truncates at 100 steps.

### Phase 2: Agent Algorithms
- **Files:** [`src/agents/ppo.py`](file:///c:/Red_queen/src/agents/ppo.py), [`src/agents/dqn.py`](file:///c:/Red_queen/src/agents/dqn.py)
- **Proximal Policy Optimization (PPO):** Actor-Critic architecture with clipped objective ($\epsilon = 0.2$), Generalized Advantage Estimation (GAE), standard deviation stability guards (preventing NaN updates on edge cases), and greedy deterministic evaluation (`predict`).

### Phase 3: Baseline Single-Agent RL vs. Scripted Opponents
- **File:** [`src/train_phase3.py`](file:///c:/Red_queen/src/train_phase3.py)
- Attacker learns against a static / rule-based camping defender to establish baseline pathfinding and avoidance capability.
- Generates baseline replay [`replays/fixed_opponent.json`](file:///c:/Red_queen/replays/fixed_opponent.json) and learning curve [`docs/phase3_training_curve.png`](file:///c:/Red_queen/docs/phase3_training_curve.png).

### Phase 4: Simultaneous MARL Self-Play (The Red Queen Dynamic)
- **File:** [`src/train_phase4.py`](file:///c:/Red_queen/src/train_phase4.py)
- Both Attacker and Defender train concurrently via PPO over 2,500 episodes.
- **Strictly Zero-Sum Payoffs:** Symmetric potential-based reward shaping $F(s, s')$ applied as $+F$ to the Attacker and $-F$ to the Defender, ensuring $R_{\text{Attacker}} + R_{\text{Defender}} \equiv 0$ at every timestep.
- Synchronized episode-boundary batch updates and mutual win-rate tracking.
- Generates replay [`replays/red_queen.json`](file:///c:/Red_queen/replays/red_queen.json) and curve [`docs/phase4_marl_curve.png`](file:///c:/Red_queen/docs/phase4_marl_curve.png).

### Phase 6: Double History Archive Training (Stability & Anti-Forgetting)
- **File:** [`src/train_phase6.py`](file:///c:/Red_queen/src/train_phase6.py)
- Maintains rolling memory archives of historical policies for both Attacker and Defender (`attacker_archive`, `defender_archive`).
- **Strictly Zero-Sum Payoffs:** Symmetric potential shaping without ad-hoc asymmetric bonuses; opponent matchups preserve $R_A + R_D = 0$.
- **Empirical Sampling Split:**
  - $60\%$ current Attacker vs. current Defender (both policies actively train).
  - $20\%$ current Attacker vs. sampled historical Defender (Attacker learns against historical defense).
  - $20\%$ sampled historical Attacker vs. current Defender (Defender learns against historical attack).
- **Persistent Disk Checkpoints:** Automatically serializes snapshot weights to `models/archive/attacker_ep{N}.pth` and `defender_ep{N}.pth`.
- **Symmetrical 2-Panel Diagnostic Curve:**
  - **Top Panel:** Dual-agent rolling win rates (Attacker WR vs Defender WR) plotted on a shared timeline.
  - **Bottom Panel:** Timeout rate diagnostic curve (distinguishing true defensive catches from attacker navigation timeouts).
- Generates replay [`replays/red_queen_history.json`](file:///c:/Red_queen/replays/red_queen_history.json) and learning curve [`docs/phase6_archive_curve.png`](file:///c:/Red_queen/docs/phase6_archive_curve.png).

### Phase 7: Robustness & Generalization Evaluation
- **File:** [`src/evaluate_robustness.py`](file:///c:/Red_queen/src/evaluate_robustness.py)
- Benchmarks the Phase 4 self-play attacker vs. the Phase 6 archive attacker against 4 fixed rule-based defender archetypes:
  1. **North Camper:** Patrols and blocks $(6, 1)$.
  2. **Center Camper:** Patrols and blocks $(6, 3)$.
  3. **South Camper:** Patrols and blocks $(6, 5)$.
  4. **River Patroller:** Dynamically guards the river line based on vertical distance.
- Uses **paired random seeds** and **deterministic greedy actions** (`predict`) to report mean win rates with 95% confidence intervals.

### Phase 8: Interactive Visualization & Analytical Tooling
- **Streamlit Analytics Dashboard ([`src/dashboard.py`](file:///c:/Red_queen/src/dashboard.py)):**
  - Interactive multi-page dashboard featuring game theory breakdown, empirical training curves, robustness comparisons, and step-by-step 2D episode playback.
- **3D Tactical Arena Viewer ([`dashboard_3d.html`](file:///c:/Red_queen/dashboard_3d.html)):**
  - Hardware-accelerated Three.js / WebGL tactical board rendering agents, obstacle pillars, river crossings, bridge choke-points, and path trajectories directly from replay JSON logs.

### Key Rigor & Bug Fixes Applied During Static Review
1. **Strictly Zero-Sum Formulation Enforced:** Eliminated asymmetric attacker shaping and one-sided breakthrough bonuses. Implemented symmetric potential-based shaping ($+F$ to Attacker, $-F$ to Defender) so that $R_{\text{Attacker}} + R_{\text{Defender}} \equiv 0$ holds identically at every timestep, preserving zero-sum Markov game theoretical guarantees (Ng et al. 1999; Devlin & Kudenko 2011).
2. **Sampling vs. Executed Action Fix:** Ensured PPO training buffers store the actual executed action rather than pre-override samples.
3. **Numerical Stability:** Guarded PPO return normalization against single-sample batch zero-division NaNs.
4. **Coordinate Denormalization:** Fixed distance-shaping coordinate scaling to strictly use `env._norm_x` and `env._norm_y`.
5. **Dual Win-Counting Guard:** Ensured defender win is counted strictly when the attacker was intercepted *and* did not simultaneously reach the target.
6. **Phase 6 Cadence Correction:** Fixed PPO update interval from a mistaken 2,000-episode modulus to an episode cadence of 20, ensuring archived policies received proper training.
7. **Scientific Claim Calibration:** Replaced overconfident "proved Nash equilibrium / cured catastrophic forgetting" marketing language with hypothesis-framed empirical analysis aligned with multi-agent literature.

---

## 📂 3. Project Structure

```
Red_queen/
├── dashboard_3d.html          # WebGL / Three.js 3D Tactical Arena replay viewer
├── requirements.txt           # Core Python dependencies
├── README.md                  # This file
│
├── docs/
│   ├── phase1_game_theory_spec.md # Complete formal game-theoretic environment spec
│   ├── TO_DO_NEXT.md              # Roadmap and planned experiment extensions
│   ├── phase3_training_curve.png  # Baseline single-agent training plot
│   ├── phase4_marl_curve.png      # Naive self-play training plot (Red Queen)
│   └── phase6_archive_curve.png   # Double history archive training plot
│
├── models/
│   ├── phase4_attacker.pth        # Trained weights from Phase 4
│   ├── phase6_attacker.pth        # Trained weights from Phase 6
│   └── archive/                   # Serialized policy snapshots from archive milestones
│
├── replays/
│   ├── fixed_opponent.json        # Trajectory log: Phase 3
│   ├── red_queen.json             # Trajectory log: Phase 4
│   └── red_queen_history.json     # Trajectory log: Phase 6
│
└── src/
    ├── environment.py             # Three Bridges MARL Gym/PettingZoo environment
    ├── replay_logger.py           # JSON step & trajectory serialization
    ├── evaluate_robustness.py     # Deterministic paired-seed evaluation against archetypes
    ├── dashboard.py               # Streamlit analytics & 2D replay dashboard
    │
    ├── agents/
    │   ├── ppo.py                 # PPO implementation with Actor-Critic networks
    │   └── dqn.py                 # Deep Q-Network baseline agent
    │
    ├── train_phase3.py            # Baseline training: Attacker vs Scripted Defender
    ├── train_phase4.py            # Multi-agent self-play training (Red Queen)
    └── train_phase6.py            # Double History Archive multi-agent training
```

---

## 🚀 4. Quickstart & How to Run

### Installation
```bash
# Clone the repository
git clone https://github.com/Nava-deep-18/Red_queen.git
cd Red_queen

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Training
```bash
# Phase 3: Train Attacker against fixed camping defender
python src/train_phase3.py

# Phase 4: Run simultaneous Red Queen MARL self-play
python src/train_phase4.py

# Phase 6: Run Double History Archive MARL self-play
python src/train_phase6.py
```

### Running Evaluation & Dashboards
```bash
# 1. Run Robustness Benchmark (Phase 4 vs Phase 6 against 4 archetypes)
python src/evaluate_robustness.py

# 2. Launch the Streamlit Analytics Dashboard
streamlit run src/dashboard.py

# 3. View the 3D WebGL Arena
# Simply open dashboard_3d.html in any modern browser (or serve via local HTTP server)
```

---

## 🔮 5. Next Steps & Roadmap

See [`docs/TO_DO_NEXT.md`](file:///c:/Red_queen/docs/TO_DO_NEXT.md) for details:
- **Randomized Arena Spawns:** Add stochastic spawn coordinates to test spatial reasoning beyond memorized start states.
- **Exploitability & Best-Response Metrics:** Measure true exploitability ($L_1$ best-response gap) rather than raw win rate to formally test Nash equilibrium convergence.
- **Strategy Route Clustering:** Measure spatial diversity and entropy across bridge choices to quantify exploration breadth over training time.
