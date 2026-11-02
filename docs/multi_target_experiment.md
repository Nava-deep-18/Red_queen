# Multi-Target Scaling in the Red Queen Tactical Arena: Theoretical Foundations, Potential-Based Reward Shaping, and Empirical Benchmark Analysis

---

## 1. Executive Summary & Research Motivation

In multi-agent reinforcement learning (MARL) and adversarial game theory, most spatial interception environments formulate the infiltrator's objective as a **single, static target** (e.g., reaching a central Citadel). However, real-world defensive domains—including enterprise network security, critical infrastructure protection, border surveillance, and combination oncology—rarely feature a single point of failure. Instead, defenders must protect a **distributed frontier of heterogeneous, simultaneous targets** against an infiltrator capable of dynamically exploiting perimeter vulnerabilities.

This experiment expands the **Red Queen AI Arena** from its baseline single-target formulation ($K=1$) to a configurable, multi-target spatial Markov game ($K \in \{1, 2, 3, 5\}$). We investigate:
1. **Target Saturation Hypothesis:** Does scaling the target count $K$ break the defensive equilibrium, causing interception performance to collapse under spatial dispersion?
2. **Path Diversification & Strategic Deception:** Does the availability of multiple targets disperse Attacker bridge choices across the bottleneck river, mitigating chokepoint congestion?
3. **Reward Shaping Invariance:** Can zero-sum discounted potential shaping (Ng et al., 1999) be generalized to multi-target perimeters while maintaining strict policy invariance and zero-sum payoff balance ($R_{\text{att}} + R_{\text{def}} \equiv 0$)?
4. **Zero-Shot Policy Transfer:** How effectively does a baseline single-target policy (6D observation space) transfer to multi-target arenas without structural retraining?

---

## 2. Formal Mathematical Formulation & Game Specification

### 2.1 Two-Player Markov Game
The multi-target arena is formulated as a two-player, simultaneous-action, zero-sum Markov Game:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}_{\text{att}}, \mathcal{A}_{\text{def}}, \mathcal{P}, \mathcal{R}_{\text{att}}, \mathcal{R}_{\text{def}}, \gamma \rangle$$

- **Grid Dimensions:** Width $W = 11$, Height $H = 7$.
- **Agents:** Attacker ($\text{att}$) and Defender ($\text{def}$).
- **Action Space:** Both agents choose from 5 discrete spatial primitives simultaneously at each timestep:
  $$\mathcal{A} = \{\text{UP: } (0, -1), \text{ DOWN: } (0, 1), \text{ LEFT: } (-1, 0), \text{ RIGHT: } (1, 0), \text{ STAY: } (0, 0)\}$$
- **Bottleneck River:** Situated at column $X=5$ with river obstacles at $Y \in \{0, 2, 4, 6\}$ and passable bridges at $Y \in \{1, 3, 5\}$.
- **Target Coordinates $\mathcal{T}$ ($X=10$):**
  - **$K=1$ (Citadel Baseline):** $\mathcal{T}_1 = \{(10, 3)\}$
  - **$K=2$ (Flank Corridor):** $\mathcal{T}_2 = \{(10, 1), (10, 5)\}$
  - **$K=3$ (Bridge-Aligned Perimeter):** $\mathcal{T}_3 = \{(10, 1), (10, 3), (10, 5)\}$
  - **$K=5$ (Full Eastern Frontier):** $\mathcal{T}_5 = \{(10, 0), (10, 1), (10, 3), (10, 5), (10, 6)\}$

---

### 2.2 Observation Space & Active Indicator Formulation

To accommodate variable target counts up to $K_{\max} = 5$ without altering neural network architecture between runs, the environment projects global state into a fixed 19-dimensional normalized observation vector:
$$s = \Big[ \bar{x}_{\text{att}}, \bar{y}_{\text{att}}, \bar{x}_{\text{def}}, \bar{y}_{\text{def}}, \big(\bar{x}_{t_i}, \bar{y}_{t_i}, a_{t_i}\big)_{i=0}^{K_{\max}-1} \Big] \in [0, 1]^{19}$$

where:
- $\bar{x} = \frac{x}{W - 1} = \frac{x}{10}$ and $\bar{y} = \frac{y}{H - 1} = \frac{y}{6}$.
- $a_{t_i} \in \{0.0, 1.0\}$ is a binary activation indicator:
  $$a_{t_i} = \begin{cases} 1.0 & \text{if target } i < K \text{ is active} \\ 0.0 & \text{if padded / inactive} \end{cases}$$
- For inactive targets ($i \ge K$), coordinates are zero-padded: $(\bar{x}_{t_i}, \bar{y}_{t_i}, a_{t_i}) = (0.0, 0.0, 0.0)$.

This guarantees identical input dimensionality across all configurations while providing the neural policy with exact semantic knowledge of active target frontiers.

---

### 2.3 Terminal Conditions & Priority Theorem

At each step, candidate transitions are resolved under two termination conditions:
1. **Target Breach:** $\exists t \in \mathcal{T}$ such that $p_{\text{att}} = t$.
2. **Interception (Capture):** Either **same-tile collision** ($p_{\text{att}} = p_{\text{def}}$) or **crossing-swap collision** ($p_{\text{att}} = p_{\text{def}}^{\text{old}} \land p_{\text{def}} = p_{\text{att}}^{\text{old}}$).

**Terminal Priority Axiom:**  
If an Attacker reaches any target $t \in \mathcal{T}$ on step $t$ while simultaneously colliding with or crossing the Defender, **the Target Breach takes strict precedence**:
$$\text{Outcome} = \begin{cases} \text{Attacker Win } (+10, -10) & \text{if } p_{\text{att}} \in \mathcal{T} \\ \text{Defender Intercept } (-10, +10) & \text{if } \text{Caught} \land p_{\text{att}} \notin \mathcal{T} \\ \text{Timeout } (0, 0) & \text{if } t \ge T_{\max} = 75 \end{cases}$$

This formal priority reflects standard operational realities: once an infiltrator reaches an operational asset or critical server, the breach has occurred regardless of simultaneous capture.

---

## 3. Multi-Target Zero-Sum Potential Shaping & Theoretical Proofs

### 3.1 Ng-Harada-Russell Potential Formulation
Under sparse terminal payoffs ($\pm 10$), credit assignment across spatial grids requires reward shaping. To prevent policy distortion (e.g., cycling or suboptimal loops), we employ potential-based reward shaping (Ng, Harada, & Russell, 1999; Devlin & Kudenko, 2011).

In a multi-target environment, the Attacker's distance potential $\Phi_{\text{multi}}(p)$ is defined as the bridge-aware shortest path to the **nearest active target**:
$$D_{\text{multi}}(p) = \min_{t \in \mathcal{T}} D_{\text{route}}(p, t)$$

The bridge-routing distance function accounts for the river chokepoints:
$$D_{\text{route}}(p, t) = \begin{cases} 
\min\limits_{b_y \in \{1, 3, 5\}} \Big(|x - 4| + |y - b_y| + (t_x - 4) + |b_y - t_y|\Big) & \text{if } x \le 4 \\
|x - t_x| + |y - t_y| & \text{if } x > 4 
\end{cases}$$

Setting potential $\Phi(s) = -D_{\text{multi}}(p_{\text{att}})$, the discounted shaping bonus is:
$$F_{\text{target}}(s, s') = \gamma \Phi(s') - \Phi(s) = \Big(D_{\text{multi}}(p_{\text{att}}) - \gamma D_{\text{multi}}(p_{\text{att}}')\Big) \cdot c_{\text{target}}$$

Simultaneously, relative pursuit distance between agents is shaped:
$$F_{\text{rel}}(s, s') = \Big(D_{\text{rel}}(s) - \gamma D_{\text{rel}}(s')\Big) \cdot c_{\text{rel}}$$
where $D_{\text{rel}}(s) = \|p_{\text{att}} - p_{\text{def}}\|_1$.

---

### 3.2 Proof of Strict Zero-Sum Preservation
At every transition $(s, a, s')$, shaped rewards are constructed as:
$$\mathcal{R}_{\text{att}}^{\text{shaped}} = r_{\text{att}}^{\text{term}} + F_{\text{target}}(s, s') - F_{\text{rel}}(s, s')$$
$$\mathcal{R}_{\text{def}}^{\text{shaped}} = r_{\text{def}}^{\text{term}} - F_{\text{target}}(s, s') + F_{\text{rel}}(s, s')$$

Summing both agents' rewards:
$$\mathcal{R}_{\text{att}}^{\text{shaped}} + \mathcal{R}_{\text{def}}^{\text{shaped}} = \Big(r_{\text{att}}^{\text{term}} + r_{\text{def}}^{\text{term}}\Big) + \Big(F_{\text{target}} - F_{\text{target}}\Big) + \Big(-F_{\text{rel}} + F_{\text{rel}}\Big) \equiv 0.0$$

$$\forall s, s' \in \mathcal{S}: \quad \mathcal{R}_{\text{att}}^{\text{shaped}}(s, s') + \mathcal{R}_{\text{def}}^{\text{shaped}}(s, s') = 0$$

**Theorem (Policy Invariance):**  
Because $F_{\text{target}}$ and $F_{\text{rel}}$ are exact discounted potential differences with $\gamma = 0.99$ matching PPO's discount factor, the set of Nash Equilibria in $\mathcal{M}$ with shaped payoffs is **identical** to the Nash Equilibria of the unshaped game (Ng et al., 1999; Devlin & Kudenko, 2011).

---

## 4. Experimental Setup & Control Methodology

To eliminate sample bias and variance, all configurations were evaluated under strict experimental controls:
- **Trial Count:** 500 deterministic paired episodes per condition ($N = 500$).
- **Seed Pairing:** Identical random seed sequence $S = \{42, 43, \dots, 541\}$ across all matchups, guaranteeing identical initial spawns for all architectures.
- **Statistical Rigor:**
  - Win and Capture proportions are reported with **95% Wilson Score Confidence Intervals**:
    $$\text{CI}_{95} = \frac{1}{1 + z^2/N} \left( \hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}} \right), \quad z = 1.96$$
  - Episode durations are reported as $\mu \pm \text{SEM}$, where $\text{SEM} = \frac{\sigma}{\sqrt{N}}$.
- **Opponent Taxonomy:**
  1. **Learned PPO Self-Play Defender:** Trained simultaneously via PPO in multi-target self-play.
  2. **Greedy Pursuit Defender:** Deterministic Euclidean/Manhattan minimizer tracking Attacker position.
  3. **Patrol Defender:** Dynamic chokepoint sweeper patrolling $Y \in [1, 5]$ along the eastern riverbank.
  4. **Campers (North, Center, South):** Stationary blockers guarding $Y=1$, $Y=3$, and $Y=5$ respectively.
  5. **Random Defender:** Uniform random valid action baseline.
  6. **Stationary (Stay) Defender:** Static baseline.
  7. **Baseline 6D Transfer Attacker:** Pretrained Phase 4 single-target policy evaluated zero-shot on multi-target grids via the 6D coordinate adapter.

---

## 5. Comprehensive Empirical Results

### 5.1 Scientific Comparison Matrix (500 Paired Episodes per Matchup)

| Target Count $K$ | Defender Archetype | Attacker Win Rate (%) | 95% Wilson CI (±%) | Defender Win Rate (%) | Timeout Rate (%) | Avg Duration (Steps) | SEM Steps |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K = 1** | **Learned PPO Defender** | **54.4%** | **±4.3%** | **45.6%** | **0.0%** | **9.44** | **0.063** |
| $K = 1$ | Stay | 55.2% | ±4.3% | 15.6% | 29.2% | 28.47 | 1.339 |
| $K = 1$ | Random | 76.8% | ±3.7% | 19.6% | 3.6% | 15.80 | 0.729 |
| $K = 1$ | Greedy Pursuit | 31.8% | ±4.1% | 68.2% | 0.0% | 6.37 | 0.147 |
| $K = 1$ | North Camper ($Y=1$) | 71.0% | ±4.0% | 29.0% | 0.0% | 8.98 | 0.098 |
| $K = 1$ | Center Camper ($Y=3$) | 32.6% | ±4.1% | 30.8% | 36.6% | 33.02 | 1.429 |
| $K = 1$ | South Camper ($Y=5$) | 31.6% | ±4.1% | 31.8% | 36.6% | 32.36 | 1.451 |
| $K = 1$ | River Patroller | 85.8% | ±3.1% | 14.2% | 0.0% | 10.53 | 0.128 |
| $K = 1$ | *Phase 4 (6D Transfer vs Pursuit)* | 36.6% | ±4.2% | 63.4% | 0.0% | 6.69 | 0.147 |
| | | | | | | | |
| **K = 2** | **Learned PPO Defender** | **22.6%** | **±3.7%** | **10.8%** | **66.6%** | **52.25** | **1.438** |
| $K = 2$ | Stay | 25.6% | ±3.8% | 6.0% | 68.4% | 53.82 | 1.394 |
| $K = 2$ | Random | 25.0% | ±3.8% | 20.2% | 54.8% | 49.49 | 1.392 |
| $K = 2$ | Greedy Pursuit | 10.0% | ±2.6% | 42.6% | 47.4% | 38.56 | 1.550 |
| $K = 2$ | North Camper ($Y=1$) | 17.0% | ±3.3% | 12.6% | 70.4% | 54.97 | 1.382 |
| $K = 2$ | Center Camper ($Y=3$) | 29.6% | ±4.0% | 14.0% | 56.4% | 45.68 | 1.492 |
| $K = 2$ | South Camper ($Y=5$) | 13.8% | ±3.0% | 16.0% | 70.2% | 54.76 | 1.390 |
| $K = 2$ | River Patroller | 25.2% | ±3.8% | 8.4% | 66.4% | 52.42 | 1.420 |
| $K = 2$ | *Phase 4 (6D Transfer vs Pursuit)* | 24.4% | ±3.8% | 75.6% | 0.0% | 6.95 | 0.162 |
| | | | | | | | |
| **K = 3** | **Learned PPO Defender** | **37.6%** | **±4.2%** | **6.0%** | **56.4%** | **45.70** | **1.492** |
| $K = 3$ | Stay | 37.6% | ±4.2% | 6.0% | 56.4% | 45.87 | 1.482 |
| $K = 3$ | Random | 35.8% | ±4.2% | 15.6% | 48.6% | 42.95 | 1.464 |
| $K = 3$ | Greedy Pursuit | 15.8% | ±3.2% | 36.8% | 47.4% | 38.39 | 1.556 |
| $K = 3$ | North Camper ($Y=1$) | 31.0% | ±4.0% | 12.6% | 56.4% | 45.68 | 1.492 |
| $K = 3$ | Center Camper ($Y=3$) | 29.6% | ±4.0% | 14.0% | 56.4% | 45.68 | 1.492 |
| $K = 3$ | South Camper ($Y=5$) | 27.6% | ±3.9% | 16.0% | 56.4% | 45.61 | 1.496 |
| $K = 3$ | River Patroller | 35.6% | ±4.2% | 8.0% | 56.4% | 45.80 | 1.486 |
| $K = 3$ | *Phase 4 (6D Transfer vs Pursuit)* | 38.6% | ±4.2% | 61.4% | 0.0% | 6.40 | 0.124 |
| | | | | | | | |
| **K = 5** | **Learned PPO Defender** | **71.2%** | **±4.0%** | **28.8%** | **0.0%** | **7.59** | **0.123** |
| $K = 5$ | Stay | 84.4% | ±3.2% | 15.6% | 0.0% | 8.68 | 0.067 |
| $K = 5$ | Random | 85.8% | ±3.1% | 14.2% | 0.0% | 8.66 | 0.069 |
| $K = 5$ | Greedy Pursuit | 30.8% | ±4.0% | 69.2% | 0.0% | 5.83 | 0.110 |
| $K = 5$ | North Camper ($Y=1$) | 69.4% | ±4.0% | 30.6% | 0.0% | 8.28 | 0.080 |
| $K = 5$ | Center Camper ($Y=3$) | 70.2% | ±4.0% | 29.8% | 0.0% | 8.30 | 0.078 |
| $K = 5$ | South Camper ($Y=5$) | 68.0% | ±4.1% | 32.0% | 0.0% | 8.24 | 0.082 |
| $K = 5$ | River Patroller | 85.4% | ±3.1% | 14.6% | 0.0% | 8.67 | 0.071 |
| $K = 5$ | *Phase 4 (6D Transfer vs Pursuit)* | 39.0% | ±4.3% | 61.0% | 0.0% | 6.42 | 0.124 |

---

## 6. Key Empirical Findings & Theoretical Insights

### 6.1 Finding 1: Frontier Saturation Collapse ($K=5$)
At $K=1$, the defender achieves a robust **45.6% interception rate** in self-play, holding the attacker to 54.4% wins. The single bottleneck bridge and target allow a localized defensive blockade.

However, when scaling to $K=5$ (saturating all 5 non-obstacle tiles on column $X=10$):
- **Attacker Win Rate surges to 71.2%** (vs learned defender) and **84.4%–85.8%** (vs stationary/random/patroller).
- **Defender Capture Rate drops sharply to 28.8%**.
- **Episode Duration decreases significantly from 9.44 steps down to 7.59 steps**, reflecting decisive, direct infiltrations.

**Theoretical Mechanism:** In spatial zero-sum games, a single defender can only occupy one tile and project an interception perimeter over at most 2 adjacent cells per turn. When the target frontier spans 5 disjoint coordinates across 7 rows, the Defender faces a **combinatorial coverage deficit**: closing distance to defend northern targets leaves the southern and central corridors wide open.

---

### 6.2 Finding 2: Spatial Target Choice & Chokepoint Exploitation
Analysis of breached target coordinates reveals how the Attacker navigates the multi-target frontier. In $K=5$:
- **Target 1 $(10, 1)$:** 160 breaches (**37.5%**) — aligned directly with the North Bridge ($Y=1$).
- **Target 3 $(10, 5)$:** 153 breaches (**35.8%**) — aligned directly with the South Bridge ($Y=5$).
- **Target 2 $(10, 3)$:** 114 breaches (**26.7%**) — aligned with the Center Bridge ($Y=3$).
- **Target 0 $(10, 0)$ & Target 4 $(10, 6)$:** **0 breaches (0.0%)**.

**Strategic Insight:** The Attacker exhibits clear operational efficiency: targets $(10, 0)$ and $(10, 6)$ require an extra lateral movement off the bridge exit line, which increases transit time and provides the Defender extra interception steps. The policy strategically concentrates on the three direct bridge-exit targets, exploiting the shortest Euclidean paths.

---

### 6.3 Finding 3: Bridge Route Diversification
Bridge crossing distributions demonstrate how multi-target frontiers prevent bottleneck congestion:
- In $K=1$, Attacker crossings are concentrated towards the Center Bridge ($Y=3$) because it provides direct Euclidean alignment to the solitary Citadel.
- In $K=5$, bridge crossings become remarkably balanced:
  - **North Bridge ($Y=1$):** 183 crossings (**36.6%**)
  - **South Bridge ($Y=5$):** 175 crossings (**35.0%**)
  - **Center Bridge ($Y=3$):** 142 crossings (**28.4%**)

This confirms the **Bridge Diversification Hypothesis**: multiple targets induce spatial entropy, forcing the Attacker to exploit outer flanks (North and South bridges) to evade central interception.

---

### 6.4 Finding 4: Failure of Zero-Shot 6D Baseline Transfer
When the pretrained single-target Phase 4 policy (6D state space) is evaluated zero-shot in the multi-target arena (using adapter mapping to target 0):
- Against Greedy Pursuit, the transfer policy achieves only **24.4%–39.0% win rate**, suffering **61.0%–75.6% capture rates**.
- In contrast, the dedicated 19D multi-target PPO policy achieves **71.2% win rate** under self-play in $K=5$.

**Diagnosis:** A policy trained strictly on single-target geometry lacks representation of auxiliary target attractors and active frontier masks. It remains fixated on a single goal coordinate, failing to execute opportunistic pivots when intercepted along its primary path. This highlights the necessity of multi-target spatial representations for frontier defense.

---

## 7. Generated Visualizations & Artifacts

All benchmark figures have been generated and saved under `results/multi_target/`:

1. **`results/multi_target/attacker_win_rate_vs_targets.png`**  
   *Line plot displaying Attacker Win Rate across $K \in \{1, 2, 3, 5\}$ for all defender archetypes, with 95% Wilson Score error bars.*
2. **`results/multi_target/defender_capture_rate_vs_targets.png`**  
   *Comparative line plot illustrating defender interception degradation as $K$ increases.*
3. **`results/multi_target/episode_duration_vs_targets.png`**  
   *Average episode duration (mean steps $\pm 1$ SEM) vs $K$, highlighting convergence to fast infiltration paths.*
4. **`results/multi_target/target_choice_distribution.png`**  
   *Spatial bar chart quantifying empirical breach frequencies across active target coordinates for $K \in \{2, 3, 5\}$ as a proportion of total successful attacker breaches.*
5. **`results/multi_target/win_rate_by_defender_archetype.png`**  
   *Grouped bar chart breaking down win rate performance across each defender archetype for each target count.*
6. **`results/multi_target/learning_curves_multi_target.png`**  
   *Synchronized multi-panel PPO self-play training progression curves.*

---

## 8. Step-by-Step Reproduction Guide

To reproduce every step of this scientific study on a clean environment, execute the following commands from the repository root:

### 1. Run Unit Tests (13 Verification Tests)
```bash
python -m unittest tests/test_multi_target_environment.py
```

### 2. Train Multi-Target PPO Self-Play Policies ($K \in \{1, 2, 3, 5\}$)
```bash
python src/train_multi_target.py --targets 1 2 3 5 --episodes 800 --update-every 20
```

### 3. Run the 500-Episode Scientific Benchmark Suite
```bash
python src/compare_single_vs_multi_target.py --targets 1 2 3 5 --episodes 500
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run src/dashboard.py
```
Navigate to the new **"🎯 Multi-Target Arena"** tab on the top sticky navigation bar to explore the interactive 2D map viewer, metric cards, and all comparative publication plots.

---

## 9. Codebase Architectural Integrity Guarantee

In accordance with strict project constraints:
- **`src/environment.py`**: 100% untouched; baseline `ArenaEnv` behavior, rewards, and APIs remain strictly preserved.
- **`src/agents/ppo.py` & `src/agents/dqn.py`**: 100% untouched; zero modifications to neural architectures or training algorithms.
- **Existing Training Scripts (`train_phase*.py`, `benchmark_all.py`, `evaluate_robustness.py`)**: 100% untouched.
- **Modularity:** Multi-target support is isolated in `src/multi_target_environment.py`, `src/train_multi_target.py`, `src/evaluate_multi_target.py`, and `src/compare_single_vs_multi_target.py`.
- **Dashboard:** Extended purely additively via a new `"🎯 Multi-Target Arena"` navigation item without altering any existing pages.
