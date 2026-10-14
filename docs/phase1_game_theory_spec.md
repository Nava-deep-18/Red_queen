# RED QUEEN AI ARENA: Game Theory & Arena Specification

## 1. Executive Summary & Objective
The **Red Queen AI Arena** is a zero-sum, multi-agent reinforcement learning environment designed to study non-stationary competitive dynamics, **Catastrophic Forgetting**, and convergence toward a **Nash Equilibrium**. 

The environment features an asymmetric, bottlenecked topology called the **Three Bridges Map**, structured specifically to create a spatial **Rock-Paper-Scissors Dilemma**.

---

## 2. Arena Geometry & Topology (The Three Bridges)

The environment consists of a discrete 2D grid of size **$11 \times 7$** ($X \in [0, 10]$, $Y \in [0, 6]$).

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

* **Left Basin ($X \in [0, 4]$):** Infiltration staging area where the Attacker spawns randomly ($X \in [0, 3]$, $Y \in [0, 6]$).
* **The River ($X = 5$):** A natural impassable boundary splitting the board into two separate territories.
* **4 Obstacle Pillars (🔒):** Located at $(5, 0)$, $(5, 2)$, $(5, 4)$, and $(5, 6)$. Agents cannot occupy or pass through these cells.
* **3 Bridges (🌉):** Open river crossing choke points located at:
  * 🌉 **North Bridge:** $(5, 1)$
  * 🌉 **Center Bridge:** $(5, 3)$
  * 🌉 **South Bridge:** $(5, 5)$
* **Right Basin ($X \in [6, 10]$):** Defensive territory where the Defender spawns ($X \in [6, 7]$, $Y \in [0, 6]$).
* **Citadel Target (🎯):** Fixed at $(10, 3)$ (far right, center).

---

## 3. Game Theoretic Structure: Why 3 Bridges?

The 3 Bridges architecture is intentionally designed as an unresolvable pure-strategy dilemma:

$$\text{North Bridge } (Y=1) \longleftrightarrow \text{Center Bridge } (Y=3) \longleftrightarrow \text{South Bridge } (Y=5)$$

1. **Why Not 1 Bridge?** Trivial. A single chokepoint allows the Defender to park in place, reducing the game to a static pathfinding obstacle.
2. **Why Not 2 Bridges?** A simple binary coin flip.
3. **The 3-Bridge Dilemma:**
   * The Defender occupies only **1 cell** and has equal step speed to the Attacker. It is mathematically impossible for one Defender to block all 3 bridges at the same time.
   * If the Defender camps North, the Attacker exploits South.
   * If the Defender shifts South, the Attacker breaches Center or North.
   * In naive self-play, this creates the **Red Queen Effect**: agents endlessly cycle through bridge preferences, perpetually forgetting past counter-strategies.
   * The only unexploitable resolution is a **Nash Equilibrium mixed strategy**: probabilistic routing and dynamic zone coverage.

---

## 4. Players & Objectives

* **Player 1 (Attacker 🔴):** Goal is to navigate across the river via one of the 3 bridges and secure the Citadel at $(10, 3)$.
* **Player 2 (Defender 🔵):** Goal is to intercept (occupy the same cell as) the Attacker before it reaches the Citadel, or delay it until timeout.

---

## 5. State Space & Observations

The state is fully observable and represented as a normalized continuous vector of dimension $6$:

$$S = \left[ \frac{a_x}{11}, \; \frac{a_y}{7}, \; \frac{d_x}{11}, \; \frac{d_y}{7}, \; \frac{t_x}{11}, \; \frac{t_y}{7} \right]$$

* $a_x, a_y$: Attacker current coordinates.
* $d_x, d_y$: Defender current coordinates.
* $t_x, t_y$: Citadel target coordinates $(10, 3)$.
* All features are normalized between $0.0$ and $1.0$ for neural network stability.

---

## 6. Action Space & Movement Dynamics

Both agents select discrete actions simultaneously from an action space of size 5:

| Action ID | Direction | Vector Offset $(\Delta X, \Delta Y)$ |
| :---: | :---: | :---: |
| `0` | UP | $(0, -1)$ |
| `1` | DOWN | $(0, 1)$ |
| `2` | LEFT | $(-1, 0)$ |
| `3` | RIGHT | $(1, 0)$ |
| `4` | STAY | $(0, 0)$ |

### Movement Physics & Validation:
$$\text{pos}_{\text{new}} = \text{pos}_{\text{current}} + \Delta \text{pos}$$

A move is accepted if and only if:
1. $0 \le X_{\text{new}} < 11$ and $0 \le Y_{\text{new}} < 7$ (within arena bounds).
2. $(X_{\text{new}}, Y_{\text{new}}) \notin \{(5,0), (5,2), (5,4), (5,6)\}$ (not into river obstacle pillars).

If an agent attempts an illegal move or walks into a river obstacle, the position remains unchanged ($\text{pos}_{\text{new}} = \text{pos}_{\text{current}}$).

---

## 7. Zero-Sum Payoff Structure

The environment is strictly zero-sum: $\text{Payoff}(A) + \text{Payoff}(D) = 0$.

| Terminal Outcome | Condition | Attacker Payoff | Defender Payoff |
| :--- | :--- | :---: | :---: |
| **Citadel Secured** | Attacker reaches $(10, 3)$ | **$+10.0$** | **$-10.0$** |
| **Attacker Intercepted** | Attacker pos $==$ Defender pos | **$-10.0$** | **$+10.0$** |
| **Timeout Draw** | $75$ steps elapsed | **$0.0$** | **$0.0$** |

### Symmetric Potential-Based Zero-Sum Reward Shaping:
To guide reinforcement learning without introducing strategic bias toward any specific bridge while strictly preserving zero-sum Markov game dynamics (Ng et al. 1999; Devlin & Kudenko 2011), the agents calculate a symmetric distance potential $D(x, y)$:

* **Before River ($X \le 4$):** Distance to the closest of the 3 bridges:
  $$D(x, y) = \min(|x - 4| + |y - b_y|) + 6 \quad \text{for } b_y \in \{1, 3, 5\}$$

* **After River ($X > 4$):** Direct distance to Citadel target $(10, 3)$:
  $$D(x, y) = |x - 10| + |y - 3|$$

* **Zero-Sum Step Shaping:**
  $$F(s, s') = (D_{\text{old}} - D_{\text{new}}) \times 0.5$$
  $$R_A = R_{\text{env}, A} + F(s, s')$$
  $$R_D = R_{\text{env}, D} - F(s, s')$$

$$\implies R_A + R_D = (R_{\text{env}, A} + F(s, s')) + (R_{\text{env}, D} - F(s, s')) = R_{\text{env}, A} + R_{\text{env}, D} = 0$$

Because $R_A + R_D \equiv 0$ holds identically at every transition, the game remains strictly zero-sum, and potential-based shaping preserves all Nash equilibria and optimal policies without distorting relative payoffs.

