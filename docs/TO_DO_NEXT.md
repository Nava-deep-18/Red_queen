# Red Queen MARL: Future Work & Planned Extensions

This document outlines planned theoretical and experimental extensions for the Three Bridges MARL testbed.

---

## 1. Multi-Seed Training Aggregation
- **Current Status:** Models are evaluated across 150 paired random seeds per evaluation matchup from representative training runs.
- **Planned Extension:** Train 5 independent runs per architecture (with distinct random initialization seeds) to report mean training curves and standard error bounds across training runs, ensuring publication-grade statistical power.

## 2. Dynamic League Architectures & Prioritized Opponent Sampling
- **Current Status:** Phase 6 deploys a uniform historical snapshot reservoir $\mathcal{U}(\mathcal{H})$ with a 60/20/20 active-archive split.
- **Planned Extension:** Implement prioritized fictitious play or regret-weighted sampling to avoid allocating policy updates against obsolete early-training checkpoints.
- **Specialized Exploiters:** Introduce dedicated main agents and exploiters inspired by AlphaStar league training.

## 3. Stochastic Spawn & Dynamic Obstacle Topologies
- **Current Status:** Spawning occurs uniformly over $X \in [0, 3]$ for the Attacker and $X \in [6, 9]$ for the Defender.
- **Planned Extension:** Add moving obstacles, variable river widths, and asymmetric agent speed parameters to study spatial game theory in non-Euclidean topologies.

## 4. Full Population Double Oracle (PSRO)
- **Current Status:** Meta-game solved via Fictitious Play over a representative 6-defender suite.
- **Planned Extension:** Integrate Policy-Space Response Oracles (PSRO) to compute exact subgame Nash equilibria and empirical best-response distances in continuous policy space.

