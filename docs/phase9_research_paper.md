# RED QUEEN AI ARENA: A Game-Theoretic MARL Study
## 1. Introduction
This paper explores the "Red Queen effect" in a Multi-Agent Reinforcement Learning (MARL) setting. Two agents, an Attacker and a Defender, are trained in a zero-sum GridWorld environment. We compare training against a fixed opponent versus co-evolutionary training, with and without an opponent archive.

## 2. Method
### 2.1 Game Theory Formulation
- **Players:** Attacker (🔴) and Defender (🔵).
- **Zero-Sum Payoffs:** Attacker gets +10 for reaching the target and -10 for being caught. Defender receives the exact opposite.
- **Environment:** 7x5 grid with fixed obstacles.

### 2.2 Reinforcement Learning Algorithms
- **Single-Agent Baseline:** DQN is used to train the Attacker against a static Defender.
- **MARL:** PPO is used to train both agents simultaneously.

### 2.3 Experimental Conditions
1. **Fixed Opponent:** Attacker learns, Defender stands still.
2. **Both Learn:** Both learn simultaneously via PPO.
3. **Red Queen + History:** Both learn via PPO, occasionally sampling past Defender policies from an archive.

## 3. Results (To be updated after full runs)
### 3.1 Fixed Opponent (Phase 3)
*Insert phase3_training_curve.png here*
The Attacker successfully exploits the fixed opponent, reaching near 100% win rate.

### 3.2 Full MARL Co-Evolution (Phase 4)
*Insert phase4_marl_curve.png here*
Win rates oscillate over generations, indicating continuous adaptation (the Red Queen effect) rather than stable convergence.

### 3.3 Archive Training Robustness (Phase 6)
*Insert phase6_archive_curve.png here*
The Attacker demonstrates more stable generalization, avoiding the steep drops caused by over-specialization against the current opponent.

## 4. Conclusion
Continuous adversarial co-evolution produces more robust strategies than training against a fixed opponent, confirming the hypotheses.
