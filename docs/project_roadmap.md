# Red Queen AI Arena: Project Roadmap & Next Steps

This document outlines the planned upgrades and extensions for the Red Queen AI Arena based on our initial findings and design discussions.

## 1. Visual Replay Dashboard (Top Priority)
**Concept:** Win-rate graphs only tell half the story. We need to *see* the strategies evolving.
**Action Items:**
*   Modify the training scripts to record the exact (X, Y) coordinate paths of the Attacker and Defender at specific milestone epochs (e.g., Episode 100, 500, 1000, 2000).
*   Upgrade the Streamlit dashboard (`src/dashboard.py`) to include a visual grid.
*   Add a slider that allows the user to select an epoch and watch an animated playback of the game to visually confirm the strategies the agents learned.

## 2. Double Opponent Archives (True Phase 6)
**Concept:** Currently, only the Attacker practices against historical Defenders. For a complete theoretical model, both agents must use history to prevent catastrophic forgetting.
**Action Items:**
*   Update `src/train_phase6.py` so that the Defender *also* maintains an archive of past Attackers.
*   Train both agents against a mix of current and archived opponents simultaneously.

## 3. The True Robustness Test (Cross-Evaluation)
**Concept:** We need to mathematically prove Hypothesis 3 (that archive training creates better generalization than standard MARL).
**Action Items:**
*   Create a new evaluation script that pits the final Attacker from Phase 4 against a completely unseen Defender.
*   Pit the final Attacker from Phase 6 (Archive) against the same unseen Defender.
*   Compare their win rates to prove the Archive-trained agent is more robust.

## 4. Randomized Arena Spawns
**Concept:** Right now, the agents and target spawn in the exact same coordinates every time, which can lead to memorization rather than true spatial reasoning.
**Action Items:**
*   Update `ArenaEnv` to randomize the starting positions of the Attacker, Defender, and Target at the start of every episode (while ensuring they don't spawn on top of obstacles or each other).

## 5. Strategy Diversity Metric
**Concept:** The Red Queen effect should force agents to be creative and try multiple paths.
**Action Items:**
*   Write a clustering or path-tracking algorithm that measures how many *distinct* routes the Attacker takes to the target.
*   Graph this "Diversity Score" to prove that MARL and Archive training create wider behavioral diversity than training against a fixed opponent.
