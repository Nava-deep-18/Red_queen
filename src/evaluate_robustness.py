import torch
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from environment import ArenaEnv
from agents.ppo import PPOAgent


def evaluate_archetype(attacker_path: str, archetype: str, seeds: list[int]) -> tuple[float, float]:
    """
    Evaluate a trained attacker against a fixed hand-coded defender archetype.

    Evaluation is deterministic: the attacker uses greedy action selection
    (agent.predict) rather than stochastic sampling.  Episodes are run with
    the provided seeds so that both Phase 4 and Phase 6 models face identical
    starting positions for a fair, paired comparison.

    Parameters
    ----------
    attacker_path : str
        Path to the saved attacker model state dict.
    archetype : str
        Name of the fixed hand-coded defender policy to evaluate against.
    seeds : list[int]
        Random seeds; one episode is played per seed.

    Returns
    -------
    win_rate : float   Average win rate across all seeds.
    ci_95    : float   Half-width of the 95 % Wilson-score-ish CI (approx).

    Note on validity: the four archetype policies below are fixed, deterministic,
    hand-coded rule-based defenders.  They are not trained adaptive opponents and
    they were not held out from training, so the results measure generalisation to
    specific scripted behaviours rather than to unseen learned strategies.
    To support a Nash-equilibrium claim, exploitability against a full population
    of defenders would need to be measured; win-rate charts against four scripted
    archetypes are insufficient for that claim.
    """
    env = ArenaEnv()
    state_dim = 6
    action_dim = env.action_space("attacker").n

    attacker = PPOAgent(state_dim, action_dim)
    if os.path.exists(attacker_path):
        attacker.policy.load_state_dict(torch.load(attacker_path, weights_only=True))
        attacker.policy_old.load_state_dict(torch.load(attacker_path, weights_only=True))
    else:
        print(f"Model not found: {attacker_path}")
        return 0.0, 0.0

    wins = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        state = obs["attacker"]
        done = False
        while not done:
            # Deterministic greedy action with obstacle collision masking
            mask = [
                env._is_valid_pos(env.attacker_pos + env.action_to_move[a])
                for a in range(5)
            ]
            att_action = attacker.predict(state, action_mask=mask)

            dx, dy = env.defender_pos
            ax, ay = env.attacker_pos

            # --- Fixed rule-based archetype defenders ---
            # These are scripted, hand-coded policies that move toward a fixed
            # target row and then stay (campers) or patrol (patroller).
            # They are NOT trained or adaptive opponents.
            if archetype == "NORTH_CAMPER":
                if   dy > 1: def_action = 0   # Move Up
                elif dy < 1: def_action = 1   # Move Down
                else:        def_action = 4   # Stay at Y=1
            elif archetype == "CENTER_CAMPER":
                if   dy > 3: def_action = 0
                elif dy < 3: def_action = 1
                else:        def_action = 4   # Stay at Y=3
            elif archetype == "SOUTH_CAMPER":
                if   dy > 5: def_action = 0
                elif dy < 5: def_action = 1
                else:        def_action = 4   # Stay at Y=5
            elif archetype == "PATROLLER":
                # Patrols between North and South bridges along riverbank
                if   dy <= 1: env.patrol_dir = 1
                elif dy >= 5: env.patrol_dir = -1
                def_action = 1 if env.patrol_dir == 1 else 0
            else:
                def_action = 4

            state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

        if info.get("reached_target", False):
            wins += 1

    n = len(seeds)
    win_rate = wins / n
    # Approximate 95% confidence interval (normal approximation to binomial)
    ci_95 = 1.96 * math.sqrt(win_rate * (1 - win_rate) / n) if n > 0 else 0.0
    return win_rate, ci_95


def main():
    archetypes = ["NORTH_CAMPER", "CENTER_CAMPER", "SOUTH_CAMPER", "PATROLLER"]
    # Fix #9: both models share the same seeds → paired comparison
    num_eval_games = 100
    seeds = list(range(num_eval_games))

    p4_scores, p4_cis = [], []
    p6_scores, p6_cis = [], []

    print("Evaluating Phase 4 (Red Queen Self-Play)...")
    print("  (Greedy actions, paired seeds, 95% CI reported)")
    for arch in archetypes:
        score, ci = evaluate_archetype("models/phase4_attacker.pth", arch, seeds)
        p4_scores.append(score)
        p4_cis.append(ci)
        print(f"  P4 vs {arch}: {score*100:.1f}% ± {ci*100:.1f}%")

    print("Evaluating Phase 6 (Double History Archive)...")
    for arch in archetypes:
        score, ci = evaluate_archetype("models/phase6_attacker.pth", arch, seeds)
        p6_scores.append(score)
        p6_cis.append(ci)
        print(f"  P6 vs {arch}: {score*100:.1f}% ± {ci*100:.1f}%")

    p4_avg = sum(p4_scores) / len(p4_scores) if p4_scores else 0
    p6_avg = sum(p6_scores) / len(p6_scores) if p6_scores else 0

    print(f"\nFinal Averages -> Phase 4: {p4_avg*100:.1f}% | Phase 6: {p6_avg*100:.1f}%")
    print("\nCaveat: opponents are fixed, hand-coded rule-based defenders.")
    print("  Results measure robustness to scripted behaviours, not to unseen trained policies.")

    # Generate Research-Grade Grouped Bar Chart (compact 9.5x4.2 @ 100 dpi)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [2.3, 1]})

    x = np.arange(len(archetypes))
    width = 0.35

    rects1 = ax1.bar(x - width/2, [s*100 for s in p4_scores], width,
                     label="Red Queen Self-Play", color="#e74c3c",
                     yerr=[c*100 for c in p4_cis], capsize=3)
    rects2 = ax1.bar(x + width/2, [s*100 for s in p6_scores], width,
                     label="Double History Archive", color="#2ecc71",
                     yerr=[c*100 for c in p6_cis], capsize=3)

    ax1.set_title(f"Robustness Test: Win Rates vs Fixed Archetype Defenders (n={num_eval_games})",
                  fontsize=10, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["North Camper\n(Y=1)", "Center Camper\n(Y=3)", "South Camper\n(Y=5)", "River Patroller\n(Dynamic)"],
        fontsize=9
    )
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", fontsize=8.5)

    for bar in rects1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#c0392b")
    for bar in rects2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#27ae60")

    # Overall summary bar chart on the right
    avg_bars = ax2.bar(["Red Queen", "Double Archive"], [p4_avg*100, p6_avg*100],
                       color=["#e74c3c", "#2ecc71"], width=0.55)
    ax2.set_title("Overall Robustness\n(4 Archetypes)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Average Win Rate (%)", fontsize=10)
    ax2.set_ylim(0, 115)

    for bar in avg_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/robustness_chart.png", dpi=100)
    print("Saved compact robustness chart to docs/robustness_chart.png")


if __name__ == "__main__":
    main()
