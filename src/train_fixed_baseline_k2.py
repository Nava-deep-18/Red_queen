import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM
from agents.ppo import PPOAgent


def train_fixed_baseline_k2(num_episodes=500, update_every=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiTargetArenaEnv(num_targets=2)
    state_dim = STATE_DIM  # 19
    action_dim = 5

    attacker = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    att_win_rates = []
    def_win_rates = []
    timeout_rates = []
    recent_att_wins = []
    recent_def_wins = []
    recent_timeouts = []

    print("Training K=2 Fixed Baseline (Attacker vs Stationary Defender)...")
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = obs["attacker"]
        done = False

        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            att_action = attacker.select_action(state, action_mask=att_mask)
            def_action = 4  # Stationary camper (STAY)

            old_att = env.attacker_pos.copy()
            old_def = env.defender_pos.copy()

            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Zero-sum potential shaping
            att_shaping, _ = env.compute_zero_sum_shaping(
                old_att, env.attacker_pos, old_def, env.defender_pos, gamma=0.99
            )
            att_reward = rewards[0] + att_shaping
            attacker.store_reward(att_reward, done)

            state = next_state

        if (episode + 1) % update_every == 0:
            attacker.update()

        att_win = 1 if info["reached_target"] else 0
        def_win = 1 if info["caught"] and not info["reached_target"] else 0
        timed = 1 if info["timeout"] else 0

        recent_att_wins.append(att_win)
        recent_def_wins.append(def_win)
        recent_timeouts.append(timed)

        if len(recent_att_wins) > 50:
            recent_att_wins.pop(0)
            recent_def_wins.pop(0)
            recent_timeouts.pop(0)

        att_win_rate = sum(recent_att_wins) / len(recent_att_wins)
        def_win_rate = sum(recent_def_wins) / len(recent_def_wins)
        timeout_rate = sum(recent_timeouts) / len(recent_timeouts)

        att_win_rates.append(att_win_rate)
        def_win_rates.append(def_win_rate)
        timeout_rates.append(timeout_rate)

        if (episode + 1) % 100 == 0:
            print(f"Ep {episode+1:3d}/{num_episodes} | Att Win: {att_win_rate:.2f} | Def Win: {def_win_rate:.2f} | Timeout: {timeout_rate:.2f}")

    # Plot curve
    os.makedirs("results/multi_target_k2", exist_ok=True)
    plot_path = "results/multi_target_k2/fixed_baseline_k2_curve.png"

    plt.figure(figsize=(8.5, 4.4))
    plt.plot(att_win_rates, color="#e63946", linewidth=1.6, label="Attacker Wins (Target Reached)")
    plt.plot(def_win_rates, color="#1d3557", linewidth=1.5, label="Defender Wins (Accidental Intercept)")
    plt.plot(timeout_rates, color="#f4a261", linewidth=1.3, linestyle="--", label="Timeouts (Standoff)")
    plt.axvspan(0, 50, color="gray", alpha=0.15, label="Warmup Window (N < 50)")
    plt.title("Fixed Opponent Baseline: PPO Attacker vs Stationary Defender (K=2 Flank Arena)", fontsize=11, fontweight="bold")
    plt.xlabel("Training Episode", fontsize=10)
    plt.ylabel("Outcome Rate (Rolling 50 Episodes)", fontsize=10)
    plt.ylim(-0.02, 1.05)
    plt.legend(loc="center right", fontsize=8.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    train_fixed_baseline_k2()

