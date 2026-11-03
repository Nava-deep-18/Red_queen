import os
import sys
import copy
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM
from agents.ppo import PPOAgent


def train_dual_archive_k2(
    num_episodes: int = 600,
    warmup_episodes: int = 100,
    archive_interval: int = 50,
    update_every_episodes: int = 20,
    seed: int = 42,
    save_dir: str = "models/multi_target_k2",
    results_dir: str = "results/multi_target_k2",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("models/archive_k2", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    env = MultiTargetArenaEnv(num_targets=2)
    state_dim = STATE_DIM  # 19
    action_dim = 5

    attacker = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    defender = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    attacker_archive = []
    defender_archive = []

    att_win_rates = []
    def_win_rates = []
    timeout_rates = []
    recent_att_wins = []
    recent_def_wins = []
    recent_timeouts = []

    print("=" * 70)
    print("  Starting Dual-Archive Co-Evolution Training (K=2 Dual-Flank Arena)")
    print(f"  Episodes: {num_episodes} | Warmup: {warmup_episodes} | Snapshot every: {archive_interval}")
    print("  Historical Opponent Regularization: 60% Self-Play, 20% Past Def, 20% Past Att")
    print("=" * 70)

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode if seed is not None else None)
        state = obs["attacker"]
        done = False

        mode = random.random()
        train_attacker = True
        train_defender = True
        active_attacker = attacker
        active_defender = defender

        if len(defender_archive) > 0 and mode < 0.20:
            active_defender = PPOAgent(state_dim, action_dim)
            active_defender.policy.load_state_dict(random.choice(defender_archive))
            active_defender.policy_old.load_state_dict(active_defender.policy.state_dict())
            train_defender = False
        elif len(attacker_archive) > 0 and mode < 0.40:
            active_attacker = PPOAgent(state_dim, action_dim)
            active_attacker.policy.load_state_dict(random.choice(attacker_archive))
            active_attacker.policy_old.load_state_dict(active_attacker.policy.state_dict())
            train_attacker = False

        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            if train_attacker:
                att_action = attacker.select_action(state, action_mask=att_mask)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state)
                    att_action, _ = active_attacker.policy_old.act(state_t, action_mask=att_mask)
                    att_action = att_action.item()

            if train_defender:
                def_action = defender.select_action(state, action_mask=def_mask)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state)
                    def_action, _ = active_defender.policy_old.act(state_t, action_mask=def_mask)
                    def_action = def_action.item()

            old_att = env.attacker_pos.copy()
            old_def = env.defender_pos.copy()

            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Zero-sum potential shaping
            att_shaping, def_shaping = env.compute_zero_sum_shaping(
                old_att, env.attacker_pos, old_def, env.defender_pos, gamma=0.99
            )
            att_reward = rewards[0] + att_shaping
            def_reward = rewards[1] + def_shaping

            if train_attacker:
                attacker.store_reward(att_reward, done)
            if train_defender:
                defender.store_reward(def_reward, done)

            state = next_state

        if train_attacker and (episode + 1) % update_every_episodes == 0:
            attacker.update()
        if train_defender and (episode + 1) % update_every_episodes == 0:
            defender.update()

        # Archive checkpointing
        if (episode + 1) >= warmup_episodes and (episode + 1) % archive_interval == 0:
            att_w = copy.deepcopy(attacker.policy.state_dict())
            def_w = copy.deepcopy(defender.policy.state_dict())
            attacker_archive.append(att_w)
            defender_archive.append(def_w)
            torch.save(att_w, f"models/archive_k2/attacker_ep{episode + 1}.pth")
            torch.save(def_w, f"models/archive_k2/defender_ep{episode + 1}.pth")

        att_win = 1 if info["reached_target"] else 0
        def_win = 1 if info["caught"] and not info["reached_target"] else 0
        timed = 1 if info["timeout"] else 0

        recent_att_wins.append(att_win)
        recent_def_wins.append(def_win)
        recent_timeouts.append(timed)

        if len(recent_att_wins) > 100:
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
            print(f"Episode {episode+1:4d}/{num_episodes} | Att Win: {att_win_rate:.2f} | Def Win: {def_win_rate:.2f} | Timeout: {timeout_rate:.2f} | Archive Size: {len(defender_archive)}")

    attacker.update()
    defender.update()

    # Save trained checkpoints
    att_save_path = os.path.join(save_dir, "ppo_k2_dual_archive_attacker.pth")
    def_save_path = os.path.join(save_dir, "ppo_k2_dual_archive_defender.pth")
    torch.save(attacker.policy.state_dict(), att_save_path)
    torch.save(defender.policy.state_dict(), def_save_path)
    # Also update active multi-target models
    torch.save(attacker.policy.state_dict(), "models/multi_target/ppo_k2_attacker.pth")
    torch.save(defender.policy.state_dict(), "models/multi_target/ppo_k2_defender.pth")
    print(f"Saved Dual-Archive K=2 models to {att_save_path} and {def_save_path}")

    # Plot synchronized Dual-Archive Learning Curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True, facecolor="#090d16")
    ax1.set_facecolor("#0d131f")
    ax2.set_facecolor("#0d131f")

    for ax in (ax1, ax2):
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color=(1, 1, 1, 0.08), linestyle="--")

    ax1.plot(att_win_rates, label="Attacker Win Rate (Target Breached)", color="#ef4444", linewidth=1.6)
    ax1.plot(def_win_rates, label="Defender Win Rate (Intercept)", color="#38bdf8", linewidth=1.6)
    ax1.axvspan(0, warmup_episodes, color="gray", alpha=0.15, label=f"Warmup Window (N < {warmup_episodes})")
    ax1.set_title("Dual-Archive Co-Evolution Training Dynamics (K=2 Dual-Flank Perimeter)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Win Rate (Rolling 100)", color="#94a3b8", fontsize=9.5)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)

    ax2.plot(timeout_rates, label="Timeout Rate (Standoff Cycles)", color="#f59e0b", linewidth=1.4, linestyle="--")
    ax2.axvspan(0, warmup_episodes, color="gray", alpha=0.15)
    ax2.set_title("Timeout Rate (Synchronous Closure: Attacker + Defender + Timeout = 100%)", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    ax2.set_ylabel("Timeout Fraction", color="#94a3b8", fontsize=9.5)
    ax2.set_xlabel("Training Episode", color="#94a3b8", fontsize=9.5)
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)

    plt.tight_layout()
    curve_path = os.path.join(results_dir, "dual_archive_k2_curve.png")
    plt.savefig(curve_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved Dual-Archive K=2 curve to {curve_path}")


if __name__ == "__main__":
    train_dual_archive_k2()

