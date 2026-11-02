import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure local imports work whether run from repo root or src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM
from agents.ppo import PPOAgent


def train_single_configuration(
    k: int,
    num_episodes: int = 1500,
    update_every_episodes: int = 20,
    save_dir: str = "models/multi_target",
    results_dir: str = "results/multi_target",
    seed: int = 42,
):
    """
    Train paired PPO Attacker & Defender policies in MultiTargetArenaEnv with K targets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiTargetArenaEnv(num_targets=k)
    state_dim = STATE_DIM  # 19
    action_dim = env.action_space("attacker").n  # 5

    attacker = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    defender = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    att_win_rates = []
    def_win_rates = []
    timeout_rates = []
    recent_att_wins = []
    recent_def_wins = []
    recent_timeouts = []

    print(f"\n=======================================================")
    print(f"  Training Multi-Target PPO Arena: K={k} Target(s)")
    print(f"  Target Positions: {[tuple(t) for t in env.targets]}")
    print(f"  Episodes: {num_episodes} | Batch update: every {update_every_episodes} eps")
    print(f"=======================================================")

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode if seed is not None else None)
        state = obs["attacker"]
        done = False

        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            att_action = attacker.select_action(state, action_mask=att_mask)
            def_action = defender.select_action(state, action_mask=def_mask)

            old_att = env.attacker_pos.copy()
            old_def = env.defender_pos.copy()

            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Strictly Zero-Sum Potential Shaping
            att_shaping, def_shaping = env.compute_zero_sum_shaping(
                old_att, env.attacker_pos, old_def, env.defender_pos, gamma=0.99
            )
            att_reward = rewards[0] + att_shaping
            def_reward = rewards[1] + def_shaping

            attacker.store_reward(att_reward, done)
            defender.store_reward(def_reward, done)

            state = next_state

        # Update at episode boundaries
        if (episode + 1) % update_every_episodes == 0:
            attacker.update()
            defender.update()

        # Track rolling rates
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

        if (episode + 1) % 100 == 0 or (episode + 1) == num_episodes:
            print(
                f"[K={k}] Ep {episode+1:4d}/{num_episodes} | "
                f"Att Win: {att_win_rate:.2f} | Def Win: {def_win_rate:.2f} | Timeout: {timeout_rate:.2f}"
            )

    # Flush any remaining buffer transitions
    attacker.update()
    defender.update()

    # Save checkpoints
    os.makedirs(save_dir, exist_ok=True)
    att_path = os.path.join(save_dir, f"ppo_k{k}_attacker.pth")
    def_path = os.path.join(save_dir, f"ppo_k{k}_defender.pth")
    torch.save(attacker.policy.state_dict(), att_path)
    torch.save(defender.policy.state_dict(), def_path)
    print(f"Saved checkpoints:\n  Attacker: {att_path}\n  Defender: {def_path}")

    return {
        "k": k,
        "att_win_rates": att_win_rates,
        "def_win_rates": def_win_rates,
        "timeout_rates": timeout_rates,
    }


def plot_combined_learning_curves(history: dict, results_dir: str):
    """Generates comparative multi-panel training progression figure."""
    os.makedirs(results_dir, exist_ok=True)
    k_values = sorted(history.keys())
    fig, axes = plt.subplots(len(k_values), 1, figsize=(9.5, 2.8 * len(k_values)), sharex=True)
    if len(k_values) == 1:
        axes = [axes]

    colors = {"att": "#e63946", "def": "#1d3557", "time": "#f4a261"}

    for idx, k in enumerate(k_values):
        ax = axes[idx]
        data = history[k]
        ax.plot(data["att_win_rates"], label="Attacker Win Rate", color=colors["att"], linewidth=1.5)
        ax.plot(data["def_win_rates"], label="Defender Win Rate", color=colors["def"], linewidth=1.5)
        ax.plot(data["timeout_rates"], label="Timeout Rate", color=colors["time"], linestyle="--", linewidth=1.2)
        ax.axvspan(0, 100, color="gray", alpha=0.12, label="Warmup (N<100)" if idx == 0 else "")
        ax.set_ylabel(f"K={k} Rates", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_title("Multi-Target MARL Self-Play Learning Dynamics (Rolling 100 Episodes)", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8.5, ncol=4)

    axes[-1].set_xlabel("Training Episode", fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "learning_curves_multi_target.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved combined learning curve plot: {plot_path}")

    # Generate standalone plots for each K with full axis labels
    for k in k_values:
        fig_s, ax_s = plt.subplots(figsize=(8.5, 4.2))
        data = history[k]
        ax_s.plot(data["att_win_rates"], label="Attacker Win Rate", color=colors["att"], linewidth=1.6)
        ax_s.plot(data["def_win_rates"], label="Defender Win Rate", color=colors["def"], linewidth=1.6)
        ax_s.plot(data["timeout_rates"], label="Timeout Rate", color=colors["time"], linestyle="--", linewidth=1.3)
        ax_s.axvspan(0, 100, color="gray", alpha=0.12, label="Warmup Window (N<100)")
        ax_s.set_title(f"Self-Play Dynamics: K = {k} Target(s) (Rolling 100 Episodes)", fontsize=11, fontweight="bold")
        ax_s.set_xlabel("Training Episode", fontsize=10)
        ax_s.set_ylabel("Outcome Rate", fontsize=10)
        ax_s.set_ylim(-0.05, 1.05)
        ax_s.legend(loc="upper right", fontsize=9)
        ax_s.grid(True, alpha=0.3)
        plt.tight_layout()
        k_path = os.path.join(results_dir, f"learning_curve_k{k}.png")
        plt.savefig(k_path, dpi=120)
        plt.close()

    # Save raw training history json
    json_path = os.path.join(results_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved raw training history to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agents in Multi-Target Arena.")
    parser.add_argument("--targets", nargs="+", type=int, default=[1, 2, 3, 5],
                        help="Target count configurations to train (e.g. 1 2 3 5)")
    parser.add_argument("--episodes", type=int, default=1500,
                        help="Number of training episodes per target configuration")
    parser.add_argument("--update-every", type=int, default=20,
                        help="Number of episodes per PPO policy update")
    parser.add_argument("--save-dir", type=str, default="models/multi_target",
                        help="Directory to save trained model state dicts")
    parser.add_argument("--results-dir", type=str, default="results/multi_target",
                        help="Directory to save plots and training logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed")
    args = parser.parse_args()

    all_history = {}
    for k in args.targets:
        hist = train_single_configuration(
            k=k,
            num_episodes=args.episodes,
            update_every_episodes=args.update_every,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            seed=args.seed,
        )
        all_history[k] = hist

    plot_combined_learning_curves(all_history, args.results_dir)
    print("\nAll multi-target training runs completed successfully!")


if __name__ == "__main__":
    main()
