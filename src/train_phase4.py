from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def main():
    env = ArenaEnv()

    state_dim = 6
    action_dim = env.action_space("attacker").n

    # Initialize both agents with tuned learning rates
    attacker = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    defender = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    # Initialize logger
    replay_logger = ReplayLogger("replays/red_queen.json")

    # Fix #3: update at episode boundaries.
    # update_every_episodes controls how many episodes are batched per PPO step.
    update_every_episodes = 20
    num_episodes = 1500

    att_win_rates = []
    def_win_rates = []
    timeout_rates = []
    recent_att_wins = []
    recent_def_wins = []
    recent_timeouts = []

    print("Starting Phase 4 Training: Full MARL with PPO (Both Learn)")
    print("  payoffs: Strictly zero-sum with relative pursuit & advance potential.")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs["attacker"]
        done = False

        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            # Both agents select valid actions
            att_action = attacker.select_action(state, action_mask=att_mask)
            def_action = defender.select_action(state, action_mask=def_mask)

            old_att = env.attacker_pos.copy()
            old_def = env.defender_pos.copy()

            # Step in environment
            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Strictly Zero-Sum Potential-Based Shaping with Relative Pursuit & Advance
            # Guarantees R_att + R_def == 0 at every timestep (Ng et al. 1999; Devlin & Kudenko 2011)
            att_shaping, def_shaping = env.compute_zero_sum_shaping(
                old_att, env.attacker_pos, old_def, env.defender_pos
            )
            att_reward = rewards[0] + att_shaping
            def_reward = rewards[1] + def_shaping

            attacker.store_reward(att_reward, done)
            defender.store_reward(def_reward, done)

            state = next_state

        # Update at episode boundaries (complete trajectories only)
        if (episode + 1) % update_every_episodes == 0:
            attacker.update()
            defender.update()

        # Track mutually-exclusive outcomes synchronously across all episodes
        att_win = 1 if info["reached_target"] else 0
        def_win = 1 if info["caught"] and not info["reached_target"] else 0
        timed   = 1 if info["timeout"] else 0

        recent_att_wins.append(att_win)
        recent_def_wins.append(def_win)
        recent_timeouts.append(timed)

        if len(recent_att_wins) > 100:
            recent_att_wins.pop(0)
            recent_def_wins.pop(0)
            recent_timeouts.pop(0)

        # Exact synchronous rolling rates: att_win_rate + def_win_rate + timeout_rate == 1.00
        att_win_rate = sum(recent_att_wins) / len(recent_att_wins)
        def_win_rate = sum(recent_def_wins) / len(recent_def_wins)
        timeout_rate = sum(recent_timeouts) / len(recent_timeouts)

        att_win_rates.append(att_win_rate)
        def_win_rates.append(def_win_rate)
        timeout_rates.append(timeout_rate)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Att WR: {att_win_rate:.2f} | Def WR: {def_win_rate:.2f} | Timeout: {timeout_rate:.2f}")

    # Flush any remaining partial batches at end of training
    attacker.update()
    defender.update()

    # Save BOTH trained models
    os.makedirs("models", exist_ok=True)
    torch.save(attacker.policy.state_dict(), "models/phase4_attacker.pth")
    torch.save(defender.policy.state_dict(), "models/phase4_defender.pth")
    print("Saved trained models to models/phase4_attacker.pth and models/phase4_defender.pth")

    # Record clean deterministic evaluation episodes for visual dashboard
    print("Recording clean deterministic evaluation replay games...")
    eval_logger = ReplayLogger("replays/red_queen.json")
    for eval_ep in range(50):
        obs, _ = env.reset(seed=eval_ep * 17 + 1)
        state = obs["attacker"]
        eval_logger.log_step(str(eval_ep + 1), env.attacker_pos, env.defender_pos)
        done = False
        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]
            att_act = attacker.predict(state, action_mask=att_mask)
            def_act = defender.predict(state, action_mask=def_mask)
            next_state, _, term, trunc, _ = env.step(att_act, def_act)
            done = term or trunc
            eval_logger.log_step(str(eval_ep + 1), env.attacker_pos, env.defender_pos)
            state = next_state
    eval_logger.save()
    print("Saved clean deterministic replay games to replays/red_queen.json")

    # Plot results: 2-panel synchronized compact figure (8.5x4.6 @ 100 dpi)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True)

    ax1.plot(att_win_rates, label="Attacker Win Rate", color="red", linewidth=1.4)
    ax1.plot(def_win_rates, label="Defender Win Rate", color="blue", linewidth=1.4)
    ax1.axvspan(0, 100, color="gray", alpha=0.15, label="Warmup Window (N < 100)")
    ax1.set_title("Phase 4 — Red Queen Self-Play (Rolling 100 Episodes)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Win Rate", fontsize=9.5)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(timeout_rates, label="Timeout Rate", color="darkorange", linewidth=1.2)
    ax2.axvspan(0, 100, color="gray", alpha=0.15)
    ax2.set_title("Timeout Rate (Synchronous closure: Attacker + Defender + Timeout = 100%)", fontsize=9.5)
    ax2.set_ylabel("Timeout Fraction", fontsize=9.5)
    ax2.set_xlabel("Episode", fontsize=9.5)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/phase4_marl_curve.png", dpi=100)
    print("Saved compact learning curve to docs/phase4_marl_curve.png")


if __name__ == "__main__":
    main()
