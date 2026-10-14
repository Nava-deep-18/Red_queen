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
    action_dim = env.action_space.n

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
    recent_att_wins = []
    recent_def_wins = []

    print("Starting Phase 4 Training: Full MARL with PPO (Both Learn)")

    # Fix #4 — Design note: this experiment is GENERAL-SUM, not zero-sum.
    # The environment reward for a target-reach/catch is zero-sum (+10/-10).
    # However, the Attacker receives an additional distance-shaping bonus that
    # the Defender does NOT receive.  The two agents therefore optimise
    # different effective reward scales and objectives.
    #
    # Interpretation: the win-rate curves show a Red Queen cycling dynamic,
    # but they cannot be interpreted as evidence of Nash equilibrium in a
    # pure zero-sum game.  If a zero-sum comparison is desired, the Defender
    # should receive the negative of the Attacker's FULL reward (including
    # shaping), or shaping should be removed from both.
    print("  Note: Attacker uses distance-shaping reward; Defender uses raw env reward only.")
    print("  This is a general-sum MARL experiment, not a strict zero-sum comparison.")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        # Fix #11: log initial spawn positions as the first replay frame
        replay_logger.log_step(str(episode + 1), env.attacker_pos, env.defender_pos)

        while not done:
            # Both agents select actions
            att_action = attacker.select_action(state)
            def_action = defender.select_action(state)

            old_att = env.attacker_pos.copy()

            # Step in environment
            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Log position for replay (after move)
            replay_logger.log_step(str(episode + 1), env.attacker_pos, env.defender_pos)

            # Strictly Zero-Sum Potential-Based Shaping (Ng et al. 1999; Devlin & Kudenko 2011)
            # Both agents receive exactly opposing shaping: R_att + R_def == 0 at every timestep.
            shaping = env.compute_zero_sum_shaping(old_att, env.attacker_pos, scale=0.5)
            att_reward = rewards[0] + shaping
            def_reward = rewards[1] - shaping

            attacker.store_reward(att_reward, done)
            defender.store_reward(def_reward, done)

            state = next_state

        # Fix #3: update at episode boundaries (complete trajectories only)
        if (episode + 1) % update_every_episodes == 0:
            attacker.update()
            defender.update()

        # Track win rates.
        # Use mutually-exclusive flags: the environment guarantees caught=False
        # when reached_target=True (target takes priority), so these are already
        # disjoint, but we make the intent explicit here.
        recent_att_wins.append(1 if info["reached_target"] else 0)
        recent_def_wins.append(1 if info["caught"] and not info["reached_target"] else 0)

        if len(recent_att_wins) > 100:
            recent_att_wins.pop(0)
            recent_def_wins.pop(0)

        att_win_rate = sum(recent_att_wins) / len(recent_att_wins)
        def_win_rate = sum(recent_def_wins) / len(recent_def_wins)

        att_win_rates.append(att_win_rate)
        def_win_rates.append(def_win_rate)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Attacker Win Rate: {att_win_rate:.2f} | Defender Win Rate: {def_win_rate:.2f}")

    # Fix #3: flush any remaining partial batches at end of training
    attacker.update()
    defender.update()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(att_win_rates, label="Attacker Win Rate", color="red")
    plt.plot(def_win_rates, label="Defender Win Rate", color="blue")
    plt.title("Attacker vs Defender Win Rates over Episodes (PPO MARL)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (last 100 ep)")
    plt.legend()

    os.makedirs("docs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    torch.save(attacker.policy.state_dict(), "models/phase4_attacker.pth")

    replay_logger.save()
    plt.savefig("docs/phase4_marl_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase4_marl_curve.png")


if __name__ == "__main__":
    main()
