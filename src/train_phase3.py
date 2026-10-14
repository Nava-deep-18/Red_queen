from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.distributions import Categorical


def main():
    env = ArenaEnv()

    # State dim is 6 (ax, ay, dx, dy, tx, ty)
    state_dim = 6
    action_dim = env.action_space.n

    # Tuned learning rates for fast, stable PPO convergence
    agent = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    replay_logger = ReplayLogger("replays/fixed_opponent.json")

    # PPO updates after this many complete episodes (episode-boundary updates
    # keep Monte Carlo returns well-defined — no partial trajectories).
    update_every_episodes = 10
    num_episodes = 1200
    win_rates = []
    recent_wins = []

    print("Starting Training: Single-Agent RL (PPO) with Fixed Opponent Baseline")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        # Fix #11: log the initial spawn positions as the first replay frame
        replay_logger.log_step(str(episode + 1), env.attacker_pos, env.defender_pos)

        while not done:
            state_tensor = torch.FloatTensor(state)

            # Fix #1: determine the final (safe) action BEFORE writing anything
            # to PPO's buffer so that the stored action and its log-prob match
            # exactly what is passed to env.step().
            with torch.no_grad():
                raw_action, raw_logprob = agent.policy_old.act(state_tensor)
                att_action = raw_action.item()

            # Suicide Prevention: if the raw action would walk onto the Defender,
            # choose the closest safe alternative and recompute its log-prob.
            intended_pos = env.attacker_pos + env.action_to_move[att_action]
            if np.array_equal(intended_pos, env.defender_pos):
                safe_actions = [
                    a for a in range(5)
                    if (not np.array_equal(env.attacker_pos + env.action_to_move[a], env.defender_pos)
                        and env._is_valid_pos(env.attacker_pos + env.action_to_move[a]))
                ]
                if safe_actions:
                    att_action = min(
                        safe_actions,
                        key=lambda a: (
                            abs((env.attacker_pos + env.action_to_move[a])[0] - env.target_pos[0])
                            + abs((env.attacker_pos + env.action_to_move[a])[1] - env.target_pos[1])
                        )
                    )
                    # Recompute log-prob for the replacement action
                    with torch.no_grad():
                        action_probs = agent.policy_old.actor(state_tensor)
                        dist = Categorical(action_probs)
                        raw_logprob = dist.log_prob(torch.tensor(att_action))

            # Record the *final* action and its correct log-prob into the buffer
            agent.states.append(state_tensor)
            agent.actions.append(torch.tensor(att_action))
            agent.logprobs.append(raw_logprob.detach())

            def_action = 4  # STAY — fixed-policy baseline

            old_att = env.attacker_pos.copy()
            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Bridge-unbiased distance shaping reward
            dist_reward = env.compute_zero_sum_shaping(old_att, env.attacker_pos, scale=0.5)

            # Proximity Danger Bubble: penalise moving within 1 tile of the Defender
            dist_to_def = (abs(env.attacker_pos[0] - env.defender_pos[0])
                           + abs(env.attacker_pos[1] - env.defender_pos[1]))
            danger_penalty = -2.0 if dist_to_def <= 1 and not info["caught"] else 0.0

            total_reward = rewards[0] + dist_reward + danger_penalty

            replay_logger.log_step(str(episode + 1), env.attacker_pos, env.defender_pos)
            agent.store_reward(total_reward, done)

            state = next_state

        # Fix #3: update only at episode boundaries so every batch contains
        # only complete trajectories and Monte Carlo returns are unbiased.
        if (episode + 1) % update_every_episodes == 0:
            agent.update()

        # Track win rate (attacker win = reached target)
        recent_wins.append(1 if info["reached_target"] else 0)
        if len(recent_wins) > 50:
            recent_wins.pop(0)

        win_rate = sum(recent_wins) / len(recent_wins)
        win_rates.append(win_rate)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Win Rate (last 50): {win_rate:.2f}")

    # Fix #3: flush any remaining episodes at the end of training
    agent.update()

    # Plot training results
    plt.plot(win_rates, color="red")
    plt.title("Attacker Win Rate over Episodes (PPO Baseline)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (last 50 ep)")
    os.makedirs("docs", exist_ok=True)
    replay_logger.save()
    plt.savefig("docs/phase3_training_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase3_training_curve.png")


if __name__ == "__main__":
    main()
