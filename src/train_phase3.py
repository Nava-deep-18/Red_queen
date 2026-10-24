from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import matplotlib.pyplot as plt
import os
import torch


def main():
    env = ArenaEnv()

    # State dim is 6 (ax, ay, dx, dy, tx, ty)
    state_dim = 6
    action_dim = env.action_space("attacker").n

    agent = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    # PPO updates after this many complete episodes (episode-boundary updates
    # keep Monte Carlo returns well-defined: no partial trajectories).
    update_every_episodes = 10
    num_episodes = 1200

    att_win_rates = []
    def_win_rates = []
    timeout_rates = []
    recent_att = []
    recent_def = []
    recent_timeout = []

    print("Starting Training: Single-Agent RL (PPO) with Fixed Opponent Baseline")
    print("  Defender always STAYS. Nothing stops the attacker from walking into it:")
    print("  if it does, the Defender wins. We only observe who wins.")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs["attacker"]
        done = False

        while not done:
            # Only walls / river pillars are masked (a move into them does nothing).
            # Moving onto the Defender is allowed and ends the episode as a catch.
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            att_action = agent.select_action(state, action_mask=att_mask)

            def_action = 4  # STAY: fixed-policy baseline

            old_att = env.attacker_pos.copy()
            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Reward = terminal result (+10 reach target / -10 caught) + distance shaping
            dist_reward, _ = env.compute_zero_sum_shaping(old_att, env.attacker_pos, gamma=0.99)
            total_reward = rewards[0] + dist_reward

            agent.store_reward(total_reward, done)
            state = next_state

        # Update only at episode boundaries so every batch holds complete trajectories.
        if (episode + 1) % update_every_episodes == 0:
            agent.update()

        # Record the outcome of this episode. Exactly one of the three is 1.
        recent_att.append(1 if info["reached_target"] else 0)
        recent_def.append(1 if info["caught"] else 0)
        recent_timeout.append(1 if info["timeout"] else 0)
        if len(recent_att) > 50:
            recent_att.pop(0)
            recent_def.pop(0)
            recent_timeout.pop(0)

        att_win_rates.append(sum(recent_att) / len(recent_att))
        def_win_rates.append(sum(recent_def) / len(recent_def))
        timeout_rates.append(sum(recent_timeout) / len(recent_timeout))

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes} "
                  f"| Attacker wins: {att_win_rates[-1]:.2f} "
                  f"| Defender wins: {def_win_rates[-1]:.2f} "
                  f"| Timeouts: {timeout_rates[-1]:.2f}  (last 50)")

    # Flush any remaining episodes at the end of training
    agent.update()

    # Save trained baseline model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.policy.state_dict(), "models/phase3_attacker.pth")
    print("Saved trained model to models/phase3_attacker.pth")

    # Record clean deterministic evaluation episodes for visual dashboard (50 games)
    print("Recording clean deterministic evaluation replay games for Fixed Opponent...")
    eval_logger = ReplayLogger.from_env("replays/fixed_opponent.json", env)
    for eval_ep in range(50):
        obs, _ = env.reset(seed=eval_ep * 17 + 1)
        state = obs["attacker"]
        eval_logger.log_step(str(eval_ep + 1), env.attacker_pos, env.defender_pos)
        done = False
        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            att_act = agent.predict(state, action_mask=att_mask)
            next_state, _, term, trunc, _ = env.step(att_act, 4)
            done = term or trunc
            eval_logger.log_step(str(eval_ep + 1), env.attacker_pos, env.defender_pos)
            state = next_state
    eval_logger.save()
    print("Saved clean deterministic replay games to replays/fixed_opponent.json")

    # Plot results with exact metric closure and warmup window shading
    plt.figure(figsize=(8.5, 4.4))
    plt.plot(att_win_rates, color="red", linewidth=1.5, label="Attacker wins (reached target)")
    plt.plot(def_win_rates, color="blue", linewidth=1.5, label="Defender wins (caught attacker)")
    plt.plot(timeout_rates, color="darkorange", linewidth=1.2, label="Timeouts")
    plt.axvspan(0, 50, color="gray", alpha=0.15, label="Warmup Window (N < 50)")
    plt.title("Phase 3: PPO Attacker vs Standing Defender", fontsize=11, fontweight="bold")
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Rate (last 50 ep)", fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(loc="center right", fontsize=8.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/phase3_training_curve.png", dpi=100)
    print("Saved learning curve to docs/phase3_training_curve.png")


if __name__ == "__main__":
    main()