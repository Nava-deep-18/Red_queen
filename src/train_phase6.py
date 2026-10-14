from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
import copy

def main():
    env = ArenaEnv()
    
    state_dim = 6
    action_dim = env.action_space.n
    
    attacker = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    defender = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    
    replay_logger = ReplayLogger("replays/red_queen_history.json")
    
    # Archives for storing past versions
    defender_archive = []
    attacker_archive = []
    
    update_every_episodes = 20   # PPO update cadence: update every N complete episodes
    num_episodes = 1500
    archive_interval = 100  # snapshot both agents every 100 episodes after warmup
    warmup_episodes = 300   # run current vs current before enabling archive matchups
    
    att_win_rates  = []
    def_win_rates  = []
    timeout_rates  = []        # fraction of episodes that ended in timeout
    recent_att_wins = []
    recent_def_wins = []
    recent_timeouts = []
    
    os.makedirs("models/archive", exist_ok=True)
    
    print("Starting Phase 6 Training: Red Queen + Double History (Opponent Archives)")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        # Fix #11: log initial spawn positions as the first replay frame
        replay_logger.log_step(str(episode + 1), env.attacker_pos, env.defender_pos)

        # Fix #5: Decide training mode with corrected thresholds.
        # Intended split (once archives exist):
        #   60%  — current attacker vs current defender  (both train)
        #   20%  — current attacker vs archived defender  (attacker trains only)
        #   20%  — archived attacker vs current defender  (defender trains only)
        # Pre-archive warmup: all episodes are current vs current.
        mode = random.random()

        train_attacker = True
        train_defender = True
        active_attacker = attacker
        active_defender = defender

        if len(defender_archive) > 0 and mode < 0.20:
            # 20%: current attacker trains against a random archived defender
            active_defender = PPOAgent(state_dim, action_dim)
            active_defender.policy.load_state_dict(random.choice(defender_archive))
            active_defender.policy_old.load_state_dict(active_defender.policy.state_dict())
            train_defender = False

        elif len(attacker_archive) > 0 and mode < 0.40:
            # 20%: random archived attacker faces current defender (defender trains)
            active_attacker = PPOAgent(state_dim, action_dim)
            active_attacker.policy.load_state_dict(random.choice(attacker_archive))
            active_attacker.policy_old.load_state_dict(active_attacker.policy.state_dict())
            train_attacker = False
        # else: 60% (or 100% during warmup) — current vs current, both train
            
        while not done:
            # Select actions
            if train_attacker:
                att_action = attacker.select_action(state)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    att_action, _ = active_attacker.policy_old.act(state_tensor)
                    att_action = att_action.item()
                    
            if train_defender:
                def_action = defender.select_action(state)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    def_action, _ = active_defender.policy_old.act(state_tensor)
                    def_action = def_action.item()
            
            old_att = env.attacker_pos.copy()
            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            replay_logger.log_step(str(episode + 1), env.attacker_pos, env.defender_pos)

            # Strictly Zero-Sum Potential-Based Shaping (Ng et al. 1999; Devlin & Kudenko 2011)
            # R_att + R_def == 0 at every timestep. Both agents receive opposing shaping.
            shaping = env.compute_zero_sum_shaping(old_att, env.attacker_pos, scale=0.5)
            att_reward = rewards[0] + shaping
            def_reward = rewards[1] - shaping

            if train_attacker:
                attacker.store_reward(att_reward, done)

            if train_defender:
                defender.store_reward(def_reward, done)

            state = next_state
                
        # Fix #3: update at episode boundaries (complete trajectories only)
        # update_every_episodes controls PPO batch cadence; archive_interval
        # controls how frequently policy snapshots are saved.
        if train_attacker and (episode + 1) % update_every_episodes == 0:
            attacker.update()
        if train_defender and (episode + 1) % update_every_episodes == 0:
            defender.update()

        # Save mature agents to archive (both in-memory and persisted to disk)
        if (episode + 1) >= warmup_episodes and (episode + 1) % archive_interval == 0:
            att_weights = copy.deepcopy(attacker.policy.state_dict())
            def_weights = copy.deepcopy(defender.policy.state_dict())
            attacker_archive.append(att_weights)
            defender_archive.append(def_weights)
            torch.save(att_weights, f"models/archive/attacker_ep{episode + 1}.pth")
            torch.save(def_weights, f"models/archive/defender_ep{episode + 1}.pth")

        # Track win rates symmetrically for both agents.
        att_win = 1 if info["reached_target"] else 0
        def_win = 1 if info["caught"] and not info["reached_target"] else 0
        timed   = 1 if info["timeout"] else 0

        if train_attacker:
            recent_att_wins.append(att_win)
            if len(recent_att_wins) > 100:
                recent_att_wins.pop(0)

        if train_defender:
            recent_def_wins.append(def_win)
            if len(recent_def_wins) > 100:
                recent_def_wins.pop(0)

        # Timeout tracked every episode regardless of mode
        recent_timeouts.append(timed)
        if len(recent_timeouts) > 100:
            recent_timeouts.pop(0)

        att_win_rate = sum(recent_att_wins) / len(recent_att_wins) if recent_att_wins else 0.0
        def_win_rate = sum(recent_def_wins) / len(recent_def_wins) if recent_def_wins else 0.0
        timeout_rate = sum(recent_timeouts) / len(recent_timeouts) if recent_timeouts else 0.0

        # Record running rates every episode so all series share the exact episode index
        att_win_rates.append(att_win_rate)
        def_win_rates.append(def_win_rate)
        timeout_rates.append(timeout_rate)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} "
                  f"| Att WR: {att_win_rate:.2f} "
                  f"| Def WR: {def_win_rate:.2f} "
                  f"| Timeout: {timeout_rate:.2f} "
                  f"| Archive: {len(defender_archive)}")

    # Fix #3: flush any remaining complete episodes at end of training
    attacker.update()
    defender.update()

    # Plot results: show both agents + timeout rate so all outcomes are accounted for
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(att_win_rates, label="Attacker Win Rate",  color="purple", linewidth=1.5)
    ax1.plot(def_win_rates, label="Defender Win Rate",  color="royalblue", linewidth=1.5)
    ax1.set_title("Phase 6 — Double History Archive Training")
    ax1.set_ylabel("Win Rate (rolling 100 ep)")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timeout_rates, label="Timeout Rate (all episodes)", color="darkorange", linewidth=1.2)
    ax2.set_title("Timeout rate — distinguishes weak policy from strong defense")
    ax2.set_ylabel("Timeout fraction (rolling 100 ep)")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    torch.save(attacker.policy.state_dict(), "models/phase6_attacker.pth")

    replay_logger.save()
    plt.savefig("docs/phase6_archive_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase6_archive_curve.png")


if __name__ == "__main__":
    main()
