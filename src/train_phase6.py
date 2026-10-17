from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import copy

def main():
    env = ArenaEnv()
    
    state_dim = 6
    action_dim = env.action_space("attacker").n
    
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
        obs, _ = env.reset()
        state = obs["attacker"]
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
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            # Select actions with action masking
            if train_attacker:
                att_action = attacker.select_action(state, action_mask=att_mask)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    att_action, _ = active_attacker.policy_old.act(state_tensor, action_mask=att_mask)
                    att_action = att_action.item()

            if train_defender:
                def_action = defender.select_action(state, action_mask=def_mask)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    def_action, _ = active_defender.policy_old.act(state_tensor, action_mask=def_mask)
                    def_action = def_action.item()
            
            old_att = env.attacker_pos.copy()
            old_def = env.defender_pos.copy()
            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            # Strictly Zero-Sum Potential-Based Shaping with Relative Pursuit & Advance
            # R_att + R_def == 0 at every timestep. Both agents receive opposing shaping.
            att_shaping, def_shaping = env.compute_zero_sum_shaping(
                old_att, env.attacker_pos, old_def, env.defender_pos
            )
            att_reward = rewards[0] + att_shaping
            def_reward = rewards[1] + def_shaping

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

        # Synchronously record all mutually-exclusive outcomes every episode
        recent_att_wins.append(att_win)
        recent_def_wins.append(def_win)
        recent_timeouts.append(timed)

        if len(recent_att_wins) > 100:
            recent_att_wins.pop(0)
            recent_def_wins.pop(0)
            recent_timeouts.pop(0)

        # Exact synchronous rolling rates: sum is mathematically guaranteed to equal 1.00 (100%)
        att_win_rate = sum(recent_att_wins) / len(recent_att_wins)
        def_win_rate = sum(recent_def_wins) / len(recent_def_wins)
        timeout_rate = sum(recent_timeouts) / len(recent_timeouts)

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

    # Flush any remaining complete episodes at end of training
    attacker.update()
    defender.update()

    # Save BOTH final models (Item 2)
    os.makedirs("models", exist_ok=True)
    torch.save(attacker.policy.state_dict(), "models/phase6_attacker.pth")
    torch.save(defender.policy.state_dict(), "models/phase6_defender.pth")
    print("Saved final models to models/phase6_attacker.pth and models/phase6_defender.pth")

    # Record clean deterministic evaluation episodes for visual dashboard (Item 3)
    print("Recording clean deterministic evaluation replay games for Double Archive...")
    eval_logger = ReplayLogger("replays/red_queen_history.json")
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
    print("Saved clean deterministic replay games to replays/red_queen_history.json")

    # Plot results: 2-panel synchronized compact figure (8.5x4.6 @ 100 dpi)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True)

    ax1.plot(att_win_rates, label="Attacker Win Rate",  color="purple", linewidth=1.4)
    ax1.plot(def_win_rates, label="Defender Win Rate",  color="royalblue", linewidth=1.4)
    ax1.axvspan(0, 100, color="gray", alpha=0.15, label="Warmup Window (N < 100)")
    ax1.set_title("Phase 6 — Double History Archive Training (Rolling 100 Episodes)", fontsize=10, fontweight="bold")
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
    plt.savefig("docs/phase6_archive_curve.png", dpi=100)
    print("Saved compact learning curve to docs/phase6_archive_curve.png")


if __name__ == "__main__":
    main()
