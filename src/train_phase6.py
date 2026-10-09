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
    action_dim = env.action_space.n
    
    attacker = PPOAgent(state_dim, action_dim)
    defender = PPOAgent(state_dim, action_dim)
    
    replay_logger = ReplayLogger("replays/red_queen_history.json")
    
    # Archives for storing past versions
    defender_archive = []
    attacker_archive = []
    
    update_timestep = 2000
    num_episodes = 5000
    archive_interval = 200 # Save agents to archive every 200 episodes
    time_step = 0
    
    att_win_rates = []
    recent_att_wins = []
    
    os.makedirs("models/archive", exist_ok=True)
    
    print("Starting Phase 6 Training: Red Queen + Double History (Opponent Archives)")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        # Decide training mode
        # 60% Both Current
        # 20% Current Attacker vs Archived Defender
        # 20% Archived Attacker vs Current Defender
        mode = random.random()
        
        train_attacker = True
        train_defender = True
        active_attacker = attacker
        active_defender = defender
        
        if mode < 0.2 and len(defender_archive) > 0:
            active_defender = PPOAgent(state_dim, action_dim)
            active_defender.policy.load_state_dict(random.choice(defender_archive))
            active_defender.policy_old.load_state_dict(active_defender.policy.state_dict())
            train_defender = False
            
        elif mode < 0.4 and len(attacker_archive) > 0:
            active_attacker = PPOAgent(state_dim, action_dim)
            active_attacker.policy.load_state_dict(random.choice(attacker_archive))
            active_attacker.policy_old.load_state_dict(active_attacker.policy.state_dict())
            train_attacker = False
            
        while not done:
            time_step += 1
            
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
            
            next_state, rewards, done, info = env.step(att_action, def_action)
            
            replay_logger.log_step(str(episode+1), env.attacker_pos, env.defender_pos)
            
            if train_attacker:
                # Distance-based shaping reward for Attacker
                old_x, old_y = state[0]*env.width, state[1]*env.height
                new_x, new_y = next_state[0]*env.width, next_state[1]*env.height
                tx, ty = state[4]*env.width, state[5]*env.height
                
                old_dist = abs(old_x - tx) + abs(old_y - ty)
                new_dist = abs(new_x - tx) + abs(new_y - ty)
                att_dist_reward = (old_dist - new_dist) * 0.5
                
                attacker.store_reward(rewards[0] + att_dist_reward, done)
            if train_defender:
                defender.store_reward(rewards[1], done)
            
            state = next_state
            
            # Update current agents
            if time_step % update_timestep == 0:
                attacker.update()
                defender.update()
                time_step = 0
                
        # Save to archive occasionally
        if (episode + 1) % archive_interval == 0:
            defender_archive.append(copy.deepcopy(defender.policy.state_dict()))
            attacker_archive.append(copy.deepcopy(attacker.policy.state_dict()))
                
        # Track win rates (only track when Attacker is actually training to get its true learning curve)
        if train_attacker:
            recent_att_wins.append(1 if info['reached_target'] else 0)
            
            if len(recent_att_wins) > 100:
                recent_att_wins.pop(0)
                
            att_win_rate = sum(recent_att_wins) / len(recent_att_wins)
            att_win_rates.append(att_win_rate)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Attacker Win Rate: {att_win_rate:.2f} | Archive Size: {len(defender_archive)}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(att_win_rates, label="Attacker Win Rate (Double Archive)", color="purple")
    plt.title("Attacker Win Rate over Episodes (Archive Training)")
    plt.xlabel("Episode (where Attacker trained)")
    plt.ylabel("Win Rate (last 100 ep)")
    plt.legend()
    
    os.makedirs("docs", exist_ok=True)
    replay_logger.save()
    plt.savefig("docs/phase6_archive_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase6_archive_curve.png")
    
if __name__ == "__main__":
    main()
