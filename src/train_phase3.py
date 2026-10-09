from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    env = ArenaEnv()
    
    # State dim is 6 (ax, ay, dx, dy, tx, ty)
    state_dim = 6
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    replay_logger = ReplayLogger("replays/fixed_opponent.json")
    
    update_timestep = 2000
    num_episodes = 3000
    time_step = 0
    win_rates = []
    recent_wins = []
    
    print("Starting Phase 3 Training: Single-Agent RL (PPO) with Fixed Defender")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            time_step += 1
            
            # Attacker uses PPO policy
            att_action = agent.select_action(state)
            
            # Defender stays still (fixed policy)
            def_action = 4 # STAY
            
            next_state, rewards, done, info = env.step(att_action, def_action)
            
            # Distance-based shaping reward to help PPO find the target through the bridges
            old_x, old_y = state[0]*env.width, state[1]*env.height
            new_x, new_y = next_state[0]*env.width, next_state[1]*env.height
            tx, ty = state[4]*env.width, state[5]*env.height
            
            old_dist = abs(old_x - tx) + abs(old_y - ty)
            new_dist = abs(new_x - tx) + abs(new_y - ty)
            dist_reward = (old_dist - new_dist) * 0.5  # Positive if moved closer, negative if moved away
            
            total_reward = rewards[0] + dist_reward
            
            replay_logger.log_step(str(episode+1), env.attacker_pos, env.defender_pos)
            
            # Store reward for Attacker
            agent.store_reward(total_reward, done)
            
            # Update network
            if time_step % update_timestep == 0:
                agent.update()
                time_step = 0
                
            state = next_state
            steps += 1
            
        # Track win rate (attacker win = reached target)
        recent_wins.append(1 if info['reached_target'] else 0)
        if len(recent_wins) > 50:
            recent_wins.pop(0)
            
        win_rate = sum(recent_wins) / len(recent_wins)
        win_rates.append(win_rate)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Win Rate (last 50): {win_rate:.2f}")

    # Plot training results
    plt.plot(win_rates, color="red")
    plt.title("Attacker Win Rate over Episodes (PPO)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (last 50 ep)")
    os.makedirs("docs", exist_ok=True)
    replay_logger.save()
    plt.savefig("docs/phase3_training_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase3_training_curve.png")

if __name__ == "__main__":
    main()
