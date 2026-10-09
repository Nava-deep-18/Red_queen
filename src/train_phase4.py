from environment import ArenaEnv
from agents.ppo import PPOAgent
from replay_logger import ReplayLogger
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    env = ArenaEnv()
    
    state_dim = 6
    action_dim = env.action_space.n
    
    # Initialize both agents
    attacker = PPOAgent(state_dim, action_dim)
    defender = PPOAgent(state_dim, action_dim)
    
    # Initialize logger
    replay_logger = ReplayLogger("replays/red_queen.json")
    
    update_timestep = 2000
    num_episodes = 4000
    time_step = 0
    
    att_win_rates = []
    def_win_rates = []
    recent_att_wins = []
    recent_def_wins = []

    
    print("Starting Phase 4 Training: Full MARL with PPO (Both Learn)")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            time_step += 1
            
            # Both agents select actions
            att_action = attacker.select_action(state)
            def_action = defender.select_action(state)
            
            # Step in environment
            next_state, rewards, done, info = env.step(att_action, def_action)
            
            # Log position for replay
            replay_logger.log_step(str(episode+1), env.attacker_pos, env.defender_pos)
            
            # Distance-based shaping reward for Attacker
            old_x, old_y = state[0]*env.width, state[1]*env.height
            new_x, new_y = next_state[0]*env.width, next_state[1]*env.height
            tx, ty = state[4]*env.width, state[5]*env.height
            
            old_dist = abs(old_x - tx) + abs(old_y - ty)
            new_dist = abs(new_x - tx) + abs(new_y - ty)
            att_dist_reward = (old_dist - new_dist) * 0.5
            
            # Store rewards
            attacker.store_reward(rewards[0] + att_dist_reward, done)
            defender.store_reward(rewards[1], done)
            
            state = next_state
            
            # Update agents
            if time_step % update_timestep == 0:
                attacker.update()
                defender.update()
                time_step = 0
                
        # Track win rates
        recent_att_wins.append(1 if info['reached_target'] else 0)
        recent_def_wins.append(1 if info['caught'] else 0)
        
        if len(recent_att_wins) > 100:
            recent_att_wins.pop(0)
            recent_def_wins.pop(0)
            
        att_win_rate = sum(recent_att_wins) / len(recent_att_wins)
        def_win_rate = sum(recent_def_wins) / len(recent_def_wins)
        
        att_win_rates.append(att_win_rate)
        def_win_rates.append(def_win_rate)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Attacker Win Rate: {att_win_rate:.2f} | Defender Win Rate: {def_win_rate:.2f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(att_win_rates, label="Attacker Win Rate", color="red")
    plt.plot(def_win_rates, label="Defender Win Rate", color="blue")
    plt.title("Attacker vs Defender Win Rates over Episodes (PPO MARL)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (last 100 ep)")
    plt.legend()
    
    os.makedirs("docs", exist_ok=True)
    replay_logger.save()
    plt.savefig("docs/phase4_marl_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase4_marl_curve.png")
    
if __name__ == "__main__":
    main()
