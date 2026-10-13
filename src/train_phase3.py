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
    
    # Tuned learning rates for fast, stable PPO convergence
    agent = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    
    replay_logger = ReplayLogger("replays/fixed_opponent.json")
    
    update_timestep = 512
    num_episodes = 1200
    time_step = 0
    win_rates = []
    recent_wins = []
    
    print("Starting Training: Single-Agent RL (PPO) with Fixed Opponent Baseline")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            time_step += 1
            
            # Attacker selects action using trained PPO policy
            att_action = agent.select_action(state)
            
            # Suicide Prevention: Never step directly onto the Defender's occupied tile
            intended_pos = env.attacker_pos + env.action_to_move[att_action]
            if np.array_equal(intended_pos, env.defender_pos):
                safe_actions = [
                    a for a in range(5)
                    if not np.array_equal(env.attacker_pos + env.action_to_move[a], env.defender_pos)
                    and env._is_valid_pos(env.attacker_pos + env.action_to_move[a])
                ]
                if safe_actions:
                    att_action = min(
                        safe_actions,
                        key=lambda a: abs((env.attacker_pos + env.action_to_move[a])[0] - env.target_pos[0]) + abs((env.attacker_pos + env.action_to_move[a])[1] - env.target_pos[1])
                    )
            
            # Defender stays still (fixed policy baseline)
            def_action = 4 # STAY
            
            next_state, rewards, done, info = env.step(att_action, def_action)
            
            # Distance-based shaping reward to help PPO find the target through the bridges
            old_x, old_y = state[0]*env.width, state[1]*env.height
            new_x, new_y = next_state[0]*env.width, next_state[1]*env.height
            tx, ty = state[4]*env.width, state[5]*env.height
            
            old_dist = abs(old_x - tx) + abs(old_y - ty)
            new_dist = abs(new_x - tx) + abs(new_y - ty)
            dist_reward = (old_dist - new_dist) * 0.5  # Positive if moved closer, negative if moved away
            
            # Proximity Danger Bubble: penalize moving within 1 tile of the Defender
            dist_to_def = abs(env.attacker_pos[0] - env.defender_pos[0]) + abs(env.attacker_pos[1] - env.defender_pos[1])
            danger_penalty = -2.0 if dist_to_def <= 1 and not info['caught'] else 0.0
            
            total_reward = rewards[0] + dist_reward + danger_penalty
            
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

    # Plot training results (no 'Phase' in visible title)
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
