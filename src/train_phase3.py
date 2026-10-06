from environment import ArenaEnv
from agents.dqn import DQNAgent
from replay_logger import ReplayLogger
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    env = ArenaEnv()
    
    # State dim is 6 (ax, ay, dx, dy, tx, ty)
    state_dim = 6
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    replay_logger = ReplayLogger("replays/fixed_opponent.json")
    
    num_episodes = 3000
    win_rates = []
    recent_wins = []
    
    print("Starting Phase 3 Training: Single-Agent RL (DQN) with Fixed Defender")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_loss = 0
        steps = 0
        
        while not done:
            # Attacker uses DQN policy
            att_action = agent.select_action(state)
            
            # Defender stays still (fixed policy)
            def_action = 4 # STAY
            
            next_state, rewards, done, info = env.step(att_action, def_action)
            
            replay_logger.log_step(str(episode+1), env.attacker_pos, env.defender_pos)
            
            # Store transition for Attacker
            agent.store_transition(state, att_action, rewards[0], next_state, done)
            
            # Update network
            loss = agent.update()
            if loss is not None:
                total_loss += loss
                
            state = next_state
            steps += 1
            
        agent.decay_epsilon()
        
        # Track win rate (attacker win = reached target)
        recent_wins.append(1 if info['reached_target'] else 0)
        if len(recent_wins) > 50:
            recent_wins.pop(0)
            
        win_rate = sum(recent_wins) / len(recent_wins)
        win_rates.append(win_rate)
        
        if (episode + 1) % 50 == 0:
            avg_loss = total_loss / steps if steps > 0 else 0
            print(f"Episode {episode+1}/{num_episodes} | Win Rate (last 50): {win_rate:.2f} | Epsilon: {agent.epsilon:.2f} | Loss: {avg_loss:.2f}")

    # Plot training results
    plt.plot(win_rates)
    plt.title("Attacker Win Rate over Episodes (DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (last 50 ep)")
    os.makedirs("docs", exist_ok=True)
    replay_logger.save()
    plt.savefig("docs/phase3_training_curve.png")
    print("\nTraining complete! Saved learning curve to docs/phase3_training_curve.png")

if __name__ == "__main__":
    main()
