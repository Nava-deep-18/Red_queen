from environment import ArenaEnv
import random
import time

def main():
    env = ArenaEnv()
    state = env.reset()
    
    print("Starting Random Play Test (Phase 2)")
    env.render()
    
    done = False
    while not done:
        # Both agents choose random actions
        att_action = env.action_space.sample()
        def_action = env.action_space.sample()
        
        print(f"\nStep {env.step_count + 1}")
        print(f"Attacker action: {att_action}, Defender action: {def_action}")
        
        state, rewards, done, info = env.step(att_action, def_action)
        env.render()
        
        time.sleep(0.5)
        
    print("\nEpisode finished!")
    print(f"Rewards - Attacker: {rewards[0]}, Defender: {rewards[1]}")
    print(f"Info: {info}")

if __name__ == "__main__":
    main()
