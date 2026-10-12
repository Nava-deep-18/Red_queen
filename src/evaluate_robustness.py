import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from environment import ArenaEnv
from agents.ppo import PPOAgent

def evaluate_archetype(attacker_path, archetype, num_games=25):
    env = ArenaEnv()
    state_dim = 6
    action_dim = env.action_space.n
    
    attacker = PPOAgent(state_dim, action_dim)
    if os.path.exists(attacker_path):
        attacker.policy.load_state_dict(torch.load(attacker_path))
        attacker.policy_old.load_state_dict(torch.load(attacker_path))
    else:
        print(f"Model not found: {attacker_path}")
        return 0.0
        
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                att_action, _ = attacker.policy_old.act(state_tensor)
                att_action = att_action.item()
            
            dx, dy = env.defender_pos
            ax, ay = env.attacker_pos
            
            # Hardcoded archetypes on the river bank (X=6)
            if archetype == "NORTH_CAMPER":
                if dy > 1: def_action = 0 # Up
                elif dy < 1: def_action = 1 # Down
                else: def_action = 4 # Stay
            elif archetype == "CENTER_CAMPER":
                if dy > 3: def_action = 0 # Up
                elif dy < 3: def_action = 1 # Down
                else: def_action = 4 # Stay
            elif archetype == "SOUTH_CAMPER":
                if dy > 5: def_action = 0 # Up
                elif dy < 5: def_action = 1 # Down
                else: def_action = 4 # Stay
            elif archetype == "PATROLLER":
                # Patrols between North and South bridges along riverbank
                if not hasattr(env, 'patrol_dir'): env.patrol_dir = 1
                if dy <= 1: env.patrol_dir = 1
                elif dy >= 5: env.patrol_dir = -1
                def_action = 1 if env.patrol_dir == 1 else 0
            else:
                def_action = 4
                
            state, rewards, done, info = env.step(att_action, def_action)
            if done and info.get('reached_target', False):
                wins += 1
    return wins / num_games

def main():
    archetypes = ["NORTH_CAMPER", "CENTER_CAMPER", "SOUTH_CAMPER", "PATROLLER"]
    num_eval_games = 50
    p4_scores = []
    p6_scores = []
    
    print("Evaluating Phase 4...")
    for arch in archetypes:
        score = evaluate_archetype("models/phase4_attacker.pth", arch, num_games=num_eval_games)
        p4_scores.append(score)
        print(f"  P4 vs {arch}: {score*100:.1f}%")
        
    print("Evaluating Phase 6...")
    for arch in archetypes:
        score = evaluate_archetype("models/phase6_attacker.pth", arch, num_games=num_eval_games)
        p6_scores.append(score)
        print(f"  P6 vs {arch}: {score*100:.1f}%")
        
    p4_avg = sum(p4_scores) / len(p4_scores) if p4_scores else 0
    p6_avg = sum(p6_scores) / len(p6_scores) if p6_scores else 0
    
    print(f"\nFinal Averages -> Phase 4: {p4_avg*100:.1f}% | Phase 6: {p6_avg*100:.1f}%")
    
    # Generate Research-Grade Grouped Bar Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.5, 1]})
    
    x = np.arange(len(archetypes))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, [s*100 for s in p4_scores], width, label='Red Queen Self-Play', color='#e74c3c')
    rects2 = ax1.bar(x + width/2, [s*100 for s in p6_scores], width, label='Double History Archive', color='#2ecc71')
    
    ax1.set_title('Robustness Test: Win Rates by Defender Archetype', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['North Camper\n(Y=1)', 'Center Camper\n(Y=3)', 'South Camper\n(Y=5)', 'River Patroller\n(Dynamic)'], fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right')
    
    for bar in rects1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')
    for bar in rects2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')
        
    # Overall summary bar chart on the right
    avg_bars = ax2.bar(['Red Queen', 'Double Archive'], [p4_avg*100, p6_avg*100], color=['#e74c3c', '#2ecc71'], width=0.55)
    ax2.set_title('Overall Robustness', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Win Rate (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    
    for bar in avg_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/robustness_chart.png", dpi=180)
    print("Saved comprehensive robustness chart to docs/robustness_chart.png")

if __name__ == "__main__":
    main()
