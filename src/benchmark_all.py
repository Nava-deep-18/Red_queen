import os
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import ArenaEnv
from agents.ppo import PPOAgent


def run_matchup(attacker, defender, num_episodes=100, seeds=None):
    """
    Run paired episodes between an attacker and a defender.
    Returns: (att_win_rate, def_win_rate, timeout_rate, avg_steps)
    """
    env = ArenaEnv()
    if seeds is None:
        seeds = list(range(num_episodes))

    att_wins = 0
    def_wins = 0
    timeouts = 0
    steps_list = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        state = obs["attacker"]
        done = False
        step_count = 0

        while not done:
            step_count += 1
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            # Attacker decision
            if attacker == "random":
                att_action = int(np.random.randint(0, 5))
            else:
                att_action = attacker.predict(state, action_mask=att_mask)

            # Defender decision
            if defender == "stay":
                def_action = 4
            elif defender == "random":
                def_action = int(np.random.randint(0, 5))
            elif defender == "pursuit":
                dx = env.attacker_pos[0] - env.defender_pos[0]
                dy = env.attacker_pos[1] - env.defender_pos[1]
                if abs(dx) > abs(dy):
                    candidate = 3 if dx > 0 else 2
                else:
                    candidate = 1 if dy > 0 else 0
                def_action = candidate if env._is_valid_pos(env.defender_pos + env.action_to_move[candidate]) else 4
            elif defender == "north_camper":
                dy = env.defender_pos[1]
                def_action = 0 if dy > 1 else (1 if dy < 1 else 4)
            elif defender == "center_camper":
                dy = env.defender_pos[1]
                def_action = 0 if dy > 3 else (1 if dy < 3 else 4)
            elif defender == "south_camper":
                dy = env.defender_pos[1]
                def_action = 0 if dy > 5 else (1 if dy < 5 else 4)
            elif defender == "patroller":
                dy = env.defender_pos[1]
                if dy <= 1: env.patrol_dir = 1
                elif dy >= 5: env.patrol_dir = -1
                def_action = 1 if env.patrol_dir == 1 else 0
            else:
                def_action = defender.predict(state, action_mask=def_mask)

            next_state, _, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated
            state = next_state

        if info.get("reached_target", False):
            att_wins += 1
        elif info.get("caught", False):
            def_wins += 1
        elif info.get("timeout", False):
            timeouts += 1
        steps_list.append(step_count)

    n = len(seeds)
    return (
        att_wins / n,
        def_wins / n,
        timeouts / n,
        float(np.mean(steps_list)),
    )


def main():
    print("=" * 75)
    print("RED QUEEN MARL: SCIENTIFIC BENCHMARK SUITE")
    print("=" * 75)

    env = ArenaEnv()
    state_dim = 6
    action_dim = env.action_space("attacker").n
    seeds = list(range(100))

    # Helper to load agents
    def load_agent(path):
        if not os.path.exists(path):
            return None
        agent = PPOAgent(state_dim, action_dim)
        agent.policy.load_state_dict(torch.load(path, weights_only=True))
        agent.policy_old.load_state_dict(torch.load(path, weights_only=True))
        return agent

    p4_att = load_agent("models/phase4_attacker.pth")
    p4_def = load_agent("models/phase4_defender.pth")
    p6_att = load_agent("models/phase6_attacker.pth")
    p6_def = load_agent("models/phase6_defender.pth")
    p_imp_att = load_agent("models/improved_attacker.pth")
    p_imp_def = load_agent("models/improved_defender.pth")

    benchmark_table = []

    print("\n--- 1. Attacker Evaluation vs Standard Defender Baselines ---")
    attackers = [
        ("Phase 4 Attacker", p4_att),
        ("Phase 6 Attacker", p6_att),
        ("Improved PPO Attacker", p_imp_att),
    ]
    defenders = [
        ("STAY Defender", "stay"),
        ("RANDOM Defender", "random"),
        ("GREEDY PURSUIT Defender", "pursuit"),
        ("NORTH CAMPER (Y=1)", "north_camper"),
        ("CENTER CAMPER (Y=3)", "center_camper"),
        ("SOUTH CAMPER (Y=5)", "south_camper"),
        ("RIVER PATROLLER", "patroller"),
    ]

    for att_name, att_agent in attackers:
        if att_agent is None:
            print(f"Skipping {att_name} (checkpoint not found)")
            continue
        print(f"\nEvaluating: {att_name}")
        for def_name, def_rule in defenders:
            aw, dw, to, steps = run_matchup(att_agent, def_rule, seeds=seeds)
            print(f"  vs {def_name:<25s} -> Att Win: {aw*100:4.1f}% | Def Catch: {dw*100:4.1f}% | Timeout: {to*100:4.1f}% | Avg Steps: {steps:.1f}")
            benchmark_table.append((att_name, def_name, aw, dw, to, steps))

    print("\n--- 2. Learned Self-Play & Cross-Play Matchups ---")
    matchups = [
        ("Phase 4 Self-Play", p4_att, p4_def),
        ("Phase 6 Self-Play", p6_att, p6_def),
        ("Improved PPO Self-Play", p_imp_att, p_imp_def),
        ("Cross: P6 Att vs P4 Def", p6_att, p4_def),
        ("Cross: P4 Att vs P6 Def", p4_att, p6_def),
        ("Cross: Imp Att vs P6 Def", p_imp_att, p6_def),
    ]

    for m_name, att, dfe in matchups:
        if att is None or dfe is None:
            print(f"Skipping {m_name} (checkpoints not complete)")
            continue
        aw, dw, to, steps = run_matchup(att, dfe, seeds=seeds)
        print(f"  {m_name:<28s} -> Att Win: {aw*100:4.1f}% | Def Catch: {dw*100:4.1f}% | Timeout: {to*100:4.1f}% | Avg Steps: {steps:.1f}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
