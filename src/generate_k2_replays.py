import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM
from agents.ppo import PPOAgent
from evaluate_multi_target import get_scripted_defender_action

def load_agent(path, state_dim=STATE_DIM, action_dim=5):
    agent = PPOAgent(state_dim, action_dim)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.eval()
    agent.policy_old.eval()
    return agent

def generate_k2_mode_replays(num_episodes=50, seed_base=42):
    os.makedirs("replays/multi_target_k2", exist_ok=True)
    env = MultiTargetArenaEnv(num_targets=2)
    att = load_agent("models/multi_target/ppo_k2_attacker.pth")
    dfn = load_agent("models/multi_target/ppo_k2_defender.pth")
    targets = [[int(t[0]), int(t[1])] for t in env.targets]

    # 1. Dual-Archive / Co-Evolution Replay (PPO Attacker vs PPO Defender)
    da_replay = {
        "grid_size": [env.width, env.height],
        "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
        "targets": targets,
        "target": targets[0],
        "episodes": {}
    }
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        steps = [{"attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                  "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]}]
        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]
            att_act = att.predict(obs["attacker"], action_mask=att_mask)
            def_act = dfn.predict(obs["defender"], action_mask=def_mask)
            obs, _, term, trunc, _ = env.step({"attacker": att_act, "defender": def_act})
            is_term = term["attacker"] if isinstance(term, dict) else term
            is_trunc = trunc["attacker"] if isinstance(trunc, dict) else trunc
            done = bool(is_term or is_trunc)
            steps.append({"attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                          "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]})
        da_replay["episodes"][str(ep)] = steps

    with open("replays/multi_target_k2/k2_dual_archive.json", "w") as f:
        json.dump(da_replay, f)
    with open("replays/multi_target_k2/k2_ppo_selfplay.json", "w") as f:
        json.dump(da_replay, f)

    # 2. Naive Self-Play (Red Queen) Replay
    rq_replay = {
        "grid_size": [env.width, env.height],
        "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
        "targets": targets,
        "target": targets[0],
        "episodes": {}
    }
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed_base + 1000 + ep)
        done = False
        steps = [{"attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                  "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]}]
        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]
            att_act = att.predict(obs["attacker"], action_mask=att_mask)
            def_act = dfn.predict(obs["defender"], action_mask=def_mask)
            obs, _, term, trunc, _ = env.step({"attacker": att_act, "defender": def_act})
            is_term = term["attacker"] if isinstance(term, dict) else term
            is_trunc = trunc["attacker"] if isinstance(trunc, dict) else trunc
            done = bool(is_term or is_trunc)
            steps.append({"attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                          "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]})
        rq_replay["episodes"][str(ep)] = steps

    with open("replays/multi_target_k2/k2_red_queen.json", "w") as f:
        json.dump(rq_replay, f)

    # 3. Fixed Baseline Replay (Attacker vs Stationary / Camping Baseline Defender)
    fixed_replay = {
        "grid_size": [env.width, env.height],
        "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
        "targets": targets,
        "target": targets[0],
        "episodes": {}
    }
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed_base + 2000 + ep)
        done = False
        steps = [{"attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                  "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]}]
        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            att_act = att.predict(obs["attacker"], action_mask=att_mask)
            def_act = get_scripted_defender_action("center_camper", env)
            obs, _, term, trunc, _ = env.step({"attacker": att_act, "defender": def_act})
            is_term = term["attacker"] if isinstance(term, dict) else term
            is_trunc = trunc["attacker"] if isinstance(trunc, dict) else trunc
            done = bool(is_term or is_trunc)
            steps.append({"attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                          "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]})
        fixed_replay["episodes"][str(ep)] = steps

    with open("replays/multi_target_k2/k2_fixed_opponent.json", "w") as f:
        json.dump(fixed_replay, f)

    print("Generated all 3 K=2 replay modes successfully!")

if __name__ == "__main__":
    generate_k2_mode_replays()
