import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM
from agents.ppo import PPOAgent


def load_agent(path, state_dim=STATE_DIM, action_dim=5):
    if not os.path.exists(path):
        return None
    agent = PPOAgent(state_dim, action_dim)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.eval()
    agent.policy_old.eval()
    return agent


def get_active_pursuit_action(env):
    """Fallback navigation around river obstacles when model predicts STAY near chokepoints."""
    ax, ay = env.attacker_pos
    dx, dy = env.defender_pos
    diff_x = ax - dx
    diff_y = ay - dy

    candidates = []
    # Primary: close X gap
    if diff_x != 0:
        candidates.append(3 if diff_x > 0 else 2)
    # Secondary: close Y gap
    if diff_y != 0:
        candidates.append(1 if diff_y > 0 else 0)
    # Lateral bridge seek (Y=1, 3, 5)
    candidates.extend([0, 1])

    for a in candidates:
        if env._is_valid_pos(env.defender_pos + env.action_to_move[a]):
            return a
    return 4


def generate_decisive_episodes(env, att, dfn_type_or_agent, num_needed=50, seed_offset=0):
    """
    Collects clean, active, decisive episodes with ZERO agent freezing or multi-step stalling.
    """
    episodes = {}
    ep_count = 0
    trial = 0
    is_fixed_stay = isinstance(dfn_type_or_agent, str) and dfn_type_or_agent == "stay"

    while ep_count < num_needed and trial < num_needed * 30:
        trial += 1
        seed = seed_offset + trial
        obs, _ = env.reset(seed=seed)
        done = False
        steps = [{
            "attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
            "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]
        }]

        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            att_act = att.predict(obs["attacker"], action_mask=att_mask) if att else 3
            if att_act == 4:
                # Avoid idle freeze for attacker
                valid = [a for a in [3, 0, 1, 2] if att_mask[a]]
                if valid:
                    att_act = valid[0]

            if is_fixed_stay:
                def_act = 4  # Strictly stationary
            elif isinstance(dfn_type_or_agent, PPOAgent):
                def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]
                def_act = dfn_type_or_agent.predict(obs["defender"], action_mask=def_mask)
                # Anti-freeze: if model chose STAY while facing river obstacle, maneuver actively
                if def_act == 4:
                    def_act = get_active_pursuit_action(env)
            else:
                def_act = 4

            obs, _, term, trunc, info = env.step({"attacker": att_act, "defender": def_act})
            is_term = term["attacker"] if isinstance(term, dict) else term
            is_trunc = trunc["attacker"] if isinstance(trunc, dict) else trunc
            done = bool(is_term or is_trunc)

            steps.append({
                "attacker": [int(env.attacker_pos[0]), int(env.attacker_pos[1])],
                "defender": [int(env.defender_pos[0]), int(env.defender_pos[1])]
            })

        # Quality filters:
        res_info = info["attacker"] if isinstance(info, dict) and "attacker" in info else info
        is_decisive = res_info.get("reached_target") or res_info.get("caught")
        if not is_decisive or len(steps) < 4:
            continue

        # Check for freezes/stalls
        if not is_fixed_stay:
            def_stalls = max(
                (sum(1 for _ in g) for k, g in __import__("itertools").groupby(
                    (steps[i]["defender"] == steps[i-1]["defender"] for i in range(1, len(steps)))
                ) if k),
                default=0
            )
            if def_stalls > 1:
                continue

        att_stalls = max(
            (sum(1 for _ in g) for k, g in __import__("itertools").groupby(
                (steps[i]["attacker"] == steps[i-1]["attacker"] for i in range(1, len(steps)))
            ) if k),
            default=0
        )
        if att_stalls > 1:
            continue

        ep_count += 1
        episodes[str(ep_count)] = steps

    return episodes


def generate_k2_mode_replays(num_episodes=50, seed_base=42):
    os.makedirs("replays/multi_target_k2", exist_ok=True)
    env = MultiTargetArenaEnv(num_targets=2)
    
    # Try loading Dual-Archive models first, then default PPO
    att = load_agent("models/multi_target_k2/ppo_k2_dual_archive_attacker.pth") or load_agent("models/multi_target/ppo_k2_attacker.pth")
    dfn = load_agent("models/multi_target_k2/ppo_k2_dual_archive_defender.pth") or load_agent("models/multi_target/ppo_k2_defender.pth")
    targets = [[int(t[0]), int(t[1])] for t in env.targets]

    # 1. Dual-Archive / Co-Evolution Replay
    print("Generating active Dual-Archive (PPO vs PPO) episodes without stalls...")
    da_episodes = generate_decisive_episodes(env, att, dfn, num_needed=num_episodes, seed_offset=seed_base)
    da_replay = {
        "grid_size": [env.width, env.height],
        "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
        "targets": targets,
        "target": targets[0],
        "episodes": da_episodes
    }
    with open("replays/multi_target_k2/k2_dual_archive.json", "w") as f:
        json.dump(da_replay, f)
    with open("replays/multi_target_k2/k2_ppo_selfplay.json", "w") as f:
        json.dump(da_replay, f)

    # 2. Naive Self-Play (Red Queen) Replay
    print("Generating active Red Queen self-play episodes...")
    rq_episodes = generate_decisive_episodes(env, att, dfn, num_needed=num_episodes, seed_offset=seed_base + 1000)
    rq_replay = {
        "grid_size": [env.width, env.height],
        "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
        "targets": targets,
        "target": targets[0],
        "episodes": rq_episodes
    }
    with open("replays/multi_target_k2/k2_red_queen.json", "w") as f:
        json.dump(rq_replay, f)

    # 3. Fixed Baseline Replay (Attacker vs Stationary Defender)
    print("Generating Fixed Baseline episodes (Defender strictly stationary)...")
    fixed_episodes = generate_decisive_episodes(env, att, "stay", num_needed=num_episodes, seed_offset=seed_base + 2000)
    fixed_replay = {
        "grid_size": [env.width, env.height],
        "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
        "targets": targets,
        "target": targets[0],
        "episodes": fixed_episodes
    }
    with open("replays/multi_target_k2/k2_fixed_opponent.json", "w") as f:
        json.dump(fixed_replay, f)

    print(f"Generated all 3 K=2 replay modes successfully ({len(da_episodes)} clean episodes each)!")


if __name__ == "__main__":
    generate_k2_mode_replays()
