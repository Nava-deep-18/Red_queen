import os
import sys
import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union

# Ensure imports work regardless of execution location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM
from agents.ppo import PPOAgent


def calculate_wilson_ci(successes: int, total: int, z: float = 1.96) -> float:
    """
    Computes half-width of the 95% Wilson Score confidence interval for proportions.
    Provides robust bounds even near 0.0 and 1.0.
    """
    if total == 0:
        return 0.0
    p = successes / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denom
    return spread


def get_scripted_defender_action(defender_type: str, env: MultiTargetArenaEnv) -> int:
    """
    Generates action for fixed scripted defender archetypes.
    """
    dtype = defender_type.lower()
    dx, dy = env.defender_pos[0], env.defender_pos[1]
    ax, ay = env.attacker_pos[0], env.attacker_pos[1]

    if dtype == "stay":
        return 4

    elif dtype == "random":
        valid_actions = [a for a in range(5) if env._is_valid_pos(env.defender_pos + env.action_to_move[a])]
        return int(np.random.choice(valid_actions)) if valid_actions else 4

    elif dtype == "pursuit" or dtype == "greedy_pursuit":
        diff_x = ax - dx
        diff_y = ay - dy
        if abs(diff_x) > abs(diff_y):
            candidate = 3 if diff_x > 0 else 2
        else:
            candidate = 1 if diff_y > 0 else 0
        if env._is_valid_pos(env.defender_pos + env.action_to_move[candidate]):
            return candidate
        # Fallback to perpendicular move
        alt = 1 if diff_y > 0 else 0
        if env._is_valid_pos(env.defender_pos + env.action_to_move[alt]):
            return alt
        return 4

    elif dtype == "north_camper":
        return 0 if dy > 1 else (1 if dy < 1 else 4)

    elif dtype == "center_camper":
        return 0 if dy > 3 else (1 if dy < 3 else 4)

    elif dtype == "south_camper":
        return 0 if dy > 5 else (1 if dy < 5 else 4)

    elif dtype == "patroller":
        if not hasattr(env, "patrol_dir"):
            env.patrol_dir = 1
        if dy <= 1:
            env.patrol_dir = 1
        elif dy >= 5:
            env.patrol_dir = -1
        return 1 if env.patrol_dir == 1 else 0

    else:
        return 4


def evaluate_matchup(
    env: MultiTargetArenaEnv,
    attacker: Any,
    defender: Any,
    num_episodes: int = 500,
    seeds: Optional[List[int]] = None,
    is_attacker_6d: bool = False,
) -> Dict[str, Any]:
    """
    Executes deterministic paired evaluation episodes between Attacker and Defender.

    Parameters
    ----------
    env : MultiTargetArenaEnv
        Environment configured with K targets.
    attacker : PPOAgent or str ('random')
        Attacker policy. If PPOAgent, uses deterministic predict().
    defender : PPOAgent or str ('stay', 'random', 'pursuit', 'north_camper', etc.)
        Defender policy.
    num_episodes : int
        Number of paired benchmark episodes.
    seeds : list of int, optional
        Predefined seed list for fair paired trial control.
    is_attacker_6d : bool
        If True, passes 6D state representation via env.get_state_single(0)
        to evaluate zero-shot transfer of single-target baseline policy.

    Returns
    -------
    dict of scientific metrics including win rates, CIs, duration, target distribution.
    """
    if seeds is None:
        seeds = [42 + i for i in range(num_episodes)]

    att_wins = 0
    def_wins = 0
    timeouts = 0
    step_counts = []
    target_hit_counts = {i: 0 for i in range(len(env.targets))}
    bridge_crossings = {"north": 0, "center": 0, "south": 0}

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        crossed_bridge = None

        while not done:
            step += 1
            old_ax = env.attacker_pos[0]
            old_ay = env.attacker_pos[1]

            # Track bridge crossings (moving from X=4 to X=5 or X=5 to X=6)
            if old_ax == 4 and env.attacker_pos[0] == 4:
                # Agent on threshold
                pass

            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            # Attacker action
            if isinstance(attacker, str) and attacker.lower() == "random":
                valid = [a for a in range(5) if att_mask[a]]
                att_act = int(np.random.choice(valid)) if valid else 4
            else:
                if is_attacker_6d:
                    state_vec = env.get_state_single(0)
                else:
                    state_vec = obs["attacker"]
                att_act = attacker.predict(state_vec, action_mask=att_mask)

            # Defender action
            if isinstance(defender, str):
                def_act = get_scripted_defender_action(defender, env)
            else:
                def_act = defender.predict(obs["defender"], action_mask=def_mask)

            # Step
            obs, rewards, term, trunc, info = env.step({"attacker": att_act, "defender": def_act})
            is_term = term["attacker"] if isinstance(term, dict) else term
            is_trunc = trunc["attacker"] if isinstance(trunc, dict) else trunc
            done = bool(is_term or is_trunc)

            if old_ax <= 4 and env.attacker_pos[0] >= 5 and crossed_bridge is None:
                bridge_y = env.attacker_pos[1]
                if bridge_y == 1:
                    crossed_bridge = "north"
                elif bridge_y == 3:
                    crossed_bridge = "center"
                elif bridge_y == 5:
                    crossed_bridge = "south"

        if crossed_bridge:
            bridge_crossings[crossed_bridge] += 1

        res_info = info["attacker"] if isinstance(info, dict) and "attacker" in info else info
        if res_info.get("reached_target", False):
            att_wins += 1
            hit_idx = res_info.get("reached_target_idx")
            if hit_idx is not None and hit_idx in target_hit_counts:
                target_hit_counts[hit_idx] += 1
        elif res_info.get("caught", False):
            def_wins += 1
        elif res_info.get("timeout", False):
            timeouts += 1

        step_counts.append(step)

    n = len(seeds)
    att_wr = att_wins / n
    def_wr = def_wins / n
    timeout_r = timeouts / n
    avg_steps = float(np.mean(step_counts))
    std_steps = float(np.std(step_counts))
    sem_steps = float(std_steps / math.sqrt(n))

    att_ci = calculate_wilson_ci(att_wins, n)
    def_ci = calculate_wilson_ci(def_wins, n)

    return {
        "num_episodes": n,
        "k_targets": len(env.targets),
        "target_coords": [tuple(t) for t in env.targets],
        "attacker_win_rate": att_wr,
        "attacker_ci_95": att_ci,
        "defender_win_rate": def_wr,
        "defender_ci_95": def_ci,
        "timeout_rate": timeout_r,
        "avg_episode_steps": avg_steps,
        "std_episode_steps": std_steps,
        "sem_episode_steps": sem_steps,
        "target_breach_distribution": target_hit_counts,
        "bridge_crossing_distribution": bridge_crossings,
    }
