import os
import sys
import json
import csv
import math
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Optional

# Ensure local imports work whether executed from root or src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment import ArenaEnv
from agents.ppo import PPOAgent


def calculate_wilson_ci(successes: int, total: int, z: float = 1.96) -> float:
    """Computes half-width of the 95% Wilson score confidence interval."""
    if total == 0:
        return 0.0
    p = successes / total
    denom = 1 + (z**2) / total
    spread = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denom
    return spread


def get_scripted_defender_action(defender_type: str, env: ArenaEnv) -> int:
    """Returns discrete action for fixed hand-coded defender archetypes in ArenaEnv."""
    dtype = defender_type.lower()
    dx, dy = env.defender_pos[0], env.defender_pos[1]
    ax, ay = env.attacker_pos[0], env.attacker_pos[1]

    if dtype == "stay":
        return 4
    elif dtype == "random":
        valid = [a for a in range(5) if env._is_valid_pos(env.defender_pos + env.action_to_move[a])]
        return int(np.random.choice(valid)) if valid else 4
    elif dtype in ("pursuit", "greedy_pursuit"):
        diff_x = ax - dx
        diff_y = ay - dy
        if abs(diff_x) > abs(diff_y):
            candidate = 3 if diff_x > 0 else 2
        else:
            candidate = 1 if diff_y > 0 else 0
        if env._is_valid_pos(env.defender_pos + env.action_to_move[candidate]):
            return candidate
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


def evaluate_single_target_matchup(
    env: ArenaEnv,
    attacker: Any,
    defender: Any,
    seeds: List[int],
) -> Dict[str, Any]:
    """Runs paired deterministic evaluation on the original single-target ArenaEnv."""
    att_wins = 0
    def_wins = 0
    timeouts = 0
    step_counts = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0

        while not done:
            step += 1
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            # Attacker greedy action
            if isinstance(attacker, str) and attacker.lower() == "random":
                valid = [a for a in range(5) if att_mask[a]]
                att_act = int(np.random.choice(valid)) if valid else 4
            else:
                att_act = attacker.predict(obs["attacker"], action_mask=att_mask)

            # Defender action
            if isinstance(defender, str):
                def_act = get_scripted_defender_action(defender, env)
            else:
                def_act = defender.predict(obs["defender"], action_mask=def_mask)

            # Step in original ArenaEnv
            obs, rewards, term, trunc, info = env.step({"attacker": att_act, "defender": def_act})
            is_term = term["attacker"] if isinstance(term, dict) else term
            is_trunc = trunc["attacker"] if isinstance(trunc, dict) else trunc
            done = bool(is_term or is_trunc)

        res_info = info["attacker"] if isinstance(info, dict) and "attacker" in info else info
        if res_info.get("reached_target", False):
            att_wins += 1
        elif res_info.get("caught", False):
            def_wins += 1
        elif res_info.get("timeout", False):
            timeouts += 1

        step_counts.append(step)

    n = len(seeds)
    return {
        "num_episodes": n,
        "target_pos": [10, 3],
        "attacker_win_rate": float(att_wins / n),
        "attacker_ci_95": float(calculate_wilson_ci(att_wins, n)),
        "defender_win_rate": float(def_wins / n),
        "defender_ci_95": float(calculate_wilson_ci(def_wins, n)),
        "timeout_rate": float(timeouts / n),
        "avg_episode_steps": float(np.mean(step_counts)),
        "std_episode_steps": float(np.std(step_counts)),
        "sem_episode_steps": float(np.std(step_counts) / math.sqrt(n)),
    }


def to_serializable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(to_serializable(k)): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj


def load_agent(path: str, state_dim: int = 6, action_dim: int = 5) -> Optional[PPOAgent]:
    if not os.path.exists(path):
        return None
    agent = PPOAgent(state_dim, action_dim)
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        agent.policy.load_state_dict(sd)
        agent.policy_old.load_state_dict(sd)
        agent.policy.eval()
        agent.policy_old.eval()
        return agent
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def run_single_target_benchmark(
    num_episodes: int = 500,
    att_model_path: str = "models/phase4_attacker.pth",
    def_model_path: str = "models/phase4_defender.pth",
    results_dir: str = "results/single_target",
    seed_base: int = 42,
):
    """Executes the standard 500-episode paired evaluation on original single-target ArenaEnv."""
    os.makedirs(results_dir, exist_ok=True)
    seeds = [seed_base + i for i in range(num_episodes)]

    print("=" * 80)
    print(f"  ORIGINAL SINGLE-TARGET BENCHMARK EVALUATION (ArenaEnv)")
    print(f"  Target: (10, 3) Citadel | Observation Space: 6D")
    print(f"  Attacker Model: {att_model_path}")
    print(f"  Defender Model: {def_model_path}")
    print(f"  Episodes: {num_episodes} | Seeds: [{seeds[0]} ... {seeds[-1]}]")
    print("=" * 80)

    env = ArenaEnv()
    attacker = load_agent(att_model_path, state_dim=6)
    defender = load_agent(def_model_path, state_dim=6)

    if attacker is None:
        raise FileNotFoundError(f"Attacker model not found: {att_model_path}")

    archetypes = [
        "learned_defender",
        "stay",
        "random",
        "pursuit",
        "north_camper",
        "center_camper",
        "south_camper",
        "patroller",
    ]

    all_results = {}
    csv_rows = []

    for arch in archetypes:
        def_opponent = defender if arch == "learned_defender" else arch
        if arch == "learned_defender" and defender is None:
            print(f"  [Skip] Learned defender {def_model_path} not found.")
            continue

        metrics = evaluate_single_target_matchup(env, attacker, def_opponent, seeds)
        all_results[arch] = metrics

        print(
            f"  Def: {arch:<18} | Att Win: {metrics['attacker_win_rate']*100:5.1f}% (±{metrics['attacker_ci_95']*100:4.1f}%) "
            f"| Def Win: {metrics['defender_win_rate']*100:5.1f}% | Timeout: {metrics['timeout_rate']*100:4.1f}% | Steps: {metrics['avg_episode_steps']:5.1f}"
        )

        csv_rows.append({
            "target_system": "original_single_target",
            "attacker_model": att_model_path,
            "defender_type": arch,
            "episodes": num_episodes,
            "att_win_rate": round(metrics["attacker_win_rate"], 4),
            "att_ci_95": round(metrics["attacker_ci_95"], 4),
            "def_win_rate": round(metrics["defender_win_rate"], 4),
            "def_ci_95": round(metrics["defender_ci_95"], 4),
            "timeout_rate": round(metrics["timeout_rate"], 4),
            "avg_steps": round(metrics["avg_episode_steps"], 2),
            "sem_steps": round(metrics["sem_episode_steps"], 3),
        })

    json_path = os.path.join(results_dir, "comparison_metrics.json")
    with open(json_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nSaved single-target results to {json_path}")

    csv_path = os.path.join(results_dir, "comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved single-target tabular metrics to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Original Single-Target Baseline System.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of paired test episodes")
    parser.add_argument("--att-model", type=str, default="models/phase4_attacker.pth",
                        help="Path to Phase 4 single-target attacker checkpoint")
    parser.add_argument("--def-model", type=str, default="models/phase4_defender.pth",
                        help="Path to Phase 4 single-target defender checkpoint")
    parser.add_argument("--results-dir", type=str, default="results/single_target",
                        help="Directory to save single-target results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for paired trials")
    args = parser.parse_args()

    run_single_target_benchmark(
        num_episodes=args.episodes,
        att_model_path=args.att_model,
        def_model_path=args.def_model,
        results_dir=args.results_dir,
        seed_base=args.seed,
    )


if __name__ == "__main__":
    main()
