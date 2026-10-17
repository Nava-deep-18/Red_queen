import sys
sys.path.insert(0, "c:/Red_queen/src")
import torch
import numpy as np
import random
from environment import ArenaEnv
from agents.ppo import PPOAgent


def evaluate_policy(att_agent, num_eval_episodes=50):
    env = ArenaEnv()
    opponents = {
        "Stay": lambda env: 4,
        "GreedyPursuit": lambda env: (
            0 if env.attacker_pos[1] < env.defender_pos[1] else (
                1 if env.attacker_pos[1] > env.defender_pos[1] else (
                    2 if env.attacker_pos[0] < env.defender_pos[0] else (
                        3 if env.attacker_pos[0] > env.defender_pos[0] else 4
                    )
                )
            )
        ),
        "NorthCamper": lambda env: (
            0 if env.defender_pos[1] > 1 else (1 if env.defender_pos[1] < 1 else 4)
        ),
        "CenterCamper": lambda env: (
            0 if env.defender_pos[1] > 3 else (1 if env.defender_pos[1] < 3 else 4)
        ),
        "SouthCamper": lambda env: (
            0 if env.defender_pos[1] > 5 else (1 if env.defender_pos[1] < 5 else 4)
        ),
    }

    results = {}
    for opp_name, opp_fn in opponents.items():
        wins, catches, timeouts = 0, 0, 0
        for ep in range(num_eval_episodes):
            obs, _ = env.reset(seed=ep * 100 + 7)
            state = obs["attacker"]
            done = False
            while not done:
                att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
                def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]
                att_act = att_agent.predict(state, action_mask=att_mask)
                def_act = opp_fn(env)
                if not def_mask[def_act]:
                    def_act = 4
                next_state, _, term, trunc, info = env.step(att_act, def_act)
                done = term or trunc
                state = next_state

            if info["reached_target"]:
                wins += 1
            elif info["caught"]:
                catches += 1
            else:
                timeouts += 1

        results[opp_name] = {
            "win_rate": wins / num_eval_episodes,
            "catch_rate": catches / num_eval_episodes,
            "timeout_rate": timeouts / num_eval_episodes,
        }
    return results


def train_marl(seed=42, num_episodes=500, update_every=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = ArenaEnv()
    state_dim = 6
    action_dim = env.action_space("attacker").n

    attacker = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)
    defender = PPOAgent(state_dim, action_dim, lr_actor=0.001, lr_critic=0.002)

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed * 10000 + episode)
        state = obs["attacker"]
        done = False

        while not done:
            att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
            def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

            att_action = attacker.select_action(state, action_mask=att_mask)
            def_action = defender.select_action(state, action_mask=def_mask)

            old_att = env.attacker_pos.copy()
            old_def = env.defender_pos.copy()

            next_state, rewards, terminated, truncated, info = env.step(att_action, def_action)
            done = terminated or truncated

            att_shaping, def_shaping = env.compute_zero_sum_shaping(
                old_att, env.attacker_pos, old_def, env.defender_pos
            )
            att_reward = rewards[0] + att_shaping
            def_reward = rewards[1] + def_shaping

            attacker.store_reward(att_reward, done)
            defender.store_reward(def_reward, done)
            state = next_state

        if (episode + 1) % update_every == 0:
            attacker.update()
            defender.update()

    attacker.update()
    defender.update()
    return attacker, defender


def main():
    seeds = [42, 123, 999]
    all_results = []

    print(f"Training across {len(seeds)} independent seeds (500 episodes each)...")
    for s in seeds:
        print(f"  Training seed {s}...")
        att, def_agent = train_marl(seed=s, num_episodes=500)
        eval_res = evaluate_policy(att, num_eval_episodes=50)
        all_results.append(eval_res)
        print(f"  Seed {s} evaluation vs SouthCamper: Win={eval_res['SouthCamper']['win_rate']*100:.1f}%, Timeout={eval_res['SouthCamper']['timeout_rate']*100:.1f}%")

    print("\n=== MULTI-SEED EVALUATION SUMMARY (Mean +/- Std) ===")
    opponents = ["Stay", "NorthCamper", "CenterCamper", "SouthCamper", "GreedyPursuit"]
    for opp in opponents:
        wrs = [r[opp]["win_rate"] * 100 for r in all_results]
        tos = [r[opp]["timeout_rate"] * 100 for r in all_results]
        mean_wr = np.mean(wrs)
        std_wr = np.std(wrs)
        mean_to = np.mean(tos)
        std_to = np.std(tos)
        print(f"  vs {opp:14s}: Win Rate = {mean_wr:5.1f}% +/- {std_wr:4.1f}% | Timeout = {mean_to:5.1f}% +/- {std_to:4.1f}%")

    # Save representative improved model
    torch.save(att.policy.state_dict(), "models/improved_attacker.pth")
    torch.save(def_agent.policy.state_dict(), "models/improved_defender.pth")
    print("\nSaved improved models to models/improved_attacker.pth and models/improved_defender.pth")


if __name__ == "__main__":
    main()
