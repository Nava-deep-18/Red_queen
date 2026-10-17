import sys
sys.path.insert(0, "c:/Red_queen/src")
import torch
import numpy as np
from environment import ArenaEnv
from agents.ppo import PPOAgent


def run_deterministic_scenario(att_start, def_start, att_agent, def_agent, max_steps=20):
    env = ArenaEnv(max_steps=max_steps)
    env.reset()
    env.attacker_pos = np.array(att_start, dtype=int)
    env.defender_pos = np.array(def_start, dtype=int)

    trajectory = []
    done = False
    step = 0
    while not done and step < max_steps:
        state = env._get_state()
        att_mask = [env._is_valid_pos(env.attacker_pos + env.action_to_move[a]) for a in range(5)]
        def_mask = [env._is_valid_pos(env.defender_pos + env.action_to_move[a]) for a in range(5)]

        att_act = att_agent.predict(state, action_mask=att_mask)
        def_act = def_agent.predict(state, action_mask=def_mask)

        trajectory.append((env.attacker_pos.copy(), env.defender_pos.copy()))
        _, _, term, trunc, info = env.step(att_act, def_act)
        done = term or trunc
        step += 1

    trajectory.append((env.attacker_pos.copy(), env.defender_pos.copy()))
    outcome = "ATTACKER_WIN" if info["reached_target"] else ("DEFENDER_CATCH" if info["caught"] else "TIMEOUT")
    return outcome, step, trajectory


def main():
    p4_att = PPOAgent(6, 5)
    p4_def = PPOAgent(6, 5)
    p6_att = PPOAgent(6, 5)
    p6_def = PPOAgent(6, 5)

    p4_att.policy.load_state_dict(torch.load("models/phase4_attacker.pth"))
    p4_def.policy.load_state_dict(torch.load("models/phase4_defender.pth"))
    p6_att.policy.load_state_dict(torch.load("models/phase6_attacker.pth"))
    p6_def.policy.load_state_dict(torch.load("models/phase6_defender.pth"))

    scenarios = [
        ("A. Open Path to Citadel", [8, 3], [6, 1]),
        ("B. Close Interception",   [8, 3], [8, 4]),
        ("C. North Bridge Crossing",[4, 1], [8, 5]),
        ("D. South Bridge Crossing",[4, 5], [8, 1]),
    ]

    print("=== DETERMINISTIC CONTROLLED SCENARIOS ===")
    for name, att_s, def_s in scenarios:
        print(f"\n--- {name} (Att: {att_s}, Def: {def_s}) ---")
        out4, steps4, traj4 = run_deterministic_scenario(att_s, def_s, p4_att, p4_def)
        out6, steps6, traj6 = run_deterministic_scenario(att_s, def_s, p6_att, p6_def)
        print(f"  Phase 4: Outcome={out4:14s} Steps={steps4}")
        print(f"           Attacker traj: {[p[0].tolist() for p in traj4[:6]]}")
        print(f"  Phase 6: Outcome={out6:14s} Steps={steps6}")
        print(f"           Attacker traj: {[p[0].tolist() for p in traj6[:6]]}")


if __name__ == "__main__":
    main()
