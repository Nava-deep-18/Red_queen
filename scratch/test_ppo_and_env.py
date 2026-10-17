import sys
sys.path.insert(0, "c:/Red_queen/src")
import torch
import numpy as np
from environment import ArenaEnv
from agents.ppo import PPOAgent


def test_distance_potential_monotonicity():
    """Verify that distance potential decreases monotonically by 1.0 along all 3 bridge paths."""
    env = ArenaEnv()
    env.reset()

    for y in [1, 3, 5]:
        path = [(x, y) for x in range(1, 11)]
        dists = [env.calc_distance_potential(np.array(p)) for p in path]
        deltas = [dists[i] - dists[i + 1] for i in range(len(dists) - 1)]
        for i, delta in enumerate(deltas):
            assert delta == 1.0, f"Bridge Y={y} at step {path[i]}->{path[i+1]} delta is {delta}, expected 1.0"


def test_zero_sum_shaping_invariant():
    """Verify R_att + R_def == 0.0 identically at every timestep."""
    env = ArenaEnv()
    env.reset()

    for _ in range(100):
        old_att = env.attacker_pos.copy()
        old_def = env.defender_pos.copy()
        att_act = np.random.randint(0, 5)
        def_act = np.random.randint(0, 5)

        next_obs, rewards, term, trunc, info = env.step(att_act, def_act)
        att_shaping, def_shaping = env.compute_zero_sum_shaping(
            old_att, env.attacker_pos, old_def, env.defender_pos
        )
        total_att = rewards[0] + att_shaping
        total_def = rewards[1] + def_shaping

        assert abs(total_att + total_def) < 1e-6, f"Zero-sum violated: {total_att} + {total_def} != 0"

        if term or trunc:
            env.reset()


def test_ppo_masked_action_selection():
    """Verify that masked actions are strictly never chosen."""
    agent = PPOAgent(state_dim=6, action_dim=5)
    state = np.random.rand(6).astype(np.float32)

    # Allow only action 2
    mask = [False, False, True, False, False]
    for _ in range(30):
        act = agent.select_action(state, action_mask=mask)
        assert act == 2, f"Expected action 2, got {act}"

    pred = agent.predict(state, action_mask=mask)
    assert pred == 2, f"Predict expected action 2, got {pred}"


def test_ppo_update_and_gradient_clipping():
    """Verify that update() runs, computes advantages, clips gradients, and optimizes without NaN."""
    agent = PPOAgent(state_dim=6, action_dim=5, lr_actor=0.001, lr_critic=0.002)

    for i in range(25):
        s = np.random.rand(6).astype(np.float32)
        mask = [True, True, True, False, True]
        agent.select_action(s, action_mask=mask)
        done = (i == 24)
        agent.store_reward(1.0 if done else 0.0, done)

    # Ensure update runs smoothly
    agent.update()

    # Check buffer was cleared
    assert len(agent.states) == 0
    assert len(agent.rewards) == 0
    assert len(agent.masks) == 0

    # Ensure no NaN parameters
    for p in agent.policy.parameters():
        assert not torch.isnan(p).any(), "NaN found in policy parameters after update"


def test_backward_compatibility_checkpoint_loading():
    """Verify that existing checkpoints load without error and can predict."""
    agent = PPOAgent(state_dim=6, action_dim=5)
    sd = torch.load("models/phase6_attacker.pth")
    agent.policy.load_state_dict(sd)

    s = np.array([0.9, 0.5, 0.6, 0.5, 1.0, 0.5], dtype=np.float32)
    mask = [True, True, True, False, True]
    act = agent.predict(s, action_mask=mask)
    assert act in [0, 1, 2, 4], f"Chosen masked action {act} is invalid"


if __name__ == "__main__":
    print("Running tests...")
    test_distance_potential_monotonicity()
    print("  [PASS] test_distance_potential_monotonicity")
    test_zero_sum_shaping_invariant()
    print("  [PASS] test_zero_sum_shaping_invariant")
    test_ppo_masked_action_selection()
    print("  [PASS] test_ppo_masked_action_selection")
    test_ppo_update_and_gradient_clipping()
    print("  [PASS] test_ppo_update_and_gradient_clipping")
    test_backward_compatibility_checkpoint_loading()
    print("  [PASS] test_backward_compatibility_checkpoint_loading")
    print("ALL TESTS PASSED!")
