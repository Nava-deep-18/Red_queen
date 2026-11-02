import unittest
import numpy as np
import gymnasium as gym
from src.environment import ArenaEnv
from src.multi_target_environment import MultiTargetArenaEnv, PRESET_TARGETS, STATE_DIM, MAX_TARGETS


class TestMultiTargetEnvironment(unittest.TestCase):
    """Unit test suite for MultiTargetArenaEnv."""

    def test_preset_initialization(self):
        """Test initialization of preset target configurations K in {1, 2, 3, 5}."""
        for k in [1, 2, 3, 5]:
            env = MultiTargetArenaEnv(num_targets=k)
            self.assertEqual(len(env.targets), k)
            expected_positions = [np.array(p) for p in PRESET_TARGETS[k]]
            for actual, expected in zip(env.targets, expected_positions):
                np.testing.assert_array_equal(actual, expected)

    def test_custom_targets(self):
        """Test initialization with custom valid target positions."""
        custom = [(10, 2), (10, 4)]
        env = MultiTargetArenaEnv(target_positions=custom)
        self.assertEqual(len(env.targets), 2)
        np.testing.assert_array_equal(env.targets[0], np.array([10, 2]))
        np.testing.assert_array_equal(env.targets[1], np.array([10, 4]))

    def test_invalid_target_validation(self):
        """Test that invalid target positions raise appropriate ValueErrors."""
        # Out of bounds
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(11, 3)])
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(-1, 3)])
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(10, 7)])

        # River obstacle collision (e.g. (5, 0) or (5, 2))
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(5, 0)])
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(5, 2)])

        # Duplicate targets
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(10, 3), (10, 3)])

        # Empty targets
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[])

        # Exceeds MAX_TARGETS
        with self.assertRaises(ValueError):
            MultiTargetArenaEnv(target_positions=[(10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5)])

    def test_dynamic_set_targets(self):
        """Test set_targets reconfigures targets correctly."""
        env = MultiTargetArenaEnv(num_targets=1)
        self.assertEqual(len(env.targets), 1)

        env.set_targets(3)
        self.assertEqual(len(env.targets), 3)

        env.set_targets([(10, 0), (10, 6)])
        self.assertEqual(len(env.targets), 2)
        np.testing.assert_array_equal(env.targets[0], np.array([10, 0]))
        np.testing.assert_array_equal(env.targets[1], np.array([10, 6]))

    def test_observation_space_and_reset(self):
        """Test PettingZoo observation space shape and normalized bounds [0, 1]."""
        env = MultiTargetArenaEnv(num_targets=3)
        obs, infos = env.reset(seed=42)

        self.assertIn("attacker", obs)
        self.assertIn("defender", obs)
        self.assertEqual(obs["attacker"].shape, (STATE_DIM,))
        self.assertEqual(obs["defender"].shape, (STATE_DIM,))

        # All values should be within [0.0, 1.0]
        self.assertTrue(np.all(obs["attacker"] >= 0.0))
        self.assertTrue(np.all(obs["attacker"] <= 1.0))

        # Check target active flags
        # Targets 0, 1, 2 are active (slots 6, 9, 12)
        self.assertEqual(obs["attacker"][4 + 0 * 3 + 2], 1.0)
        self.assertEqual(obs["attacker"][4 + 1 * 3 + 2], 1.0)
        self.assertEqual(obs["attacker"][4 + 2 * 3 + 2], 1.0)
        # Targets 3, 4 are inactive (slots 15, 18)
        self.assertEqual(obs["attacker"][4 + 3 * 3 + 2], 0.0)
        self.assertEqual(obs["attacker"][4 + 4 * 3 + 2], 0.0)

    def test_single_target_adapter(self):
        """Test 6D adapter for backward-compatible evaluation."""
        env = MultiTargetArenaEnv(num_targets=2)
        env.reset(seed=123)
        env.attacker_pos = np.array([2, 3])
        env.defender_pos = np.array([8, 3])

        state6d_0 = env.get_state_single(target_idx=0)
        self.assertEqual(state6d_0.shape, (6,))
        self.assertAlmostEqual(state6d_0[0], 2.0 / 10.0)
        self.assertAlmostEqual(state6d_0[1], 3.0 / 6.0)
        self.assertAlmostEqual(state6d_0[2], 8.0 / 10.0)
        self.assertAlmostEqual(state6d_0[3], 3.0 / 6.0)
        self.assertAlmostEqual(state6d_0[4], 10.0 / 10.0)
        self.assertAlmostEqual(state6d_0[5], 1.0 / 6.0)

        state6d_1 = env.get_state_single(target_idx=1)
        self.assertAlmostEqual(state6d_1[5], 5.0 / 6.0)

    def test_dual_step_api(self):
        """Test PettingZoo dictionary step vs legacy positional step."""
        env = MultiTargetArenaEnv(num_targets=1)
        env.reset(seed=10)

        # Legacy positional call: step(att, def)
        obs, rewards, term, trunc, info = env.step(3, 4)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(rewards, tuple)
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(term, (bool, np.bool_))

        # PettingZoo dictionary call: step({"attacker": a, "defender": d})
        env.reset(seed=10)
        pz_obs, pz_rewards, pz_term, pz_trunc, pz_infos = env.step({"attacker": 3, "defender": 4})
        self.assertIsInstance(pz_obs, dict)
        self.assertIn("attacker", pz_obs)
        self.assertIn("defender", pz_obs)
        self.assertIn("attacker", pz_rewards)
        self.assertIn("defender", pz_rewards)

    def test_attacker_reach_any_target_wins(self):
        """Test that reaching ANY active target yields an attacker victory (+10 / -10)."""
        env = MultiTargetArenaEnv(num_targets=3)
        # Targets are (10, 1), (10, 3), (10, 5)
        for target_idx, target_pos in enumerate(env.targets):
            env.reset()
            # Place attacker one tile to the left of the target
            env.attacker_pos = np.array([target_pos[0] - 1, target_pos[1]])
            env.defender_pos = np.array([0, 0])  # Far away

            obs, rewards, term, trunc, info = env.step(3, 4)  # Attacker moves RIGHT (3), Defender STAY (4)
            self.assertTrue(term)
            self.assertTrue(info["reached_target"])
            self.assertFalse(info["caught"])
            self.assertEqual(info["reached_target_idx"], target_idx)
            self.assertEqual(rewards[0], 10.0)
            self.assertEqual(rewards[1], -10.0)

    def test_defender_capture_same_tile(self):
        """Test that same-tile interception yields defender victory (-10 / +10)."""
        env = MultiTargetArenaEnv(num_targets=2)
        env.reset()
        env.attacker_pos = np.array([2, 2])
        env.defender_pos = np.array([2, 3])

        # Defender moves UP (0), Attacker STAY (4)
        obs, rewards, term, trunc, info = env.step(4, 0)
        self.assertTrue(term)
        self.assertTrue(info["caught"])
        self.assertFalse(info["reached_target"])
        self.assertEqual(rewards[0], -10.0)
        self.assertEqual(rewards[1], 10.0)

    def test_defender_capture_crossing_swap(self):
        """Test that crossing swap interception yields defender victory (-10 / +10)."""
        env = MultiTargetArenaEnv(num_targets=2)
        env.reset()
        env.attacker_pos = np.array([2, 2])
        env.defender_pos = np.array([3, 2])

        # Attacker moves RIGHT (3), Defender moves LEFT (2) -> swap!
        obs, rewards, term, trunc, info = env.step(3, 2)
        self.assertTrue(term)
        self.assertTrue(info["caught"])
        self.assertFalse(info["reached_target"])
        self.assertEqual(rewards[0], -10.0)
        self.assertEqual(rewards[1], 10.0)

    def test_priority_target_breach_over_capture(self):
        """Test strict priority: Attacker reaching target takes precedence even if Defender steps on same tile."""
        env = MultiTargetArenaEnv(num_targets=2)
        env.reset()
        # Target 0 is at (10, 1)
        # Attacker at (9, 1) moving RIGHT (3) into (10, 1)
        # Defender at (10, 2) moving UP (0) into (10, 1)
        env.attacker_pos = np.array([9, 1])
        env.defender_pos = np.array([10, 2])

        obs, rewards, term, trunc, info = env.step(3, 0)
        self.assertTrue(term)
        self.assertTrue(info["reached_target"])
        self.assertFalse(info["caught"])
        self.assertEqual(rewards[0], 10.0)
        self.assertEqual(rewards[1], -10.0)

    def test_zero_sum_potential_shaping(self):
        """Test that potential shaping rewards are identically zero-sum (R_att + R_def == 0.0)."""
        env = MultiTargetArenaEnv(num_targets=3)
        env.reset(seed=999)

        test_positions = [
            (np.array([2, 3]), np.array([3, 3]), np.array([8, 3]), np.array([7, 3])),
            (np.array([4, 1]), np.array([5, 1]), np.array([6, 1]), np.array([5, 1])),
            (np.array([7, 2]), np.array([8, 2]), np.array([9, 5]), np.array([9, 4])),
        ]

        for old_att, new_att, old_def, new_def in test_positions:
            att_shape, def_shape = env.compute_zero_sum_shaping(old_att, new_att, old_def, new_def)
            self.assertAlmostEqual(att_shape + def_shape, 0.0, places=7)

    def test_1target_behavioral_equivalence_with_baseline(self):
        """Test that K=1 MultiTargetArenaEnv matches single-target ArenaEnv exactly."""
        base_env = ArenaEnv()
        base_env.reset()
        multi_env = MultiTargetArenaEnv(num_targets=1)
        multi_env.reset()

        # Baseline Citadel is at (10, 3)
        np.testing.assert_array_equal(base_env.target_pos, multi_env.targets[0])

        # Test distance potentials across grid
        for x in [0, 2, 4, 6, 8]:
            for y in [1, 3, 5]:
                pos = np.array([x, y])
                base_dist = base_env.calc_distance_potential(pos)
                multi_dist = multi_env.calc_distance_potential(pos)
                self.assertEqual(base_dist, multi_dist)


if __name__ == "__main__":
    unittest.main()
