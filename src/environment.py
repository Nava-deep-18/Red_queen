import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
import copy
import random


class ArenaEnv(ParallelEnv):
    """
    Two-player zero-sum arena implemented with Farama PettingZoo's Parallel API:
    - Attacker tries to reach the Citadel target at (10, 3).
    - Defender tries to intercept (catch) the Attacker before it reaches the Citadel.

    Standard Multi-Agent Interface
    ------------------------------
    Inherits from `pettingzoo.ParallelEnv`, the gold-standard API for simultaneous-action
    Markov games. Both agents act in parallel at every timestep.

    Signatures (PettingZoo Parallel API)
    ------------------------------------
    reset(seed=None, options=None)
        -> (observations: dict[str, np.ndarray], infos: dict[str, dict])

    step(actions: dict[str, int])
        -> (observations: dict[str, np.ndarray],
            rewards: dict[str, float],
            terminations: dict[str, bool],
            truncations: dict[str, bool],
            infos: dict[str, dict])

    Backward Compatibility
    ----------------------
    `step(att_action, def_action)` is also supported when two integers are passed,
    returning: `(obs, (reward_att, reward_def), terminated, truncated, info)`.

    State Vector (6 floats in [0, 1]):
        [ax, ay, dx, dy, tx, ty]
    Coordinates are normalised by (dim - 1) so corners map to exactly 0.0 and 1.0.

    Terminal Priority & Capture Rule
    --------------------------------
    - Reached Citadel target takes priority over caught.
    - Catch occurs when both agents occupy the same tile, or swap tiles in a single step (crossing paths).
    """

    metadata = {
        "render_modes": ["ansi"],
        "name": "three_bridges_arena_v0",
    }

    def __init__(self, width: int = 11, height: int = 7, max_steps: int = 75):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Multi-agent identity list
        self.possible_agents = ["attacker", "defender"]
        self.agents = copy.copy(self.possible_agents)

        # Action spaces: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        self.action_spaces = {
            agent: gym.spaces.Discrete(5) for agent in self.possible_agents
        }

        # Observation spaces: 6 normalised coordinates in [0, 1]
        self.observation_spaces = {
            agent: gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Convenient single-space accessors
        self.action_space_sample = self.action_spaces["attacker"]
        self.observation_space_sample = self.observation_spaces["attacker"]

        # Actions → movement deltas
        self.action_to_move = {
            0: np.array([0, -1]),   # UP
            1: np.array([0,  1]),   # DOWN
            2: np.array([-1, 0]),   # LEFT
            3: np.array([1,  0]),   # RIGHT
            4: np.array([0,  0]),   # STAY
        }

        # Normalisation denominators: use (dim - 1) so corners map to 0/1 exactly
        self._norm_x = float(width - 1)
        self._norm_y = float(height - 1)

        # Internal state
        self.attacker_pos: np.ndarray = np.zeros(2, dtype=int)
        self.defender_pos: np.ndarray = np.zeros(2, dtype=int)
        self.target_pos:   np.ndarray = np.zeros(2, dtype=int)
        self.obstacles:    list = []
        self.step_count:   int = 0
        self.patrol_dir:   int = 1
        self.np_random = None

    def observation_space(self, agent: str) -> gym.spaces.Space:
        """PettingZoo required observation space getter."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        """PettingZoo required action space getter."""
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        """PettingZoo centralized global state getter."""
        return self._get_state()

    def reset(self, seed: int = None, options: dict = None):
        """Reset the arena; return (observations, infos)."""
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, _ = seeding.np_random()

        self.agents = copy.copy(self.possible_agents)
        self.step_count = 0
        self.patrol_dir = 1

        # Citadel Target: far-right, middle row (10, 3)
        self.target_pos = np.array([10, 3])

        # River at X=5; bridges at Y=1, 3, 5
        self.obstacles = [
            np.array([5, y]) for y in range(self.height) if y not in [1, 3, 5]
        ]

        self.attacker_pos = self._sample_attacker_spawn()
        self.defender_pos = self._sample_defender_spawn()

        while np.array_equal(self.attacker_pos, self.defender_pos):
            self.defender_pos = self._sample_defender_spawn()

        obs_state = self._get_state()
        observations = {
            "attacker": obs_state.copy(),
            "defender": obs_state.copy(),
        }
        infos = {
            "attacker": {},
            "defender": {},
        }
        return observations, infos

    def step(self, actions, defender_action=None):
        """
        Advance one step. Supports both PettingZoo dictionary calls and legacy positional calls.

        PettingZoo call:
            step({"attacker": 3, "defender": 0})
            -> (observations, rewards, terminations, truncations, infos)

        Legacy positional call:
            step(att_action, def_action)
            -> (obs, (reward_att, reward_def), terminated, truncated, info)
        """
        is_legacy = defender_action is not None or not isinstance(actions, dict)

        if is_legacy:
            attacker_action = int(actions)
            defender_action = int(defender_action)
        else:
            attacker_action = int(actions.get("attacker", 4))
            defender_action = int(actions.get("defender", 4))

        self.step_count += 1

        # Remember old positions to detect crossing swaps
        old_att = self.attacker_pos.copy()
        old_def = self.defender_pos.copy()

        # Compute candidate new positions
        new_att = old_att + self.action_to_move[attacker_action]
        new_def = old_def + self.action_to_move[defender_action]

        # Apply moves (invalid moves bounce/stay in place)
        if self._is_valid_pos(new_att):
            self.attacker_pos = new_att
        if self._is_valid_pos(new_def):
            self.defender_pos = new_def

        # Capture detection:
        # 1. Same-tile: both agents end on the same cell
        same_tile = np.array_equal(self.attacker_pos, self.defender_pos)
        # 2. Crossing-swap: agents cross paths by occupying each other's previous cells
        crossing = (
            np.array_equal(self.attacker_pos, old_def)
            and np.array_equal(self.defender_pos, old_att)
        )
        raw_caught = same_tile or crossing
        reached_target = np.array_equal(self.attacker_pos, self.target_pos)

        # Terminal priority: Citadel target takes priority over being caught
        caught = raw_caught and not reached_target
        terminated = caught or reached_target
        truncated = (not terminated) and self.step_count >= self.max_steps
        timeout = truncated

        # Zero-sum terminal rewards: exactly one branch fires on termination
        reward_attacker = 0.0
        reward_defender = 0.0
        if reached_target:
            reward_attacker = 10.0
            reward_defender = -10.0
        elif caught:
            reward_attacker = -10.0
            reward_defender = 10.0

        info = {
            "caught": caught,
            "reached_target": reached_target,
            "timeout": timeout,
        }

        obs_state = self._get_state()

        if is_legacy:
            return obs_state, (reward_attacker, reward_defender), terminated, truncated, info

        # PettingZoo Parallel API return structure
        observations = {
            "attacker": obs_state.copy(),
            "defender": obs_state.copy(),
        }
        rewards = {
            "attacker": reward_attacker,
            "defender": reward_defender,
        }
        terminations = {
            "attacker": terminated,
            "defender": terminated,
        }
        truncations = {
            "attacker": truncated,
            "defender": truncated,
        }
        infos = {
            "attacker": info.copy(),
            "defender": info.copy(),
        }

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = [["⬜" for _ in range(self.width)] for _ in range(self.height)]
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = "🟦"
        grid[self.target_pos[1]][self.target_pos[0]] = "🎯"
        if np.array_equal(self.attacker_pos, self.defender_pos):
            grid[self.defender_pos[1]][self.defender_pos[0]] = "💥"
        else:
            grid[self.defender_pos[1]][self.defender_pos[0]] = "🛡️"
            grid[self.attacker_pos[1]][self.attacker_pos[0]] = "😈"
        print("\nArena State:")
        for row in grid:
            print(" ".join(row))
        print("-------------")

    def _get_state(self) -> np.ndarray:
        """
        Return the normalised state vector [ax, ay, dx, dy, tx, ty].
        Divides x by (width-1) and y by (height-1) so every position maps to [0, 1].
        """
        return np.array([
            self.attacker_pos[0] / self._norm_x,
            self.attacker_pos[1] / self._norm_y,
            self.defender_pos[0] / self._norm_x,
            self.defender_pos[1] / self._norm_y,
            self.target_pos[0]   / self._norm_x,
            self.target_pos[1]   / self._norm_y,
        ], dtype=np.float32)

    def _is_valid_pos(self, pos: np.ndarray) -> bool:
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return False
        return not any(np.array_equal(pos, obs) for obs in self.obstacles)

    def _sample_attacker_spawn(self) -> np.ndarray:
        """Attacker spawns on the left side (X: 0–3); uses self.np_random when available."""
        while True:
            if self.np_random is not None:
                pos = np.array([
                    int(self.np_random.integers(0, 4)),
                    int(self.np_random.integers(0, self.height)),
                ])
            else:
                pos = np.array([random.randint(0, 3), random.randint(0, self.height - 1)])
            if self._is_valid_pos(pos):
                return pos

    def _sample_defender_spawn(self) -> np.ndarray:
        """Defender spawns on the right side (X: 6–7); uses self.np_random when available."""
        while True:
            if self.np_random is not None:
                pos = np.array([
                    int(self.np_random.integers(6, 8)),
                    int(self.np_random.integers(0, self.height)),
                ])
            else:
                pos = np.array([random.randint(6, 7), random.randint(0, self.height - 1)])
            if not np.array_equal(pos, self.target_pos) and self._is_valid_pos(pos):
                return pos

    def calc_distance_potential(self, pos: np.ndarray) -> float:
        """
        Calculates symmetric, bridge-unbiased distance potential D(x, y).
        - Before river (x <= 4): Shortest path Manhattan distance to the Citadel target (10, 3)
          routing through the optimal bridge entrance (4, by) where by in {1, 3, 5}:
          dist = min(abs(x - 4) + abs(y - by) + 6 + abs(by - target_y) for by in {1, 3, 5}).
          This guarantees exact monotonicity (step delta = 1.0) along all three bridge paths.
        - After river (x > 4): Manhattan distance directly to Citadel target (10, 3).
        Potential Phi(pos) = -D(pos), so moving closer increases potential.
        """
        x, y = int(pos[0]), int(pos[1])
        target_y = int(self.target_pos[1]) if self.target_pos is not None else 3
        if x <= 4:
            return float(min(abs(x - 4) + abs(y - by) + 6 + abs(by - target_y) for by in [1, 3, 5]))
        else:
            return float(abs(x - self.target_pos[0]) + abs(y - self.target_pos[1]))

    def compute_zero_sum_shaping(
        self,
        old_attacker_pos: np.ndarray,
        new_attacker_pos: np.ndarray,
        old_defender_pos: np.ndarray = None,
        new_defender_pos: np.ndarray = None,
        scale_target: float = 0.3,
        scale_relative: float = 0.2,
    ) -> tuple[float, float]:
        """
        Calculates strictly zero-sum potential-based reward shaping:
        1. Target advance potential: Attacker gains +F_target for moving toward Citadel;
           Defender receives -F_target (opposing objective: prevent breach).
        2. Relative distance potential: Defender gains +F_rel for closing distance to Attacker;
           Attacker receives -F_rel (opposing objective: evade interception).

        Guarantees R_att + R_def == 0 identically at every timestep:
        - att_shaping = +F_target - F_rel
        - def_shaping = -F_target + F_rel
        Sum = 0.0 identically (Ng et al. 1999; Devlin & Kudenko 2011).
        """
        old_dist_tgt = self.calc_distance_potential(old_attacker_pos)
        new_dist_tgt = self.calc_distance_potential(new_attacker_pos)
        f_target = (old_dist_tgt - new_dist_tgt) * scale_target

        f_rel = 0.0
        if old_defender_pos is not None and new_defender_pos is not None:
            old_rel = float(abs(old_attacker_pos[0] - old_defender_pos[0]) + abs(old_attacker_pos[1] - old_defender_pos[1]))
            new_rel = float(abs(new_attacker_pos[0] - new_defender_pos[0]) + abs(new_attacker_pos[1] - new_defender_pos[1]))
            f_rel = (old_rel - new_rel) * scale_relative

        att_shaping = f_target - f_rel
        def_shaping = -f_target + f_rel
        return att_shaping, def_shaping
