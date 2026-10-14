import numpy as np
import gymnasium as gym
import random


class ArenaEnv(gym.Env):
    """
    Two-player zero-sum arena: Attacker tries to reach the target;
    Defender tries to catch the Attacker.

    Interface note
    --------------
    step() accepts *two simultaneous actions* and returns *two rewards*.
    This is a **multi-agent** interface, not a single-agent one.
    The closest standard is PettingZoo's Parallel API
    (https://pettingzoo.farama.org/api/parallel/), which is designed for
    exactly this signature.  We inherit from gym.Env only to reuse its
    space definitions and random-number-generator seeding; the full
    single-agent Gymnasium contract is NOT satisfied.

    Signatures
    ----------
    reset(seed, options) -> (obs, info)
    step(att_action, def_action)
         -> (obs, (reward_att, reward_def), terminated, truncated, info)

    State vector (6 floats, all in [0, 1]):
      [ax, ay, dx, dy, tx, ty]
    Coordinates are normalised by (dim - 1) so corners map to exactly
    0.0 and 1.0.

    Terminal priority
    -----------------
    reached_target takes priority over caught.  If the attacker lands on
    the target and the defender simultaneously occupies the same cell, the
    outcome is an attacker win (target reached), not a defender win.
    This prevents dual win-counting in the training scripts.

    Capture rule
    ------------
    A capture is registered when both agents occupy the same cell after
    their moves, OR when they swap cells in a single step (crossing paths
    counts as a catch), UNLESS the attacker simultaneously reaches the
    target (see Terminal priority above).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, width: int = 11, height: int = 7, max_steps: int = 75):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        self.action_space = gym.spaces.Discrete(5)

        # Observation space: 6 normalised coordinates, each in [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

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

        # Internal state (populated by reset)
        self.attacker_pos: np.ndarray = np.zeros(2, dtype=int)
        self.defender_pos: np.ndarray = np.zeros(2, dtype=int)
        self.target_pos:   np.ndarray = np.zeros(2, dtype=int)
        self.obstacles:    list = []
        self.step_count:   int = 0
        # Extra attribute used by PATROLLER archetype in evaluate_robustness
        self.patrol_dir:   int = 1

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset the environment; return (observation, info)."""
        super().reset(seed=seed)   # seeds self.np_random for reproducibility
        self.step_count = 0
        self.patrol_dir = 1

        # Target: far-right, middle row
        self.target_pos = np.array([10, 3])

        # River at X=5; bridges at Y=1, 3, 5
        self.obstacles = [
            np.array([5, y]) for y in range(self.height) if y not in [1, 3, 5]
        ]

        self.attacker_pos = self._sample_attacker_spawn()
        self.defender_pos = self._sample_defender_spawn()

        # Guarantee no overlap (virtually impossible given disjoint X ranges)
        while np.array_equal(self.attacker_pos, self.defender_pos):
            self.defender_pos = self._sample_defender_spawn()

        return self._get_state(), {}

    def step(self, attacker_action: int, defender_action: int):
        """
        Advance one step.

        Returns
        -------
        obs        : np.ndarray  (6 floats in [0, 1])
        rewards    : tuple[float, float]   (attacker_reward, defender_reward)
        terminated : bool  – game event ended the episode (catch or target reached)
        truncated  : bool  – step limit reached without a terminal event
        info       : dict
        """
        self.step_count += 1

        # Remember old positions to detect crossing swaps
        old_att = self.attacker_pos.copy()
        old_def = self.defender_pos.copy()

        # Compute candidate new positions
        new_att = old_att + self.action_to_move[attacker_action]
        new_def = old_def + self.action_to_move[defender_action]

        # Apply moves (invalid → stay)
        if self._is_valid_pos(new_att):
            self.attacker_pos = new_att
        if self._is_valid_pos(new_def):
            self.defender_pos = new_def

        # --- Capture detection ---
        # 1. Same-tile: both agents end on the same cell
        same_tile = np.array_equal(self.attacker_pos, self.defender_pos)
        # 2. Crossing-swap: each agent moved into the other's previous cell
        crossing  = (
            np.array_equal(self.attacker_pos, old_def)
            and np.array_equal(self.defender_pos, old_att)
        )
        raw_caught = same_tile or crossing

        reached_target = np.array_equal(self.attacker_pos, self.target_pos)
        timeout = self.step_count >= self.max_steps

        # Terminal priority: reaching the target beats being caught.
        # If the attacker steps onto the target cell and the defender is
        # simultaneously there (or they swap to that cell), it is counted
        # as an attacker win only — preventing dual win-counting.
        caught = raw_caught and not reached_target

        terminated = caught or reached_target
        truncated  = (not terminated) and timeout

        # Zero-sum terminal rewards — exactly one branch fires per episode end
        reward_attacker = 0.0
        reward_defender = 0.0
        if reached_target:
            reward_attacker =  10.0
            reward_defender = -10.0
        elif caught:
            reward_attacker = -10.0
            reward_defender =  10.0

        info = {
            "caught":         caught,
            "reached_target": reached_target,
            "timeout":        timeout,
        }

        return self._get_state(), (reward_attacker, reward_defender), terminated, truncated, info

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        Return the normalised state vector [ax, ay, dx, dy, tx, ty].
        Divides x by (width-1) and y by (height-1) so every position in
        [0, width-1] × [0, height-1] maps to exactly [0, 1].
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
        """Attacker spawns on the left side (X: 0–3); uses self.np_random when seeded."""
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
        """Defender spawns on the right side (X: 6–7); uses self.np_random when seeded."""
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
        - Before river (x <= 4): Manhattan distance to the closest bridge entrance (x=4, y in {1,3,5}) + 6.
        - After river (x > 4): Manhattan distance to the Citadel target (10, 3).
        Potential Phi(pos) = -D(pos), so moving closer increases potential.
        """
        x, y = int(pos[0]), int(pos[1])
        if x <= 4:
            return float(min(abs(x - 4) + abs(y - by) for by in [1, 3, 5]) + 6)
        else:
            return float(abs(x - self.target_pos[0]) + abs(y - self.target_pos[1]))

    def compute_zero_sum_shaping(self, old_attacker_pos: np.ndarray, new_attacker_pos: np.ndarray, scale: float = 0.5) -> float:
        """
        Returns symmetric potential-based shaping reward F = (D_old - D_new) * scale.
        In a zero-sum game, attacker receives +F and defender receives -F,
        guaranteeing R_att + R_def == 0 at every transition (Ng et al. 1999).
        """
        old_dist = self.calc_distance_potential(old_attacker_pos)
        new_dist = self.calc_distance_potential(new_attacker_pos)
        return (old_dist - new_dist) * scale

