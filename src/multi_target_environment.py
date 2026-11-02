import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
import copy
import random
from typing import List, Tuple, Dict, Optional, Union


# Standard target configurations by target count K
PRESET_TARGETS: Dict[int, List[Tuple[int, int]]] = {
    1: [(10, 3)],
    2: [(10, 1), (10, 5)],
    3: [(10, 1), (10, 3), (10, 5)],
    5: [(10, 0), (10, 1), (10, 3), (10, 5), (10, 6)],
}

MAX_TARGETS: int = 5
# Observation vector: [ax, ay, dx, dy, (tx_i, ty_i, active_i) * MAX_TARGETS] -> 4 + 3 * 5 = 19
STATE_DIM: int = 4 + 3 * MAX_TARGETS


class MultiTargetArenaEnv(ParallelEnv):
    """
    Two-player zero-sum multi-target arena adhering to PettingZoo Parallel API:
    - Attacker attempts to reach ANY of K active targets on the Eastern front (X=10).
    - Defender attempts to intercept the Attacker before any target is breached.

    Target Configurations
    ---------------------
    Supports K in {1, 2, 3, 5} targets or custom coordinates:
      - K=1: [(10, 3)] (Standard Citadel baseline)
      - K=2: [(10, 1), (10, 5)] (North and South bridge corridors)
      - K=3: [(10, 1), (10, 3), (10, 5)] (Aligned with all 3 bridge exits)
      - K=5: [(10, 0), (10, 1), (10, 3), (10, 5), (10, 6)] (All obstacle-free tiles on row X=10)

    Observation Space (19D Box in [0, 1]):
    --------------------------------------
        [ax, ay, dx, dy,
         t0_x, t0_y, t0_active,
         t1_x, t1_y, t1_active,
         t2_x, t2_y, t2_active,
         t3_x, t3_y, t3_active,
         t4_x, t4_y, t4_active]
    Coordinates are normalized by (width - 1) and (height - 1).
    Inactive targets are padded with (0.0, 0.0, 0.0).

    Terminal Priority & Capture Rule
    --------------------------------
    - Target breach takes strict priority over capture: if the Attacker reaches any target
      on the same step that the Defender intercepts it, the Attacker wins (+10 / -10).
    - Capture occurs when both agents occupy the same tile, or swap tiles in a single step.
    - Zero-sum payoff: R_att + R_def == 0 at all times.
    """

    metadata = {
        "render_modes": ["ansi"],
        "name": "multi_target_arena_v0",
    }

    def __init__(
        self,
        width: int = 11,
        height: int = 7,
        max_steps: int = 75,
        num_targets: int = 1,
        target_positions: Optional[List[Tuple[int, int]]] = None,
    ):
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

        # Observation spaces: fixed 19D vector for up to MAX_TARGETS=5 targets
        self.observation_spaces = {
            agent: gym.spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Accessors
        self.action_space_sample = self.action_spaces["attacker"]
        self.observation_space_sample = self.observation_spaces["attacker"]

        # Actions -> movement deltas [dx, dy]
        self.action_to_move = {
            0: np.array([0, -1]),   # UP
            1: np.array([0,  1]),   # DOWN
            2: np.array([-1, 0]),   # LEFT
            3: np.array([1,  0]),   # RIGHT
            4: np.array([0,  0]),   # STAY
        }

        # Normalization factors
        self._norm_x = float(width - 1)
        self._norm_y = float(height - 1)

        # River at X=5; bridges at Y=1, 3, 5
        self.obstacles: List[np.ndarray] = [
            np.array([5, y]) for y in range(self.height) if y not in [1, 3, 5]
        ]

        # Target configuration initialization & validation
        self.num_targets = num_targets
        self.target_positions_config = target_positions
        self.targets: List[np.ndarray] = []
        self._init_targets(num_targets, target_positions)

        # Internal state
        self.attacker_pos: np.ndarray = np.zeros(2, dtype=int)
        self.defender_pos: np.ndarray = np.zeros(2, dtype=int)
        self.step_count: int = 0
        self.np_random = None

    def _init_targets(
        self,
        num_targets: int,
        target_positions: Optional[List[Tuple[int, int]]],
    ):
        """Validates and sets active target coordinates."""
        if target_positions is not None:
            if len(target_positions) == 0:
                raise ValueError("target_positions cannot be empty.")
            if len(target_positions) > MAX_TARGETS:
                raise ValueError(
                    f"target_positions length {len(target_positions)} exceeds MAX_TARGETS={MAX_TARGETS}"
                )
            coords = target_positions
        else:
            if num_targets not in PRESET_TARGETS:
                raise ValueError(
                    f"num_targets must be in {list(PRESET_TARGETS.keys())}, got {num_targets}"
                )
            coords = PRESET_TARGETS[num_targets]

        # Validate each coordinate
        seen = set()
        self.targets = []
        for c in coords:
            pos = np.array([int(c[0]), int(c[1])], dtype=int)
            if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
                raise ValueError(f"Target position {c} is out of arena bounds ({self.width}x{self.height}).")
            if any(np.array_equal(pos, obs) for obs in self.obstacles):
                raise ValueError(f"Target position {c} collides with river obstacle.")
            coord_tuple = (int(pos[0]), int(pos[1]))
            if coord_tuple in seen:
                raise ValueError(f"Duplicate target position {coord_tuple}.")
            seen.add(coord_tuple)
            self.targets.append(pos)

    def set_targets(self, targets: Union[int, List[Tuple[int, int]]]):
        """Dynamically reconfigure targets without recreating the environment."""
        if isinstance(targets, int):
            self.num_targets = targets
            self.target_positions_config = None
            self._init_targets(targets, None)
        else:
            self.num_targets = len(targets)
            self.target_positions_config = targets
            self._init_targets(len(targets), targets)

    def observation_space(self, agent: str) -> gym.spaces.Space:
        """PettingZoo required observation space getter."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        """PettingZoo required action space getter."""
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        """PettingZoo centralized global state getter."""
        return self._get_state()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the arena; return (observations, infos)."""
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, _ = seeding.np_random()

        self.agents = copy.copy(self.possible_agents)
        self.step_count = 0

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

        # Remember old positions for crossing-swap capture detection
        old_att = self.attacker_pos.copy()
        old_def = self.defender_pos.copy()

        # Compute candidate new positions
        new_att = old_att + self.action_to_move[attacker_action]
        new_def = old_def + self.action_to_move[defender_action]

        # Apply valid moves
        if self._is_valid_pos(new_att):
            self.attacker_pos = new_att
        if self._is_valid_pos(new_def):
            self.defender_pos = new_def

        # Capture detection:
        # 1. Same-tile: both agents end on identical coordinates
        same_tile = np.array_equal(self.attacker_pos, self.defender_pos)
        # 2. Crossing-swap: agents cross paths across adjacent tiles
        crossing = (
            np.array_equal(self.attacker_pos, old_def)
            and np.array_equal(self.defender_pos, old_att)
        )
        raw_caught = same_tile or crossing

        # Target reach detection: check against all active targets
        reached_target = False
        reached_target_idx = None
        reached_target_pos = None
        for idx, tgt in enumerate(self.targets):
            if np.array_equal(self.attacker_pos, tgt):
                reached_target = True
                reached_target_idx = idx
                reached_target_pos = (int(tgt[0]), int(tgt[1]))
                break

        # Terminal priority: Reaching ANY target takes strict precedence over capture
        caught = raw_caught and not reached_target
        terminated = caught or reached_target
        truncated = (not terminated) and self.step_count >= self.max_steps
        timeout = truncated

        # Terminal rewards: Zero-sum payoff
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
            "reached_target_idx": reached_target_idx,
            "reached_target_pos": reached_target_pos,
            "timeout": timeout,
            "num_targets": len(self.targets),
            "step_count": self.step_count,
        }

        obs_state = self._get_state()

        if is_legacy:
            return obs_state, (reward_attacker, reward_defender), terminated, truncated, info

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

    def _get_state(self) -> np.ndarray:
        """
        Return the 19D normalised state vector:
        [ax, ay, dx, dy, t0_x, t0_y, t0_act, ..., t4_x, t4_y, t4_act]
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = self.attacker_pos[0] / self._norm_x
        state[1] = self.attacker_pos[1] / self._norm_y
        state[2] = self.defender_pos[0] / self._norm_x
        state[3] = self.defender_pos[1] / self._norm_y

        for i, tgt in enumerate(self.targets):
            if i >= MAX_TARGETS:
                break
            base = 4 + i * 3
            state[base] = tgt[0] / self._norm_x
            state[base + 1] = tgt[1] / self._norm_y
            state[base + 2] = 1.0  # active indicator

        return state

    def get_state_single(self, target_idx: int = 0) -> np.ndarray:
        """
        Adapter returning the classic 6D state [ax, ay, dx, dy, tx, ty]
        for a specific target index, enabling direct compatibility with
        single-target 6D neural networks.
        """
        tgt = self.targets[target_idx] if target_idx < len(self.targets) else self.targets[0]
        return np.array([
            self.attacker_pos[0] / self._norm_x,
            self.attacker_pos[1] / self._norm_y,
            self.defender_pos[0] / self._norm_x,
            self.defender_pos[1] / self._norm_y,
            tgt[0] / self._norm_x,
            tgt[1] / self._norm_y,
        ], dtype=np.float32)

    def _is_valid_pos(self, pos: np.ndarray) -> bool:
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return False
        return not any(np.array_equal(pos, obs) for obs in self.obstacles)

    def _sample_attacker_spawn(self) -> np.ndarray:
        """Attacker spawns in the Western staging zone (X: 0-3)."""
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
        """Defender spawns in the Eastern sector (X: 6-9), not on any target."""
        while True:
            if self.np_random is not None:
                pos = np.array([
                    int(self.np_random.integers(6, 10)),
                    int(self.np_random.integers(0, self.height)),
                ])
            else:
                pos = np.array([random.randint(6, 9), random.randint(0, self.height - 1)])
            if self._is_valid_pos(pos) and not any(np.array_equal(pos, tgt) for tgt in self.targets):
                return pos

    def calc_distance_potential_to_target(self, pos: np.ndarray, target_pos: np.ndarray) -> float:
        """
        Calculates bridge-aware shortest Manhattan distance from pos to target_pos:
        - When pos is west of river (x <= 4): routes via the optimal bridge (4, by), by in {1, 3, 5}.
          dist = min(abs(x - 4) + abs(y - by) + (target_pos[0] - 4) + abs(by - target_pos[1]))
        - When pos is east of river (x > 4): direct Manhattan distance to target_pos.
        """
        x, y = int(pos[0]), int(pos[1])
        tx, ty = int(target_pos[0]), int(target_pos[1])
        if x <= 4:
            return float(min(
                abs(x - 4) + abs(y - by) + (tx - 4) + abs(by - ty)
                for by in [1, 3, 5]
            ))
        else:
            return float(abs(x - tx) + abs(y - ty))

    def calc_distance_potential(self, pos: np.ndarray) -> float:
        """
        Calculates the distance potential to the NEAREST active target.
        Potential is non-negative distance; lower distance = closer to victory.
        """
        if not self.targets:
            return 0.0
        return float(min(
            self.calc_distance_potential_to_target(pos, tgt)
            for tgt in self.targets
        ))

    def compute_zero_sum_shaping(
        self,
        old_attacker_pos: np.ndarray,
        new_attacker_pos: np.ndarray,
        old_defender_pos: Optional[np.ndarray] = None,
        new_defender_pos: Optional[np.ndarray] = None,
        scale_target: float = 0.3,
        scale_relative: float = 0.2,
        gamma: float = 0.99,
    ) -> Tuple[float, float]:
        """
        Multi-target zero-sum potential shaping:
        F_target = (D_nearest(old_att) - gamma * D_nearest(new_att)) * scale_target
        F_rel    = (D_rel(old) - gamma * D_rel(new)) * scale_relative

        Attacker reward = +F_target - F_rel
        Defender reward = -F_target + F_rel
        Guarantees R_att + R_def == 0.0 at every step (Ng et al. 1999).
        """
        old_dist_tgt = self.calc_distance_potential(old_attacker_pos)
        new_dist_tgt = self.calc_distance_potential(new_attacker_pos)
        f_target = (old_dist_tgt - gamma * new_dist_tgt) * scale_target

        f_rel = 0.0
        if old_defender_pos is not None and new_defender_pos is not None:
            old_rel = float(abs(old_attacker_pos[0] - old_defender_pos[0]) + abs(old_attacker_pos[1] - old_defender_pos[1]))
            new_rel = float(abs(new_attacker_pos[0] - new_defender_pos[0]) + abs(new_attacker_pos[1] - new_defender_pos[1]))
            f_rel = (old_rel - gamma * new_rel) * scale_relative

        att_shaping = f_target - f_rel
        def_shaping = -f_target + f_rel
        return att_shaping, def_shaping

    def render(self):
        """Render ASCII/Unicode view of the multi-target arena."""
        grid = [["⬜" for _ in range(self.width)] for _ in range(self.height)]
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = "🟦"
        for i, tgt in enumerate(self.targets):
            grid[tgt[1]][tgt[0]] = f"🎯"

        if np.array_equal(self.attacker_pos, self.defender_pos):
            grid[self.defender_pos[1]][self.defender_pos[0]] = "💥"
        else:
            grid[self.defender_pos[1]][self.defender_pos[0]] = "🛡️"
            grid[self.attacker_pos[1]][self.attacker_pos[0]] = "😈"

        print(f"\nMulti-Target Arena ({len(self.targets)} Targets) - Step {self.step_count}:")
        for row in grid:
            print(" ".join(row))
        print("-" * 30)
