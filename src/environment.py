import numpy as np
import gymnasium as gym
import random

class ArenaEnv:
    def __init__(self, width=11, height=7, max_steps=75):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        self.action_space = gym.spaces.Discrete(5)
        
        # Actions to movement mapping
        self.action_to_move = {
            0: np.array([0, -1]), # UP
            1: np.array([0, 1]),  # DOWN
            2: np.array([-1, 0]), # LEFT
            3: np.array([1, 0]),  # RIGHT
            4: np.array([0, 0])   # STAY
        }
        
        self.reset()

    def reset(self):
        self.step_count = 0
        
        # Target at far right, middle height
        self.target_pos = np.array([10, 3])
        
        # Build the River and 3 Bridges
        # River is at X=5. Bridges are at Y=1, 3, 5
        self.obstacles = []
        for y in range(self.height):
            if y not in [1, 3, 5]: # If not a bridge, it's an obstacle
                self.obstacles.append(np.array([5, y]))
        
        # Attacker spawns on the left side (X from 0 to 3)
        def get_attacker_spawn():
            while True:
                pos = np.array([random.randint(0, 3), random.randint(0, self.height-1)])
                if not any(np.array_equal(pos, obs) for obs in self.obstacles):
                    return pos
                    
        # Defender spawns on the right side patrolling the river (X from 6 to 7)
        def get_defender_spawn():
            while True:
                pos = np.array([random.randint(6, 7), random.randint(0, self.height-1)])
                if not np.array_equal(pos, self.target_pos) and not any(np.array_equal(pos, obs) for obs in self.obstacles):
                    return pos

        self.attacker_pos = get_attacker_spawn()
        self.defender_pos = get_defender_spawn()
        
        # Ensure no overlap (virtually impossible given different X ranges, but safe)
        while np.array_equal(self.attacker_pos, self.defender_pos):
            self.defender_pos = get_defender_spawn()
        
        return self._get_state()

    def _get_state(self):
        # Flattened state vector: [ax, ay, dx, dy, tx, ty]
        # Normalized by width/height
        return np.array([
            self.attacker_pos[0] / self.width, self.attacker_pos[1] / self.height,
            self.defender_pos[0] / self.width, self.defender_pos[1] / self.height,
            self.target_pos[0] / self.width, self.target_pos[1] / self.height
        ], dtype=np.float32)

    def _is_valid_pos(self, pos):
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return False
        for obs in self.obstacles:
            if np.array_equal(pos, obs):
                return False
        return True

    def step(self, attacker_action, defender_action):
        self.step_count += 1
        
        # Calculate new positions
        new_att_pos = self.attacker_pos + self.action_to_move[attacker_action]
        new_def_pos = self.defender_pos + self.action_to_move[defender_action]
        
        # Apply moves if valid, else stay
        if self._is_valid_pos(new_att_pos):
            self.attacker_pos = new_att_pos
        if self._is_valid_pos(new_def_pos):
            self.defender_pos = new_def_pos

        # Check win/loss conditions
        caught = np.array_equal(self.attacker_pos, self.defender_pos)
        reached_target = np.array_equal(self.attacker_pos, self.target_pos)
        timeout = self.step_count >= self.max_steps
        
        done = caught or reached_target or timeout
        
        # Calculate rewards (Zero-Sum)
        reward_attacker = 0.0
        reward_defender = 0.0
        
        if reached_target:
            reward_attacker = 10.0
            reward_defender = -10.0
        elif caught:
            reward_attacker = -10.0
            reward_defender = 10.0
            
        info = {
            'caught': caught,
            'reached_target': reached_target,
            'timeout': timeout
        }
            
        return self._get_state(), (reward_attacker, reward_defender), done, info

    def render(self):
        grid = [['⬜' for _ in range(self.width)] for _ in range(self.height)]
        
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = '🟦'
            
        grid[self.target_pos[1]][self.target_pos[0]] = '🎯'
        
        # If they overlap, defender is on top (caught)
        if np.array_equal(self.attacker_pos, self.defender_pos):
            grid[self.defender_pos[1]][self.defender_pos[0]] = '💥'
        else:
            grid[self.defender_pos[1]][self.defender_pos[0]] = '🛡️'
            grid[self.attacker_pos[1]][self.attacker_pos[0]] = '😈'
            
        print("\nArena State:")
        for row in grid:
            print(" ".join(row))
        print("-------------")
