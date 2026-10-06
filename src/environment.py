import numpy as np
import gymnasium as gym

class ArenaEnv:
    def __init__(self, width=7, height=5, max_steps=50):
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
        
        # Simple obstacles list (x, y)
        self.obstacles = [np.array([3, 1]), np.array([3, 4])]
        self.target_pos = np.array([6, 0])
        
        # Randomize Attacker and Defender positions
        import random
        def get_random_empty_pos():
            while True:
                pos = np.array([random.randint(0, self.width-1), random.randint(0, self.height-1)])
                # Check if it's on an obstacle or target
                if np.array_equal(pos, self.target_pos):
                    continue
                is_obstacle = False
                for obs in self.obstacles:
                    if np.array_equal(pos, obs):
                        is_obstacle = True
                if not is_obstacle:
                    return pos
                    
        self.attacker_pos = get_random_empty_pos()
        self.defender_pos = get_random_empty_pos()
        
        # Ensure they don't spawn on each other
        while np.array_equal(self.attacker_pos, self.defender_pos):
            self.defender_pos = get_random_empty_pos()
        
        return self._get_state()

    def _get_state(self):
        # Flattened state vector: [ax, ay, dx, dy, tx, ty]
        # In a real DL model, we'd normalize these coordinates
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
        grid = [['·' for _ in range(self.width)] for _ in range(self.height)]
        
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = '🧱'
            
        grid[self.target_pos[1]][self.target_pos[0]] = '🎯'
        
        # If they overlap, defender is on top (caught)
        if np.array_equal(self.attacker_pos, self.defender_pos):
            grid[self.defender_pos[1]][self.defender_pos[0]] = '💥'
        else:
            grid[self.defender_pos[1]][self.defender_pos[0]] = '🔵'
            grid[self.attacker_pos[1]][self.attacker_pos[0]] = '🔴'
            
        print("\nArena State:")
        for row in grid:
            print(" ".join(row))
        print("-------------")
