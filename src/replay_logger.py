import json
import os

class ReplayLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            "grid_size": [11, 7],
            "obstacles": [[5, 0], [5, 2], [5, 4], [5, 6]],
            "target": [10, 3],
            "episodes": {}
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def log_step(self, episode_str, attacker_pos, defender_pos):
        if episode_str not in self.data["episodes"]:
            self.data["episodes"][episode_str] = []
            
        self.data["episodes"][episode_str].append({
            "attacker": [int(attacker_pos[0]), int(attacker_pos[1])],
            "defender": [int(defender_pos[0]), int(defender_pos[1])]
        })

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)
