import json
import os


class ReplayLogger:
    """Records agent positions step by step and saves them as JSON for the dashboards."""

    def __init__(self, filename, grid_size=(11, 7), obstacles=None, target=(10, 3)):
        self.filename = filename
        if obstacles is None:
            obstacles = [[5, 0], [5, 2], [5, 4], [5, 6]]
        self.data = {
            "grid_size": list(grid_size),
            "obstacles": [[int(o[0]), int(o[1])] for o in obstacles],
            "target": [int(target[0]), int(target[1])],
            "episodes": {},
        }
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)

    @classmethod
    def from_env(cls, filename, env):
        """Build a logger whose map (size, obstacles, target) is read from the environment."""
        env.reset(seed=0)
        return cls(
            filename,
            grid_size=(env.width, env.height),
            obstacles=env.obstacles,
            target=env.target_pos,
        )

    def log_step(self, episode_str, attacker_pos, defender_pos):
        self.data["episodes"].setdefault(episode_str, []).append({
            "attacker": [int(attacker_pos[0]), int(attacker_pos[1])],
            "defender": [int(defender_pos[0]), int(defender_pos[1])],
        })

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f)