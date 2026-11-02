import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environment import ArenaEnv
from pettingzoo.test import parallel_api_test


def test_arena_parallel_api():
    """Verify that ArenaEnv strictly adheres to Farama PettingZoo ParallelEnv API."""
    env = ArenaEnv()
    parallel_api_test(env, num_cycles=50)
    print("Passed Farama PettingZoo Parallel API test.")


if __name__ == "__main__":
    test_arena_parallel_api()
