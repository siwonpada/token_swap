from typing import Optional, Any
import numpy as np
import gymnasium as gym
import random

from src.utils.generateGraph import generate_random_graph


class TokenSwapEnv(gym.Env):
    def __init__(self, node_num: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self.node_num = node_num
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.action_space = gym.spaces.Box(
            low=0, high=self.node_num - 1, shape=(2,), dtype=np.int32
        )  # Select two nodes to swap
        self.observation_space = gym.spaces.Dict(
            {
                "graph": gym.spaces.Graph(
                    edge_space=None,
                    node_space=gym.spaces.Box(
                        low=0, high=1, shape=(1,), dtype=np.int32
                    ),
                ),
                "current_map": gym.spaces.Box(
                    low=0,
                    high=self.node_num - 1,
                    shape=(self.node_num,),
                    dtype=np.int32,
                ),
                "final_map": gym.spaces.Box(
                    low=0,
                    high=self.node_num - 1,
                    shape=(self.node_num,),
                    dtype=np.int32,
                ),
            }
        )
        return

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.graph = generate_random_graph(
            self.node_num, random.random(), seed=random.randint(0, 10000)
        )
        self.initial_map = np.arange(self.node_num)
        self.current_map = self.initial_map.copy()
        self.final_map = np.random.permutation(self.initial_map)
        return self.observation(), {}

    def step(
        self, action: tuple[int, int]
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        error = self.swap(action[0], action[1])
        done = np.array_equal(self.current_map, self.final_map)
        if done:
            reward = 1.0
        else:
            reward = -0.1  # Penalty for each step taken
        if error:
            reward -= 0.5  # Penalty for invalid swap
        return self.observation(), reward, done, False, {}

    def observation(self) -> dict[str, Any]:
        return {
            "graph": gym.spaces.GraphInstance(
                nodes=np.array(list(self.graph)),
                edge_links=np.array([e for e in self.graph.edges]),
                edges=None,
            ),
            "current_map": self.current_map,
            "final_map": self.final_map,
        }

    def swap(self, u: int, v: int) -> bool:
        """Swap the positions of nodes u and v in the current map."""
        if not self.graph.has_edge(u, v):  # Ensure the edge exists
            return True  # Cannot swap if there is an edge between them
        self.current_map[u], self.current_map[v] = (
            self.current_map[v],
            self.current_map[u],
        )
        return False  # Successful swap


if __name__ == "__main__":
    env = TokenSwapEnv(node_num=4, seed=100)
    obs, _ = env.reset()
    print("Initial Observation:", obs)
    while True:
        action = (int(input("first node: ")), int(input("second node: ")))
        obs, reward, done, _, _ = env.step(action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done)
