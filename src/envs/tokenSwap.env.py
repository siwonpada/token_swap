from typing import Optional, Any
import numpy as np
import networkx as nx
import gymnasium as gym


class TokenSwapEnv(gym.Env):
    def __init__(self, node_num: int) -> None:
        super().__init__()
        self.node_num = node_num
        self.action_space = gym.spaces.Box(
            low=0, high=self.node_num - 1, shape=(2,), dtype=np.int32
        )  # Two actions: swap or hold
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
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0  # Initial state
        return self.state, {}

    def step(
        self, action: tuple[int, int]
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        super().step(action)
        if action == 0:  # Swap
            self.state = 1
        else:  # Hold
            self.state = 0
        return self.state, 0.0, False, False, {}
