import gymnasium as gym

from src.envs.tokenSwap_env import TokenSwapEnv


class candidateDistanceWrapper(gym.Wrapper):
    def __init__(self, env: TokenSwapEnv):
        super().__init__(env)

        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.env.node_num, self.env.node_num),
            dtype=np.float32,
        )
