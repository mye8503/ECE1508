from ppo_env import TestStockTradingEnv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# env = TestStockTradingEnv(n_env=4)
lookback = 60

env = TestStockTradingEnv(lookback_period=lookback)

# ppo = PPO('MlpPolicy', env=env)

state,_ = env.reset()

print(state)

# print(ppo.predict(state))

