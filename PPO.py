from ppo_env import StockTradingEnv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# env = TestStockTradingEnv(n_env=4)
dataset_path='data/train/2005-2009.pkl'
lookback = 60

env_kwargs = {'lookback-period': lookback, 
              'dataset_path': dataset_path}

env = make_vec_env(StockTradingEnv, n_env = 5, )

# env = StockTradingEnv(lookback_period=lookback, dataset_path='data/train/2005-2009.pkl')
# print(env.data)

# print(env.current_day_idx)
# print(len(env._get_stock_lrs()))
# print(env._get_obs())

# ppo = PPO('MlpPolicy', env=env)

# state,_ = env.reset()

# print(state)

# print(ppo.predict(state))

