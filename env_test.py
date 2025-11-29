import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
import pickle

class TestStockTradingEnv(gym.Env):

    def __init__(self, num_companies=5, num_stocks=1000, lookback_period=60, dataset_path='dataset.pkl'):
        # load dataset from pickle file
        self.dataset_path = dataset_path
        self.data = self._load_dataset()

        self.data_lrs = self.data['Log return']
        self.data_open = self.data['Open']

        # number of stocks to trade
        self.num_companies = num_companies
        self.num_stocks = num_stocks

        # set start date for the environment
        self.current_day_idx = lookback_period
        self.start_date = self.data.index[self.current_day_idx]
        
        self.lookback_period = lookback_period

        # array representing each stock's fraction of the total portfolio
        # initialize with equal allocation
        self._portfolio_allocation = np.array([1/self.num_companies] * self.num_companies)

        # define what the agent can observe
        # dictionary containing current portfolio allocation plus
        # stock log returns for the duration of the lookback period
        # plus volatility metrics (optional)
        self.observation_space = gym.spaces.Dict({
            'portfolio_allocation': gym.spaces.Box(low=0, high=1, shape=(self.num_companies,), dtype=np.float32),
            'stock_lrs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_companies, lookback_period), dtype=np.float32),
            'stock_opens': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_companies, lookback_period), dtype=np.float32),
            # 'volatility_metrics': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_companies, lookback_period), dtype=np.float32)
        })

        # define what actions are available
        # sum must be 1 (100% of portfolio)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_companies,), dtype=np.float32)

    
    def _get_obs(self):
        # return the current observation
        return {
            'portfolio_allocation': self._portfolio_allocation,
            'stock_lrs': [] if self.current_day_idx >= len(self.data) else self._get_stock_lrs(),
            'stock_opens': [] if self.current_day_idx >= len(self.data) else self._get_stock_opens(),
            'current_date': None if self.current_day_idx >= len(self.data) else self.data.index[self.current_day_idx]
            # 'volatility_metrics': self._get_volatility_metrics()
        }
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # reset portfolio allocation to equal distribution
        self._portfolio_allocation = np.array([1/self.num_companies] * self.num_companies)

        # reset day
        self.current_day_idx = self.lookback_period

        # reset data
        self.data_lrs = self._get_stock_lrs()
        self.data_open = self._get_stock_opens()

        # return initial observation
        return self._get_obs(), {}
    

    def step (self, action):
        # normalize action to ensure it sums to 1
        action = action / np.sum(action)

        # calculate reward based on change in portfolio value
        reward = self._calculate_reward(action)

        # update portfolio allocation
        self._portfolio_allocation = action

        # update date
        self.current_day_idx += 1

        # check if episode is done
        done = self.current_day_idx >= len(self.data)

        # return observation, reward, done, and info
        return self._get_obs(), reward, done, {}
    

    def _load_dataset(self):
        # load dataset from pickle file
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        return data


    def _get_stock_lrs(self):
        # get stock prices for the lookback period
        start_date = self.data.index[self.current_day_idx - self.lookback_period]
        end_date = self.data.index[self.current_day_idx - 1]

        return self.data_lrs[start_date:end_date]
    

    def _get_stock_opens(self):
        # get stock prices for the lookback period
        start_date = self.data.index[self.current_day_idx - self.lookback_period]
        end_date = self.data.index[self.current_day_idx - 1]

        return self.data_open[start_date:end_date]
    

    def _calculate_reward(self, new_allocation):
        # calculate portfolio return based on new allocation and stock opening prices
        # portfolio return for today
        today = self.data.index[self.current_day_idx]
        today_open = self.data['Open'][today:today]
        today_open = np.array(today_open)[0]

        yesterday = self.data.index[self.current_day_idx - 1]
        yesterday_open = self.data['Open'][yesterday:yesterday]
        yesterday_open = np.array(yesterday_open)[0]
        
        old_value = np.dot(np.rint(self._portfolio_allocation * self.num_stocks), yesterday_open)
        new_value = np.dot(np.rint(new_allocation * self.num_stocks), today_open)

        if today.day % 10 == 0:
            print(f"Date: {today}, Old Value: {old_value}, New Value: {new_value}, Reward: {new_value - old_value}")
            # print(len(self.data_lrs), len(self.data_open))

        reward = new_value - old_value
        return reward