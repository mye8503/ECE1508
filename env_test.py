import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
import pickle

class TestStockTradingEnv(gym.Env):

    def __init__(self, num_stocks=5, lookback_period=60, dataset_path='dataset.pkl'):
        # load dataset from pickle file
        self.dataset_path = dataset_path
        self.data = self._load_dataset()

        self.data_lrs = self.data['Log return']
        self.data_open = self.data['Open']

        # number of stocks to trade
        self.num_stocks = num_stocks

        # set start date for the environment
        day = pd.Timedelta(days=1)
        self.start_date = self.data.index[0] + day * lookback_period
        self.current_date = self.start_date

        self.lookback_period = lookback_period

        # array representing each stock's fraction of the total portfolio
        # initialize with equal allocation
        self._portfolio_allocation = np.array([1/self.num_stocks] * self.num_stocks)

        # define what the agent can observe
        # dictionary containing current portfolio allocation plus
        # stock log returns for the duration of the lookback period
        # plus volatility metrics (optional)
        self.observation_space = gym.spaces.Dict({
            'portfolio_allocation': gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32),
            'stock_lrs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks, lookback_period), dtype=np.float32),
            'stock_opens': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_stocks, lookback_period), dtype=np.float32),
            # 'volatility_metrics': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_stocks, lookback_period), dtype=np.float32)
        })

        # define what actions are available
        # sum must be 1 (100% of portfolio)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)

    
    def _get_obs(self):
        # return the current observation
        return {
            'portfolio_allocation': self._portfolio_allocation,
            'stock_lrs': self._get_stock_lrs(),
            'stock_opens': self._get_stock_opens(),
            'current_date': self.current_date
            # 'volatility_metrics': self._get_volatility_metrics()
        }
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # reset portfolio allocation to equal distribution
        self._portfolio_allocation = np.array([1/self.num_stocks] * self.num_stocks)

        # reset log returns
        self.data_lrs = self._get_stock_lrs()

        self.current_date = self.start_date

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
        self.increment_date()

        # check if episode is done
        done = self.current_date is None 

        # return observation, reward, done, and info
        return self._get_obs(), reward, done, {}
    

    def _load_dataset(self):
        print("hello", self.dataset_path)
        # load dataset from pickle file
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        return data


    def _get_stock_lrs(self):
        # get stock prices for the lookback period
        start_date = self.subtract_dates(self.current_date, self.lookback_period)

        print("hello", start_date, self.current_date)
        # print(self.data_lrs[start_date:current_date])
        return self.data_lrs[start_date:self.current_date]
    

    def _get_stock_opens(self):
        # get stock prices for the lookback period
        start_date = self.subtract_dates(self.current_date, self.lookback_period)

        print("hello", start_date, self.current_date)
        # print(self.data_open[start_date:current_date])
        return self.data_open[start_date:self.current_date]
    

    def _calculate_reward(self, new_allocation):
        # calculate portfolio return based on new allocation and stock opening prices
        # portfolio return for today
        today_open = self.data_open[self.current_date:self.current_date]
        today_open = np.array(today_open)[0]

        yesterday = self.subtract_dates(self.current_date, 1)
        yesterday_open = self.data_open[yesterday:yesterday]
        yesterday_open = np.array(yesterday_open)[0]

        old_value = np.dot(self._portfolio_allocation, yesterday_open)
        new_value = np.dot(new_allocation, today_open)
        reward = new_value - old_value
        return reward
    

    def subtract_dates(self, date, num_days):
        day = pd.Timedelta(days=1)

        return date - num_days * day


    def increment_date(self):
        day = pd.Timedelta(days=1)

        self.current_date += day

        # check if this is an omitted date
        while self.current_date not in self.data_open.index:
            self.current_date += day

            # check if we have finished the dataset
            if self.current_date > self.data_open.index[-1]:
                self.current_date = None
                return  

