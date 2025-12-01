import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
import pickle

class StockTradingEnv(gym.Env):

    def __init__(self, investment=100000, lookback_period=60, 
                 eta=1/252, dataset_path='dataset.pkl'):
        # load dataset from pickle file
        self.dataset_path = dataset_path
        self.data = self._load_dataset()


        self.data_lrs = self.data['Log return']
        # self.data_open = self.data['Open']

        # number of assets in the portfolio
        self.num_companies = len(self.data.columns.levels[1])

        # set start date for the environment
        self.current_day_idx = lookback_period - 1 # account for indexing starting from 0
        self.start_date = self.data.index[self.current_day_idx]
        
        self.budget = investment
        self.portfolio_val = investment

        self.lookback_period = lookback_period
        self.eta = eta # used in reward

        # array representing each stock's fraction of the total portfolio
        # initialize with equal allocation
        self._portfolio_allocation = np.array([1/self.num_companies] * self.num_companies)

        # define what the agent can observe
        # matrix containing the asset weights and log returns of the assets over the look back period
        # plus volatility metrics (optional)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(self.num_companies, self.lookback_period), dtype=np.float32)

        # define what actions are available
        # sum must be 1 (100% of portfolio)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_companies,), dtype=np.float32)

    
    def _get_obs(self):
        # return the current observation
        # which is: the weights and the log-returns of the assets over lookback period
        # and also technical indicators (optionally)
        data = {
            'portfolio_allocation': self._portfolio_allocation,
            'stock_lrs': [] if self.current_day_idx > len(self.data) else self._get_stock_lrs(),
            # 'volatility_metrics': self._get_volatility_metrics()
        }

        state = np.zeros((self.num_companies, self.lookback_period))

        state[:,0] = data['portfolio_allocation'] # first column consists of portfolio weights
        
        if type(data['stock_lrs']) != list:
            state[:,1:] = data['stock_lrs'].fillna(0).T # the log return of each asset
        else:
            state = None

        return state
        
    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.portfolio_val = self.budget

        self.A = 0
        self.B = 0
        # reset portfolio allocation to equal distribution
        self._portfolio_allocation = np.array([1/self.num_companies] * self.num_companies)

        # reset day
        self.current_day_idx = self.lookback_period - 1

        # return initial observation
        return self._get_obs(), {}
    

    def step (self, action):
        # normalize action to ensure it sums to 1
        action = np.exp(action) / np.sum(np.exp(action))

        # calculate reward based on change in portfolio value
        reward = self._calculate_diff_sharpe_ratio(new_weights=action)

        # update portfolio allocation
        self._portfolio_allocation = action

        # update date
        self.current_day_idx += 1

        # check if episode is done
        done = self.current_day_idx >= len(self.data)

        # return observation, reward, done, truncated and info
        return self._get_obs(), reward, done, False, {}
    

    def _load_dataset(self):
        # load dataset from pickle file
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

        return data


    def _get_stock_lrs(self):
        # get stock prices for the lookback period
        start_date = self.data.index[self.current_day_idx - self.lookback_period + 1]
        end_date = self.data.index[self.current_day_idx - 1]

        return self.data_lrs[start_date:end_date]
    
    
    def _calculate_shares(self, weights, date):
        asset_budget = self.portfolio_val * weights # the amount of money allocated to each asset
        prices = self.data['Close'][date:date].values.flatten() # price of each asset at the time

        shares = np.rint(np.divide(asset_budget, prices)) # number of shares of each asset

        return shares



    def _calculate_diff_sharpe_ratio(self, new_weights):
        # calculating the reward, which is the differential sharpe ratio
        # we are assuming that at the end of the trading day, we are able to reallocate our portfolio
        today = self.data.index[self.current_day_idx]
        yesterday = self.data.index[self.current_day_idx-1]

        shares = self._calculate_shares(new_weights, yesterday) 
        new_prices = self.data['Close'][today:today].values.flatten()

        new_portfolio_val = np.dot(shares, new_prices)
        old_portfolio_val = self.portfolio_val

        portfolio_return = (new_portfolio_val-old_portfolio_val)/old_portfolio_val

        delta_A = portfolio_return - self.A
        delta_B = portfolio_return**2 - self.B

        reward = ((self.B*delta_A)-1/2*(self.A*delta_B))/((self.B-self.A**2)**(3/2)+1e-8)

        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        self.portfolio_val = new_portfolio_val

        return reward
