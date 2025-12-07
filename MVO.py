from ppo_env import StockTradingEnv

from scipy.optimize import minimize

import numpy as np




class MVO():
    def __init__(self, env, annual_rf=0, trading_days=252):
        '''
        Class that implements MVO with sharpe ratio as the objective function.

        env must be and environment that given weights returns as the next state the returns of each 
        asset in the lookback period.

        daily_rf is the daily risk free rate.

        trading_days  is the number of trading days that are in a year.
        '''
        self.env = env
        self.annual_rf=annual_rf
        self.trading_days = trading_days

    
    def _get_daily_rf(self):
        daily_rf = (1+self.annual_rf)**(1/self.trading_days) - 1

        return daily_rf
    
    
    def _objective_func(self, weights, mu, sigma):
        '''
        The objective function to be optimized.
        '''

        weights = np.array(weights)
        daily_rf = self._get_daily_rf()

        excess_ret = np.dot(weights, mu) - daily_rf
        vol = np.dot(np.dot(weights, sigma), weights)

        sharpe_ratio = excess_ret/vol

        return -sharpe_ratio
    
    
    def _get_weights(self, mu, sigma):
        '''
        Calculates weights as to minimize the negative sharpe ratio (maximize sharpe ration)
        '''

        n_assets = len(mu)
        #enforce weights to be between 0 and 1 (long only)
        bound = [(0,1)]*n_assets
        # enforce weights to add up to 1
        cosntraints = {'type':'eq', 
                       'fun':lambda x:np.sum(x)-1,}
        
        init_weights = np.ones(n_assets)/n_assets # start with same weight
        
        solution = minimize(self._objective_func, 
                                   x0=init_weights, args=(mu,sigma),
                                   method='SLSQP', bounds=bound, constraints=cosntraints,
                                   options={'maxiter':1000, 'disp':True})
        
        print(f'optimization success stat: {solution.success}')
        
        # in case of divergnece revert to equal weighing
        if not solution.success:
            return init_weights
        

        return solution.x


    def _estimate_mu_sigma(self, returns):
        mu = np.mean(returns, 1)
        sigma = np.cov(returns)

        return mu, sigma
    

    def test(self):
        '''
        tests the model on the given environment and returns the portfolio values.
        '''


        env = self.env
        returns, _ = env.reset()

        portfolio_val = [env.portfolio_val]

        done = False
        
        while not done:
            
            mu, sigma = self._estimate_mu_sigma(returns)
            weights = self._get_weights(mu, sigma)

            new_returns, _, done, _, _ = env.step(weights)

            portfolio_val.append(env.portfolio_val)

            returns = new_returns

        return portfolio_val




class StockEnvMVO(StockTradingEnv):
    def __init__(self, investment=100000, lookback_period=60, eta=1 / 252, dataset_path='dataset.pkl'):
        super().__init__(investment, lookback_period, eta, dataset_path)


    def _get_obs(self):
        '''
        return the daily returns of the assets in the lookback period.
        '''

        data = [] if self.current_day_idx > len(self.data) else self._get_stock_lrs()

        state = np.zeros((self.num_companies, self.lookback_period-1))

        if type(data) != list:
            state[:,:] = data.fillna(0).T # the log return of each asset
        else:
            state = None

        return state
    

    def step(self, action):
        self.current_day_idx += 1
        
        # calculate reward based on change in portfolio value
        reward = self._calculate_simple_return(new_weights=action)
        # reward = self._calculate_diff_sharpe_ratio(new_weights=action)

        # check if episode is done
        done = self.current_day_idx >= len(self.data) -1 # account for indexing starting from 0

        # update portfolio allocation
        self._portfolio_allocation = action

        # return observation, reward, done, truncated and info
        return self._get_obs(), reward, done, False, {}



    
