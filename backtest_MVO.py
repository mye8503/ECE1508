import os
import pickle


import numpy as np
import matplotlib.pyplot as plt

from MVO import StockEnvMVO, MVO
from backtest_ppo import *



if __name__=='__main__':

    # environment params
    investment = 100_000
    eta = 1/252
    lookback = 60


    # get the path to the testsets
    testset_paths = sorted([os.path.join('data/test', f) for f in os.listdir('data/test') if not os.path.isfile(f)])


    save_results_path = 'MVO_test_results'
    os.makedirs(save_results_path, exist_ok=True)

    portfolio_vals_sets = []
    years = []

    for i in range(len(testset_paths)):

        file_name = testset_paths[i].split('/')[-1]
        year = file_name.split('.')[0]

        env = StockEnvMVO(investment=investment, lookback_period=lookback,
                          eta=eta, dataset_path=testset_paths[i])
        
        mvo_agent = MVO(env)

        portfolio_vals = mvo_agent.test()
        
        print(f'portfolio value at the end of {year}: {portfolio_vals[-1]: .2f}')

        portfolio_vals_sets.append(portfolio_vals)
        years.append(year)

    annual_returns_plot_name = f'{save_results_path}/annual_returns'
    annual_sharpe_plot_name = f'{save_results_path}/annual_sharpe_ratio'
    monthly_returns_plot_name = f'{save_results_path}/monthly_returns'
    monthly_returns_plot_name = f'{save_results_path}/monthly_returns'
    monthly_returns_dist_name = f'{save_results_path}/monthly_returns_dist'
    plot_annual_returns(portfolio_vals_sets, years, annual_returns_plot_name)
    plot_sharpe_ratio(portfolio_vals_sets, years, annual_sharpe_plot_name)
    plot_monthly_returns(portfolio_vals_sets, years, monthly_returns_plot_name)
    monthly_retuns_hist(portfolio_vals_sets, path_to_save=monthly_returns_dist_name)

    with open(f'{save_results_path}/portfolio_vals.pkl', 'wb') as f:
        pickle.dump(portfolio_vals_sets, f)
