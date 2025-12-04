import matplotlib.pyplot as plt
import pickle
import os

from ppo_env import StockTradingEnv

from backtest_ppo import plot_annual_returns
from backtest_ppo import plot_sharpe_ratio
from backtest_ppo import plot_monthly_returns
from backtest_ppo import monthly_retuns_hist


if __name__ == "__main__":
    investment = 100000
    
    testset_paths = sorted([os.path.join('data/test', f) for f in os.listdir('data/test') if not os.path.isfile(f)])
    save_results_path = 'BH_test_results'

    portfolio_vals_sets = []
    years = []
    for i in range(len(testset_paths)):
        file_name = testset_paths[i].split('/')[-1]
        year = file_name.split('.')[0]

        with open(testset_paths[i], 'rb') as f:
            data = pickle.load(f)

        env = StockTradingEnv(investment=investment, lookback_period=60,
                            eta=1/252, dataset_path=testset_paths[i])
        
        num_companies = env.num_companies
        target_weight = investment / num_companies
        
        current_day_idx = env.current_day_idx

        # array representing each stock's fraction of the total portfolio
        # initialize with equal allocation
        portfolio_allocation = env._portfolio_allocation


        done = False
        state, _ = env.reset()


        portfolio_vals = []
        while not done:
            # don't change anything, just hold
            action = portfolio_allocation
            new_state, reward, done, _, _ = env.step(action)

            portfolio_allocation = env._portfolio_allocation
            current_day_idx = env.current_day_idx

            portfolio_vals.append(env.portfolio_val)

            # print("Day:", current_day_idx, 
            #     "\nPortfolio Value:", env.portfolio_val,
            #     "\nAllocation:", portfolio_allocation,
            #     "\nReward:", reward, "\n\n")

            state = new_state


        print(f'portfolio value at the end of {year}: {portfolio_vals[-1]: .2f}')
        years.append(year)
        portfolio_vals_sets.append(portfolio_vals)

    
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