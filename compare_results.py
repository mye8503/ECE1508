import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backtest_ppo import calculate_annual_sharpe_ratio


ppo_daily_vals_sets = 'PPO_test_results/portfolio_vals.pkl'
ddpg_daily_vals_sets = 'DDPG_test_results/portfolio_vals.pkl'
a2c_daily_vals_sets = 'A2C_test_results/portfolio_vals.pkl'
acer_daily_vals_sets = 'ACER_test_results/portfolio_vals.pkl'
EQ_daily_vals_sets = 'EQ_test_results/portfolio_vals.pkl'

comparison_graphs_path = 'comparison_graphs' # dir to save graphs
os.makedirs(comparison_graphs_path, exist_ok=True)


def open_file(path):
    '''
    loads pickle file
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


data_dict = {
    'ppo_val_sets':open_file(ppo_daily_vals_sets),
    'ddpg_val_sets':open_file(ddpg_daily_vals_sets),
    'a2c_val_sets':open_file(a2c_daily_vals_sets),
    'acer_val_sets':open_file(acer_daily_vals_sets),
    'eq_val_sets':open_file(EQ_daily_vals_sets)
}


def plot_sharpe(data_dict: dict, path_to_save=None):
    '''
    plots and saves the sharpe ratio over the years.
    '''

    first_test_year = 2011

    sharpe_ratio = {}

    for key, vals in data_dict.items():

        algo_name = key.split('_')[0] # get the name of the algorithm from the key

        if 'a2c' not in algo_name and 'acer' not in algo_name: # these two have different formating
            sharpe_ratio[algo_name]=[calculate_annual_sharpe_ratio(val) for val in vals]
        else:
            sharpe_ratio[algo_name]=[calculate_annual_sharpe_ratio(val) for val in vals['values']]


    fig, ax = plt.subplots()
    x_idx = np.arange(len(sharpe_ratio[algo_name]))

    for key, annual_sharpes in sharpe_ratio.items():
    
        ax.plot(x_idx, annual_sharpes, label=key)

    ax.axhline(0, color='black')

    ax.set_xticks(x_idx, x_idx+first_test_year)

    ax.set_xlabel('Years')
    ax.set_ylabel('Sharpe ratio')

    ax.set_title('annual Sharpe ratio')

    plt.legend()
    plt.grid()
    fig.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)

    plt.show()


def get_max_drawdown(daily_vals):
    '''
    helper function to calculate the max drawdown given daily values.
    '''
    
    daily_vals = pd.Series(daily_vals)
    running_max = daily_vals.cummax()
    drawdowns = (daily_vals-running_max)/running_max

    return np.min(drawdowns)


def plot_max_drawdown(data_dict, path_to_save):
    '''
    plot and save the maximum drawdown of algorithm.
    '''



    first_test_year = 2011

    max_drawdowns = {}

    for key, vals in data_dict.items():

        algo_name = key.split('_')[0] # get the name of the algorithm from the key

        if 'a2c' not in algo_name and 'acer' not in algo_name: # these two have different formating
            max_drawdowns[algo_name]=[get_max_drawdown(val) for val in vals]
        else:
            max_drawdowns[algo_name]=[get_max_drawdown(val) for val in vals['values']]


    fig, ax = plt.subplots()
    x_idx = np.arange(len(max_drawdowns[algo_name]))

    for key, max_drawdown in max_drawdowns.items():
    
        ax.plot(x_idx, max_drawdown, label=key)

    ax.set_xticks(x_idx, x_idx+first_test_year)

    ax.set_xlabel('Years')
    ax.set_ylabel('Max dradown')

    ax.set_title('Maximum drawdown')

    plt.legend()
    plt.grid()
    fig.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)

    plt.show()


if __name__=='__main__':
    plot_sharpe(data_dict, os.path.join(comparison_graphs_path, 'sharpe_ratio'))
    plot_max_drawdown(data_dict, os.path.join(comparison_graphs_path, 'max_drawdown'))