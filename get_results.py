import os
import pickle

import pandas as pd
import numpy as np

from compare_results import data_dict, get_max_drawdown
from backtest_ppo import calculate_annual_sharpe_ratio



"""
this script saves the annual return, volatility, max drawdown, sharpe ratio as a csv file.
(add any other metric)
"""


def get_annual_return(daily_val):
    '''
    helper function to get the annual return given a sequence of daily values.
    '''


    annual_return = (daily_val[-1]-daily_val[0])/daily_val[0]

    return annual_return



def get_metrics(data_dict):
    metrics_dict = {}

    for key, vals in data_dict.items():

        algo_name = key.split('_')[0]

        if 'a2c' not in algo_name and 'acer' not in algo_name:    
            return_set = [get_annual_return(val) for val in vals]

            mean_return = np.mean(return_set)
            std_return = np.std(return_set)

            drawdown_set = [get_max_drawdown(val) for val in vals]
            max_drawdown = np.min(drawdown_set)

            sharpe_set = [calculate_annual_sharpe_ratio(val) for val in vals]
            mean_sharpe = np.mean(sharpe_set)

            metrics = [mean_return, std_return, max_drawdown, mean_sharpe]

            metrics_dict[algo_name] = metrics
        else:
            return_set = [get_annual_return(val) for val in vals['values']]

            mean_return = np.mean(return_set)
            std_return = np.std(return_set)

            drawdown_set = [get_max_drawdown(val) for val in vals['values']]
            max_drawdown = np.min(drawdown_set)

            sharpe_set = [calculate_annual_sharpe_ratio(val) for val in vals['values']]
            mean_sharpe = np.mean(sharpe_set)

            metrics = [mean_return, std_return, max_drawdown, mean_sharpe]

            metrics_dict[algo_name] = metrics

    
    return metrics_dict


if __name__=='__main__':
    path_to_save = 'simple_return_metrics_table.csv'
    
    metrics_dict = get_metrics(data_dict=data_dict)

    df = pd.DataFrame(metrics_dict)
    df.index = ['Annual Return', 'Annual Volatility', 'Max Drawdown', 'Sharpe Ratio']
    df.index.name = 'Metric'

    df.to_csv(path_to_save)