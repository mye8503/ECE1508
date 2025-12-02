import os

import numpy as np
import matplotlib.pyplot as plt

from ppo_env import StockTradingEnv

from stable_baselines3 import PPO



# params of the env
investment = 100_000
eta = 1/252
lookback = 60


# get the path for the best trained model for each training window
best_model_paths = sorted([os.path.join('best_model', f) for f in os.listdir('best_model') if not os.path.isfile(f)])

# get the path to the testsets
testset_paths = sorted([os.path.join('data/test', f) for f in os.listdir('data/test') if not os.path.isfile(f)])


def test_model(model_path, testset_path, 
               investment=100_000, eta=1/252, lookback=60):
    """
    test the model on the test set.
    returns the portfolio value and weights over time.
    """
    
    portfolio_weights = []
    portfolio_vals = [investment]
    done = False

    model_path = os.path.join(model_path, 'best_model.zip')
    

    env = StockTradingEnv(investment=investment, lookback_period=lookback,
                          eta=eta, dataset_path=testset_path)
    
    
    state, _ = env.reset()

    model = PPO.load(model_path, env)

    portfolio_weights.append(env._portfolio_allocation)

    while not done:
        actions = model.predict(state, deterministic=True)
        new_state, _, done, _, _ = env.step(actions[0])

        portfolio_vals.append(env.portfolio_val)
        portfolio_weights.append(env._portfolio_allocation)

        state = new_state

    return portfolio_vals, portfolio_weights


def plot_annual_returns(portfolio_vals_sets, years, path_to_save=None):
    '''
    plots and saves the annual return for a set of daily portfolio val.
    '''
    
    annual_return_set = [(port_val[-1]-port_val[0])/port_val[0] for port_val in portfolio_vals_sets]
    y_pos = np.arange(len(years))
    avg_annual_return = np.mean(annual_return_set)

    fig, ax = plt.subplots()

    ax.barh(y_pos, annual_return_set, align='center', alpha=.7)
    ax.axvline(avg_annual_return, label='Mean', ls='--')
    ax.axvline(0, color='black')
    
    ax.set_yticks(y_pos, labels=years)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    ax.set_xlabel('Returns')
    ax.set_ylabel('Year')
    ax.set_title('annual return')

    plt.legend()
    
    if path_to_save is not None:
        plt.savefig(path_to_save)
    
    plt.show()


def calculate_annual_sharpe_ratio(portfolio_vals: list, days=252, annual_risk_free_rate=0):
    '''
    calculates the annual Sharpe ratio from daily values of of a portfolio for a year.
    '''
    
    #calculate the daily rf
    daily_rf = (1+annual_risk_free_rate)**(1/days) - 1

    daily_returns = np.diff(np.array(portfolio_vals))
    excess_daily_returns = daily_returns - daily_rf

    avg_excess_return = np.mean(excess_daily_returns)
    std_excess_return = np.std(excess_daily_returns)

    annual_sharpe = avg_excess_return/std_excess_return * np.sqrt(days)

    return annual_sharpe


def plot_sharpe_ratio(portfolio_vals_sets, years, path_to_save=None):
    '''
    plots the annual Sharpe ratio ever the years
    '''

    annual_sharpes = [calculate_annual_sharpe_ratio(port_vals) for port_vals in portfolio_vals_sets]

    fig, ax = plt.subplots()
    x_idx = np.arange(len(years))

    ax.plot(x_idx, annual_sharpes)
    ax.axhline(np.mean(annual_sharpes), ls='--', label='mean')
    ax.axhline(0, color='black')

    ax.set_xticks(x_idx, years)

    ax.set_xlabel('Years')
    ax.set_ylabel('Sharpe ratio')

    ax.set_title('annual Sharpe ratio')

    plt.legend()
    plt.grid()
    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)

    plt.show()



if __name__=='__main__':
    
    save_results_path = 'PPO_test_results'
    os.makedirs(save_results_path, exist_ok=True)

    portfolio_vals_sets = []
    portfolio_weights_sets = []

    years = []

    for i in range(len(testset_paths)):

        file_name = testset_paths[i].split('/')[-1]
        year = file_name.split('.')[0]

        portfolio_vals, portfolio_weights = test_model(best_model_paths[i], testset_paths[i], 
                                    investment=investment, eta=eta, lookback=lookback)
        
        print(f'portfolio value at the end of {year}: {portfolio_vals[-1]: .2f}')

        portfolio_vals_sets.append(portfolio_vals)
        portfolio_weights_sets.append(portfolio_weights)

        years.append(year)

    annual_returns_plot_name = f'{save_results_path}/annual_returns'
    annual_sharpe_plot_name = f'{save_results_path}/annual_sharpe_ratio'
    plot_annual_returns(portfolio_vals_sets, years, annual_returns_plot_name)
    plot_sharpe_ratio(portfolio_vals_sets, years, annual_sharpe_plot_name)