import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from env import StockTradingEnv

from stable_baselines3 import PPO



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
    print("Loaded model from:", model_path)
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
    
    annual_return_set = [(port_val[-1]-port_val[0])/port_val[0]*100 for port_val in portfolio_vals_sets]
    y_pos = np.arange(len(years))
    avg_annual_return = np.mean(annual_return_set)

    fig, ax = plt.subplots()

    ax.barh(y_pos, annual_return_set, align='center', alpha=.7)
    ax.axvline(avg_annual_return, label='Mean', ls='--')
    ax.axvline(0, color='black')

    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks, labels=[f'{int(returns)}%' for returns in x_ticks]) 
    
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
    calculates the annual Sharpe ratio from daily values of a portfolio for a year.
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
    plots and saves the annual Sharpe ratio ever the years
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
    fig.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)

    plt.show()



def get_monthly_returns(portfolio_val_sets:list):
    '''
    helper function to get the monthly return from daily portfolio values.
    '''

    n_days_a_year = 252
    n_days_a_month = n_days_a_year // 12

    monthly_returns_sets = []
    
    for prices in portfolio_val_sets:
        monthly_returns = []
        for i in range(0, len(prices), n_days_a_month):
            if i + n_days_a_month < len(prices):
                monthly_returns.append((prices[i+n_days_a_month]-prices[i])/prices[i])
            elif i + 10 < len(prices):
                monthly_returns.append((prices[-1]-prices[i])/prices[i])
                break

        monthly_returns_sets.append(monthly_returns)
    
    return monthly_returns_sets




def plot_monthly_returns(portfolio_val_sets:list, years, path_to_save=None):
    '''
    plots and saves the monthly return given daily portfolio values.
    '''


    n_years = len(years)

    # get the monthly return for each test year
    monthly_returns_sets = get_monthly_returns(portfolio_val_sets)

    n_months = len(monthly_returns_sets[0])

    figure, ax = plt.subplots()

    min_norm = min(-np.max(monthly_returns_sets), np.min(monthly_returns_sets))
    max_norm = max(-np.min(monthly_returns_sets), np.max(monthly_returns_sets))

    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min_norm, vcenter=0., vmax=max_norm)
    ax.imshow(monthly_returns_sets, cmap='RdYlGn', norm=divnorm, aspect='auto')
    
    ax.set_xticks(np.arange(n_months+1, step=2), labels=np.arange(1, n_months+1, step=2)+3) # the first 3 months are used as observation
    ax.set_yticks(np.arange(n_years), labels=years)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

    ax.set_title('Monthly returns')

    for i in range(n_years):
        for j in range(n_months):
            
            ax.text(j,i, f'{monthly_returns_sets[i][j]*100:.1f}', 
                    ha='center', va='center', color='black')
            
    figure.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)

    plt.show()



def monthly_retuns_hist(portfolio_val_sets, n_bins=20, path_to_save=None):
    '''
    plots and saves the distribution of monthly returns given the daily porfolio values.
    '''


    monthly_returns_sets = np.array(get_monthly_returns(portfolio_val_sets)).flatten()*100
    mean_monthly_return = np.mean(monthly_returns_sets)

    fig, ax = plt.subplots()

    ax.hist(monthly_returns_sets, bins=n_bins, 
            alpha=.4, color='red', edgecolor='white')
    ax.axvline(0, color='black')
    ax.axvline(mean_monthly_return, ls='--', color='red')

    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks, labels=[f'{int(returns)}%' for returns in x_ticks])
    
    ax.set_xlabel('Returns')
    ax.set_ylabel('Number of Months')
    ax.set_title('distribution of monthly returns')

    fig.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)
    
    plt.show()





if __name__=='__main__':

    # params of the env
    investment = 100_000
    eta = 1/252
    lookback = 60


    # get the path for the best trained model for each training window
    best_model_paths = sorted([os.path.join('PPO_best_model', f) for f in os.listdir('PPO_best_model') if not os.path.isfile(f)])

    # get the path to the testsets
    # testset_paths = sorted([os.path.join('data/test', f) for f in os.listdir('data/test') if not os.path.isfile(f)])
    testset_paths = sorted([os.path.join('new_data/test', f) for f in os.listdir('new_data/test') if not os.path.isfile(f)])

    
    save_results_path = 'PPO_test_results'
    os.makedirs(save_results_path, exist_ok=True)

    portfolio_vals_sets = []
    portfolio_weights_sets = []

    years = []

    for i in range(len(testset_paths)):
        print(i)

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
    monthly_returns_plot_name = f'{save_results_path}/monthly_returns'
    monthly_returns_plot_name = f'{save_results_path}/monthly_returns'
    monthly_returns_dist_name = f'{save_results_path}/monthly_returns_dist'
    plot_annual_returns(portfolio_vals_sets, years, annual_returns_plot_name)
    plot_sharpe_ratio(portfolio_vals_sets, years, annual_sharpe_plot_name)
    plot_monthly_returns(portfolio_vals_sets, years, monthly_returns_plot_name)
    monthly_retuns_hist(portfolio_vals_sets, path_to_save=monthly_returns_dist_name)

    with open(f'{save_results_path}/portfolio_vals.pkl', 'wb') as f:
        pickle.dump(portfolio_vals_sets, f)