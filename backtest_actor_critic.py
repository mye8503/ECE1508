import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from ppo_env2 import StockTradingEnv
from a2c import ActorCritic as A2CActorCritic, train_a2c
from acer import ActorCritic as ACERActorCritic, train_acer

# Params of the env
investment = 100_000
eta = 1/252
lookback = 60

BEST_DIR_A2C = 'A2C_best_model'
BEST_DIR_ACER = 'ACER_best_model'
os.makedirs(BEST_DIR_A2C, exist_ok=True)
os.makedirs(BEST_DIR_ACER, exist_ok=True)


def list_pickle_files(directory: str):
    return sorted(glob.glob(os.path.join(directory, '*.pkl')))


def train_and_select_best(algo: str, train_paths):
    best_models = []
    for i, train_path in enumerate(train_paths):
        env = StockTradingEnv(investment=investment, lookback_period=lookback, eta=eta, dataset_path=train_path)
        num_assets = env.num_companies
        lb = env.lookback_period
        # Train for fixed episodes; select the last checkpoint as "best" for now
        if algo == 'a2c':
            model, opt = train_a2c(env, num_assets, lb, episodes=50, gamma=0.99, lr=3e-4, device='cpu', model=None, optimizer=None)
            save_path = os.path.join(BEST_DIR_A2C, f'{os.path.basename(train_path).split(".")[0]}_best.pt')
        else:
            model, opt = train_acer(env, num_assets, lb, episodes=50, gamma=0.99, lr=3e-4, device='cpu', model=None, optimizer=None)
            save_path = os.path.join(BEST_DIR_ACER, f'{os.path.basename(train_path).split(".")[0]}_best.pt')
        torch.save({'model': model.state_dict(), 'algo': algo, 'dataset_path': train_path, 'num_assets': num_assets, 'lookback': lb}, save_path)
        best_models.append(save_path)
        print(f'Saved best {algo.upper()} model: {save_path}')
    return best_models


def find_saved_best_paths(algo: str):
    """Return list of saved best model file paths for an algorithm, sorted."""
    best_dir = BEST_DIR_A2C if algo == 'a2c' else BEST_DIR_ACER
    paths = sorted(glob.glob(os.path.join(best_dir, '*_best.pt')))
    if not paths:
        print(f'No saved best {algo.upper()} models found in {best_dir}.')
    else:
        print(f'Found {len(paths)} saved best {algo.upper()} models in {best_dir}.')
    return paths


def load_model_from_file(algo: str, model_path: str):
    """Load a model instance with weights from a saved best file."""
    ckpt = torch.load(model_path, map_location='cpu')
    num_assets = ckpt.get('num_assets')
    lookback_ckpt = ckpt.get('lookback')
    if num_assets is None or lookback_ckpt is None:
        raise ValueError(f'Missing num_assets/lookback in checkpoint: {model_path}')
    if algo == 'a2c':
        model = A2CActorCritic(num_assets, lookback_ckpt)
    else:
        model = ACERActorCritic(num_assets, lookback_ckpt)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, num_assets, lookback_ckpt


def test_model(algo: str, model_path: str, testset_path: str,
               investment=investment, eta=eta, lookback=lookback):
    """Test the model on the test set. Returns portfolio values and weights over time."""
    portfolio_weights = []
    portfolio_vals = [investment]
    done = False

    env = StockTradingEnv(investment=investment, lookback_period=lookback, eta=eta, dataset_path=testset_path)
    state, _ = env.reset()

    model, num_assets, lb = load_model_from_file(algo, model_path)

    portfolio_weights.append(env._portfolio_allocation)

    while not done:
        # Build state vector like training
        weights = state[:,0]
        lrs = state[:,1:]
        flat = np.concatenate([weights.reshape(-1,1), lrs], axis=1).flatten().astype(np.float32)
        x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(x)
        alpha = F.softplus(logits) + 1e-3
        action_vec = (alpha / alpha.sum()).detach().cpu().numpy().squeeze(0)

        new_state, _, done, _, _ = env.step(action_vec)
        portfolio_vals.append(env.portfolio_val)
        portfolio_weights.append(env._portfolio_allocation)
        state = new_state

    return portfolio_vals, portfolio_weights


def plot_annual_returns(portfolio_vals_sets, years, path_to_save=None):
    annual_return_set = [(port_val[-1]-port_val[0])/port_val[0]*100 for port_val in portfolio_vals_sets]
    y_pos = np.arange(len(years))
    avg_annual_return = np.mean(annual_return_set)

    fig, ax = plt.subplots()

    ax.barh(y_pos, annual_return_set, align='center', alpha=.7)
    ax.axvline(avg_annual_return, label='Mean', ls='--')
    ax.axvline(0, color='black')
    ax.set_xticks(np.arange(-40, 100, 20), labels=[f'{returns}%' for returns in np.arange(-40, 100, 20)])
    ax.set_yticks(y_pos, labels=years)
    ax.invert_yaxis()
    ax.set_xlabel('Returns')
    ax.set_ylabel('Year')
    ax.set_title('Annual return')
    plt.legend()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()


def calculate_annual_sharpe_ratio(portfolio_vals: list, days=252, annual_risk_free_rate=0):
    daily_rf = (1+annual_risk_free_rate)**(1/days) - 1
    daily_returns = np.diff(np.array(portfolio_vals))
    excess_daily_returns = daily_returns - daily_rf
    avg_excess_return = np.mean(excess_daily_returns)
    std_excess_return = np.std(excess_daily_returns)
    if std_excess_return == 0:
        return 0.0
    annual_sharpe = avg_excess_return/std_excess_return * np.sqrt(days)
    return annual_sharpe


def plot_sharpe_ratio(portfolio_vals_sets, years, path_to_save=None):
    annual_sharpes = [calculate_annual_sharpe_ratio(port_vals) for port_vals in portfolio_vals_sets]
    fig, ax = plt.subplots()
    x_idx = np.arange(len(years))
    ax.plot(x_idx, annual_sharpes)
    ax.axhline(np.mean(annual_sharpes), ls='--', label='mean')
    ax.axhline(0, color='black')
    ax.set_xticks(x_idx, years)
    ax.set_xlabel('Years')
    ax.set_ylabel('Sharpe ratio')
    ax.set_title('Annual Sharpe ratio')
    plt.legend()
    plt.grid()
    fig.tight_layout()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()


def get_monthly_returns(portfolio_val_sets:list):
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
    n_years = len(years)
    monthly_returns_sets = get_monthly_returns(portfolio_val_sets)
    n_months = len(monthly_returns_sets[0])
    figure, ax = plt.subplots()
    ax.imshow(monthly_returns_sets, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(np.arange(n_months+1, step=2), labels=np.arange(1, n_months+1, step=2)+3)
    ax.set_yticks(np.arange(n_years), labels=years)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_title('Monthly returns')
    for i in range(n_years):
        for j in range(n_months):
            ax.text(j,i, f'{monthly_returns_sets[i][j]*100:.1f}', ha='center', va='center', color='black')
    figure.tight_layout()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()


def monthly_retuns_hist(portfolio_val_sets, n_bins=20, path_to_save=None):
    monthly_returns_sets = np.array(get_monthly_returns(portfolio_val_sets)).flatten()
    mean_monthly_return = np.mean(monthly_returns_sets)
    fig, ax = plt.subplots()
    ax.hist(monthly_returns_sets, bins=n_bins, alpha=.4, color='red', edgecolor='white')
    ax.axvline(0, color='black')
    ax.axvline(mean_monthly_return, ls='--', color='red')
    ax.set_xticks(np.arange(-15, 25, 5)/100, labels=[f'{returns}%' for returns in np.arange(-15, 25, 5)])
    ax.set_xlabel('Returns')
    ax.set_ylabel('Number of Months')
    ax.set_title('Distribution of monthly returns')
    fig.tight_layout()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()


if __name__=='__main__':
    import argparse
    save_results_path = 'AC_test_results'
    os.makedirs(save_results_path, exist_ok=True)

    trainset_paths = list_pickle_files('data/train')
    testset_paths = list_pickle_files('data/test')

    parser = argparse.ArgumentParser(description='Backtest A2C/ACER with saved best models')
    parser.add_argument('--train', action='store_true', help='Train and save best models')
    parser.add_argument('--load', action='store_true', help='Load existing saved best models')
    parser.add_argument('--skip-a2c', action='store_true', help='Skip A2C (only run ACER)')
    args = parser.parse_args()

    if args.train:
        best_a2c = [] if args.skip_a2c else train_and_select_best('a2c', trainset_paths)
        best_acer = train_and_select_best('acer', trainset_paths)
    elif args.load:
        best_a2c = [] if args.skip_a2c else find_saved_best_paths('a2c')
        best_acer = find_saved_best_paths('acer')
    else:
        # Default: try to load; if none, train
        best_a2c = [] if args.skip_a2c else find_saved_best_paths('a2c')
        best_acer = find_saved_best_paths('acer')
        if (best_acer is None or len(best_acer) == 0) or ((not args.skip_a2c) and (best_a2c is None or len(best_a2c) == 0)):
            print('No saved models found; training now...')
            best_a2c = [] if args.skip_a2c else train_and_select_best('a2c', trainset_paths)
            best_acer = train_and_select_best('acer', trainset_paths)

    portfolio_vals_sets_a2c = []
    portfolio_vals_sets_acer = []
    years = []

    for i in range(len(testset_paths)):
        file_name = os.path.basename(testset_paths[i])
        year = file_name.split('.')[0]
        years.append(year)
        # Evaluate A2C (if not skipped)
        if not args.skip_a2c:
            vals_a2c, _ = test_model('a2c', best_a2c[i], testset_paths[i])
            print(f'[A2C] portfolio value at the end of {year}: {vals_a2c[-1]: .2f}')
            portfolio_vals_sets_a2c.append(vals_a2c)
        # Evaluate ACER
        vals_acer, _ = test_model('acer', best_acer[i], testset_paths[i])
        print(f'[ACER] portfolio value at the end of {year}: {vals_acer[-1]: .2f}')
        portfolio_vals_sets_acer.append(vals_acer)

    # Plots per algorithm
    if not args.skip_a2c and len(portfolio_vals_sets_a2c) > 0:
        plot_annual_returns(portfolio_vals_sets_a2c, years, os.path.join(save_results_path, 'a2c_annual_returns'))
        plot_sharpe_ratio(portfolio_vals_sets_a2c, years, os.path.join(save_results_path, 'a2c_annual_sharpe_ratio'))
        plot_monthly_returns(portfolio_vals_sets_a2c, years, os.path.join(save_results_path, 'a2c_monthly_returns'))
        monthly_retuns_hist(portfolio_vals_sets_a2c, path_to_save=os.path.join(save_results_path, 'a2c_monthly_returns_dist'))

    plot_annual_returns(portfolio_vals_sets_acer, years, os.path.join(save_results_path, 'acer_annual_returns'))
    plot_sharpe_ratio(portfolio_vals_sets_acer, years, os.path.join(save_results_path, 'acer_annual_sharpe_ratio'))
    plot_monthly_returns(portfolio_vals_sets_acer, years, os.path.join(save_results_path, 'acer_monthly_returns'))
    monthly_retuns_hist(portfolio_vals_sets_acer, path_to_save=os.path.join(save_results_path, 'acer_monthly_returns_dist'))
