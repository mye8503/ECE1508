import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
from typing import Dict, Any, Optional, Tuple

# --------------------------------------------------
# Utility: build state matrix (reused from A2C)
# --------------------------------------------------

def build_state(obs: Any, num_assets: int, lookback: int) -> np.ndarray:
    """Build state from either dict observation or ndarray matrix (ppo_env2).

    - ndarray: shape (num_assets, lookback) possibly needs transpose/padding
    - dict: expects 'portfolio_allocation' and 'stock_lrs'
    """
    if isinstance(obs, np.ndarray):
        arr = obs
        if arr.ndim != 2:
            arr = np.array(arr)
        if arr.shape[0] != num_assets and arr.shape[1] == num_assets:
            arr = arr.T
        if arr.shape[1] < lookback:
            pad = np.zeros((num_assets, lookback - arr.shape[1]))
            arr = np.concatenate([arr, pad], axis=1)
        else:
            arr = arr[:, -lookback:]
        return arr.astype(np.float32)

    weights = obs.get('portfolio_allocation', np.zeros(num_assets))
    lrs = obs.get('stock_lrs', [])
    if isinstance(lrs, list) or lrs is None or len(lrs) == 0:
        lr_matrix = np.zeros((num_assets, lookback))
    else:
        try:
            vals = lrs.values
        except AttributeError:
            vals = np.array(lrs)
        if vals.shape[0] != num_assets and vals.shape[1] == num_assets:
            vals = vals.T
        if vals.shape[1] < lookback:
            pad = np.zeros((num_assets, lookback - vals.shape[1]))
            lr_matrix = np.concatenate([vals, pad], axis=1)
        else:
            lr_matrix = vals[:, -lookback:]
    weights = np.asarray(weights).reshape(num_assets, 1)
    state = np.concatenate([weights, lr_matrix], axis=1)
    return state.astype(np.float32)

# --------------------------------------------------
# Actor-Critic Network (same as A2C)
# --------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, num_assets: int, lookback: int, hidden: int = 256):
        super().__init__()
        # Match ppo_env2 ndarray observation: shape (num_assets, lookback)
        # We flatten to num_assets * lookback for the MLP input.
        input_dim = num_assets * lookback
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, num_assets)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

# --------------------------------------------------
# Simplified ACER (on-policy only for this environment)
# --------------------------------------------------

def train_acer(env, num_assets: int, lookback: int, episodes: int = 100, gamma: float = 0.99,
               lr: float = 3e-4, device: str = 'cpu', model: Optional[ActorCritic] = None,
               optimizer: Optional[optim.Optimizer] = None) -> Tuple[ActorCritic, optim.Optimizer]:
    if model is None:
        model = ActorCritic(num_assets, lookback).to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, episodes + 1):
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        done = False
        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_values = []
        traj_log_probs = []

        # Collect trajectory
        while not done:
            state_np = build_state(obs, num_assets, lookback)
            state_t = torch.tensor(state_np.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(state_t)
            alpha = F.softplus(logits) + 1e-3
            dist = Dirichlet(alpha)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t)
            action = action_t.squeeze(0).detach().cpu().numpy()

            step_ret = env.step(action)
            if isinstance(step_ret, tuple) and len(step_ret) >= 4:
                next_obs, reward, done = step_ret[:3]
            else:
                next_obs = step_ret[0]
                reward = step_ret[1] if len(step_ret) > 1 else 0.0
                done = step_ret[2] if len(step_ret) > 2 else False

            traj_states.append(state_t)
            traj_actions.append(action_t)
            traj_rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            traj_values.append(value.squeeze(0))
            traj_log_probs.append(log_prob.squeeze(0))
            obs = next_obs

        # Compute returns and advantages (reverse pass)
        returns = []
        R = torch.tensor(0.0, device=device)
        for r in reversed(traj_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.stack(returns)
        values_t = torch.stack(traj_values)
        log_probs_t = torch.stack(traj_log_probs)
        advantages = returns_t - values_t

        # Policy & value losses
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean() * 0.5
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            ep_return = sum([r.item() for r in traj_rewards])
            print(f"ACER Episode {ep}: return={ep_return:.4f} loss={loss.item():.4f}")

    return model, optimizer


def list_pickle_files(directory: str):
    return sorted(glob.glob(os.path.join(directory, '*.pkl')))


def train_over_directory(data_root: str = 'data', split: str = 'train', episodes_per_file: int = 50,
                         device: str = 'cpu', lr: float = 3e-4, gamma: float = 0.99):
    from env_test import TestStockTradingEnv
    split_dir = os.path.join(data_root, split)
    files = list_pickle_files(split_dir)
    if not files:
        raise FileNotFoundError(f'No .pkl files found in {split_dir}')

    model = None
    optimizer = None
    for idx, fpath in enumerate(files, 1):
        env = TestStockTradingEnv(dataset_path=fpath)
        num_assets = env.num_companies
        lookback = env.lookback_period
        print(f'Training on file {idx}/{len(files)}: {fpath}')
        model, optimizer = train_acer(env, num_assets, lookback,
                                      episodes=episodes_per_file, gamma=gamma, lr=lr,
                                      device=device, model=model, optimizer=optimizer)
    return model

# --------------------------------------------------
# Simple test runner
# --------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ACER-like loop over directory of datasets')
    parser.add_argument('--data-root', type=str, default='data', help='Root data directory containing splits')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--episodes-per-file', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    args = parser.parse_args()

    try:
        _ = train_over_directory(data_root=args.data_root, split=args.split,
                                 episodes_per_file=args.episodes_per_file,
                                 device=args.device, lr=args.lr, gamma=args.gamma)
    except Exception as e:
        print(f'Runner failed: {e}')
