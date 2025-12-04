import os
import shutil

from typing import Callable

from ppo_env import StockTradingEnv
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import torch
import torch.nn as nn



def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func



class DDPG_Agent():
    def __init__(self, policy, train_env, val_env, gamma, 
                initial_lr, final_lr, batch_size, tau, action_noise,
                n_steps, total_steps, policy_kwargs, seed, eval_freq,
                time_window, device='cpu', trained_weights_path=None):
        
        
        # create directory for saving train tensorboards
        self.train_log_dir = f'logs/train logs/winodw_{time_window}_seed_{seed}'
        os.makedirs(self.train_log_dir, exist_ok=True)

        # create directory to save validation logs
        self.val_log_dir = f'logs/val logs/winodw_{time_window}_seed_{seed}'
        os.makedirs(self.val_log_dir, exist_ok=True)

        # create directory the best model (of a time window and a seed)
        self.model_dir = f'models/winodw_{time_window}_seed_{seed}'
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.train_env = train_env
        self.val_env = val_env

        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.device = device
        
        if trained_weights_path is not None and os.path.exists(trained_weights_path):
            self.model = DDPG.load(trained_weights_path, train_env, device)

            self.model.set_random_seed(seed)

        else:
            self.model = DDPG(
                policy=policy,
                env=train_env,
                learning_rate=linear_schedule(initial_lr, final_lr),
                buffer_size=1000000,
                learning_starts=100,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,                
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                n_steps=n_steps,
                tensorboard_log=self.train_log_dir,
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=seed,
                device=device
            )
            
    
    def train(self):
        callbacks = EvalCallback(self.val_env, eval_freq=self.eval_freq,
                                log_path=self.val_log_dir, best_model_save_path=self.model_dir)
        
        self.model.learn(self.total_steps, callback=callbacks)

    
    def eval(self):
        self.best_model_path = f'{self.model_dir}/best_model'

        best_model = DDPG.load(self.best_model_path, self.val_env, device=self.device)

        mean_val_reward, _ = evaluate_policy(best_model, self.val_env)

        return mean_val_reward

        
            



trainsets = sorted(os.listdir('data/train'))
valsets = sorted(os.listdir('data/val'))

trainset_paths = [f'data/train/{trainset}' for trainset in trainsets]
valset_paths = [f'data/val/{valset}' for valset in valsets]



# environment parameters
lookback = 60
eta = 1/252
investment = 100_000
n_env = 1 #5


# model hyperparameters
policy = 'MlpPolicy'
initial_lr = 3e-4
final_lr = 1e-5

# hyperparameters from paper https://pmc.ncbi.nlm.nih.gov/articles/PMC8512099/#:~:text=The%20DDPG%20consists%20of%20a,continuous%20action%20at%20each%20step. 
buffer_size = 100000
learning_starts = 100
batch_size = 128
tau = 0.01
gamma = 0.8
train_freq = 1
gradient_steps = 1
layers = [32, 64, 32]
activation_func = nn.Tanh

policy_kwargs = {
    'net_arch': layers,
    'activation_fn': activation_func,
    'n_critics': 1,
}

# environment parameters
lookback = 60
eta = 1/252
investment = 100_000
n_env = 10 #5

action_noise = None
replay_buffer_class = None
replay_buffer_kwargs = None
optimize_memory_usage = False
n_steps = 252 * n_env
episodes = 100
total_steps = 252 * 1 * n_env * episodes 

tensorboard_log = None
verbose = 1
seed = None
device = 'cpu'

eval_freq = 100_000 // n_env # used in evalcallback
# eval_freq = 252*5 // n_env # used in evalcallback
device = 'cpu'
seeds = [i*111 for i in range(1, 6)]

best_seed = None

best_models = []


if __name__=='__main__':
    
    for i in range(len(valset_paths)):

        print("------------------------------------------------------------")
        print("Training on dataset:", trainset_paths[i])
        print("------------------------------------------------------------")
    
        best_model = best_seed
        
        best_eval_reward = -np.inf
        best_seed = None
        
        for seed in seeds:
            print("------------------------------------------------------------")
            print("Seed:", seed)
            print("------------------------------------------------------------")

            train_env_kwargs = {'investment': investment, 'lookback_period': lookback, 
                                'eta': eta, 'dataset_path': trainset_paths[i]}

            val_env_kwargs = {'investment': investment, 'lookback_period': lookback, 
                            'eta': eta, 'dataset_path': valset_paths[i]}
            
            train_env = make_vec_env(env_id=StockTradingEnv, n_envs = n_env, 
                            vec_env_cls=SubprocVecEnv, env_kwargs=train_env_kwargs)

            val_env = make_vec_env(env_id=StockTradingEnv, n_envs=1, 
                                vec_env_cls=SubprocVecEnv, env_kwargs=val_env_kwargs)

            agent = DDPG_Agent(policy, train_env, val_env, gamma, initial_lr,
                            final_lr, batch_size, tau, action_noise, n_steps, 
                            total_steps, policy_kwargs, seed, eval_freq, time_window=i, 
                            device=device, trained_weights_path=best_model)
            
            agent.train()

            eval_reward = agent.eval()

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_seed = agent.best_model_path

            
            train_env.close()
            val_env.close()

        best_models.append(best_seed)

        print("------------------------------------------------------------")
        print("DONE TRAINING on dataset:", trainset_paths[i])
        print("------------------------------------------------------------")


    # move the best model of each training window to the same dir
    for model in best_models:
        destination = f'best_model/{model.split('/')[-2]}'
        os.makedirs(destination, exist_ok=True)
        shutil.move(model, f'{destination}/{model.split('/')[-1]}')

        