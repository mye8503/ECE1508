import os

from typing import Callable

from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


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
                time_window, device='cuda', trained_weights_path=None):
    
        # create directory for saving train tensorboards
        self.train_log_dir = f'ddpg/logs/train_logs/window_{time_window}_seed_{seed}'
        os.makedirs(self.train_log_dir, exist_ok=True)

        # create directory to save validation logs
        self.val_log_dir = f'ddpg/logs/val_logs/window_{time_window}_seed_{seed}'
        os.makedirs(self.val_log_dir, exist_ok=True)

        # create directory the best model (of a time window and a seed)
        self.model_dir = f'models/window_{time_window}_seed_{seed}'
        os.makedirs(self.model_dir, exist_ok=True)  

        self.train_env = train_env
        self.val_env = val_env 

        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.device = device

        if trained_weights_path is not None and os.path.exists(trained_weights_path):
            self.model = DDPG.load(trained_weights_path, train_env, device)

            self.model.set_random_seed(seed)


        # policy, env, learning_rate=0.001, buffer_size=1000000, 
        # learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, 
        # train_freq=1, gradient_steps=1, action_noise=None, 
        # replay_buffer_class=None, replay_buffer_kwargs=None, 
        # optimize_memory_usage=False, n_steps=1, tensorboard_log=None, 
        # policy_kwargs=None, verbose=0, seed=None, device='auto', 
        # _init_setup_model=True
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

        # if os.path.exists(self.best_model_path):
        best_model = DDPG.load(self.best_model_path, self.val_env, device=self.device)
        # else:
            # Fall back to current model if best model file not found
            # best_model = self.model

        mean_val_reward, _ = evaluate_policy(best_model, self.val_env)

        return mean_val_reward
