import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from dqn_hyperparameters import Hyperparameters
from dqn_agent import Agent

from env import StockTradingEnv

class DQL():
    def __init__(self, hyperparameters:Hyperparameters, train_mode):

        if train_mode:
            render = None
        else:
            render = "human"

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters

        # Load the environment
        # self.env = gym.make('FrozenLake-v1', map_name=f"{self.hp.map_size}x{self.hp.map_size}", is_slippery=False, render_mode = render)
        self.env = StockTradingEnv(csv_path="dataset.csv")

        # Initiate the Agent
        self.agent = Agent(env = self.env, hyperparameters = self.hp)
                
        
    def feature_representation(self, state:int):
        """
        represent the feature of the state
        We simply use tokenization
        """
        ## COMPLETE ##
        # Write a function that get the state and returns the feature
        # We simply use tokenization
        feature = np.zeros(self.env.observation_space.shape[0])
        feature[state] = 1
        return feature
    
    
    def train(self): 
        """                
        Traing the DQN via DQL
        """
        
        total_steps = 0
        self.collected_rewards = []
        
        # Training loop
        for episode in range(1, self.hp.num_episodes+1):
            
            # sample a new state
            # state, _ = self.env.reset()
            state = self.env.reset()
            state = self.feature_representation(state)
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0
                                                
            while not ended and not truncated:
                ## COMPLETE ##
                # find <action> via epsilon greedy 
                # use what you implemented in Class Agent
                action = self.agent.epsilon_greedy(state)

                # find nest state and reward
                next_state, reward, ended, truncated, _ = self.env.step(action)

                ## COMPLETE ##
                # Find the feature of <next_state> using your implementation <self.feature_representation>
                next_state = self.feature_representation(next_state)
                
                # Put it into replay buffer
                self.agent.replay_buffer.store(state, action, next_state, reward, ended) 
                

                if len(self.agent.replay_buffer) > self.hp.batch_size and sum(self.collected_rewards) > 0:
                    
                    ## COMPLETE ##
                    # use <self.agent.apply_SGD> implementation to update the online DQN
                    self.agent.apply_SGD(ended)
                
                    # Update target-network weights
                    if total_steps % self.hp.targetDQN_update_rate == 0:
                        ## COMPLETE ##
                        # Copy the online DQN into the Target DQN using what you implemented in Class Agent
                        self.agent.update_target()
                
                state = next_state
                episode_reward += reward
                step_size +=1
                            
            # 
            self.collected_rewards.append(episode_reward)                     
            total_steps += step_size
                                                                           
            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()
                            
            # Print Results of the Episode
            printout = (f"Episode: {episode}, "
                      f"Total Time Steps: {total_steps}, "
                      f"Trajectory Length: {step_size}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.2f}")
            print(printout)
        self.agent.save(self.hp.save_path + '.pth')
        self.plot_learning_curves()
                                                                    

    def play(self):  
        """                
        play with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        """
           
        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        self.agent.onlineDQN.eval()
        
        # Playing 
        for episode in range(1, self.hp.num_test_episodes+1):         
            state, _ = self.env.reset()
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0
                                                           
            while not ended and not truncated:
                ## COMPLETE ##
                # Find the feature of <state> using your implementation <self.feature_representation>
                # Act greedy and find <action> using what you implemented in Class Agent
                state = self.feature_representation(state)
                action = self.agent.greedy(state)
                
                
                next_state, reward, ended, truncated, _ = self.env.step(action)
                                
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print Results of Episode            
            printout = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, ")
            print(printout)
            
        pygame.quit()
        
    ############## THIS METHOD HAS BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        moving_average = np.convolve(self.collected_rewards, np.ones(100)/100, mode='valid')
        
        plt.figure()
        plt.title("Reward")
        plt.plot(self.collected_rewards, label='Reward', color='gray')
        plt.plot(moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Save the figure
        plt.savefig(f'./Reward_vs_Episode_{self.hp.map_size}_x_{self.hp.map_size}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close() 
        
                
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")
        
       # Save the figure
        plt.savefig(f'./Learning_Curve_{self.hp.map_size}_x_{self.hp.map_size}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()        
        

