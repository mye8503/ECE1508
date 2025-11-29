import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from Hyperparameters import Hyperparameters

# implement a model to trade stocks using the equal weighting strategy
class EQ_Agent():
    def __init__(self, hyperparameters:Hyperparameters):
        self.hp = hyperparameters


    def feature_representation(self, state:int):
        """
        represent the feature of the state
        We simply use tokenization
        """
        ## COMPLETE ##
        # Write a function that get the state and returns the feature
        # We simply use tokenization
        allocation = self.env.observation_space['portfolio_allocation']
        opens = self.env.observation_space['stock_opens']
        feature[state] = 1
        return feature


    def play(self):                   
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
                # state = self.feature_representation(state)
                # action = self.agent.greedy(state)  


                
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
