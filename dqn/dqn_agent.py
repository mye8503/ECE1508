import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections import deque

from dqn_hyperparameters import Hyperparameters


######### YOU DON'T NEED TO MODIFY THIS CLASS -- IT'S ALREADY COMPLETED ###############
class Replay_Buffer():
    """
    Experience Replay Buffer to store experiences
    """
    def __init__(self, size, device):


        self.device = device
        
        self.size = size # size of the buffer
        
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.terminals = deque(maxlen=size)
        
        
    def store(self, state, action, next_state, reward, terminal):
        """
        Store experiences to their respective queues
        """      
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        
        
    def sample(self, batch_size):
        """
        Sample from the buffer
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, terminals
    
    
    def __len__(self):
        return len(self.terminals)
    
########################################################

################### START FROM HERE ################################
class DQN(nn.Module):
    """
    The Deep Q-Network (DQN) model 
    Implement the MLP described in the assignment 
    """
    
    def __init__(self, num_actions, feature_size):
        super(DQN, self).__init__()
                                                          
        ## COMPLETE
        self.num_actions = num_actions
        self.feature_size = feature_size

        # First hidden layer has width 16 with all neurons activated by ReLU
        self.fc1 = nn.Linear(self.feature_size, 16)
        # Second hidden layer has width 8 with all neurons activated by ReLU
        self.fc2 = nn.Linear(16, 8)
        # Output layer has size number of actions and has no activation
        self.output = nn.Linear(8, self.num_actions)

        self.relu = nn.ReLU()        
        
    def forward(self, x):
        ## COMPLETE
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x
        
class Agent:
    """
    Implementing Agent DQL Algorithm
    """
    
    def __init__(self, env:gym.Env, hyperparameters:Hyperparameters, device = False):
        
        # Some Initializations
        if not device:
            if torch.backends.cuda.is_built():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters  


        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0
        self.episode_counts = 0

        self.action_space  = env.action_space
        self.feature_space = env.observation_space
        self.replay_buffer = Replay_Buffer(self.hp.buffer_size, device = self.device)
        
        # Initiate the online and Target DQNs
        ## COMPLETE ##
        self.onlineDQN = DQN(self.action_space.n, self.feature_space.shape[0]).to(self.device)
        self.targetDQN = DQN(self.action_space.n, self.feature_space.shape[0]).to(self.device)
        self.update_target()

        self.loss_function = nn.MSELoss()

        ## COMPLETE ## 
        # set the optimizer to Adam and call it <self.optimizer>, i.e., self.optimizer = optim.Adam()
        self.optimizer = torch.optim.Adam(self.onlineDQN.parameters(), lr=self.hp.learning_rate)
                

    def epsilon_greedy(self, state):
        """
        Implement epsilon-greedy policy
        """
        ## COMPLETE ##
        # This function should return the action chosen by epsilon greedy algorithm # 
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.greedy(state)

    def greedy(self, state):
        """
        Implement greedy policy
        """ 
        ## COMPLETE ##
        # This function should return the action chosen by greedy algorithm # 
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.onlineDQN(state)
        action = torch.argmax(q_values).item()
        return action
   

    def apply_SGD(self, ended):
        """
        Train DQN
            ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
            calculate the loss of the episode.
        """ 
        
        # Sample from the replay buffer
        states, actions, next_states, rewards, terminals = self.replay_buffer.sample(self.hp.batch_size)
                    
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)       
      
        ## COMPLETE ##
        # Compute <Q_hat> using the online DQN
        Q_hat = self.onlineDQN(states).gather(1, actions)

        with torch.no_grad():   
            ## COMPLETE ##         
            # Compute the maximum Q-value for off-policy update and call it <next_target_q_value> 
            next_target_q_value = self.targetDQN(next_states).max(1)[0].unsqueeze(1)
            
        
        next_target_q_value[terminals] = 0 # Set Q-value for terminal states to zero
        ## COMPLETE ##
        # Compute the Q-estimator and call it <y>
        y = rewards + self.hp.discount_factor * next_target_q_value
        loss = self.loss_function(Q_hat, y) # Compute the loss
        
        # Update the running loss and learned counts for logging and plotting
        self.current_loss += loss.item()
        self.episode_counts += 1

        if ended:
            episode_loss = self.current_loss / self.episode_counts # Average loss per episode
            # Track the loss for final graph
            self.loss_list.append(episode_loss) 
            self.current_loss = 0
            self.episode_counts = 0
        
        # Apply backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        

        
        # Clip the gradients
        # It's just to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)
        
        ## COMPLETE ###
        # Update DQN by using the optimizer: <self.optimizer>
        self.optimizer.step()
 
    ############## THE REMAINING METHODS HAVE BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def update_target(self):
        """
        Update the target network 
        """
        # Copy the online DQN into target DQN
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

    
    def update_epsilon(self):
        """
        reduce epsilon by the decay factor
        """
        # Gradually reduce epsilon
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)
        

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention
        This can be used for later test of the trained agent
        """
        torch.save(self.onlineDQN.state_dict(), path)