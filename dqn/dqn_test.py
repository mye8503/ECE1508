import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims, fc2_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()

        # Record hyperparameters
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name

        # Create checkpoint path
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Create an input layer, a hidden layer, an output layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # Set optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Set loss function to be mean squared error
        self.loss = nn.MSELoss()

        # Set device type
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # The last layer do not require activation,
        # because action values Q(s,a) can be negative with large magnitude
        action_values = self.fc3(x)
        return action_values

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initialize memories for samples (s, a, r, s', done)
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        # Overwrite previous memory if #samples exceeds memory size
        index = self.mem_cntr % self.mem_size

        # Record sample into memory
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 fc1_dims, fc2_dims, mem_size, batch_size,
                 eps_min=0.01, eps_dec=5e-7,
                 replace_target_cnt=1000,
                 algo=None, env_name=None, chkpt_dir='tmp/dqn'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace_target_cnt
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # ---- online network ----
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   fc1_dims=self.fc1_dims,
                                   fc2_dims=self.fc2_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   chkpt_dir=self.chkpt_dir)

        # ---- target network ----
        self.q_target = DeepQNetwork(self.lr, self.n_actions,
                                     input_dims=self.input_dims,
                                     fc1_dims=self.fc1_dims,
                                     fc2_dims=self.fc2_dims,
                                     name=self.env_name + '_' + self.algo + '_q_target',
                                     chkpt_dir=self.chkpt_dir)

        # initial hard copy
        self.q_target.load_state_dict(self.q_eval.state_dict())

        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        return (T.tensor(state).to(self.q_eval.device),
                T.tensor(action).to(self.q_eval.device),
                T.tensor(reward).to(self.q_eval.device),
                T.tensor(new_state).to(self.q_eval.device),
                T.tensor(done).to(self.q_eval.device))

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        states, actions, rewards, new_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        # ---- Q(s,a) prediction ----
        q_pred = self.q_eval(states)[indices, actions]

        # ---- Q_target uses target network ----
        q_next = self.q_target(new_states).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        # ---- loss ----
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        # ---- update target network periodically ----
        self.learn_step_counter += 1
        self.replace_target_network()

        self.decrement_epsilon()


env = gym.make('MountainCar-v0', render_mode=None)

agent = DQNAgent(
    gamma=0.99,
    epsilon=1,
    lr=0.0001,
    input_dims=(env.observation_space.shape),
    fc1_dims=64,
    fc2_dims=64,
    n_actions=env.action_space.n,
    mem_size=50000,
    eps_min=0.1,
    batch_size=64,
    eps_dec=1e-5,
    chkpt_dir='models/',
    algo='DQNAgent',
    env_name=env.spec.id
)

n_episodes = 100000
scores, eps_history = [], []
best_score = -np.inf
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

# --- training loop ---
for episode in tqdm(range(n_episodes), desc="Training Progress"):
    observation, _ = env.reset()
    done = False
    score = 0

    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Optional reward shaping
        if new_observation[0] >= 0.5:
            reward += 100

        agent.store_transition(observation, action, reward, new_observation, done)
        agent.learn()

        observation = new_observation
        score += reward

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:pass
            # agent.save_models()

    if (episode + 1) % 5000 == 0:
        print(f'Episode {episode+1}, '
              f'Score: {score:.1f}, '
              f'Avg score (last 5000): {avg_score:.1f}, '
              f'Epsilon: {agent.epsilon:.3f}')

env.close()

# --- Save training results to pickle file ---
results_path = 'mountaincar_dqn_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump({
        'scores': scores,
        'eps_history': eps_history
    }, f)

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(scores, label='Score per Episode')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN on MountainCar-v0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()