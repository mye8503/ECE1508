class Hyperparameters():
    def __init__(self, map_size):
        self.map_size = map_size
        self.RL_load_path = f'./{map_size}x{map_size}_final_weights.pth'
        self.save_path = f'./{map_size}x{map_size}_final_weights'
        self.learning_rate = 5e-4
        self.discount_factor = 0.9
        self.batch_size = 32
        self.targetDQN_update_rate = 10
        self.num_episodes = 3000
        self.num_test_episodes = 10
        self.epsilon_decay = 0.999
        self.buffer_size = 10000

    def change(self, map_size, batch_size = 32, learning_rate = 5e-4, num_episodes = 3000, epsilon_decay = 0.999):
        '''
        This method can change
        map_size, 
        Also can change the following argument if called:
        batch_size , learning_rate , num_episodes
        '''
        self.map_size = map_size
        self.batch_size = batch_size 
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.epsilon_decay = epsilon_decay
