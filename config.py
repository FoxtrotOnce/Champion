"""Configuration file containing hyperparameters used for training the TensorFlow model.

Each parameter controls the model's training behavior.
Parameters are grouped for clarity.

Training Duration:
    episodes (int): How many episodes (games) the model will train for.

Model Learning:
    replay_buffer_size (int): Maximum stored experiences in memory.
    batch_size (int): How many experiences are sampled from memory each episode.
    learning_rate (float): Step size for gradient descent.
    target_update_rate (int): Frequency (in episodes) for target model updates.
    gamma (float): Discount factor for future rewards.

Exploration:
    epsilon (float): Initial probability of taking non-model actions.
    epsilon_min (float): Minimum exploration probability.
    epsilon_decay (float): How much the epsilon decays each episode.
    epsilon_decay_start (int): Epsilon begins to decay after this episode.
"""

episodes = 30000

replay_buffer_size = 10000
batch_size = 64
learning_rate = 0.001
target_update_rate = 100
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.00
epsilon_decay = 0.994
epsilon_decay_start = 0
