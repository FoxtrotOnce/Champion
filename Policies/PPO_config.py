"""Configuration file containing hyperparameters used for training a PPO.

Each parameter controls the PPO's training behavior.
Parameters are grouped for clarity.

Training Duration:
    episodes (int): How many episodes (games) the PPO will train for.

Model Learning:
    batch_size (int): How many episodes the PPO collects data for before fitting to it.
    learning_rate (float): Gradient step size.

Loss Computation:
    c2 (float): Weighted entropy loss.
    gamma (float): Discount factor for future rewards.
    epsilon (float): Clipping value for the policy ratio.
"""

episodes = 10000

batch_size = 1
learning_rate = 0.001

c2 = 0.03
gamma = 1.0
epsilon = 0.3
