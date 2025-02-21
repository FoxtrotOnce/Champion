"""Graphs model training data in dqn_data.csv using matplotlib.pyplot.

2 graphs are produced from this script.
The first one contains:
    - Reward per episode. Higher reward indicates better performance against
        the opponent, while lower reward indicates worse performance.
    - Max q-value across each entire episode. Increasing max q-values indicate
        the model overestimates its actions, while decreasing values indicate
        that the model is making more deliberate actions.
    - Loss per episode. Lower loss indicates the reward aligns with the model's
        prediction, while higher loss indicates the model is not accurately
        predicting the reward it receives.
    - Round length of each episode. Round length does not necessarily correlate
        to model performance, since a strong model could win games quickly or
        strategically prolong them.

The second graph shows the max q-values for each move within each episode.
The purpose of the second graph is to see if the middle is overfitting:
If certain moves are rarely played and others are played often, it suggests
that the model is overfitting to (picking) those moves regardless of the state.

Methods:
    get_moving_avg: Compute the moving average of data for each episode.
    get_moving_qnt: Compute the moving quantile of data for each episode.
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from core import all_moves
from config import batch_size, epsilon_decay_start


def get_moving_avg(data, period: int) -> list[float]:
    """Compute the moving average of data for each episode.

    If the number of data points is less than the given period,
    the average is taken using all available data points instead.
    
    Args:
        data: The data to be analyzed.
        period (int): The period of the moving average.
    
    Returns:
        list[float]: The moving average of the data for each episode.
    """
    moving_avg = []
    window_sum = 0
    for i in csv['Episode']:
        if i > period:
            window_sum -= data[i - period]
        window_sum += data[i]
        moving_avg.append(window_sum / min(len(moving_avg) + 1, period))
    return moving_avg


def get_moving_qnt(data, period: int, qnt: float) -> list[float]:
    """Compute the moving quantile of data for each episode.

    If the number of data points is less than the given period,
    the quantile is taken using all available data points instead.
    
    Args:
        data: The data to be analyzed.
        period (int): The period of the moving quantile.
        qnt (float): The quantile size. (0-1)

    Returns:
        list[float]: The moving quantile of the data for each episode.
    """
    moving_qnt = []
    for i in csv['Episode']:
        moving_qnt.append(np.quantile(data[max(0,i - period):i + 1], qnt))
    return moving_qnt


# Try to load model data.
try:
    csv = pd.read_csv('dqn_data.csv')
except FileNotFoundError:
    print('Could not find the data file. Ensure it is correctly named. Quitting...')
    quit()

# cm_ar is a colormap for Action Ratio. Blue indicates less model actions, red indicates more model actions.
cm_ar = [(ar, 0, 1 - ar) for ar in csv['Action Ratio']]

# cm_rw is a colormap for Reward. It is red if the reward is -1 (model lost) or green if it's 1 (model won).
cm_rw = ['g' if rw == 1 else 'r' for rw in csv['Reward']]


# Graph the model's reward
plt.subplot(2, 2, 1)
plt.grid(which='both')
plt.scatter(list(csv['Episode']), csv['Reward'], s=1.5, c=cm_ar, label='Reward')
rw_avg = get_moving_avg(csv['Reward'], 300)
plt.plot(list(csv['Episode']), rw_avg, '-k', label=f"Avg Reward/300 Episodes")
plt.plot(list(csv['Episode']), get_moving_avg(csv['Reward'], 100), ':k', alpha=0.3, label=f"Avg Reward/100 Episodes")
plt.title(f"Model Reward per Episode. Final AVG (300): {rw_avg[-1]:.2f}")
plt.legend()
plt.gca().set_axisbelow(True)

# Graph the model's max q-value
plt.subplot(2, 2, 2)
plt.grid(which='both')
plt.scatter(list(csv['Episode']), csv['Max Q-Value'], s=1.5, c=cm_ar, label='Max Q-Value')
plt.plot(list(csv['Episode']), get_moving_avg(csv['Max Q-Value'], 100), '-k', label='Avg Q-Value/100 Episodes')
plt.title(f"Max Q-Value per Episode")
plt.legend()
plt.gca().set_axisbelow(True)

# Graph the model's loss
ax = plt.subplot(2, 2, 3)
plt.grid(which='both')
plt.scatter(list(csv['Episode']), csv['Loss'], s=1.5, c='g', label='DQN Loss')
plt.plot(list(csv['Episode']), get_moving_avg(csv['Loss'], 200), '--k', linewidth=1.5, label='Avg DQN Loss/200 Episodes')
plt.plot(list(csv['Episode']), get_moving_qnt(csv['Loss'], 200, 0.1), '-k', linewidth=1.5, label='10th Percentile DQN Loss/200 Episodes')
plt.title('DQN Loss per Episode')
try:
    # Setting upper ylim using quantile errors for some reason if there are fewer points than the batch (I think)
    # Going to just use a try/except here until I figure out why.
    plt.ylim(top=np.quantile(csv['Loss'][max(epsilon_decay_start, batch_size):], 0.95))
except ValueError:
    pass
plt.ylim(ax.get_ylim()[1] / -20, ax.get_ylim()[1] * 1.6)
plt.legend()
plt.gca().set_axisbelow(True)

# Graph the round lengths
plt.subplot(2, 2, 4)
plt.grid(which='both')
plt.scatter(list(csv['Episode']), csv['Round Length'], s=1.5, c=cm_rw, label='Round Length')
plt.plot(list(csv['Episode']), get_moving_avg(csv['Round Length'], 100), '-k', label='Avg Round Length/100 Episodes')
plt.legend()
plt.title("Round Length per Episode")
plt.gca().set_axisbelow(True)
plt.show()


# Separately graph the max q-values for each move in each episode.
plt.grid(which='both')
# x and y need to be modified since there are multiple q-values per episode
x = []
y = []
for i, round_q_vals in zip(csv['Episode'], csv['Actions']):
    x.extend([i] * len(round_q_vals))
    y.extend(round_q_vals)

plt.scatter(x, y, s=2, label='Actions per Episode')
plt.xlabel('Episode')
plt.ylabel('Move Index')
plt.yticks(range(len(all_moves)), [f"{i} {hand}" for i, hand in enumerate(all_moves)])
plt.gca().set_axisbelow(True)
plt.legend()
plt.show()
