"""Graphs PPO training data in ppo_data.pkl using matplotlib.pyplot.

3 graphs are produced from this script.
The first one contains:
    - Reward per episode. Higher reward indicates better performance against
        the opponent, while lower reward indicates worse performance.
    - Policy log probabilities across each entire episode. Higher probabilities
        indicate the move is chosen confidently, where lower probabilities
        indicate less confidence.
    - Actor and critic losses per episode. The actor is attempting to decrease
        as much as possible, and the critic is attempting to get as close to
        0 as possible.
    - Round length of each episode. Round length does not necessarily correlate
        to policy performance, since a strong policy could win games quickly or
        strategically prolong them.

The second graph shows the average advantages (TD(0) error) for each episode.
    - High advantages indicate the model is doing better than expected, while
    low advantages indicate it is doing worse than expected.
The third graph shows the actions played throughout each episode, with
the purpose of seeing if the model is overfitting:
    - If certain moves are rarely played and others are played often, it
    suggests that the policy is overfitting to (picking) those moves
    regardless of the state.

Methods:
    get_moving_avg: Compute the moving average of data for each episode.
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from core import all_moves
from statistics import mean


def get_moving_avg(data: list, period: int) -> list[float]:
    """Compute the moving average of data for each episode.

    If the number of data points is less than the given period,
    the average is taken using all available data points instead.

    Args:
        data (list): The data to be analyzed.
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


# Try to load policy data.
try:
    csv = pd.read_pickle('../ppo_data.pkl')
except FileNotFoundError:
    print('Could not find the data file. Ensure it is correctly named. Quitting...')
    quit()

tp_map = {
    'red':   ((0.0, 0.0, 0.0),
              (0.5, 0.0, 0.0),
              (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 1.0, 1.0),
              (0.5, 0.0, 0.0),
              (1.0, 0.0, 0.0))
}
tp_map = mpl.colors.LinearSegmentedColormap('', tp_map)

# Graph the policy's reward
plt.subplot(2, 2, 1)
plt.grid(which='both')
# Takes the last reward instead of the sum, since the last is indicative of win/loss.
csv['Reward'] = [rew[-1] for rew in csv['Reward']]
plt.scatter(list(csv['Episode']), list(csv['Reward']), s=1.5, label='Reward')
rw_avg = get_moving_avg(list(csv['Reward']), 300)
plt.plot(list(csv['Episode']), rw_avg, '-k', label=f"Avg Reward/300 Episodes")
plt.plot(list(csv['Episode']), get_moving_avg(list(csv['Reward']), 100), ':k', alpha=0.3, label=f"Avg Reward/100 Episodes")
plt.title(f"Model Reward per Episode. Final AVG (300): {rw_avg[-1]:.2f}")
# Player is using leafbugs, opponent is using thugs:
# Against SampleMove Opponent: -0.089 SM, 0.243 SWM, 0.933, BFAI, 0.976 ATK+TP
# Against BugFablesAI Opponent: -0.932 SM, -0.849 SWM, 0.182, BFAI, 0.464 ATK+TP
# plt.hlines(-0.932, 0, len(csv['Reward']), 'red', zorder=1, label='Avg SampleMove Reward')
# plt.hlines(-0.849, 0, len(csv['Reward']), 'orange', zorder=1, label='Avg SampleWeightedMove Reward')
# plt.hlines(0.182, 0, len(csv['Reward']), 'yellow', zorder=1, label='Avg BugFablesAI Reward')
# plt.hlines(0.464, 0, len(csv['Reward']), 'green', zorder=1, label='Avg MaxATKandTP Reward')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncols=2)
# print(sum(csv['Reward']) / len(csv['Reward']))
plt.gca().set_axisbelow(True)

# Graph the policy's log probabilities
ax = plt.subplot(2, 2, 2)
plt.grid(which='both')
x = []
y = []
prob_colors = []
for i, probs in zip(csv['Episode'], csv['Log Probs']):
    x.extend([i] * len(probs))
    y.extend(probs)
    # Get normalized TP for each round (2-10 = 0-1)
    TP_norm = [(min(i, 8) / 8) for i in range(len(probs))]
    prob_colors.extend([tp_map(t) for t in TP_norm])
plt.scatter(x, y, s=1.5, c=prob_colors, label='Action Log Probs')
plt.plot(list(csv['Episode']), get_moving_avg([mean(a) for a in csv['Log Probs']], 100), '-k', label='Avg Action Log Probs/100 Episodes')
plt.title(f"Action Log Probs per Episode")
plt.ylim(-19.25, 0.85)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1))
plt.colorbar(mpl.cm.ScalarMappable(plt.Normalize(2, 10), tp_map), ax=ax,  aspect=35, fraction=0.02, pad=0.02, label='Round TP')
plt.gca().set_axisbelow(True)

# Graph the models' loss
ax = plt.subplot(2, 2, 3)
plt.grid(which='both')
if 'Actor Loss' in csv:  # Double-head policy
    plt.scatter(list(csv['Episode']), csv['Actor Loss'], s=1.5, c='g', label='Actor Loss')
    plt.scatter(list(csv['Episode']), csv['Critic Loss'], s=1.5, c='r', label='Critic Loss')
    plt.title('Actor and Critic Loss per Episode')
else:  # Shared policy
    plt.scatter(list(csv['Episode']), csv['Loss'], s=1.5, c='b', label='Shared Loss')
    plt.title('Shared Loss per Episode (A+V+E)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.gca().set_axisbelow(True)

# Graph the round lengths
rew_map = ['g' if rew > 0 else 'r' for rew in csv['Reward']]
plt.subplot(2, 2, 4)
plt.grid(which='both')
plt.scatter(list(csv['Episode']), list(csv['Round Length']), s=1.5, c=rew_map, label='Round Length')
plt.plot(list(csv['Episode']), get_moving_avg(list(csv['Round Length']), 100), '-k', label='Avg Round Length/100 Episodes')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.title("Round Length per Episode")
plt.gca().set_axisbelow(True)
plt.show()

# Graph the advantages, if possible.
if 'Advantages' in csv:
    plt.grid(which='both')
    x = []
    y = []
    adv_colors = []
    for i, adv in zip(csv['Episode'], csv['Advantages']):
        if adv is None:
            adv = []
        else:
            adv = [adv]
        x.extend([i] * len(adv))
        y.extend(adv)
        # Get normalized TP for each round (2-10 = 0-1)
        TP_norm = [(min(i, 8) / 8) for i in range(len(adv))]
        adv_colors.extend([tp_map(t) for t in TP_norm])
    plt.scatter(x, y, s=1.5, c=adv_colors, label='Advantages')
    plt.legend()
    plt.title("Advantages per Episode")
    plt.gca().set_axisbelow(True)
    plt.show()

# Graph each action taken in each episode.
plt.grid(which='both')
x = []
y = []
action_colors = []
for i, actions in zip(csv['Episode'], csv['Actions']):
    x.extend([i] * len(actions))
    y.extend(actions)
    # Get normalized TP for each round (2-10 -> 0-1)
    TP_norm = [(min(i, 8) / 8) for i in range(len(actions))]
    action_colors.extend([tp_map(t) for t in TP_norm])

plt.scatter(x, y, s=2, c=action_colors, label='Actions per Episode')
plt.xlabel('Episode')
plt.ylabel('Move Index')
plt.yticks(range(len(all_moves)), [f"{i} {hand}" for i, hand in enumerate(all_moves)])
plt.gca().set_axisbelow(True)
plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
plt.colorbar(mpl.cm.ScalarMappable(plt.Normalize(2, 10), tp_map), ax=plt.gca(), aspect=35, fraction=0.02, pad=0.02, label='Round TP')
plt.show()
