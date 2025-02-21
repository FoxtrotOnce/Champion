"""Defines model architecture, and implements model training for Spy Cards.

All model operations are defined and used here.
Model data is stored in a dqn_data.csv file, and can be viewed in GraphData.py.
The model trains via a training loop that continuously picks moves to play,
for multiple games.

Classes:
    DQNAgent: Create an agent to perceive the game and make decisions.
    Slice: Custom TensorFlow layer for slicing input tensors.

Methods:
    mask_q_values: Mask invalid q-values.
    create_model: Create and compile a new TensorFlow model.
"""

import core
import config
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque
import random


class DQNAgent:
    """Create an agent to perceive the game and make decisions.

    The agent interacts with the primary model to predict optimal actions by
    using q-values. The model uses epsilon-greedy to explore or exploit actions
    during training, and a replay buffer for training on seen experiences.

    Attributes:
        model (tf.keras.Model): The primary model used for predictions.
        target_model (tf.keras.Model): The target model used for more stable
            predictions.
        replay_buffer (deque): Stores seen experiences (states) for training
            the model on past decisions.
        epsilon (float): Probability of exploring (taking non-model actions).

    Methods:
        update_target_model: Sync the target model's weights with the primary
            model's.
        act: Choose an action using epsilon-greedy.
        predict: Predict the best action to take.
        replay: Train the model on batches of previous experiences.
    """

    def __init__(self, input_size: tuple, output_size: int):
        """Create new models for the agent.

        Args:
            input_size (tuple): The shape of the input array.
            output_size (int): How many moves to choose from.
        """
        self.model = create_model(input_size, output_size)
        self.target_model = create_model(input_size, output_size)
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        self.update_target_model()
        self.epsilon = epsilon

    def update_target_model(self) -> None:
        """Sync the target model's weights with the primary model's.

        The purpose of a target model is to make stable predictions.
        """
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> tuple[int, bool, float]:
        """Choose an action using epsilon-greedy.

        The model explores with a probability of epsilon, choosing an ISMCTS
        move.
        Otherwise, it exploits the action with the highest predicted Q-value.

        Args:
            state (np.ndarray): The state to predict.

        Returns:
            tuple[int, bool, float]: The results of the action:
                action (int): The chosen move's index.
                model_move (bool): If the was made by the model.
                q_val (float): The Q-value of the action.
                    This value is -inf if the action was not made by the model.
        """
        if random.random() <= self.epsilon:
            return core.ISMCTS(env, 35), False, float('-inf')

        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        mask_q_values(env.player_hand, env.TP, q_values)

        return np.argmax(q_values), True, np.max(q_values)

    def predict(self, state: np.ndarray, show_qvals: bool = False) -> tuple[int, float]:
        """Predict what action to take.

        This function always predicts using the model.
        The action with the highest predicted q-value is the chosen action.

        Args:
            state (np.ndarray): The state to predict.
            show_qvals (bool): Whether to print each action's q-value.

        Returns:
            tuple[int, float]: The results of the prediction:
                action (int): The predicted action.
                q_val (float): The q-value of the action.
        """
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        mask_q_values(env.player_hand, env.TP, q_values)
        action, q_val = np.argmax(q_values), np.max(q_values)

        if show_qvals:
            print("Predicted Q-values:")
            for move, q_val in zip(core.all_moves, q_values):
                print(move, q_val)
            print("Chosen Action:", core.all_moves[action])
        return action, q_val

    def replay(self, episode: int) -> float:
        """Train the model on batches of previous experiences.

        Sample a batch of previous experiences, predict Q-values using the
        target model, and then fit the primary model with the calculated values.

        Args:
            episode (int): The current episode number.

        Returns:
            loss (float): The loss of the model after fitting.
                Return -inf if there are not enough samples for a batch.
        """
        # Don't train if there aren't enough samples for a batch
        if len(self.replay_buffer) < config.batch_size:
            return float('-inf')
        # Samples a random batch of experiences
        minibatch = random.sample(self.replay_buffer, k=config.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # Fit the model to previous q-values
        next_q_values = self.target_model.predict(next_states, batch_size=config.batch_size, verbose=0)
        target_q_values = rewards + config.gamma * np.max(next_q_values, axis=1) * (1 - dones)
        q_values = self.model.predict(states, batch_size=config.batch_size, verbose=0)
        q_values[np.arange(config.batch_size), actions] = target_q_values
        self.model.fit(states, q_values, batch_size=config.batch_size, epochs=1, verbose=0)

        # Decay the exploration probability (epsilon)
        if episode >= config.epsilon_decay_start and self.epsilon > config.epsilon_min:
            self.epsilon *= config.epsilon_decay
        return self.model.history.history['loss'][0]


def mask_q_values(hand: list[core.Card], TP_limit: int, q_values: np.ndarray) -> None:
    """Mask invalid q-values.

    Modifies q_values in-place.
    Masks invalid q-values (actions) to -inf that either use a card not present
    in the hand, or exceed the TP limit.

    Args:
        hand (list[core.Card]): A list of the cards in the player's hand.
            Used for checking available cards.
        TP_limit (int): The total TP cost of the move cannot exceed this limit.
        q_values (np.ndarray): The Q-values to modify in-place.
    """
    for i, card_indices in enumerate(core.all_moves):
        # Mask move if it requires a card that isn't in the hand
        if i > 0 and card_indices[-1] + 1 > len(hand):
            q_values[i] = float('-inf')
            continue
        # Mask move if the cost exceeds the TP limit
        TP_used = 0
        for card_index in card_indices:
            TP_used += hand[card_index].TP
            if TP_used > TP_limit:
                q_values[i] = float('-inf')
                break


class Slice(tf.keras.layers.Layer):
    """Custom TensorFlow layer for slicing input tensors.

    Meant for slicing individual inputs of a bigger input array to be processed
    independently.

    Equivalent to:
    tf.keras.layers.Lambda(lambda inp: tf.slice(inp, self.begin, self.size))

    Args:
        begin (list | tuple): The starting indices for each dim slice. (0-indexed)
        size (list | tuple): The size of each dim slice. (1-indexed)

    See https://www.tensorflow.org/api_docs/python/tf/keras/Layer for more
    details.
    """
    def __init__(self, begin: list | tuple, size: list | tuple):
        super().__init__()
        self.begin = begin
        self.size = size

    def get_config(self) -> dict:
        """Return the layer's configuration.

        Saves layer configuration for model saving, loading, and reconstruction.

        Returns:
            config (dict): A dictionary representing the layer's configuration.
        """
        config = super().get_config()
        config.update({
            'begin': self.begin,
            'size': self.size
        })
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Slice the input tensor.

        Args:
            inputs (tf.Tensor): The tensor to be sliced

        Returns:
            tf.Tensor: The sliced input tensor.
        """
        return tf.slice(inputs, self.begin, self.size)


def create_model(input_shape: tuple, output_shape: int) -> tf.keras.Model:
    """Create and compile a new TensorFlow model.

    These layers control the model's internal behavior.
    Each input is sliced from the input array for individual processing, and
    then concatenated together to be processed again, before determining an
    output.

    The model uses the Adam optimizer and MSE loss.

    Args:
        input_shape (tuple): The shape of the input array.
        output_shape (int): How many moves to choose from.

    Returns:
        tf.keras.Model: The compiled model.
    """
    input_layer = tf.keras.layers.Input(input_shape)
    deck = Slice([0, 0, 0], [-1, 15, 54])(input_layer)
    tp = Slice([0, 15, 2], [-1, 1, 9])(input_layer)
    hand = Slice([0, 15+1, 0], [-1, 5, 54])(input_layer)
    o_hand = Slice([0, 15+1+5, 0], [-1, 3, 5])(input_layer)
    hp = Slice([0, 15+1+5+3, 0], [-1, 1, 6])(input_layer)
    o_hp = Slice([0, 15+1+5+3+1, 0], [-1, 1, 6])(input_layer)

    deck = tf.keras.layers.Dense(64, activation='relu')(deck)
    tp = tf.keras.layers.Dense(64, activation='relu')(tp)
    hand = tf.keras.layers.Dense(64, activation='relu')(hand)
    o_hand = tf.keras.layers.Dense(64, activation='relu')(o_hand)
    hp = tf.keras.layers.Dense(64, activation='relu')(hp)
    o_hp = tf.keras.layers.Dense(64, activation='relu')(o_hp)

    concat = tf.keras.layers.Concatenate(axis=1)([deck, tp, hand, o_hand, hp, o_hp])
    flatten = tf.keras.layers.Flatten()(concat)
    dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))(flatten)
    output_layer = tf.keras.layers.Dense(output_shape, activation='linear')(dense)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    # Compile model with Adam optimizer and MSE loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='mse'
    )

    # Attempt to load a model if there is one present.
    try:
        model.load_weights('dqn_model.keras')
    except (FileNotFoundError, ValueError) as err:
        if isinstance(err, ValueError):
            print('Could not load weights. Did you forget to modify the architecture?')
            print(err)
    return model


# Try to load model data, and resume training from where it left off.
try:
    model_data = pd.read_csv('dqn_data.csv')
    initial_episode = model_data['Episode'].iloc[-1] + 1
    epsilon = max(config.epsilon_min, config.epsilon_decay ** max(initial_episode - config.epsilon_decay_start, 0))
except FileNotFoundError:
    model_data = pd.DataFrame()
    initial_episode = 0

# Define player and opponent decks
p_deck = [
    core.cards_by_name['Devourer'],
    core.cards_by_name['Zasp'],
    core.cards_by_name['Mothiva'],
    core.cards_by_name['Leafbug Archer'],
    core.cards_by_name['Leafbug Archer'],
    core.cards_by_name['Leafbug Archer'],
    core.cards_by_name['Leafbug Ninja'],
    core.cards_by_name['Leafbug Ninja'],
    core.cards_by_name['Leafbug Ninja'],
    core.cards_by_name['Leafbug Ninja'],
    core.cards_by_name['Leafbug Clubber'],
    core.cards_by_name['Leafbug Clubber'],
    core.cards_by_name['Numbnail'],
    core.cards_by_name['Numbnail'],
    core.cards_by_name['Venus\' Bud']
]
o_deck = [
    core.cards_by_name['Tidal Wyrm'],
    core.cards_by_name['Astotheles'],
    core.cards_by_name['Monsieur Scarlet'],
    core.cards_by_name['Thief'],
    core.cards_by_name['Thief'],
    core.cards_by_name['Thief'],
    core.cards_by_name['Bandit'],
    core.cards_by_name['Bandit'],
    core.cards_by_name['Bandit'],
    core.cards_by_name['Burglar'],
    core.cards_by_name['Burglar'],
    core.cards_by_name['Burglar'],
    core.cards_by_name['Ironnail'],
    core.cards_by_name['Ironnail'],
    core.cards_by_name['Venus\' Bud']
]

env = core.SpyCardsEnv(p_deck, o_deck)
# There are 32 different possible card hands (nCr(5,0)+nCr(5,1)...+nCr(5,5)=32)
agent = DQNAgent(input_size=(15 + 1 + 5 + 3 + 1 + 1, 54), output_size=32)


# Train the model
for e in range(initial_episode, initial_episode + config.episodes):
    state = env.reset()
    done = False
    total_reward = 0
    model_action_sum = 0
    max_q_val = 0
    actions = []

    # Play one episode
    while not done:
        action, was_model_action, q_val = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        model_action_sum += was_model_action
        max_q_val = max(max_q_val, q_val)
        actions.append(action)

    # Save the episode to the replay buffer, and save data.
    loss = agent.replay(e)
    model_data = pd.concat([model_data, pd.DataFrame({
        'Episode': [e],
        'Action Ratio': [model_action_sum / env.total_rounds],
        'Reward': [total_reward],
        'Loss': [loss],
        'Round Length': [env.total_rounds],
        'Actions': [action],
        'Max Q-Value': [max_q_val]
    })])

    print(f"Episode: {e} - Action Ratio: {model_action_sum / env.total_rounds:.3f} ({model_action_sum}/{env.total_rounds}) - Reward: {total_reward:.2f} - Loss: {loss:.4f} - Max Q-Val: {max_q_val:.4f}")

    if (e + 1) % config.target_update_rate == 0:
        agent.update_target_model()
    if (e + 1) % 50 == 0:
        agent.model.save('dqn_model.keras')
        model_data.to_csv('dqn_data.csv', index=False)
