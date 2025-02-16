import tensorflow as tf
import numpy as np
import pandas as pd
from pickle import load, dump
import matplotlib.pyplot as plt
from collections import deque, Counter
from itertools import combinations, chain
from heapq import heappush, heappop
import random

np.set_printoptions(threshold=np.inf)
from timeit import timeit

# tf.keras.config.enable_unsafe_deserialization()

# from pyautogui import press, sleep
# while True:
#     sleep(20)
#     press(['left','right','up','down'][random.randrange(0,4)])


class Stats:
    def __init__(self):
        self.HP = 5
        self.ATK = 0
        self.DEF = 0
        self.Lifesteal = 0
        self.Pierce = 0
        self.Setup = 0
        self.Empower = {
            'Spider': 0,
            'Plant': 0,
            'Bot': 0,
            'Zombie': 0,
            'Bug': 0,
            'Chomper': 0,
            'Fungi': 0,
            'Seedling': 0,
            'Mothfly': 0,
            'Thug': 0,
            'Wasp': 0,
            'Dead Lander': 0,
            'Leafbug': 0
        }

    def reset(self):
        self.ATK = 0
        self.DEF = 0
        self.Lifesteal = 0
        self.Pierce = 0
        self.Setup = 0
        self.Empower = {
            'Spider': 0,
            'Plant': 0,
            'Bot': 0,
            'Zombie': 0,
            'Bug': 0,
            'Chomper': 0,
            'Fungi': 0,
            'Seedling': 0,
            'Mothfly': 0,
            'Thug': 0,
            'Wasp': 0,
            'Dead Lander': 0,
            'Leafbug': 0
        }

    def __repr__(self):
        return f"ATK: {self.ATK} - DEF: {self.DEF} - PI: {self.Pierce} - LS: {self.Lifesteal}"


class Card:
    card_id = 0

    def __init__(self, name: str, TP: int, category: str, effects: tuple, tribes=None):
        if tribes is None:
            tribes = set()
        self.id = Card.card_id
        Card.card_id += 1

        self.name = name
        self.TP = TP
        self.category = category
        self.effects = effects
        self.tribes = tribes
        # Tell the ATK Effects if the card is a battle card so it can be properly numbed
        if category == 'Battle':
            self.effects[0].is_battle = True
        # Tell the Unity Effects the card's name so unity can work properly
        for effect in effects:
            if isinstance(effect, Effect.Unity):
                effect.original_card_name = name
        self.one_hot()

    def one_hot(self):
        key = {
            'Boss': 0,
            'Mini-Boss': 1,
            'Effect': 2,
            'Battle': 3
        }
        # TP + Category + ATK + Effects + Tribes
        state = np.zeros(9 + 4 + 11 + 18 + 13)
        state[self.TP - 1] = 1
        state[9 + key[self.category]] = 1
        atk = 0
        for effect in self.effects:
            if isinstance(effect, Effect.ATK):
                atk = min(10, atk + effect.amount)
            state[9 + 4 + 11 + effect.id] = 1
        state[9 + 4 + atk] = 1
        tribe_indices = {key: i for i, key in enumerate(Stats().Empower)}
        for tribe in self.tribes:
            state[9 + 4 + 11 + 18 + tribe_indices[tribe]] = 1
        return state

    def __repr__(self):
        return f"{self.name} ({self.TP})"


class InheritEffect(type):
    """
    Metaclass that assigns Effect's __lt__ to its nested classes (ex. Setup)
    """
    def __new__(cls, name, bases, attrs):
        for cls_name, cls_obj in attrs.items():
            if isinstance(cls_obj, type):  # Ensure you are inheriting to the nested classes (classes are of type: type)
                setattr(cls_obj, '__lt__', attrs['__lt__'])
        return super().__new__(cls, name, bases, attrs)


class Effect(metaclass=InheritEffect):
    """
    Effect container class for all other effect subclasses.
    Uses the metaclass InheritEffect to assign default attributes in __init__ and assign default methods to all subclasses.
    """

    # __lt__ is used so you can sort effects in queue based on priority
    # All subclasses of Effect inherit this method
    def __lt__(self, other):
        return self._priority < other._priority

    class Setup:
        def __init__(self, amount):
            self._priority = 0
            self.id = 0
            self._amount = amount

        def __repr__(self):
            return f"Setup ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            stats.Setup += self._amount

    class Heal:
        def __init__(self, amount):
            self._priority = 0
            self.id = 1
            self._amount = amount

        def __repr__(self):
            return f"Heal ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            stats.HP = min(5, stats.HP + self._amount)

    class ATK:
        def __init__(self, amount):
            self._priority = 0
            self.id = 2
            self.amount = amount
            # is_battle gets assigned by the Card object
            self.is_battle = False

        def __repr__(self):
            return f"ATK ({self.amount}), Battle: {self.is_battle}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            stats.ATK += self.amount
            # Add the ATK effect to played if the card was a Battle card (for numbing)
            if self.is_battle:
                heappush(played, self.amount)

    class DEF:
        def __init__(self, amount):
            self._priority = 0
            self.id = 3
            self._amount = amount

        def __repr__(self):
            return f"DEF ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            stats.DEF += self._amount

    class Pierce:
        def __init__(self, pierce):
            self._priority = 0
            self.id = 4
            self._amount = pierce

        def __repr__(self):
            return f"Pierce ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            stats.Pierce += self._amount

    class Lifesteal:
        def __init__(self, amount):
            self._priority = 0
            self.id = 5
            self._amount = amount

        def __repr__(self):
            return f"Lifesteal ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            stats.Lifesteal += self._amount

    class Summon:
        def __init__(self, card_name):
            self._priority = 0
            self.id = 6
            self._card_name = card_name

        def __repr__(self):
            return f"Summon {self._card_name}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            card = cards_by_name[self._card_name]
            chosen.append(card)
            counter[card.name] += 1
            for effect in card.effects:
                heappush(queue, effect)

    class Summon_Wasp:
        def __init__(self, amount):
            self._priority = 0
            self.id = 7
            self._amount = amount

        def __repr__(self):
            return f"Summon Wasps ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            for _ in range(self._amount):
                card = random.sample([cards_by_name['Wasp Scout'],
                                      cards_by_name['Wasp Trooper'],
                                      cards_by_name['Wasp Bomber'],
                                      cards_by_name['Wasp Driller']], k=1)[0]
                chosen.append(card)
                counter[card.name] += 1
                for effect in card.effects:
                    heappush(queue, effect)

    class Carmina:
        def __init__(self):
            self._priority = 0
            self.id = 8

        def __repr__(self):
            return f"Carmina"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            card = random.sample([card for card in all_cards if card.category == 'Mini-Boss'], k=1)[0]
            chosen.append(card)
            counter[card.name] += 1
            for effect in card.effects:
                heappush(queue, effect)

    class Coin:
        def __init__(self, effect_heads, effect_tails=None, repetitions=1):
            self._priority = 0
            self.id = 9
            self._effect_heads = effect_heads
            self._effect_tails = effect_tails
            self._repetitions = repetitions

        def __repr__(self):
            if self._effect_tails is None:
                return f"Coin ({self._repetitions}): {self._effect_heads}"
            return f"Coin ({self._repetitions}): {self._effect_heads}/{self._effect_tails}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            for _ in range(self._repetitions):
                if random.random() >= 0.5:
                    heappush(queue, self._effect_heads)
                elif self._effect_tails is not None:
                    heappush(queue, self._effect_tails)

    class If_Card:
        def __init__(self, card_name, effect):
            self._priority = 1
            self.id = 10
            self._card_name = card_name
            self._effect = effect

        def __repr__(self):
            return f"If {self._card_name}: {self._effect}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            if counter[self._card_name] >= 1:
                heappush(queue, self._effect)

    class If_Tribe:
        def __init__(self, tribe, amount, effect):
            self._priority = 1
            self.id = 11
            self._tribe = tribe
            self._amount = amount
            self._effect = effect

        def __repr__(self):
            return f"If {self._tribe} ({self._amount}): {self._effect}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            tribe_cards = 0
            for card in chosen:
                tribe_cards += self._tribe in card.tribes
            if tribe_cards >= self._amount:
                heappush(queue, self._effect)

    class Per_Card:
        def __init__(self, card_name, effect):
            self._priority = 1
            self.id = 12
            self._card_name = card_name
            self._effect = effect

        def __repr__(self):
            return f"Per {self._card_name}: {self._effect}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            for _ in range(counter[self._card_name]):
                heappush(queue, self._effect)

    class VS:
        def __init__(self, tribe, effect):
            self._priority = 1
            self.id = 13
            self._tribe = tribe
            self._effect = effect

        def __repr__(self):
            return f"VS {self._tribe}: {self._effect}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            for card in o_chosen:
                if self._tribe in card.tribes:
                    heappush(queue, self._effect)
                    break

    class Empower:
        def __init__(self, power, tribe):
            self._priority = 1
            self.id = 14
            self._power = power
            self._tribe = tribe

        def __repr__(self):
            return f"Empower +{self._power} ({self._tribe})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            for card in chosen:
                if self._tribe in card.tribes:
                    stats.ATK += self._power

    class Unity:
        def __init__(self, power, tribe):
            self._priority = 1
            self.id = 15
            # original_card_name gets assigned by the Card object
            self.original_card_name = None
            self._power = power
            self._tribe = tribe

        def __repr__(self):
            return f"Unity (+{self._power}, {self._tribe})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            if self.original_card_name in unities:
                return
            for card in chosen:
                if self._tribe in card.tribes:
                    stats.ATK += self._power
            unities.add(self.original_card_name)

    class If_ATK:
        def __init__(self, amount, effect):
            self._priority = 2
            self.id = 16
            self._amount = amount
            self._effect = effect

        def __repr__(self):
            return f"If ATK ({self._amount}): {self._effect}"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            if stats.ATK >= self._amount:
                heappush(queue, self._effect)

    class Numb:
        def __init__(self, amount):
            self._priority = 3
            self.id = 17
            self._amount = amount

        def __repr__(self):
            return f"Numb ({self._amount})"

        def play(self, counter: Counter, queue: list, played: list, o_played: list, chosen: list, stats: Stats, o_chosen: list, o_stats: Stats, unities: set):
            # While there are cards to numb (and you can numb), remove ATK from your opponent
            amount_available = self._amount
            while o_played and amount_available > 0:
                o_stats.ATK -= heappop(o_played)
                amount_available -= 1


class SpyCardsEnv:
    def __init__(self, player_deck: list, opponent_deck: list):
        if len(player_deck) != 15 or len(opponent_deck) != 15:
            raise NotImplementedError(f"Deck size is not 15. p_deck: {len(player_deck)}, o_deck: {len(opponent_deck)}")
        self.TP = None
        self.total_rounds = None
        self.player_deck = player_deck
        self.player_hand = None
        self.player_stats = None
        self.opponent_deck = opponent_deck
        self.opponent_hand = None
        self.opponent_stats = None
        self.done = None
        # lists every possible hand that can be played, using indices
        self.moves = tuple(chain.from_iterable(combinations(range(5), r) for r in range(5+1)))

        self.reset()

    def reset(self):
        self.TP = 2
        self.total_rounds = 0
        self.player_hand = random.sample(self.player_deck, k=3)
        random.shuffle(self.player_hand)
        self.player_stats = Stats()
        self.opponent_hand = random.sample(self.opponent_deck, k=3)
        random.shuffle(self.opponent_hand)
        self.opponent_stats = Stats()
        self.done = False

        return self.get_state()

    def get_state(self):
        card_ranks = {
            'Boss': 0,
            'Mini-Boss': 0,
            'Effect': 0,
            'Battle': 0
        }
        # card info is of size (55,)
        # deck will be (15, 55)
        state = np.zeros((15 + 1 + 5 + 4 + 1 + 1, 55))
        for i, card in enumerate(self.player_deck):
            state[i] = card.one_hot()
        # tp will be (1, 9) (9 is 2-10)
        state[15, self.TP] = 1
        # hand will be (5, 55)
        for i, card in enumerate(self.player_hand):
            state[15 + 1 + i] = card.one_hot()
        # o_hand will be (4, 5) (max 5 cards of 4 categories)
        for card in self.opponent_hand:
            card_ranks[card.category] += 1
        for i, val in enumerate(card_ranks.values()):
            state[15 + 1 + 5 + i, val] = 1
        # hp will be (1, 6) (6 is 0-5)
        state[15 + 1 + 5 + 4, self.player_stats.HP] = 1
        # o_hp will be (1, 6) (6 is 0-5)
        state[15 + 1 + 5 + 4 + 1, self.opponent_stats.HP] = 1
        return state

    def random_move(self):
        """
        Picks cards at random to play.
        """
        possible_moves = []
        for i, card_indices in enumerate(self.moves):
            # Ignore first move (playing nothing) since there's no list to index
            if i == 0:
                possible_moves.append(i)
                continue
            # Set Q-Value to -inf if it requires a card that isn't in the hand
            if card_indices[-1] + 1 > len(self.player_hand):
                continue
            # Set Q-value to -inf if it requires more TP than it available
            TP_used = 0
            for card_index in card_indices:
                TP_used += self.player_hand[card_index].TP
                if TP_used > self.TP:
                    break
            possible_moves.append(i)
        return random.sample(possible_moves, k=1)[0]

    def step(self, action, play_against=False):
        """
        Processes a move given the chosen cards for the player.
        """
        self.player_stats.reset()
        self.opponent_stats.reset()
        # Use Bug Fables enemy AI to pick enemy cards (AI works by drawing left to right until TP is depleted)
        player_chosen_cards = [self.player_hand[i] for i in self.moves[action]]
        if not play_against:
            TP_used = 0
            opponent_chosen_cards = []
            for card in self.opponent_hand:
                if self.TP >= TP_used + card.TP:
                    opponent_chosen_cards.append(card)
                    TP_used += card.TP
        else:
            opponent_chosen_cards = [self.opponent_hand[i] for i in env.moves[agent.predict(self.get_state())[0]]]
        # print()
        # print(f"Player: {player_chosen_cards}")
        # print(f"Opponent: {opponent_chosen_cards}")
        # Remove chosen cards from hands
        for card in player_chosen_cards:
            self.player_hand.remove(card)
        for card in opponent_chosen_cards:
            self.opponent_hand.remove(card)

        # Actually process the round
        # Set-up counters for checking if cards are in a hand, and create queues for playing cards (effects)
        # _played is progressively filled in ONLY BATTLE CARD ATK effects for numbing
        player_counter = Counter()
        player_queue = []
        player_played = []
        player_unities = set()
        for card in player_chosen_cards:
            player_counter[card.name] += 1
            for effect in card.effects:
                heappush(player_queue, effect)
        opponent_counter = Counter()
        opponent_queue = []
        opponent_played = []
        opponent_unities = set()
        for card in opponent_chosen_cards:
            opponent_counter[card.name] += 1
            for effect in card.effects:
                heappush(opponent_queue, effect)

        # Play effects while they exist in queue, effects with the lowest priority are played first.
        while player_queue or opponent_queue:
            if not opponent_queue or (player_queue and player_queue[0] < opponent_queue[0]):
                counter, queue, played, o_played, unities = player_counter, player_queue, player_played, opponent_played, player_unities
                chosen, stats, o_chosen, o_stats = player_chosen_cards, self.player_stats, opponent_chosen_cards, self.opponent_stats
            else:
                counter, queue, played, o_played, unities = opponent_counter, opponent_queue, opponent_played, player_played, opponent_unities
                chosen, stats, o_chosen, o_stats = opponent_chosen_cards, self.opponent_stats, player_chosen_cards, self.player_stats

            p_effect = heappop(queue)
            # print(p_effect, player_counter == counter)
            # print('p:' + str(stats if player_counter == counter else o_stats))
            # print('o:' + str(o_stats if player_counter == counter else stats))
            # print()
            p_effect.play(counter, queue, played, o_played, chosen, stats, o_chosen, o_stats, unities)

        # Raise TP, draw new cards, and determine ending scores
        self.TP = min(10, self.TP + 1)
        self.total_rounds += 1
        # Pick new cards from available cards in the deck. A Counter is used here to know what cards are available.
        for hand, deck in ((self.player_hand, self.player_deck), (self.opponent_hand, self.opponent_deck)):
            can_draw = 2 if hand else 3
            available_cards = Counter(deck)
            for card in hand:
                available_cards[card] -= 1
            while len(hand) < 5 and can_draw > 0:
                picked = random.choices(*zip(*available_cards.items()))[0]
                hand.append(picked)
                available_cards[picked] -= 1
                can_draw -= 1

        # print()
        # print(f"Player Stats: {self.player_stats}")
        # print(f"Opponent Stats: {self.opponent_stats}")
        player_score = max(0, self.player_stats.ATK - max(0, self.opponent_stats.DEF - self.player_stats.Pierce))
        opponent_score = max(0, self.opponent_stats.ATK - max(0, self.player_stats.DEF - self.opponent_stats.Pierce))
        # print()
        # print(f"Player Score: {player_score}")
        # print(f"Opponent Score: {opponent_score}")
        # Return a reward of 1 if the GAME ends in a win for player, -1 if loss, and 0 otherwise.
        if player_score > opponent_score:
            # Player Win
            self.opponent_stats.HP -= 1
            self.player_stats.HP = min(5, self.player_stats.HP + self.player_stats.Lifesteal)
            if self.opponent_stats.HP == 0:
                return self.get_state(), 1, True
        elif opponent_score > player_score:
            # Opponent Win
            self.player_stats.HP -= 1
            self.opponent_stats.HP = min(5, self.opponent_stats.HP + self.opponent_stats.Lifesteal)
            if self.player_stats.HP == 0:
                return self.get_state(), -1, True
        return self.get_state(), 0, False


class DQNAgent:
    def __init__(self, input_size, output_size):
        self.model = create_dqn_model(input_size, output_size)
        self.target_model = create_dqn_model(input_size, output_size)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.update_target_model()
        self.epsilon = epsilon

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    @classmethod
    def remove_invalid(cls, q_values):
        for i, card_indices in enumerate(env.moves):
            # Ignore first move (playing nothing) since there's no list to index
            if i == 0:
                continue
            # Set Q-Value to -inf if it requires a card that isn't in the hand
            if card_indices[-1] + 1 > len(env.player_hand):
                q_values[i] = float('-inf')
                continue
            # Set Q-value to -inf if it requires more TP than it available
            TP_used = 0
            for card_index in card_indices:
                TP_used += env.player_hand[card_index].TP
                if TP_used > env.TP:
                    q_values[i] = float('-inf')
                    break

    def act(self, state):
        """
        Returns (chosen_cards, is_model_move, max_q_val)
        """
        if random.random() <= self.epsilon:
            return env.random_move(), False, float('-inf'), np.full(len(env.moves), float('-inf'))

        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        DQNAgent.remove_invalid(q_values)

        return np.argmax(q_values), True, np.max(q_values), q_values

    def predict(self, state, show_qvals=False):
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        DQNAgent.remove_invalid(q_values)
        action, q_val = np.argmax(q_values), np.max(q_values)

        if show_qvals:
            print("Predicted Q-values:")
            for move, q_val in zip(env.moves, q_values):
                print(move, q_val)
            print("Chosen Action:", env.moves[action])
        return action, q_val

    def replay(self, batch_size, episode):
        # Return the loss as well
        # Don't train if there aren't enough samples for a batch
        if len(self.replay_buffer) < batch_size:
            return None
        # Samples a random batch of experiences
        minibatch = random.sample(self.replay_buffer, k=batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        next_q_values = self.target_model.predict(next_states, batch_size=batch_size, verbose=0)
        target_q_values = rewards + gamma * np.max(next_q_values, axis=1) * (1 - dones)
        q_values = self.model.predict(states, batch_size=batch_size, verbose=0)
        q_values[np.arange(batch_size), actions] = target_q_values
        self.model.fit(states, q_values, batch_size=batch_size, epochs=1, verbose=0)

        # Decay the exploration probability (epsilon)
        if episode >= epsilon_decay_start and self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay
        return self.model.history.history['loss'][0]


class Slice(tf.keras.layers.Layer):
    def __init__(self, begin, size):
        super().__init__()
        self.begin = begin
        self.size = size

    def get_config(self):
        config = super().get_config()
        config.update({
            'begin': self.begin,
            'size': self.size
        })
        return config

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)


def create_dqn_model(input_shape, output_shape):
    # (1, 27, 55)
    # 15 + 1 + 5 + 4 + 1 + 1
    # deck will be (15, 55)
    # tp will be (1, 9) (9 is 2-10)
    # hand will be (5, 55)
    # TODO: model can see opponent's effect category. fix.
    # o_hand will be (4, 5) (max 5 cards of 4 categories)
    # hp will be (1, 6) (6 is 0-5)
    input_layer = tf.keras.layers.Input(input_shape)
    deck = Slice([0, 0, 0], [-1, 15, 55])(input_layer)
    tp = Slice([0, 15, 2], [-1, 1, 9])(input_layer)
    hand = Slice([0, 15+1, 0], [-1, 5, 55])(input_layer)
    o_hand = Slice([0, 15+1+5, 0], [-1, 4, 5])(input_layer)
    hp = Slice([0, 15+1+5+4, 0], [-1, 1, 6])(input_layer)
    o_hp = Slice([0, 15+1+5+4+1, 0], [-1, 1, 6])(input_layer)

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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    try:
        model.load_weights('dqn_model.keras')
    except (FileNotFoundError, ValueError) as err:
        print('Could not load weights. Did you forget to modify the architecture?')
        print(err)
    return model


def debug():
    def plot_text(arr):
        m, n = arr.shape
        for i in range(m):
            for j in range(n):
                plt.text(j, i, int(arr[i, j]), ha="center", va="center", color="tab:red")

    def eps_graph():
        graph = [epsilon]
        for i in csv['Episode'][1:]:
            if i < epsilon_decay_start or i < batch_size:
                graph.append(epsilon)
            else:
                graph.append(max(graph[-1] * epsilon_decay, epsilon_min))
        return graph

    def get_moving_avg(data, period):
        moving_avg = []
        window_sum = 0
        for i in csv['Episode']:
            if i > period:
                window_sum -= data[i - period]
            window_sum += data[i]
            moving_avg.append(window_sum / min(len(moving_avg) + 1, period))
        return moving_avg

    def get_moving_qnt(data, period, qnt):
        moving_qrt = []
        for i in csv['Episode']:
            moving_qrt.append(np.quantile(data[max(0,i - period):i + 1], qnt))
        return moving_qrt

    agent.model.summary()
    try:
        csv = pd.read_csv('dqn_data.csv')
    except FileNotFoundError:
        return

    ans = input('Display data graph? (Y/N): ')
    if ans.lower() in {'n', 'no'}:
        return
    plt.subplot(2, 2, 1)
    plt.grid(which='both')
    c = [(r, 0, 1 - r) for r in csv['Action Ratio']]
    c2 = ['r' if r == -1 else 'g' for r in csv['Reward']]
    plt.scatter(list(csv['Episode']), csv['Reward'], s=1.5, c=c, label='Reward')
    rw_avg = get_moving_avg(csv['Reward'], 300)
    plt.plot(list(csv['Episode']), rw_avg, '-k', label=f"Avg Reward/300 Episodes")
    plt.plot(list(csv['Episode']), get_moving_avg(csv['Reward'], 100), ':k', alpha=0.3, label=f"Avg Reward/100 Episodes")
    # plt.ylim(-1, 10)
    plt.title(f"Model Reward per Episode. Final AVG (300): {rw_avg[-1]:.2f}")
    plt.legend()
    plt.gca().set_axisbelow(True)

    # plt.twinx()
    plt.subplot(2, 2, 2)
    plt.grid(which='both')
    plt.scatter(list(csv['Episode']), csv['Max Q-Value'], s=1.5, c=c, label='Max Q-Value')
    plt.plot(list(csv['Episode']), get_moving_avg(csv['Max Q-Value'], 100), '-k', label='Avg Q-Value/100 Episodes')
    plt.title(f"Max Q-Value per Episode")
    plt.legend()
    plt.gca().set_axisbelow(True)

    ax = plt.subplot(2, 2, 3)
    plt.grid(which='both')
    plt.scatter(list(csv['Episode']), csv['Loss'], s=1.5, c='g', label='DQN Loss')
    plt.plot(list(csv['Episode']), get_moving_avg(csv['Loss'], 200), '--k', linewidth=1.5, label='Avg DQN Loss/200 Episodes')
    plt.plot(list(csv['Episode']), get_moving_qnt(csv['Loss'], 200, 0.1), '-k', linewidth=1.5, label='10th Percentile DQN Loss/200 Episodes')
    plt.title('DQN Loss per Episode')
    # plt.yscale('symlog')
    try:
        plt.ylim(top=np.quantile(csv['Loss'][max(epsilon_decay_start, batch_size):], 0.99))
    except ValueError:
        pass
    plt.ylim(ax.get_ylim()[1] / -20, ax.get_ylim()[1] * 1.6)
    plt.legend()
    plt.gca().set_axisbelow(True)

    plt.subplot(2, 2, 4)
    plt.grid(which='both')
    plt.scatter(list(csv['Episode']), csv['Round Length'], s=1.5, c=c2, label='Round Length')
    plt.plot(list(csv['Episode']), get_moving_avg(csv['Round Length'], 100), '-k', label='Avg Round Length/100 Episodes')
    plt.legend()
    plt.title("Round Length per Episode")
    plt.gca().set_axisbelow(True)
    plt.show()

    plt.grid(which='both')
    m = []
    for x in csv['Q-Values']:
        if type(x) is str:
            m.append(np.argmax([float(n) for n in x.strip('[]').split(' ') if n]))
        else:
            m.append(np.argmax([x]))
    plt.scatter(list(csv['Episode']), m, s=2, c=c2, label='Random Q-Value per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Move Index')
    plt.yticks(range(len(env.moves)), [f"{i} {hand}" for i, hand in enumerate(env.moves)])
    plt.gca().set_axisbelow(True)
    plt.legend()
    plt.show()
    quit()


all_cards = [
    Card('Spider', 3, 'Boss', (Effect.ATK(2), Effect.Summon('Inichas'), Effect.Summon('Jellyshroom')), {'Spider'}),
    Card('Venus\' Guardian', 4, 'Boss', (Effect.ATK(2), Effect.Per_Card('Venus\' Bud', Effect.ATK(3))), {'Plant'}),
    Card('Heavy Drone B-33', 5, 'Boss', (Effect.DEF(2), Effect.Empower(2, 'Bot'),), {'Bot'}),
    Card('The Watcher', 5, 'Boss', (Effect.ATK(1), Effect.Summon('Krawler'), Effect.Summon('Warden')), {'Zombie'}),
    Card('The Beast', 5, 'Boss', (Effect.ATK(3), Effect.If_Card('Kabbu', Effect.ATK(4))), {'Bug'}),
    Card('ULTIMAX Tank', 7, 'Boss', (Effect.ATK(8),), {'Bot'}),
    Card('Mother Chomper', 4, 'Boss', (Effect.Lifesteal(2), Effect.Empower(2, 'Chomper')), {'Plant', 'Chomper'}),
    # TODO: make broodmother empower only affect midge card properly
    Card('Broodmother', 4, 'Boss', (Effect.ATK(2), Effect.Empower(2, 'Midge(FIX)')), {'Bug'}),
    Card('Zommoth', 4, 'Boss', (Effect.Empower(2, 'Fungi'),), {'Fungi', 'Zombie'}),
    Card('Seedling King', 4, 'Boss', (Effect.Empower(2, 'Seedling'),), {'Seedling', 'Plant'}),
    Card('Tidal Wyrm', 5, 'Boss', (Effect.ATK(2), Effect.Numb(99))),
    Card('Peacock Spider', 5, 'Boss', (Effect.Empower(3, 'Spider'),), {'Spider'}),
    Card('Devourer', 4, 'Boss', (Effect.ATK(3), Effect.VS('Bug', Effect.ATK(3))), {'Plant'}),
    Card('False Monarch', 5, 'Boss', (Effect.Empower(3, 'Mothfly'),), {'Bug', 'Mothfly'}),
    Card('Maki', 6, 'Boss', (Effect.ATK(6),), {'Bug'}),
    Card('Wasp King', 7, 'Boss', (Effect.Summon_Wasp(2),), {'Bug'}),
    Card('The Everlasting King', 9, 'Boss', (Effect.ATK(99), Effect.Pierce(99)), {'Plant', 'Bug'}),

    Card('Acolyte Aria', 3, 'Mini-Boss', (Effect.ATK(1), Effect.Coin(Effect.Summon('Venus\' Bud'), repetitions=3)), {'Bug'}),
    Card('Mothiva', 2, 'Mini-Boss', (Effect.ATK(2), Effect.If_Card('Zasp', Effect.Heal(1))), {'Bug'}),
    Card('Zasp', 3, 'Mini-Boss', (Effect.ATK(3), Effect.If_Card('Mothiva', Effect.ATK(2))), {'Bug'}),
    Card('Ahoneynation', 5, 'Mini-Boss', (Effect.Coin(Effect.Summon('Abomihoney'), repetitions=2),)),
    Card('Astotheles', 5, 'Mini-Boss', (Effect.Empower(3, 'Thug'),), {'Bug', 'Thug'}),
    Card('Dune Scorpion', 7, 'Mini-Boss', (Effect.ATK(7),), {'Bug'}),
    # TODO: make primal weevil empower only effect weevil card properly
    Card('Primal Weevil', 6, 'Mini-Boss', (Effect.ATK(3), Effect.Empower(2, 'Weevil(FIX)')), {'Bug'}),
    Card('Cross', 3, 'Mini-Boss', (Effect.ATK(2), Effect.If_Card('Poi', Effect.ATK(2))), {'Bug'}),
    Card('Poi', 3, 'Mini-Boss', (Effect.DEF(1), Effect.If_Card('Cross', Effect.DEF(3))), {'Bug'}),
    Card('General Ultimax', 3, 'Mini-Boss', (Effect.ATK(1), Effect.Empower(1, 'Wasp')), {'Bug', 'Wasp'}),
    Card('Cenn', 3, 'Mini-Boss', (Effect.ATK(2), Effect.If_Card('Pisci', Effect.Numb(1))), {'Bug'}),
    Card('Pisci', 3, 'Mini-Boss', (Effect.DEF(2), Effect.If_Card('Cenn', Effect.DEF(4))), {'Bug'}),
    Card('Monsieur Scarlet', 3, 'Mini-Boss', (Effect.ATK(3), Effect.If_ATK(7, Effect.Heal(1))), {'Bug'}),
    Card('Kabbu', 2, 'Mini-Boss', (Effect.ATK(1), Effect.Pierce(3)), {'Bug'}),
    Card('Kali', 4, 'Mini-Boss', (Effect.DEF(2), Effect.If_Card('Kabbu', Effect.Heal(3))), {'Bug'}),
    Card('Carmina', 4, 'Mini-Boss', (Effect.Carmina(),), {'Bug'}),
    Card('Riz', 3, 'Mini-Boss', (Effect.ATK(1), Effect.Setup(2)), {'Bug'}),
    Card('Kina', 4, 'Mini-Boss', (Effect.ATK(2), Effect.If_Card('Maki', Effect.ATK(3))), {'Bug'}),
    Card('Yin', 3, 'Mini-Boss', (Effect.DEF(1), Effect.If_Card('Maki', Effect.Heal(2))), {'Bug'}),
    Card('Stratos', 6, 'Mini-Boss', (Effect.ATK(4), Effect.Pierce(3), Effect.If_Card('Delilah', Effect.ATK(1))), {'Bug'}),
    Card('Delilah', 3, 'Mini-Boss', (Effect.ATK(2), Effect.Lifesteal(2), Effect.If_Card('Stratos', Effect.DEF(1))), {'Bug'}),
    Card('Dead Lander a', 4, 'Mini-Boss', (Effect.ATK(3), Effect.DEF(1)), {'Dead Lander'}),
    Card('Dead Lander b', 4, 'Mini-Boss', (Effect.ATK(2), Effect.Coin(Effect.ATK(2), Effect.DEF(2))), {'Dead Lander'}),
    Card('Dead Lander y', 6, 'Mini-Boss', (Effect.ATK(3), Effect.DEF(3)), {'Dead Lander'}),

    Card('Seedling', 1, 'Battle', (Effect.ATK(1),), {'Plant', 'Seedling'}),
    Card('Acornling', 2, 'Effect', (Effect.Coin(Effect.DEF(3)),), {'Plant', 'Seedling'}),
    Card('Underling', 3, 'Battle', (Effect.ATK(3),), {'Plant', 'Seedling'}),
    Card('Cactiling', 3, 'Effect', (Effect.Coin(Effect.DEF(4)),), {'Plant', 'Seedling'}),
    Card('Flowerling', 1, 'Effect', (Effect.Lifesteal(1),), {'Plant', 'Seedling'}),
    Card('Plumpling', 4, 'Effect', (Effect.Coin(Effect.DEF(6)),), {'Plant', 'Seedling'}),
    Card('Golden Seedling', 9, 'Battle', (Effect.ATK(9),), {'Plant', 'Seedling'}),
    Card('Zombiant', 1, 'Battle', (Effect.ATK(1),), {'Fungi', 'Zombie'}),
    Card('Zombee', 3, 'Battle', (Effect.ATK(3),), {'Fungi', 'Zombie'}),
    Card('Zombeetle', 5, 'Battle', (Effect.ATK(5),), {'Fungi', 'Zombie'}),
    Card('Jellyshroom', 1, 'Battle', (Effect.ATK(1),), {'Fungi'}),
    Card('Bloatshroom', 4, 'Battle', (Effect.ATK(4),), {'Fungi'}),
    Card('Inichas', 1, 'Effect', (Effect.Coin(Effect.DEF(2)),), {'Bug'}),
    Card('Denmuki', 3, 'Effect', (Effect.ATK(1), Effect.Coin(Effect.Numb(1))), {'Bug'}),
    Card('Madesphy', 5, 'Effect', (Effect.Coin(Effect.DEF(3), repetitions=2),), {'Bug'}),
    Card('Numbnail', 2, 'Effect', (Effect.Numb(1),)),
    Card('Ironnail', 3, 'Effect', (Effect.DEF(1), Effect.Numb(1))),
    Card('Midge', 1, 'Battle', (Effect.ATK(1),), {'Bug'}),
    Card('Chomper', 2, 'Battle', (Effect.ATK(2),), {'Plant', 'Chomper'}),
    Card('Chomper Brute', 4, 'Battle', (Effect.ATK(4),), {'Plant', 'Chomper'}),
    Card('Wild Chomper', 3, 'Effect', (Effect.ATK(1), Effect.Coin(Effect.Summon('Chomper'))), {'Plant', 'Chomper'}),
    Card('Weevil', 2, 'Effect', (Effect.ATK(1), Effect.VS('Plant', Effect.ATK(2))), {'Bug'}),
    Card('Psicorp', 2, 'Battle', (Effect.ATK(2),), {'Bug'}),
    Card('Arrow Worm', 3, 'Battle', (Effect.ATK(3),)),
    Card('Thief', 2, 'Battle', (Effect.ATK(2),), {'Bug', 'Thug'}),
    Card('Bandit', 3, 'Battle', (Effect.ATK(3),), {'Bug', 'Thug'}),
    Card('Burglar', 4, 'Battle', (Effect.ATK(4),), {'Bug', 'Thug'}),
    Card('Ruffian', 6, 'Battle', (Effect.ATK(6),), {'Bug', 'Thug'}),
    Card('Bee-Boop', 1, 'Effect', (Effect.Coin(Effect.ATK(1), Effect.DEF(1)),), {'Bot'}),
    Card('Security Turret', 2, 'Battle', (Effect.ATK(2),), {'Bot'}),
    Card('Mender', 1, 'Effect', (Effect.If_Tribe('Bot', 6, Effect.Heal(1)),), {'Bot'}),
    Card('Abomihoney', 4, 'Battle', (Effect.ATK(4),)),
    Card('Krawler', 2, 'Battle', (Effect.ATK(2),), {'Bot'}),
    Card('Warden', 3, 'Battle', (Effect.ATK(3),), {'Bot'}),
    Card('Haunted Cloth', 4, 'Battle', (Effect.ATK(4),)),
    Card('Leafbug Ninja', 3, 'Effect', (Effect.Unity(2, 'Leafbug'),), {'Bug', 'Leafbug'}),
    Card('Leafbug Archer', 2, 'Effect', (Effect.Unity(1, 'Leafbug'),), {'Bug', 'Leafbug'}),
    Card('Leafbug Clubber', 4, 'Effect', (Effect.ATK(2), Effect.If_Tribe('Leafbug', 3, Effect.Numb(1))), {'Bug', 'Leafbug'}),
    Card('Mantidfly', 4, 'Battle', (Effect.ATK(4),), {'Bug'}),
    Card('Jumping Spider', 3, 'Battle', (Effect.ATK(3),), {'Spider'}),
    Card('Mimic Spider', 5, 'Battle', (Effect.ATK(5),), {'Spider'}),
    Card('Diving Spider', 3, 'Battle', (Effect.ATK(3),), {'Spider'}),
    Card('Water Strider', 3, 'Battle', (Effect.ATK(3),), {'Bug'}),
    Card('Belostoss', 6, 'Battle', (Effect.ATK(6),), {'Bug'}),
    Card('Mothfly', 1, 'Battle', (Effect.ATK(1),), {'Bug', 'Mothfly'}),
    Card('Mothfly Cluster', 3, 'Battle', (Effect.ATK(3),), {'Bug', 'Mothfly'}),
    Card('Wasp Scout', 2, 'Battle', (Effect.ATK(2),), {'Bug', 'Wasp'}),
    Card('Wasp Trooper', 4, 'Battle', (Effect.ATK(4),), {'Bug', 'Wasp'}),
    Card('Wasp Bomber', 5, 'Effect', (Effect.ATK(2), Effect.Coin(Effect.ATK(1)), Effect.Coin(Effect.Numb(1))), {'Bug', 'Wasp'}),
    Card('Wasp Driller', 6, 'Effect', (Effect.ATK(4), Effect.Pierce(2)), {'Bug', 'Wasp'}),
    Card('Venus\' Bud', 1, 'Effect', (Effect.Lifesteal(1),), {'Plant'})
]
cards_by_name = {card.name: card for card in all_cards}
p_deck = [
    cards_by_name['Devourer'],
    cards_by_name['Zasp'],
    cards_by_name['Mothiva'],
    cards_by_name['Leafbug Archer'],
    cards_by_name['Leafbug Archer'],
    cards_by_name['Leafbug Archer'],
    cards_by_name['Leafbug Ninja'],
    cards_by_name['Leafbug Ninja'],
    cards_by_name['Leafbug Ninja'],
    cards_by_name['Leafbug Ninja'],
    cards_by_name['Leafbug Clubber'],
    cards_by_name['Leafbug Clubber'],
    cards_by_name['Numbnail'],
    cards_by_name['Numbnail'],
    cards_by_name['Venus\' Bud']
]
o_deck = [
    cards_by_name['Tidal Wyrm'],
    cards_by_name['Astotheles'],
    cards_by_name['Monsieur Scarlet'],
    cards_by_name['Thief'],
    cards_by_name['Thief'],
    cards_by_name['Thief'],
    cards_by_name['Bandit'],
    cards_by_name['Bandit'],
    cards_by_name['Bandit'],
    cards_by_name['Burglar'],
    cards_by_name['Burglar'],
    cards_by_name['Burglar'],
    cards_by_name['Ironnail'],
    cards_by_name['Ironnail'],
    cards_by_name['Venus\' Bud']
]

# -=- HYPERPARAMETERS -=-
initial_episode = 0
episodes = 30000
batch_size = 64
learning_rate = 0.001
target_update_rate = 100
replay_buffer_size = 10000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.00
epsilon_decay = 0.994
epsilon_decay_start = 0
loss_limit = 30000
check_training_rate = 100
# -=-=-=-=-=-=-=-=-=-=-=-
try:
    model_data = pd.read_csv('dqn_data.csv')
    initial_episode = model_data['Episode'].iloc[-1] + 1
    epsilon = max(epsilon_min, epsilon_decay ** max(initial_episode - epsilon_decay_start, 0))
except FileNotFoundError:
    model_data = pd.DataFrame()

env = SpyCardsEnv(p_deck, o_deck)
# There are 32 different possible card hands (nCr(5,0)+nCr(5,1)...+nCr(5,5)=32)
agent = DQNAgent(input_size=(15 + 1 + 5 + 4 + 1 + 1, 55), output_size=32)
debug()

for e in range(initial_episode, initial_episode + episodes):
    state = env.reset()
    done = False
    total_reward = 0
    model_action_sum = 0
    max_q_val = 0
    q_values = []

    while not done:
        action, was_model_action, q_val, q_vals = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        model_action_sum += was_model_action
        max_q_val = max(max_q_val, q_val)
        q_values.append(q_vals)
    q_values = random.sample(q_values, k=1)[0]

    loss = agent.replay(batch_size, e)
    model_data = pd.concat([model_data, pd.DataFrame({
        'Episode': [e],
        'Action Ratio': [model_action_sum / env.total_rounds],
        'Reward': [total_reward],
        'Loss': [loss],
        'Round Length': [env.total_rounds],
        'Q-Values': [q_values],
        'Max Q-Value': [max_q_val]
    })])

    if loss is None:
        loss = 0
    print(f"Episode: {e} - Action Ratio: {model_action_sum / env.total_rounds:.3f} ({model_action_sum}/{env.total_rounds}) - Reward: {total_reward:.2f} - Loss: {loss:.4f} - Max Q-Val: {max_q_val:.4f}")
    if loss > loss_limit and e > max(batch_size, epsilon_decay_start):
        print(f"Loss exceeded {loss_limit}. Quitting early...")
        break

    if (e + 1) % target_update_rate == 0:
        agent.update_target_model()
    if (e + 1) % 50 == 0:
        agent.model.save('dqn_model.keras')
        model_data.to_csv('dqn_data.csv', index=False)
