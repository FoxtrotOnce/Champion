"""Core functions and methods required to interact with a Spy Cards environment.

Defines heuristics for opponent moves in Spy Cards, and ISMCTS for use with
the neural networks.
Classes/methods are grouped for clarity.

Spy Cards:
    Stats: Store statistics used for Spy Cards.
    Card: Represent a spy card.
    Effect: Container class for all other effect subclasses.
    SpyCardsEnv: Manage mechanics for playing Spy Cards.

Neural Networking:
    filter_moves: Return all valid moves that can be played.
    MaxATKandTP: Pick the valid move with the highest ATK and TP.
    SampleWeightedMove: Pick a random move, weighed by how many cards it uses.
    Node: Represent a Spy Cards env as an ISMCTS node.
    ISMCTS: Pick the best move to play, using ISMCTS.
"""

import numpy as np
from typing import Literal
from abc import abstractmethod
from copy import deepcopy
from collections import Counter
from itertools import combinations, chain
from heapq import heappush, heappop
from math import sqrt, log
import random


Literal_Category = Literal["Boss", "Mini-Boss", "Effect", "Battle"]
Literal_Tribe = Literal["Spider", "Plant", "Bot", "Zombie", "Bug", "Chomper", "Fungi", "Seedling", "Mothfly", "Thug",
                        "Wasp", "Dead Lander", "Leafbug", "Midge", "Weevil"]


class Stats:
    """Store statistics used for Spy Cards.

    Attributes:
        HP (int): How much HP the player has, at most 5.
        ATK (int): Total ATK dealt by this player.
        DEF (int): Total DEF dealt by this player. Reduces opponent ATK.
        Lifesteal (int): The player gains this much HP, if they win the round.
        Pierce (int): Ignores this much of the opponent's DEF.
        Setup (int): How much ATK the player gains next round.
        Empower (dict[str, int]): Attack bonuses for each of the player's tribes.
    """

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
            'Leafbug': 0,
            'Midge': 0,
            'Weevil': 0
        }

    def __repr__(self):
        return f"ATK: {self.ATK} - DEF: {self.DEF} - PI: {self.Pierce} - LS: {self.Lifesteal}"

    def reset(self) -> None:
        """Reset all values to 0, besides HP.

        This method is provided for use with neural networks, as a new instance
        doesn't need to be initialized to continue playing.
        """
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
            'Leafbug': 0,
            'Midge': 0,
            'Weevil': 0
        }


class Card:
    """Represent a spy card.

    This class is immutable. Attributes should not be modified after an
    instance is created.
    Holds all the required attributes to represent any unique spy card.
    Cards in the Effect or Battle category are converted to Effect/Battle after
    initialization, since they appear the same in-game.

    Methods:
        one_hot: Return a one-hot encoded representation of the Card.
    """

    def __init__(self, name: str, TP: int, category: Literal_Category, effects: tuple["Effect", ...],
                 tribes: set[Literal_Tribe] = None):
        """Create a new spy card.

        The Card's category is changed to Effect/Battle for Cards in the Effect
        or Battle category, since they appear the same in-game.

        Effect.ATK gets assigned is_battle=True if it is attached to a Battle
        card to handle numbing, before the category gets changed from Battle
        to Effect/Battle.
        Effect.Unity gets assigned original_card_name=name to properly handle
        the Unity effect.

        Args:
            name (str): The name of this Card.
            TP (int): How much TP this Card costs to play.
            category (Literal_Category): The category this Card is in.
            effects (tuple["Effect" ...]): Each effect this Card triggers.
            tribes (set[Literal_Tribe]): Each tribe this Card is in.

        Raises:
            NotImplementedError: A Battle Card's effects are something other
                than a single ATK effect.
        """
        if tribes is None:
            tribes = set()

        self.name = name
        self.TP = TP
        self.category = category
        self.effects = effects
        self.tribes = tribes
        # Tell the ATK Effects if the card is a battle card so it can be properly numbed
        if category == 'Battle':
            if len(self.effects) == 1 and isinstance(self.effects[0], Effect.ATK):
                self.effects[0].is_battle = True
            else:
                raise NotImplementedError(f"Battle card has effects other than 1 ATK effect. "
                                          f"Effect count: {len(self.effects)} - "
                                          f"Effect types: {[type(eff) for eff in self.effects]}")
        # Tell the Unity Effects the card's name so unity can work properly
        for effect in effects:
            if isinstance(effect, Effect.Unity):
                effect.original_card_name = name
        # Convert category to Effect/Battle if it is Effect or Battle,
        # since numbing (only need for Battle category) was already handled.
        if self.category == 'Effect' or self.category == 'Battle':
            self.category = 'Effect/Battle'

    def __repr__(self):
        return f"{self.name} ({self.TP})"

    def one_hot(self) -> np.ndarray:
        """Return a one-hot encoded representation of the Card.

        This function is provided for use with neural networks, encoding the
        card as a one-hot array of size (54,):

        TP cost: (9) - TP cost is between 1-9, which requires 9 slots.
        Category: (3) - 3 slots for each category (Effect/Battle are together).
        ATK: (11) - ATK is between 0-9, and 99 for ELK, which requires 11 slots.
        Effects: (18) - 18 slots for each effect.
        Tribes: (13) - 13 slots for each tribe.

        Returns:
            np.ndarray: A (54,) one-hot encoded array representing the Card.
        """
        key = {'Boss': 0, 'Mini-Boss': 1, 'Effect/Battle': 2}
        # TP + Category + ATK + Effects + Tribes
        state = np.zeros(9 + 3 + 11 + 18 + 13)
        state[self.TP - 1] = 1
        state[9 + key[self.category]] = 1
        atk = 0
        for effect in self.effects:
            if isinstance(effect, Effect.ATK):
                atk = min(10, atk + effect.amount)
            state[9 + 3 + 11 + effect.id] = 1
        state[9 + 3 + atk] = 1
        tribe_indices = {key: i for i, key in enumerate(Stats().Empower)}
        for tribe in self.tribes:
            state[9 + 3 + 11 + 18 + tribe_indices[tribe]] = 1
        return state


class Effect:
    """Base class for all other effect subclasses.
    
    This class and its subclasses are immutable. Attributes should not be
    modified after an instance is created.
    Serves as a container for specific effects that modify player stats.
    Effects with lower priorities will be played first.

    Attributes:
        _priority (int): The priority of the effect.
            Lower priorities get played first.
        id (int): Arbitrary id of the effect. Used for Card.one_hot.

    Methods:
        play: Abstract method that plays the effect onto the environment.

    Subclasses:
        Priority: 0 - Setup (int)
        Priority: 0 - Heal (int)
        Priority: 0 - ATK (int)
        Priority: 0 - DEF (int)
        Priority: 0 - Pierce (int)
        Priority: 0 - Lifesteal (int)
        Priority: 0 - Summon (str)
        Priority: 0 - Summon_Wasp (int)
        Priority: 0 - Carmina

        Priority: 1 - Coin (Effect, Effect, int)
        Priority: 1 - If_Card (str, Effect)
        Priority: 1 - If_Tribe (str, int, Effect)
        Priority: 1 - Per_Card (str, Effect)
        Priority: 1 - VS (str, Effect)
        Priority: 1 - Empower (int, str)
        Priority: 1 - Unity (int, str)

        Priority: 2 - If_ATK (int, Effect)

        Priority: 3 - Numb (int)
    """

    _priority: int
    id: int

    def __lt__(self, other: "Effect"):
        """Sort effects based on priority."""
        return self._priority < other._priority

    @abstractmethod
    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        """Applies the effect to the game environment.

        Args:
            is_player (bool): If the effect originated from the player.
            queue (list[tuple[bool, Effect]]): Min-heap of queued effects to
                be played. Each entry indicates (is_player, queued_effect).
            p_unities (set[str]): Names of the player's cards that have
                triggered Unity.
            p_played (Counter[Card, int]): How many of each card (by name) the
                player has played this round.
            p_numbable (list[int]): Min-heap list of ATK effect amounts played
                from Battle cards, by the player. Used for numbing.
            p_stats (Stats): Player statistics for this game.
            o_played (Counter[Card, int]): How many of each card (by name) the
                opponent has played this round.
            o_numbable (list[int]): Min-heap list of ATK effect amounts played
                from Battle cards, by the opponent. Used for numbing.
            o_stats (Stats): Opponent statistics for this game.
        """


class Setup(Effect):
    """Add (amount) to the player's ATK the round after it is played.

    NOTE:
        This effect is only used by Riz.
    """

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 0
        self._amount = amount

    def __repr__(self):
        return f"Setup ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        p_stats.Setup += self._amount


class Heal(Effect):
    """Heal the player by (amount) HP. Their HP will not exceed 5."""

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 1
        self._amount = amount

    def __repr__(self):
        return f"Heal ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        p_stats.HP = min(5, p_stats.HP + self._amount)


class ATK(Effect):
    """Add (amount) to the player's ATK."""

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 2
        self.amount = amount
        # is_battle gets assigned by the Card object
        self.is_battle = False

    def __repr__(self):
        return f"ATK ({self.amount}), Battle: {self.is_battle}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        p_stats.ATK += self.amount
        # Add the ATK effect to played if the card was a Battle card (for numbing)
        if self.is_battle:
            heappush(p_numbable, self.amount)


class DEF(Effect):
    """Add (amount) to the player's DEF."""

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 3
        self._amount = amount

    def __repr__(self):
        return f"DEF ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        p_stats.DEF += self._amount


class Pierce(Effect):
    """Lower the opponent's DEF by (amount). Their DEF will not be less than 0."""

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 4
        self._amount = amount

    def __repr__(self):
        return f"Pierce ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        p_stats.Pierce += self._amount


class Lifesteal(Effect):
    """Heal the player by (amount) HP, if they win the round. Their HP will not exceed 5."""

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 5
        self._amount = amount

    def __repr__(self):
        return f"Lifesteal ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        p_stats.Lifesteal += self._amount


class Summon(Effect):
    """Play the summoned card, and handle its effects."""

    def __init__(self, card_name: str):
        self._priority = 0
        self.id = 6
        self._card_name = card_name

    def __repr__(self):
        return f"Summon {self._card_name}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        card = cards_by_name[self._card_name]
        p_played[card.name] += 1
        for effect in card.effects:
            heappush(queue, (is_player, effect))


class Summon_Wasp(Effect):
    """Sample x random Wasp Scout/Trooper/Bomber/Driller cards, summon them, and handle their effects.

    NOTE:
        This effect is only used by Wasp King.
    """

    def __init__(self, amount: int):
        self._priority = 0
        self.id = 7
        self._amount = amount

    def __repr__(self):
        return f"Summon Wasps ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        for _ in range(self._amount):
            card = random.choice([cards_by_name['Wasp Scout'],
                                  cards_by_name['Wasp Trooper'],
                                  cards_by_name['Wasp Bomber'],
                                  cards_by_name['Wasp Driller']])
            p_played[card.name] += 1
            for effect in card.effects:
                heappush(queue, (is_player, effect))


class Carmina(Effect):
    """Summon a random Mini-Boss card, and handle its effects.

    NOTE:
        This effect is only used by Carmina.
    """

    def __init__(self):
        self._priority = 0
        self.id = 8

    def __repr__(self):
        return f"Carmina"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        card = random.choice([card for card in all_cards if card.category == 'Mini-Boss'])
        p_played[card.name] += 1
        for effect in card.effects:
            heappush(queue, (is_player, effect))


class Coin(Effect):
    """Flip (repetitions) coin(s). If heads, queue (effect_heads). Otherwise, (effect_tails)."""

    def __init__(self, effect_heads: "Effect", effect_tails: "Effect" = None, repetitions: int = 1):
        self._priority = 0
        self.id = 9
        self._effect_heads = effect_heads
        self._effect_tails = effect_tails
        self._repetitions = repetitions

    def __repr__(self):
        if self._effect_tails is None:
            return f"Coin ({self._repetitions}): {self._effect_heads}"
        return f"Coin ({self._repetitions}): {self._effect_heads}/{self._effect_tails}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        for _ in range(self._repetitions):
            if random.random() >= 0.5:
                heappush(queue, (is_player, self._effect_heads))
            elif self._effect_tails:
                heappush(queue, (is_player, self._effect_tails))


class If_Card(Effect):
    """If the player played (card_name), queue (effect)."""

    def __init__(self, card_name: str, effect: "Effect"):
        self._priority = 1
        self.id = 10
        self._card_name = card_name
        self._effect = effect

    def __repr__(self):
        return f"If {self._card_name}: {self._effect}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        if p_played[self._card_name]:
            heappush(queue, (is_player, self._effect))


class If_Tribe(Effect):
    """If the player played (amount) cards with the (tribe) tribe, queue (effect)."""

    def __init__(self, tribe: str, amount: int, effect: "Effect"):
        self._priority = 1
        self.id = 11
        self._tribe = tribe
        self._amount = amount
        self._effect = effect

    def __repr__(self):
        return f"If {self._tribe} ({self._amount}): {self._effect}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        tribe_cards = 0
        for card_name, occ in p_played.items():
            card = cards_by_name[card_name]
            if self._tribe in card.tribes:
                tribe_cards += occ
        if tribe_cards >= self._amount:
            heappush(queue, (is_player, self._effect))


class Per_Card(Effect):
    """For each (card_name) the player played, queue (effect).

    NOTE:
        This effect is only used by Venus' Guardian.
    """

    def __init__(self, card_name: str, effect: "Effect"):
        self._priority = 1
        self.id = 12
        self._card_name = card_name
        self._effect = effect

    def __repr__(self):
        return f"Per {self._card_name}: {self._effect}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        for _ in range(p_played[self._card_name]):
            heappush(queue, (is_player, self._effect))


class VS(Effect):
    """If the opponent played a card with the (tribe) tribe, queue (effect)."""

    def __init__(self, tribe: str, effect: "Effect"):
        self._priority = 1
        self.id = 13
        self._tribe = tribe
        self._effect = effect

    def __repr__(self):
        return f"VS {self._tribe}: {self._effect}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        for card_name in o_played:
            card = cards_by_name[card_name]
            if self._tribe in card.tribes:
                heappush(queue, (is_player, self._effect))
                break


class Empower(Effect):
    """Increase the ATK of the player by (power), for each card they played with the (tribe) tribe."""

    def __init__(self, power: int, tribe: str):
        self._priority = 1
        self.id = 14
        self.power = power
        self._tribe = tribe

    def __repr__(self):
        return f"Empower +{self.power} ({self._tribe})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        for card_name, occ in p_played.items():
            card = cards_by_name[card_name]
            if self._tribe in card.tribes:
                p_stats.ATK += self.power * occ


class Unity(Effect):
    """Increase the ATK of the player by (power), for each card they played with the (tribe) tribe.

    NOTE:
        This effect can only be triggered once per unique card.
    """

    def __init__(self, power: int, tribe: str):
        self._priority = 1
        self.id = 15
        # original_card_name gets assigned by the Card object
        self.original_card_name = None
        self.power = power
        self._tribe = tribe

    def __repr__(self):
        return f"Unity (+{self.power}, {self._tribe})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        # Don't do Unity if the same card has already played it
        if self.original_card_name in p_unities:
            return
        for card_name, occ in p_played.items():
            card = cards_by_name[card_name]
            if self._tribe in card.tribes:
                p_stats.ATK += self.power * occ
        p_unities.add(self.original_card_name)


class If_ATK(Effect):
    """If the player's ATK is (amount) or higher, queue (effect)."""

    def __init__(self, amount: int, effect: "Effect"):
        self._priority = 2
        self.id = 16
        self._amount = amount
        self._effect = effect

    def __repr__(self):
        return f"If ATK ({self._amount}): {self._effect}"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        if p_stats.ATK >= self._amount:
            heappush(queue, (is_player, self._effect))


class Numb(Effect):
    """Removes the lowest ATK effect played by a Battle card, played by the opponent (amount) times."""

    def __init__(self, amount: int) -> None:
        self._priority = 3
        self.id = 17
        self._amount = amount

    def __repr__(self):
        return f"Numb ({self._amount})"

    def play(self, is_player: bool, queue: list[tuple[bool, "Effect"]], p_unities: set[str],
             p_played: Counter["Card", int], p_numbable: list[int], p_stats: "Stats",
             o_played: Counter["Card", int], o_numbable: list[int], o_stats: "Stats") -> None:
        # While there are cards to numb (and you can numb), remove ATK from your opponent
        amount_available = self._amount
        while o_numbable and amount_available > 0:
            o_stats.ATK -= heappop(o_numbable)
            amount_available -= 1


# No other way to get IDEs to autofill subclasses without specifying them like this.
Effect.Setup = Setup
Effect.Heal = Heal
Effect.ATK = ATK
Effect.DEF = DEF
Effect.Pierce = Pierce
Effect.Lifesteal = Lifesteal
Effect.Summon = Summon
Effect.Summon_Wasp = Summon_Wasp
Effect.Carmina = Carmina
Effect.Coin = Coin
Effect.If_Card = If_Card
Effect.If_Tribe = If_Tribe
Effect.Per_Card = Per_Card
Effect.VS = VS
Effect.Empower = Empower
Effect.Unity = Unity
Effect.If_ATK = If_ATK
Effect.Numb = Numb


class SpyCardsEnv:
    """Manage mechanics for playing Spy Cards.

    Handle game statistics, and playing rounds of Spy Cards. The round system
    behaves identical to Bug Fables', with the exception that numbing always
    numbs the Battle card with the lowest ATK.
    This environment provides features for use with neural networks, like
    reward and state representation.

    Attributes:
        TP (int): The total TP available to both players.
        r (random.Random): Random generator used for consistent card drawing.
        total_rounds (int): How many rounds this game has lasted for.
        player_deck (list[Card]): Card deck the player draws from.
        player_hand (list[Card]): Cards in the player's hand, starting at 3.
        player_stats (list[Card]): The player's stats.
        opponent_deck (list[Card]): Card deck the opponent draws from.
        opponent_hand (list[Card]): Cards in the opponent's hand, starting at 3.
        opponent_stats (list[Card]): The opponent's stats.
        ismcts_mode (bool): If enabled, opponent drawing and playing behavior
            changes to be used for ISMCTS.

    Methods:
        reset: Reset the environment to an initialized state.
        get_state: Return a one-hot ndarray representing the environment.
        step (int): Play one round of Spy Cards.
    """

    def __init__(self, player_deck: list["Card"], opponent_deck: list["Card"]):
        """Create a new env with player and opponent decks.

        Both decks are required to be of size 15, otherwise NotImplementedError
        is raised.

        Args:
            player_deck (list[Card]): Card deck that the player draws from.
            opponent_deck (list[Card]): Card deck that the opponent draws from.

        Raises:
            NotImplementedError: Deck sizes are not 15.
        """
        if len(player_deck) != 15 or len(opponent_deck) != 15:
            raise NotImplementedError(f"Deck size is not 15. p_deck: {len(player_deck)}, o_deck: {len(opponent_deck)}")
        self.TP = None
        self.r = random.Random()
        self.total_rounds = None
        self.player_deck = player_deck
        self.player_hand = None
        self.player_stats = None
        self.opponent_deck = opponent_deck
        self.opponent_hand = None
        self.opponent_stats = None
        self.ismcts_mode = False

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to an initialized state.

        Reset TP, total_rounds, hands, and stats.
        This method is provided for use with neural networks, as a new env
        doesn't need to be initialized to continue playing.

        Returns:
            np.ndarray: The environment state from get_state.
        """
        self.TP = 2
        self.total_rounds = 0
        self.player_hand = random.sample(self.player_deck, k=3)
        random.shuffle(self.player_hand)
        self.player_stats = Stats()
        self.opponent_hand = random.sample(self.opponent_deck, k=3)
        random.shuffle(self.opponent_hand)
        self.opponent_stats = Stats()

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return a one-hot encoded state representation of the environment.

        This function is provided for use with neural networks, encoding the
        game state as a one-hot array of size (26, 54):

        Player deck: (15, 54) - 15 rows for 15 Card objects.
        TP: (1, 9) - TP is between 2-10, which requires 9 slots.
        Player hand: (5, 54) - 5 rows for 5 Card objects.
        Opponent hand: (3, 6) - 3 rows per visible card category, and 6 slots
            for 0-5 of the counted category.
        Player HP: (1, 6) - HP is between 0-5, which requires 6 slots.
        Opponent HP: (1, 6): - HP is between 0-5, which requires 6 slots.

        Returns:
            np.ndarray: A (26, 54) one-hot encoded array representing the environment.
        """
        card_categories = {'Boss': 0, 'Mini-Boss': 0, 'Effect/Battle': 0}
        # Player deck
        state = np.zeros((15 + 1 + 5 + 3 + 1 + 1, 54))
        for i, card in enumerate(self.player_deck):
            state[i] = card.one_hot()
        # TP
        state[15, self.TP] = 1
        # Player hand
        for i, card in enumerate(self.player_hand):
            state[15 + 1 + i] = card.one_hot()
        # Opponent hand
        for card in self.opponent_hand:
            card_categories[card.category] += 1
        for i, val in enumerate(card_categories.values()):
            state[15 + 1 + 5 + i, val] = 1
        # Player HP
        state[15 + 1 + 5 + 3, self.player_stats.HP] = 1
        # Opponent HP
        state[15 + 1 + 5 + 3 + 1, self.opponent_stats.HP] = 1
        return state

    def step(self, action: int) -> tuple[np.ndarray, int, bool]:
        """Play one round of Spy Cards.

        The round is played as follows:
        The opponent's move is generated via a heuristic, then each card's
        effects are put into a queue, and they are then played in order based
        on each effect's priority. After that, new cards are drawn from decks.

        ismcts_mode should be set to True if this function is intended to be
        used with ISMCTS. This will change how the opponent's hand gets drawn,
        and what move the opponent will play.

        Args:
            action (int): The player's chosen move index.

        Returns:
            tuple[np.ndarray, int, bool]: Tuple containing round information.
                state (np.ndarray): The environment state from get_state.
                reward (int): The reward received.
                done (bool): If the game has terminated.

        Raises:
            NotImplementedError: The TP cost of the player's move exceeds the
                environment's TP.
        """
        self.player_stats.reset()
        self.opponent_stats.reset()
        # Choose player's cards and do a check to see if the move costs an invalid amount of TP.
        player_chosen_cards = [self.player_hand[i] for i in all_moves[action]]
        TP_used = 0
        for card in player_chosen_cards:
            TP_used += card.TP
        if TP_used > self.TP:
            raise NotImplementedError(f"Player TP used is {TP_used}. Env only has {self.TP} TP.")

        if not self.ismcts_mode:
            # Use Bug Fables enemy AI to pick enemy cards (AI works by drawing left to right until TP is depleted)
            TP_used = 0
            opponent_chosen_cards = []
            for card in self.opponent_hand:
                if self.TP >= TP_used + card.TP:
                    opponent_chosen_cards.append(card)
                    TP_used += card.TP
        else:
            # Guess what the opponent will play by using a heuristic
            opponent_chosen_cards = [self.opponent_hand[i] for i in all_moves[SampleWeightedMove(self.opponent_hand, self.TP)]]

        # print()
        # print(f"Player: {player_chosen_cards}")
        # print(f"Opponent: {opponent_chosen_cards}")
        # Remove chosen cards from hands
        for card in player_chosen_cards:
            self.player_hand.remove(card)
        for card in opponent_chosen_cards:
            self.opponent_hand.remove(card)

        # Set up variables needed for processing effects
        # Queue will be (bool, Effect) pairs indicating (is_player's_effect, effect_to_play).
        queue = []

        p_played = Counter()
        p_numbable = []
        p_unities = set()
        for card in player_chosen_cards:
            p_played[card] += 1
            for effect in card.effects:
                heappush(queue, (True, effect))
        o_played = Counter()
        o_numbable = []
        o_unities = set()
        for card in opponent_chosen_cards:
            o_played[card] += 1
            for effect in card.effects:
                heappush(queue, (False, effect))

        # Play effects while they exist in queue, effects with the lowest priority are played first.
        while queue:
            is_player, effect = heappop(queue)
            if is_player:
                effect.play(is_player, queue, p_unities, p_played, p_numbable, self.player_stats, o_played, o_numbable, self.opponent_stats)
            else:
                effect.play(is_player, queue, o_unities, o_played, o_numbable, self.opponent_stats, p_played, p_numbable, self.player_stats)

        # Raise TP, draw new cards, and determine ending scores
        self.TP = min(10, self.TP + 1)
        self.total_rounds += 1
        for is_player, hand, deck in ((True, self.player_hand, self.player_deck),
                                      (False, self.opponent_hand, self.opponent_deck)):
            can_draw = 2 if hand else 3
            available_cards = Counter(deck)
            for card in hand:
                available_cards[card] -= 1
            while len(hand) < 5 and can_draw > 0:
                # player always samples from random generator. Opponent samples as well if it is not ISMCTS mode.
                if (not self.ismcts_mode) or is_player:
                    # Both players use random generator to draw cards normally
                    picked = self.r.choices(*zip(*available_cards.items()))[0]
                else:
                    # Opponent draw predictions are sampled from their deck (cheating) in ISMCTS mode.
                    # Note: This is identical to normal drawing, since normal drawing samples from a known deck.
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


def filter_moves(hand: list["Card"], TP_limit: int) -> list[int]:
    """Return all valid moves that can be played.

    Filter out moves that either use a card not present in the hand, or exceed
    the TP limit.

    Args:
        hand (list[Card]): Check move TP cost.
        TP_limit (int): The total TP cost of the move cannot exceed this limit.

    Returns:
        list[int]: All valid moves indices that can be played.
    """
    valid_moves = []
    for i, card_indices in enumerate(all_moves):
        # Skip move if it requires a card that isn't in the hand
        if i > 0 and card_indices[-1] + 1 > len(hand):
            continue
        # Skip move if it uses more TP than available
        TP_used = 0
        for card_index in card_indices:
            TP_used += hand[card_index].TP
            if TP_used > TP_limit:
                break
        else:
            valid_moves.append(i)

    return valid_moves


def MaxATKandTP(hand: list["Card"], TP_limit: int) -> int:
    """Pick the valid move with the highest ATK and TP.

    Total ATK is prioritized first, and total TP cost is prioritized second.
    Total ATK is calculated as the sum of the ATK, Unity, and Empower effect amounts.
    Total TP cost is calculated as the sum of each card's TP cost.

    Args:
        hand (list[core.Card]): A list of the cards in the player's hand.
            Used for checking available cards.
        TP_limit (int): The total TP cost of the move cannot exceed this limit.

    Returns:
        int: The index of the valid move with the highest ATK and TP.
    """
    best = -1
    atk = -1
    tp = -1
    for i, move in enumerate(all_moves):
        # Skip move if it requires hands that the move doesn't have
        if i > 0 and move[-1] + 1 > len(hand):
            continue
        TP_used = 0
        total_atk = 0
        for card_index in move:
            card = hand[card_index]
            if TP_limit >= TP_used + card.TP:
                # Add amount to total ATK if it's an ATK, Unity, or Empower effect.
                for effect in card.effects:
                    if isinstance(effect, Effect.ATK):
                        total_atk += effect.amount
                    elif isinstance(effect, Effect.Unity):
                        total_atk += effect.power
                    elif isinstance(effect, Effect.Empower):
                        total_atk += effect.power
                TP_used += card.TP

        # Pick move with highest ATK
        if total_atk > atk:
            best = i
            atk = total_atk
            tp = TP_used
        # If the ATK is equal, pick move with the highest TP
        elif total_atk == atk and TP_used > tp:
            best = i
            tp = TP_used
    return best


def SampleWeightedMove(hand: list["Card"], TP_limit: int) -> int:
    """Pick a random move, weighed by how many cards it uses.

    Moves are weighed based on how many cards they play. For example:
        0 Cards = () = Weight of 1
        2 Cards = (0,1) = Weight of 3
        4 Cards = (0,2,3,4) = Weight of 5

    Args:
        hand (list[core.Card]): A list of the cards in the player's hand.
            Used for checking available cards.
        TP_limit (int): The total TP cost of the move cannot exceed this limit.

    Returns:
        int: The index of the randomly picked move.
    """
    valid_moves = filter_moves(hand, TP_limit)
    weights = [len(all_moves[move]) + 1 for move in valid_moves]
    move = random.choices(valid_moves, weights, k=1)[0]
    return move


class Node:
    """Represent a Spy Cards game state as an ISMCTS node.

    This class is to be used with ISMCTS for building a tree.
    Each node represents a game state, and each child represents a possible
    move from the parent node's state.
    Traversing the tree should result in (close-to) reproducible results, as
    the same moves are being taken.
    Tree traversal uses the UCT formula.

    Attributes:
        move (int): The move that led to this node.
        children (list[Node]): List of each child node, representing a possible
            move that can be taken from this node.
        parent (Node): The parent node this nod ederived from.
        visits (int): How many times this node has been visited during ISMCTS.
        reward (float): Accumulated reward received from child nodes.
        untried_moves (list[int]): Possible moves that have yet to be played.

    Methods:
        repr (int): Print the node, and its children.
        select (float): Select a child using the UCT formula.
        expand (SpyCardsEnv, int): Expand the node by trying an untried move.
        backpropagate (float): Update the node with rollout rewards.
    """

    def __init__(self, env: "SpyCardsEnv" = None, move: int = None, parent: "Node" = None):
        """Create a new node representing a game state.

        The node's game state should be reachable by applying the given move
        on the parent.

        Args:
            env (SpyCardsEnv): The environment that resulted from this move.
            move (int): The move that led to this node.
            parent (Node): The parent node this node derived from.
        """
        self.move = move
        self.children = []
        self.parent = parent
        self.visits = 0
        self.reward = 0
        self.untried_moves = []

        if env:
            self.untried_moves = filter_moves(env.player_hand, env.TP)

    def repr(self, depth: int = 0) -> None:
        """Print the node, and its children.

        Recursively calls repr to print each child node, using depth for better
        visualization.

        Args:
            depth (int): PROTECTED, DO NOT CHANGE. Child depth from parent.
        """
        print(f"{'_' * depth}Node({self.move}, {self.visits}, {self.reward})")
        print(f"{'_' * depth}(")
        for child in self.children:
            child.repr(depth + 1)
        print(f"{'_' * depth})")

    def select(self, C: float = sqrt(2)) -> "Node":
        """Select a child using the UCT formula.

        Args:
            C (float): Exploration constant. Higher values encourage more
                exploration.

        Returns:
            Node: The selected child node.
        """
        return max(self.children,
                   key=lambda child: child.reward / child.visits + C * sqrt(log(self.visits) / child.visits))

    def expand(self, env: "SpyCardsEnv", move: int) -> "Node":
        """Expand the node by trying an untried move.

        Create a new child node with the given args, and return it.
        Handle removing the untried move, and adding the child node.

        Args:
            env (SpyCardsEnv): The environment that resulted from the move.
            move (int): The move played from this node.

        Returns:
            Node: The expanded child node.
        """
        child = Node(env, move, self)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def backpropagate(self, reward: float) -> None:
        """Update the node with rollout rewards.

        After rollout, the reward is propagated back to each parent node.

        Args:
            reward (float): The reward received from the rollout.
        """
        self.visits += 1
        self.reward += reward


def ISMCTS(root_env: "SpyCardsEnv", iterations: int) -> int:
    """Pick the best move to play, using ISMCTS.

    The missing information from the Spy Cards environment is the player's
    newly drawn cards, the opponent's hand, the opponent's deck, and the
    opponent's move.
    Currently, ISMCTS cheats by knowing the opponent's deck so that an extra
    model is not needed. This will be changed in the future.

    The player's newly drawn cards are filled in with a preset random generator,
    so they remain consistent if the same move is played from the root.
    The opponent's hand is predicted by sampling random cards from their deck,
    according to what card categories are in their hand. (CHEATING)
    The opponent's move is predicted by using the SampleWeightedMove heuristic
    on their predicted hand.

    Args:
        root_env (SpyCardsEnv): Env used for getting untried_moves and
            playing the first moves.
        iterations (int): The number of iterations to run ISMCTS for.

    Returns:
        int: The index of the best move to play, based on ISMCTS results.
    """
    root = Node(root_env)
    r_state = random.Random().getstate()
    # ISMCTS cheats by looking at the opponent's deck to sample guessed cards here.
    opponent_deck_categories = {'Boss': [], 'Mini-Boss': [], 'Effect/Battle': []}
    for card in root_env.opponent_deck:
        opponent_deck_categories[card.category].append(card)

    root_hand_guess = [random.choice(opponent_deck_categories[card.category]) for card in root_env.opponent_hand]

    # Run ISMCTS
    for _ in range(iterations):
        node = root
        node_env = deepcopy(root_env)
        node_env.ismcts_mode = True
        node_env.r.setstate(r_state)
        node_env.opponent_hand = root_hand_guess

        # Selection (Select node if it is non-terminal and fully expanded)
        while (not node_env.done) and len(node.untried_moves) == 0:
            node = node.select()
            node_env.step(node.move)

        # Expansion (If node is non-terminal, expand one untried move)
        if not node_env.done:
            move = random.choice(node.untried_moves)
            node_env.step(move)
            node = node.expand(node_env, move)

        # Rollout (Use a heuristic to quickly play the game to completion)
        while not node_env.done:
            move = MaxATKandTP(node_env.player_hand, node_env.TP)
            node_env.step(move)

        # Backpropagation (Update all parent nodes with the reward obtained during rollout)
        rw = 1 if node_env.opponent_stats.HP == 0 else -1
        while node:
            node.backpropagate(rw)
            node = node.parent

    # Pick ISMCTS move based on child with most visits (most tried move)
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move


# all_moves is a list of every single move that can be played, in the form of indices. For example:
#   (0,1) would indicate playing the first card, and the second card.
#   () would indicate playing no cards.
#   (0,2,3,4) would indicate playing every card except for the second one.
all_moves = tuple(chain.from_iterable(combinations(range(5), r) for r in range(5 + 1)))
all_cards = [
    Card('Spider', 3, 'Boss', (Effect.ATK(2), Effect.Summon('Inichas'), Effect.Summon('Jellyshroom')), {'Spider'}),
    Card('Venus\' Guardian', 4, 'Boss', (Effect.ATK(2), Effect.Per_Card('Venus\' Bud', Effect.ATK(3))), {'Plant'}),
    Card('Heavy Drone B-33', 5, 'Boss', (Effect.DEF(2), Effect.Empower(2, 'Bot'),), {'Bot'}),
    Card('The Watcher', 5, 'Boss', (Effect.ATK(1), Effect.Summon('Krawler'), Effect.Summon('Warden')), {'Zombie'}),
    Card('The Beast', 5, 'Boss', (Effect.ATK(3), Effect.If_Card('Kabbu', Effect.ATK(4))), {'Bug'}),
    Card('ULTIMAX Tank', 7, 'Boss', (Effect.ATK(8),), {'Bot'}),
    Card('Mother Chomper', 4, 'Boss', (Effect.Lifesteal(2), Effect.Empower(2, 'Chomper')), {'Plant', 'Chomper'}),
    Card('Broodmother', 4, 'Boss', (Effect.ATK(2), Effect.Empower(2, 'Midge')), {'Bug'}),
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
    Card('Primal Weevil', 6, 'Mini-Boss', (Effect.ATK(3), Effect.Empower(2, 'Weevil')), {'Bug'}),
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
    Card('Midge', 1, 'Battle', (Effect.ATK(1),), {'Bug', 'Midge'}),
    Card('Chomper', 2, 'Battle', (Effect.ATK(2),), {'Plant', 'Chomper'}),
    Card('Chomper Brute', 4, 'Battle', (Effect.ATK(4),), {'Plant', 'Chomper'}),
    Card('Wild Chomper', 3, 'Effect', (Effect.ATK(1), Effect.Coin(Effect.Summon('Chomper'))), {'Plant', 'Chomper'}),
    Card('Weevil', 2, 'Effect', (Effect.ATK(1), Effect.VS('Plant', Effect.ATK(2))), {'Bug', 'Weevil'}),
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
