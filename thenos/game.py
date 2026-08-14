"""Rules engine for a complete headless game."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from thenos.ai import PlayerAI, RandomAI
from thenos.cards.base import CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.models import GameStats, PlayerState


PLAYER_COUNT = 4
DAYS_PER_GAME = 6
STARTING_HAND_SIZE = 7
DAILY_ENERGY = 7
SUITCASE_SIZE = 4
DAILY_PICKS = 3


class DeckExhaustedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[int, ...]
    win_shares: tuple[float, ...]
    days_played: int


def fractional_wins(scores: Sequence[int]) -> tuple[float, ...]:
    """Award one win divided evenly among all highest-scoring players."""
    if not scores:
        return ()
    high_score = max(scores)
    winner_count = sum(score == high_score for score in scores)
    share = 1.0 / winner_count
    return tuple(share if score == high_score else 0.0 for score in scores)


class Game:
    def __init__(
        self,
        deck: list[CardInstance],
        ais: Sequence[PlayerAI],
        rng: random.Random | None = None,
    ) -> None:
        if len(ais) != PLAYER_COUNT:
            raise ValueError(f"The Nos requires exactly {PLAYER_COUNT} AIs")
        self.rng = rng or random.Random()
        self.ais = list(ais)
        self.players = [PlayerState(f"Player {i + 1}") for i in range(PLAYER_COUNT)]
        self.trunk = list(deck)
        self.discard: list[CardInstance] = []
        self.suitcase: list[CardInstance] = []
        self.stats = GameStats()
        self.day = 0
        self.starting_player = 0
        self._is_setup = False

    @classmethod
    def default(cls, seed: int | None = None) -> Game:
        rng = random.Random(seed)
        ais = [RandomAI(rng) for _ in range(PLAYER_COUNT)]
        return cls(create_default_deck(), ais, rng)

    def setup(self) -> None:
        if self._is_setup:
            raise RuntimeError("Game is already set up")
        minimum_size = PLAYER_COUNT * STARTING_HAND_SIZE + SUITCASE_SIZE
        if len(self.trunk) < minimum_size:
            raise ValueError(f"Deck needs at least {minimum_size} cards")
        self.rng.shuffle(self.trunk)
        for _ in range(STARTING_HAND_SIZE):
            for player_index in range(PLAYER_COUNT):
                self.give_card(player_index, self._draw_from_trunk())
        for _ in range(SUITCASE_SIZE):
            self.suitcase.append(self._draw_from_trunk())
        self._is_setup = True

    def _draw_from_trunk(self) -> CardInstance:
        if not self.trunk:
            if not self.discard:
                raise DeckExhaustedError("Both the Trunk and discard pile are empty")
            self.trunk = self.discard
            self.discard = []
            self.rng.shuffle(self.trunk)
        return self.trunk.pop()

    def discard_card(self, card: CardInstance) -> None:
        """Discard one card and clear state that must not survive recycling."""
        card.is_tomorrow = False
        card.markers.clear()
        self.discard.append(card)

    def player_order(self) -> list[int]:
        return [
            (self.starting_player + offset) % PLAYER_COUNT
            for offset in range(PLAYER_COUNT)
        ]

    def start_day(self) -> None:
        for player in self.players:
            player.energy = DAILY_ENERGY
            player.asleep = False
            for card in player.tomorrow_cards:
                card.effective_behavior.on_start_day(self, player, card)

    def draw_phase(self) -> None:
        for _ in range(DAILY_PICKS):
            for player_index in self.player_order():
                uses_extra_pick = self._uses_extra_suitcase_pick(player_index)
                if uses_extra_pick:
                    self.players[player_index].energy -= 1
                picks = 2 if uses_extra_pick else 1
                for _ in range(picks):
                    self.pick_from_suitcase(player_index)

    def _uses_extra_suitcase_pick(self, player_index: int) -> bool:
        player = self.players[player_index]
        if player.energy < 1:
            return False
        if not any(
            card.effective_behavior.allows_extra_suitcase_pick(
                self, player, card
            )
            for card in player.tomorrow_cards
        ):
            return False
        chooser = getattr(self.ais[player_index], "choose_extra_suitcase_pick", None)
        return chooser is not None and chooser(
            self, player_index, tuple(self.suitcase)
        )

    def pick_from_suitcase(self, player_index: int) -> CardInstance:
        choice = self.ais[player_index].choose_suitcase_card(
            self, player_index, tuple(self.suitcase)
        )
        if choice < 0 or choice >= len(self.suitcase):
            raise ValueError(f"AI returned invalid Suitcase index: {choice}")

        return self.pick_suitcase_cards(player_index, (self.suitcase[choice],))[0]

    def pick_suitcase_cards(
        self,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> list[CardInstance]:
        """Take specified physical Suitcase cards and refill their slots.

        ``cards`` must be a distinct subset of the currently visible cards.
        Refilling happens after each card is taken, so replacement cards are
        not included in the requested set.
        """
        if len({id(card) for card in cards}) != len(cards):
            raise ValueError("Cannot pick the same Suitcase card more than once")
        if any(card not in self.suitcase for card in cards):
            raise ValueError(
                "Every picked card must be currently visible in the Suitcase"
            )

        # Each physical card shown is one offer, including duplicate titles.
        self.stats.suitcase_offers.update(card.title for card in self.suitcase)
        picked_cards: list[CardInstance] = []
        for target in cards:
            choice = self.suitcase.index(target)
            card = self.suitcase.pop(choice)
            player = self.players[player_index]
            self.give_card(player_index, card)
            player.picked_cards[card.title] += 1
            self.stats.suitcase_picks[card.title] += 1
            # Refill the emptied spot immediately, as required by the rules.
            self.suitcase.insert(choice, self._draw_from_trunk())
            picked_cards.append(card)
        return picked_cards

    def choose_suitcase_target(self, player_index: int) -> CardInstance:
        """Choose a physical Suitcase card without taking or discarding it."""
        if not self.suitcase:
            raise ValueError("Cannot choose a target from an empty Suitcase")
        choice = self.ais[player_index].choose_suitcase_target(
            self, player_index, tuple(self.suitcase)
        )
        if choice < 0 or choice >= len(self.suitcase):
            raise ValueError(f"AI returned invalid Suitcase target index: {choice}")
        return self.suitcase[choice]

    def unpack(self, player_index: int, fun_delta: int = -1) -> None:
        """Discard and refill the Suitcase, applying the action's Fun change.

        The standard action passes the default ``-1``. Card effects that alter
        the reward can pass their printed value instead.
        """
        self.players[player_index].fun += fun_delta
        old_suitcase = self.suitcase
        self.suitcase = []
        for card in old_suitcase:
            self.discard_card(card)
        for _ in range(SUITCASE_SIZE):
            self.suitcase.append(self._draw_from_trunk())

    def give_card(self, player_index: int, card: CardInstance) -> None:
        """Put a card in a hand and record ownership for balance statistics.

        Card effects that acquire cards from any zone should use this method.
        """
        player = self.players[player_index]
        player.hand.append(card)
        player.acquired_cards[card.title] += 1
        self.stats.card_acquisitions[card.title] += 1

    def discard_from_hand(self, player_index: int, hand_index: int) -> CardInstance:
        """Remove one card from a player's hand and discard that copy."""
        player = self.players[player_index]
        if hand_index < 0 or hand_index >= len(player.hand):
            raise ValueError(f"Invalid hand index: {hand_index}")
        card = player.hand.pop(hand_index)
        self.discard_card(card)
        return card

    def discard_cards_from_hand(
        self,
        player_index: int,
        hand_indices: Sequence[int],
    ) -> list[CardInstance]:
        """Remove and discard several distinct cards from a player's hand."""
        player = self.players[player_index]
        indices = tuple(hand_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("Cannot discard the same hand card more than once")
        if any(index < 0 or index >= len(player.hand) for index in indices):
            raise ValueError("Invalid hand index in discard selection")

        discarded_cards: list[CardInstance] = []
        for index in sorted(indices, reverse=True):
            discarded_cards.append(self.discard_from_hand(player_index, index))
        discarded_cards.reverse()
        return discarded_cards

    def choose_player(
        self,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        """Ask a player's AI to choose one player from an eligible set."""
        if not eligible_player_indices:
            raise ValueError("Cannot choose from an empty player selection")
        choice = self.ais[player_index].choose_player(
            self, player_index, tuple(eligible_player_indices)
        )
        if choice not in eligible_player_indices:
            raise ValueError(f"AI selected ineligible player index: {choice}")
        return choice

    def choose_card_to_copy(
        self,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> CardInstance:
        """Ask a player's AI to choose one eligible physical card."""
        if not eligible_cards:
            raise ValueError("Cannot choose from an empty card selection")
        choice = self.ais[player_index].choose_card_to_copy(
            self, player_index, tuple(eligible_cards)
        )
        if choice < 0 or choice >= len(eligible_cards):
            raise ValueError(f"AI selected an invalid card index: {choice}")
        return eligible_cards[choice]

    def choose_card_target(
        self,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> CardInstance:
        """Ask a player's AI to choose one physical card as a target."""
        if not eligible_cards:
            raise ValueError("Cannot choose from an empty card selection")
        choice = self.ais[player_index].choose_card_target(
            self, player_index, tuple(eligible_cards)
        )
        if choice < 0 or choice >= len(eligible_cards):
            raise ValueError(f"AI selected an invalid card index: {choice}")
        return eligible_cards[choice]

    def energy_cost(self, player_index: int, card: CardInstance) -> int:
        player = self.players[player_index]
        cost = card.effective_cost
        for source in player.visible_cards:
            cost = source.effective_behavior.modify_energy_cost(
                self, player, source, card, cost
            )
        return max(0, cost)

    def playable_hand_indices(self, player_index: int) -> list[int]:
        player = self.players[player_index]
        return [
            index
            for index, card in enumerate(player.hand)
            if card.effective_behavior.can_play(self, player, card)
            and self.energy_cost(player_index, card) <= player.energy
        ]

    def play_card(self, player_index: int, hand_index: int) -> CardInstance:
        player = self.players[player_index]
        if hand_index < 0 or hand_index >= len(player.hand):
            raise ValueError(f"Invalid hand index: {hand_index}")
        card = player.hand[hand_index]
        if not card.effective_behavior.can_play(self, player, card):
            raise ValueError(f"{card.title} cannot legally be played")
        cost = self.energy_cost(player_index, card)
        if cost > player.energy:
            raise ValueError(f"Not enough Energy to play {card.title}")

        player.energy -= cost
        player.hand.pop(hand_index)
        player.played_today.append(card)
        self.stats.card_plays[card.title] += 1
        card.effective_behavior.on_play(self, player, card)
        for source in player.tomorrow_cards:
            source.effective_behavior.on_card_play(self, player, source, card)
        return card

    def copy_card_effect(
        self,
        player_index: int,
        source: CardInstance,
        destination: CardInstance,
    ) -> None:
        """Resolve ``source``'s effect as a new play of ``destination``."""
        player = self.players[player_index]
        source_cost = source.definition.cost
        if source_cost > player.energy:
            raise ValueError(f"Not enough Energy to copy {source.title}")
        player.energy -= source_cost

        copied_definition = CardDefinition(
            slug=destination.definition.slug,
            title=destination.definition.title,
            tags=destination.definition.tags,
            cost=source_cost,
            base_fun=source.effective_base_fun,
            behavior=source.effective_behavior,
        )
        destination.markers["_copied_definition"] = copied_definition
        destination.markers["_copying_effect"] = True
        source.effective_behavior.on_play(self, player, destination)

    def playing_phase(self) -> None:
        first_to_bed: int | None = None
        awake = PLAYER_COUNT
        order = self.player_order()

        while awake:
            for player_index in order:
                player = self.players[player_index]
                if player.asleep:
                    continue
                playable = self.playable_hand_indices(player_index)
                if not playable:
                    player.asleep = True
                    awake -= 1
                    if first_to_bed is None:
                        first_to_bed = player_index
                    continue

                choice = self.ais[player_index].choose_card_to_play(
                    self, player_index, tuple(playable)
                )
                if choice not in playable:
                    raise ValueError(f"AI selected unplayable hand index: {choice}")
                played_card = self.play_card(player_index, choice)

                # A normal turn ends after one card.  Some cards explicitly
                # let their player continue playing during this same turn.
                extra_plays = played_card.effective_behavior.allows_extra_card_plays(
                    self, player, played_card
                )
                while extra_plays:
                    playable = self.playable_hand_indices(player_index)
                    if not playable:
                        break
                    choice = self.ais[player_index].choose_card_to_play(
                        self, player_index, tuple(playable)
                    )
                    if choice not in playable:
                        raise ValueError(
                            f"AI selected unplayable hand index: {choice}"
                        )
                    self.play_card(player_index, choice)

        if first_to_bed is None:
            raise RuntimeError("Playing phase ended without a first player going to bed")
        self.starting_player = first_to_bed

    def card_fun(self, player_index: int, target: CardInstance) -> int:
        player = self.players[player_index]
        value = target.effective_behavior.fun_value(self, player, target)
        for source in player.visible_cards:
            value = source.effective_behavior.modify_fun(
                self, player, source, target, value
            )
        return value

    def end_day(self) -> None:
        for player_index, player in enumerate(self.players):
            # visible_cards orders active Tomorrow cards before today's cards.
            for card in player.visible_cards:
                player.fun += self.card_fun(player_index, card)

        for player in self.players:
            previous_tomorrow = player.tomorrow_cards
            player.tomorrow_cards = []
            for card in previous_tomorrow:
                self.discard_card(card)

            for card in player.played_today:
                if card.effective_behavior.has_tomorrow_action:
                    card.is_tomorrow = True
                    player.tomorrow_cards.append(card)
                else:
                    self.discard_card(card)
            player.played_today = []

    def run_day(self) -> None:
        if not self._is_setup:
            raise RuntimeError("Call setup() before running a day")
        if self.day >= DAYS_PER_GAME:
            raise RuntimeError("All days have already been played")
        self.day += 1
        self.start_day()
        self.draw_phase()
        self.playing_phase()
        self.end_day()

    def result(self) -> GameResult:
        scores = tuple(player.fun for player in self.players)
        return GameResult(scores, fractional_wins(scores), self.day)

    def run(self) -> GameResult:
        if not self._is_setup:
            self.setup()
        while self.day < DAYS_PER_GAME:
            self.run_day()
        return self.result()
