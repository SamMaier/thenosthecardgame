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
        self._fun_at_start_of_scoring: tuple[int, ...] | None = None

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

    def reveal_from_trunk(self, count: int) -> list[CardInstance]:
        """Remove and return ``count`` cards from the top of the Trunk."""
        if count < 0:
            raise ValueError("Cannot reveal a negative number of cards")
        return [self._draw_from_trunk() for _ in range(count)]

    def draw_from_trunk(
        self,
        player_index: int,
        count: int,
    ) -> list[CardInstance]:
        """Draw cards for a player and resolve visible pick-or-draw reactions."""
        cards = self.reveal_from_trunk(count)
        for card in cards:
            self.record_card_pick_or_draw(player_index, card)
        return cards

    def return_cards_to_trunk_top(
        self,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> list[CardInstance]:
        """Let an AI order cards and return them to the top of the Trunk.

        AI order indices are interpreted top to bottom. Internally the Trunk's
        top is the end of the list, so cards are appended in reverse order.
        """
        cards = tuple(cards)
        if not cards:
            return []
        order = tuple(
            self.ais[player_index].order_cards_for_trunk(
                self, player_index, cards
            )
        )
        if sorted(order) != list(range(len(cards))):
            raise ValueError(
                "AI must return every Trunk-order index exactly once"
            )
        ordered_cards = [cards[index] for index in order]
        self.trunk.extend(reversed(ordered_cards))
        return ordered_cards

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

    def gain_energy(
        self,
        player: PlayerState,
        amount: int,
        source: CardInstance | None = None,
    ) -> None:
        """Give Energy and remember the card that gave it, when applicable."""
        if amount > 0 and not all(
            card.effective_behavior.allows_energy_gain(
                self, player, card, source
            )
            for card in player.visible_cards
        ):
            return
        player.energy += amount
        if amount > 0 and source is not None:
            source.markers["_gave_energy"] = True

    def skip_next_turn(self, player: PlayerState) -> None:
        """Queue one skipped playing-phase turn for ``player``."""
        player.skipped_turns += 1

    def go_to_bed(self, player_index: int) -> None:
        """End a player's remaining turns in the current playing phase."""
        self.players[player_index].asleep = True

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
            self.record_card_pick_or_draw(player_index, card)
            player.picked_cards[card.title] += 1
            self.stats.suitcase_picks[card.title] += 1
            # Refill the emptied spot immediately, as required by the rules.
            self.suitcase.insert(choice, self._draw_from_trunk())
            picked_cards.append(card)
        return picked_cards

    def acquire_suitcase_cards(
        self,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> list[CardInstance]:
        """Move specified Suitcase cards to a hand without counting picks."""
        if len({id(card) for card in cards}) != len(cards):
            raise ValueError("Cannot acquire the same Suitcase card more than once")
        if any(card not in self.suitcase for card in cards):
            raise ValueError(
                "Every acquired card must be currently visible in the Suitcase"
            )

        acquired_cards: list[CardInstance] = []
        for target in cards:
            choice = self.suitcase.index(target)
            card = self.suitcase.pop(choice)
            self.give_card(player_index, card)
            self.suitcase.insert(choice, self._draw_from_trunk())
            acquired_cards.append(card)
        return acquired_cards

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
        for source in player.visible_cards:
            source.effective_behavior.on_card_acquire(
                self, player, source, card
            )

    def record_card_pick_or_draw(
        self,
        player_index: int,
        card: CardInstance,
    ) -> None:
        """Notify visible cards that their owner picked or drew ``card``."""
        player = self.players[player_index]
        for source in player.visible_cards:
            source.effective_behavior.on_card_pick_or_draw(
                self, player, source, card
            )

    def cards_played_before(
        self,
        player: PlayerState,
        card: CardInstance,
    ) -> list[CardInstance]:
        """Return today's cards before ``card`` or every play if it is in hand."""
        for position, played_card in enumerate(player.played_today):
            if played_card is card:
                return player.played_today[:position]
        return player.played_today

    def fun_at_start_of_scoring(self, player: PlayerState) -> int:
        """Return a stable total for comparisons made during end-day scoring."""
        if self._fun_at_start_of_scoring is None:
            return player.fun
        return self._fun_at_start_of_scoring[self.players.index(player)]

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

    def choose_energy_to_spend(
        self,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        """Choose an optional amount of Energy for a card effect."""
        if maximum < 0:
            raise ValueError("Maximum optional Energy cannot be negative")
        choice = self.ais[player_index].choose_energy_to_spend(
            self, player_index, card, maximum
        )
        if choice < 0 or choice > maximum:
            raise ValueError(f"AI returned invalid optional Energy amount: {choice}")
        return choice

    def choose_optional_action(self, player_index: int, action: str) -> bool:
        """Ask whether a player wants to take an optional rules action."""
        return self.ais[player_index].choose_optional_action(
            self, player_index, action
        )

    def energy_cost(self, player_index: int, card: CardInstance) -> int:
        player = self.players[player_index]
        cost = card.effective_behavior.modify_own_energy_cost(
            self, player, card, card.effective_cost
        )
        for source in player.visible_cards:
            cost = source.effective_behavior.modify_energy_cost(
                self, player, source, card, cost
            )
        return max(0, cost)

    def playable_hand_indices(self, player_index: int) -> list[int]:
        player = self.players[player_index]
        if player.asleep:
            return []
        return [
            index
            for index, card in enumerate(player.hand)
            if card.effective_behavior.can_play(self, player, card)
            and self.energy_cost(player_index, card) <= player.energy
        ]

    def play_card(self, player_index: int, hand_index: int) -> CardInstance:
        player = self.players[player_index]
        if player.asleep:
            raise ValueError(f"{player.name} has already gone to bed")
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
        for source in player.visible_cards:
            source.effective_behavior.on_card_play(self, player, source, card)
        return card

    def play_card_from_trunk(self, player_index: int) -> CardInstance:
        """Play the Trunk's top card for an effect without paying Energy.

        A card that cannot legally be played is returned to the player's hand
        without a cost, as required for cards that play cards from the Trunk.
        """
        card = self._draw_from_trunk()
        return self.play_card_for_effect(player_index, card)

    def play_card_for_effect(
        self,
        player_index: int,
        card: CardInstance,
        *,
        cost_adjustment: int = 0,
        pay_energy: bool = False,
    ) -> CardInstance:
        """Play a card for an effect, optionally adjusting its Energy cost.

        A card that cannot legally be played is returned to the player's hand
        without a cost. If ``card`` is already in the player's hand, it is
        removed when successfully played; otherwise the caller owns removing
        it from its source zone. When ``pay_energy`` is true, an adjustment is
        applied after the normal visible-card cost modifiers and the result is
        clamped at zero.
        """
        player = self.players[player_index]
        if not card.effective_behavior.can_play(self, player, card):
            if card not in player.hand:
                self.give_card(player_index, card)
            return card

        cost = (
            max(0, self.energy_cost(player_index, card) + cost_adjustment)
            if pay_energy
            else 0
        )
        if cost > player.energy:
            if card not in player.hand:
                self.give_card(player_index, card)
            return card

        acquired_from_effect = card not in player.hand
        if card in player.hand:
            player.hand.remove(card)
        player.energy -= cost
        player.played_today.append(card)
        self.stats.card_plays[card.title] += 1
        if acquired_from_effect:
            self.stats.card_plays_without_acquisition[card.title] += 1
        card.effective_behavior.on_play(self, player, card)
        for source in player.visible_cards:
            source.effective_behavior.on_card_play(self, player, source, card)
        return card

    def copy_card_effect(
        self,
        player_index: int,
        source: CardInstance,
        destination: CardInstance,
        *,
        pay_source_cost: bool = True,
    ) -> None:
        """Resolve ``source``'s effect as a new play of ``destination``."""
        player = self.players[player_index]
        copied_cost = (
            source.definition.cost
            if pay_source_cost
            else destination.effective_cost
        )
        if pay_source_cost:
            if copied_cost > player.energy:
                raise ValueError(f"Not enough Energy to copy {source.title}")
            player.energy -= copied_cost

        copied_definition = CardDefinition(
            slug=destination.definition.slug,
            title=destination.definition.title,
            tags=destination.definition.tags,
            cost=copied_cost,
            base_fun=source.effective_base_fun,
            behavior=source.effective_behavior,
        )
        destination.markers["_copied_definition"] = copied_definition
        destination.markers["_copying_effect"] = True
        previous_chain = destination.markers.get("_copy_chain", ())
        if not isinstance(previous_chain, tuple):
            previous_chain = ()
        destination.markers["_copy_chain"] = (*previous_chain, source.instance_id)
        source.effective_behavior.on_play(self, player, destination)

    def playing_phase(self) -> None:
        first_to_bed: int | None = None
        order = self.player_order()

        while any(not player.asleep for player in self.players):
            for player_index in order:
                player = self.players[player_index]
                if player.asleep:
                    continue
                if player.skipped_turns:
                    player.skipped_turns -= 1
                    continue
                playable = self.playable_hand_indices(player_index)
                if not playable:
                    self.go_to_bed(player_index)
                    if first_to_bed is None:
                        first_to_bed = player_index
                    continue

                choice = self.ais[player_index].choose_card_to_play(
                    self, player_index, tuple(playable)
                )
                if choice not in playable:
                    raise ValueError(f"AI selected unplayable hand index: {choice}")
                played_card = self.play_card(player_index, choice)

                if player.asleep:
                    if first_to_bed is None:
                        first_to_bed = player_index
                    continue

                # A normal turn ends after one card.  Some cards explicitly
                # let their player continue playing during this same turn.
                extra_plays = played_card.effective_behavior.allows_extra_card_plays(
                    self, player, played_card
                )
                while extra_plays:
                    playable = self.playable_hand_indices(player_index)
                    if not playable:
                        break
                    choice = self.ais[player_index].choose_extra_card_to_play(
                        self, player_index, tuple(playable)
                    )
                    if choice is None:
                        break
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
        self._fun_at_start_of_scoring = tuple(
            player.fun for player in self.players
        )
        for player_index, player in enumerate(self.players):
            # visible_cards orders active Tomorrow cards before today's cards.
            for card in player.visible_cards:
                player.fun += self.card_fun(player_index, card)
                card.effective_behavior.on_score(self, player, card)
        self._fun_at_start_of_scoring = None

        for player in self.players:
            for card in player.visible_cards:
                card.effective_behavior.on_end_day(self, player, card)

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
