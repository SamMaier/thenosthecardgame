"""AI decision interfaces and the baseline random player."""

from __future__ import annotations

import random
from typing import Protocol, Sequence, TYPE_CHECKING

from thenos.cards.base import CardInstance

if TYPE_CHECKING:
    from thenos.game import Game


class PlayerAI(Protocol):
    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        """Return one eligible player index for a card effect."""

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        """Return an index into eligible cards for a copying effect."""

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        """Return an index into ``suitcase``."""

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        """Return an index into ``suitcase`` without taking that card."""

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> int:
        """Return an index into the player's hand to discard."""

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> Sequence[int]:
        """Return the hand indices of any cards to discard."""

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        """Return one of ``playable_hand_indices``."""

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        """Whether to pay 1 Energy for a second pick during this selection."""


class RandomAI:
    """Pick uniformly and play until no legal affordable card remains."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        if not eligible_player_indices:
            raise ValueError("Cannot choose from an empty player selection")
        return self.rng.choice(eligible_player_indices)

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        if not eligible_cards:
            raise ValueError("Cannot choose from an empty card selection")
        return self.rng.randrange(len(eligible_cards))

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        if not suitcase:
            raise ValueError("Cannot choose from an empty Suitcase")
        return self.rng.randrange(len(suitcase))

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        if not suitcase:
            raise ValueError("Cannot choose a target from an empty Suitcase")
        return self.rng.randrange(len(suitcase))

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> int:
        if not hand:
            raise ValueError("Cannot choose a card from an empty hand")
        return self.rng.randrange(len(hand))

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> Sequence[int]:
        count = self.rng.randrange(len(hand) + 1)
        return self.rng.sample(range(len(hand)), count)

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        return self.rng.choice(playable_hand_indices)

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        # The baseline AI does not take optional Unpack-like actions.
        return False
