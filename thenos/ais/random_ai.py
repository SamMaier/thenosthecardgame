"""The baseline random player policy."""

from __future__ import annotations

import random
from typing import Sequence, TYPE_CHECKING

from thenos.cards.base import CardInstance

if TYPE_CHECKING:
    from thenos.game import Game


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

    def choose_card_target(
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

    def choose_extra_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        # The baseline AI never voluntarily stops while it can afford a card.
        return self.choose_card_to_play(
            game, player_index, playable_hand_indices
        )

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        # The baseline AI does not take optional Unpack-like actions.
        return False

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        # The baseline AI does not take optional actions.
        return False

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        return self.rng.randrange(maximum + 1)

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        """Choose uniformly among all possible orderings."""
        return self.rng.sample(range(len(cards)), len(cards))
