"""The stable decision interface implemented by every player AI."""

from __future__ import annotations

from typing import Protocol, Sequence, TYPE_CHECKING

from thenos.cards.base import CardInstance

if TYPE_CHECKING:
    from thenos.game import Game


class PlayerAI(Protocol):
    """All choices the rules engine may delegate to a player policy."""

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

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        """Return an index into eligible cards for a card-targeting effect."""

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

    def choose_to_go_to_bed(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> bool:
        """Whether to stop playing despite having a legal card available."""

    def choose_extra_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        """Return one playable index for an extra play, or ``None`` to stop."""

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        """Whether to pay 1 Energy for a second pick during this selection."""

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        """Whether to take an optional rules action after seeing its setup."""

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        """Return how much optional Energy to spend on a card effect."""

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        """Return card indices in the desired top-to-bottom Trunk order."""
