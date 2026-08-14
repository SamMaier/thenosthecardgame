"""Food cards with immediate effects based on cards played today."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class DoritosBehavior(CardBehavior):
    """Gain Energy after playing at least two earlier Food cards today."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return

        food_cards_played = sum(
            "Food" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        if food_cards_played >= 2:
            player.energy += 2


DORITOS = CardDefinition(
    slug="doritos",
    title="Doritos",
    tags=frozenset({"Food"}),
    cost=0,
    behavior=DoritosBehavior(),
)


FOOD_CARDS = (DORITOS,)
