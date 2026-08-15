"""Item card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class BoobyPrizeBehavior(CardBehavior):
    """Draw one card from the Trunk into the player's hand."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        drawn_card = game.reveal_from_trunk(1)[0]
        game.give_card(player_index, drawn_card)


BOOBY_PRIZE = CardDefinition(
    slug="booby-prize",
    title="Booby Prize",
    tags=frozenset({"Item"}),
    cost=0,
    behavior=BoobyPrizeBehavior(),
)
