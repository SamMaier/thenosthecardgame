"""Exercise card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class ZumbaBehavior(CardBehavior):
    """Start the next day with three additional Energy."""

    has_tomorrow_action = True

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 3


ZUMBA = CardDefinition(
    slug="zumba",
    title="Zumba",
    tags=frozenset({"Exercise"}),
    cost=4,
    base_fun=2,
    behavior=ZumbaBehavior(),
)
