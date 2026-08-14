"""The first three implemented cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class FajitasBehavior(CardBehavior):
    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 4


BIOGRAPHY = CardDefinition(
    slug="biography",
    title="Biography",
    tags=frozenset({"Relax"}),
    cost=1,
    base_fun=2,
)

FAJITAS = CardDefinition(
    slug="fajitas",
    title="Fajitas",
    tags=frozenset({"Food"}),
    cost=3,
    behavior=FajitasBehavior(),
)

WATERSKI = CardDefinition(
    slug="waterski",
    title="Waterski",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=6,
)

