"""Pure Energy card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class GainOneEnergyBehavior(CardBehavior):
    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 1, card)


M_AND_MS = CardDefinition(
    slug="m-ms",
    title="M&Ms",
    tags=frozenset({"Food"}),
    cost=0,
    behavior=GainOneEnergyBehavior(),
)

NAP = CardDefinition(
    slug="nap",
    title="Nap",
    tags=frozenset({"Relax", "Indoors"}),
    cost=0,
    behavior=GainOneEnergyBehavior(),
)


PURE_ENERGY_CARDS = (M_AND_MS, NAP)
