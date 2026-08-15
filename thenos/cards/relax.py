"""Relax card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class EarlyBedtimeBehavior(CardBehavior):
    """Start the next day with three additional Energy; play before the fourth card."""

    has_tomorrow_action = True

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return len(player.played_today) < 3

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 3, card)


EARLY_BEDTIME = CardDefinition(
    slug="early-bedtime",
    title="Early Bedtime",
    tags=frozenset({"Relax"}),
    cost=1,
    behavior=EarlyBedtimeBehavior(),
)
