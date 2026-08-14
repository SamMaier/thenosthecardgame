"""Event card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class SingSongBehavior(CardBehavior):
    """Score one Fun for each distinct tag played by this player today."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        unique_tags = {
            tag
            for played_card in player.played_today
            for tag in played_card.tags
        }
        return card.effective_base_fun + len(unique_tags)


SING_SONG = CardDefinition(
    slug="sing-song",
    title="Sing Song",
    tags=frozenset({"Event"}),
    cost=4,
    base_fun=1,
    behavior=SingSongBehavior(),
)
