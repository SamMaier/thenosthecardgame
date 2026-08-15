"""Social card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class TellAStoryBehavior(CardBehavior):
    """Score a bonus if this player played an Event earlier today."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return card.effective_base_fun

        has_previous_event = any(
            "Event" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        return card.effective_base_fun + (3 if has_previous_event else 0)


TELL_A_STORY = CardDefinition(
    slug="tell-a-story",
    title="Tell a Story",
    tags=frozenset({"Social"}),
    cost=3,
    base_fun=2,
    behavior=TellAStoryBehavior(),
)
