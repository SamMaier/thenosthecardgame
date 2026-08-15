"""Social card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.fun_effects import _is_after, _today_position

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class DatesFirstNozBehavior(CardBehavior):
    """Pick from the Suitcase after later plays, then boost Tomorrow's plays."""

    has_tomorrow_action = True

    def on_card_play(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        played_card: CardInstance,
    ) -> None:
        if _is_after(player, source, played_card):
            game.pick_from_suitcase(game.players.index(player))

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if source.is_tomorrow and _today_position(player, target) is not None:
            return current_fun + 1
        return current_fun


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

DATES_FIRST_NOZ = CardDefinition(
    slug="dates-first-noz",
    title="Date's First Noz",
    tags=frozenset({"Social"}),
    cost=7,
    behavior=DatesFirstNozBehavior(),
)
