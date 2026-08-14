"""Exercise card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class MorningWalkBehavior(CardBehavior):
    """Start the next day with two additional Energy; play first today."""

    has_tomorrow_action = True

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not player.played_today

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy += 2


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


class PlayWithTheKidsBehavior(CardBehavior):
    """Score a bonus when this player has the largest hand at day's end."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        has_more_cards = all(
            len(player.hand) > len(opponent.hand)
            for opponent in game.players
            if opponent is not player
        )
        bonus = 3 if has_more_cards else 0
        return card.effective_base_fun + bonus


class WrestleTheKidsBehavior(CardBehavior):
    """Score one additional Fun for each card remaining in this player's hand."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + len(player.hand)


ZUMBA = CardDefinition(
    slug="zumba",
    title="Zumba",
    tags=frozenset({"Exercise"}),
    cost=4,
    base_fun=2,
    behavior=ZumbaBehavior(),
)

PLAY_WITH_THE_KIDS = CardDefinition(
    slug="play-with-the-kids",
    title="Play With the Kids",
    tags=frozenset({"Exercise"}),
    cost=4,
    base_fun=3,
    behavior=PlayWithTheKidsBehavior(),
)

WRESTLE_THE_KIDS = CardDefinition(
    slug="wrestle-the-kids",
    title="Wrestle the Kids",
    tags=frozenset({"Exercise"}),
    cost=5,
    base_fun=1,
    behavior=WrestleTheKidsBehavior(),
)

MORNING_WALK = CardDefinition(
    slug="morning-walk",
    title="Morning Walk",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=1,
    behavior=MorningWalkBehavior(),
)
