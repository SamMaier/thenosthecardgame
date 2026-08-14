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


class TheCrewBehavior(CardBehavior):
    """Remember whether enough opponents played Board Games before this."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        opponents = [opponent for opponent in game.players if opponent is not player]
        board_game_opponents = sum(
            any(
                "Board Game" in played_card.definition.tags
                for played_card in opponent.played_today
            )
            for opponent in opponents
        )
        if board_game_opponents * 2 >= len(opponents):
            card.markers["energy_cube"] = True

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.definition.base_fun + (2 if card.markers.get("energy_cube") else 0)


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

THE_CREW = CardDefinition(
    slug="the-crew",
    title="The Crew",
    tags=frozenset({"Board Game"}),
    cost=1,
    base_fun=1,
    behavior=TheCrewBehavior(),
)

WATERSKI = CardDefinition(
    slug="waterski",
    title="Waterski",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=6,
)
