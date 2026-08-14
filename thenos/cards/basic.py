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


class SoloBehavior(CardBehavior):
    """Score a bonus when no opponent played a Board Game previously today."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        opponents = [opponent for opponent in game.players if opponent is not player]
        if not any(
            "Board Game" in played_card.definition.tags
            for opponent in opponents
            for played_card in opponent.played_today
        ):
            card.markers["energy_cube"] = True

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.definition.base_fun + (2 if card.markers.get("energy_cube") else 0)


class CarcassonneBehavior(CardBehavior):
    """Score a bonus when directly between two Board Games played today."""

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
        if card_position is None or card_position == 0:
            return card.definition.base_fun

        if card_position + 1 >= len(player.played_today):
            return card.definition.base_fun

        neighbors = (
            player.played_today[card_position - 1],
            player.played_today[card_position + 1],
        )
        if all("Board Game" in neighbor.definition.tags for neighbor in neighbors):
            return card.definition.base_fun + 4
        return card.definition.base_fun


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

SOLO = CardDefinition(
    slug="solo",
    title="Solo",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=3,
    behavior=SoloBehavior(),
)

CARCASSONNE = CardDefinition(
    slug="carcassonne",
    title="Carcassonne",
    tags=frozenset({"Board Game"}),
    cost=2,
    base_fun=1,
    behavior=CarcassonneBehavior(),
)

WATERSKI = CardDefinition(
    slug="waterski",
    title="Waterski",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=6,
)
