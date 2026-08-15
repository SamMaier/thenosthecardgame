"""Item card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class BoobyPrizeBehavior(CardBehavior):
    """Draw one card from the Trunk into the player's hand."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        drawn_card = game.reveal_from_trunk(1)[0]
        game.give_card(player_index, drawn_card)


class FishingBoatBehavior(CardBehavior):
    """Reveal through the Trunk for an Outdoors card and reorder the misses."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        revealed: list[CardInstance] = []
        while True:
            revealed_card = game.reveal_from_trunk(1)[0]
            if "Outdoors" in revealed_card.tags:
                game.give_card(player_index, revealed_card)
                break
            revealed.append(revealed_card)

        game.return_cards_to_trunk_top(player_index, revealed)


class SkiBoatBehavior(CardBehavior):
    """Reveal through the Trunk for an Exercise card and reorder the misses."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        revealed: list[CardInstance] = []
        while True:
            revealed_card = game.reveal_from_trunk(1)[0]
            if "Exercise" in revealed_card.tags:
                game.give_card(player_index, revealed_card)
                break
            revealed.append(revealed_card)

        game.return_cards_to_trunk_top(player_index, revealed)


class FancyFloatieBehavior(CardBehavior):
    """Pick the Relax cards currently visible in the Suitcase."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        targets = tuple(
            suitcase_card
            for suitcase_card in game.suitcase
            if "Relax" in suitcase_card.tags
        )
        if targets:
            game.pick_suitcase_cards(game.players.index(player), targets)


BOOBY_PRIZE = CardDefinition(
    slug="booby-prize",
    title="Booby Prize",
    tags=frozenset({"Item"}),
    cost=0,
    behavior=BoobyPrizeBehavior(),
)


FISHING_BOAT = CardDefinition(
    slug="fishing-boat",
    title="Fishing Boat",
    tags=frozenset({"Item"}),
    cost=2,
    behavior=FishingBoatBehavior(),
)


SKI_BOAT = CardDefinition(
    slug="ski-boat",
    title="Ski Boat",
    tags=frozenset({"Item"}),
    cost=2,
    behavior=SkiBoatBehavior(),
)


FANCY_FLOATIE = CardDefinition(
    slug="fancy-floatie",
    title="Fancy Floatie",
    tags=frozenset({"Item"}),
    cost=2,
    behavior=FancyFloatieBehavior(),
)
