"""Additional cards from the current catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class PotatoPancakesBehavior(CardBehavior):
    def on_play(self, game: Game, player: PlayerState, card: CardInstance) -> None:
        count = sum(
            "Relax" in previous.tags
            for previous in game.cards_played_before(player, card)
        )
        game.gain_energy(player, 2 * count, card)


class ReadTheRadarBehavior(CardBehavior):
    def on_play(self, game: Game, player: PlayerState, card: CardInstance) -> None:
        game.arrange_daily_conditions(game.players.index(player))


class PokerBehavior(CardBehavior):
    def on_play(self, game: Game, player: PlayerState, card: CardInstance) -> None:
        from thenos.cards.catalog import CARD_REGISTRY

        tags = tuple(sorted({
            tag for definition in CARD_REGISTRY.values() for tag in definition.tags
        }))
        index = game.players.index(player)
        tag = game.ais[index].choose_tag(game, index, tags)
        if tag not in tags:
            raise ValueError(f"AI selected an invalid tag: {tag}")
        revealed = game.reveal_from_trunk(1)[0]
        card.markers["poker_success"] = tag in revealed.tags
        game.discard_card(revealed)

    def fun_value(self, game: Game, player: PlayerState, card: CardInstance) -> int:
        return card.effective_base_fun + 4 * bool(card.markers.get("poker_success"))


POTATO_PANCAKES = CardDefinition(
    slug="potato-pancakes", title="Potato Pancakes", tags=frozenset({"Food"}),
    cost=2, behavior=PotatoPancakesBehavior(),
)
READ_THE_RADAR = CardDefinition(
    slug="read-the-radar", title="Read the Radar", tags=frozenset({"Relax"}),
    cost=1, behavior=ReadTheRadarBehavior(),
)
POKER = CardDefinition(
    slug="poker", title="Poker", tags=frozenset({"Board Game"}),
    cost=2, base_fun=1, behavior=PokerBehavior(),
)
