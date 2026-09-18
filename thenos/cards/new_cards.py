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


class BohnanzaBehavior(CardBehavior):
    """Trash selected earlier plays and score one Fun per trashed card."""

    def on_play(self, game: Game, player: PlayerState, card: CardInstance) -> None:
        player_index = game.players.index(player)
        previous_cards = tuple(game.cards_played_before(player, card))
        if not previous_cards:
            return

        choices = tuple(
            game.ais[player_index].choose_cards_to_discard(
                game, player_index, previous_cards
            )
        )
        if len(set(choices)) != len(choices):
            raise ValueError("Cannot trash the same played card more than once")
        if any(index < 0 or index >= len(previous_cards) for index in choices):
            raise ValueError("Invalid played-card index in Bohnanza selection")

        for index in choices:
            target = previous_cards[index]
            player.played_today.remove(target)
            game.discard_card(target)
        card.markers["energy_cubes"] = len(choices)

    def fun_value(self, game: Game, player: PlayerState, card: CardInstance) -> int:
        return card.effective_base_fun + int(card.markers.get("energy_cubes", 0))


class IstanbulBehavior(CardBehavior):
    """Discard any number of hand cards and gain two Energy per discard."""

    def on_play(self, game: Game, player: PlayerState, card: CardInstance) -> None:
        player_index = game.players.index(player)
        hand = tuple(player.hand)
        choices = (
            tuple(
                game.ais[player_index].choose_cards_to_discard(
                    game, player_index, hand
                )
            )
            if hand
            else ()
        )
        discarded = game.discard_cards_from_hand(player_index, choices)
        game.gain_energy(player, 2 * len(discarded), card)


class QuacksBehavior(CardBehavior):
    """Reveal Trunk cards for Fun, but bust if an Outdoors card appears."""

    def on_play(self, game: Game, player: PlayerState, card: CardInstance) -> None:
        player_index = game.players.index(player)
        revealed_count = 0
        outdoors_revealed = False
        revealed_ids: set[int] = set()
        while game.trunk or game.discard:
            if (
                not game.trunk
                and not any(
                    id(candidate) not in revealed_ids
                    for candidate in game.discard
                )
            ):
                break
            if not game.choose_optional_action(
                player_index, "reveal another card for Quacks"
            ):
                break
            revealed = game.reveal_from_trunk(1)[0]
            revealed_ids.add(id(revealed))
            revealed_count += 1
            outdoors_revealed |= "Outdoors" in revealed.tags
            game.discard_card(revealed)
            if outdoors_revealed:
                break
        card.markers["energy_cubes"] = revealed_count
        card.markers["_quacks_outdoors"] = outdoors_revealed

    def fun_value(self, game: Game, player: PlayerState, card: CardInstance) -> int:
        if card.markers.get("_quacks_outdoors"):
            return 0
        return card.effective_base_fun + int(card.markers.get("energy_cubes", 0))


POTATO_PANCAKES = CardDefinition(
    slug="potato-pancakes", title="Potato Pancakes",
    tags=frozenset({"Food", "Outdoors"}),
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

BOHNANZA = CardDefinition(
    slug="bohnanza",
    title="Bohnanza",
    tags=frozenset({"Board Game"}),
    cost=2,
    base_fun=1,
    behavior=BohnanzaBehavior(),
)
ISTANBUL = CardDefinition(
    slug="istanbul",
    title="Istanbul",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=1,
    behavior=IstanbulBehavior(),
)
QUACKS = CardDefinition(
    slug="quacks",
    title="Quacks",
    tags=frozenset({"Board Game"}),
    cost=1,
    behavior=QuacksBehavior(),
)
