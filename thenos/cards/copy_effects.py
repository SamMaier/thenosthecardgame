"""Cards that resolve another visible card's effect as their own."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class WeddingAnniversaryBehavior(CardBehavior):
    """Copy one legally playable opponent card from today's play area."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        # A copied Wedding Anniversary cannot recursively copy another effect.
        # Without a terminal case, two Weddings could select one another forever.
        if card.markers.get("_copying_effect"):
            return

        player_index = game.players.index(player)
        eligible_cards = [
            candidate
            for opponent_index, opponent in enumerate(game.players)
            if opponent_index != player_index
            for candidate in opponent.played_today
            if candidate.definition.cost <= player.energy
            and candidate.effective_behavior.can_play(game, player, card)
        ]
        if not eligible_cards:
            return

        target = game.choose_card_to_copy(player_index, eligible_cards)
        target.markers["energy_cube"] = True
        game.copy_card_effect(player_index, target, card)


WEDDING_ANNIVERSARY = CardDefinition(
    slug="wedding-anniversary",
    title="Wedding Anniversary",
    tags=frozenset({"Social"}),
    cost=0,
    behavior=WeddingAnniversaryBehavior(),
)


class CopyActiveItemBehavior(CardBehavior):
    """Copy one player's active Item card without copying its cost or tags."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        copy_chain = card.markers.get("_copy_chain", ())
        eligible_cards = [
            candidate
            for candidate_player in game.players
            for candidate in candidate_player.played_today
            if candidate is not card
            and candidate.instance_id not in copy_chain
            and "Item" in candidate.tags
            and candidate.effective_behavior.can_play(game, player, card)
        ]
        if not eligible_cards:
            return

        target = game.choose_card_to_copy(player_index, eligible_cards)
        target.markers["energy_cube"] = True
        game.copy_card_effect(
            player_index,
            target,
            card,
            pay_source_cost=False,
        )


NOS_SHIRT = CardDefinition(
    slug="nos-shirt",
    title="Nos Shirt",
    tags=frozenset({"Item"}),
    cost=1,
    behavior=CopyActiveItemBehavior(),
)


class LastYearsShortsBehavior(CardBehavior):
    """Take any card from the discard pile."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        if not game.discard:
            return
        player_index = game.players.index(player)
        target = game.choose_card_target(player_index, tuple(game.discard))
        game.discard.remove(target)
        game.give_card(player_index, target)


LAST_YEARS_SHORTS = CardDefinition(
    slug="last-years-shorts",
    title="Last Year's Shorts",
    tags=frozenset({"Item"}),
    cost=2,
    behavior=LastYearsShortsBehavior(),
)
