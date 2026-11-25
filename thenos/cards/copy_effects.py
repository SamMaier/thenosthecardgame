"""Cards that resolve another visible card's effect as their own."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.fun_effects import _is_next_card

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
            and candidate.definition.behavior.can_play(game, player, card)
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

class NosShirtBehavior(CardBehavior):
    """Boost the next card played today and return to hand after >7 Fun."""

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_next_card(player, source, target):
            return current_fun + 2
        return current_fun

    def on_end_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        if game.fun_scored_today(player) > 7 and card in player.played_today:
            player.played_today.remove(card)
            # A copier that became Nos Shirt must revert to its printed card
            # before it can be played again from hand.
            card.is_tomorrow = False
            card.markers.clear()
            player.hand.append(card)


NOS_SHIRT = CardDefinition(
    slug="nos-shirt",
    title="Nos Shirt",
    tags=frozenset({"Item"}),
    cost=1,
    base_fun=0,
    behavior=NosShirtBehavior(),
)


class LastYearsShirtBehavior(CardBehavior):
    """Draw one card from the Trunk, then place one hand card on top of the Trunk."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        drawn_card = game.draw_from_trunk(player_index, 1)[0]
        game.give_card(player_index, drawn_card)
        if not player.hand:
            return
        hand = tuple(player.hand)
        choice = game.ais[player_index].choose_card_to_discard(
            game, player_index, hand
        )
        if choice < 0 or choice >= len(hand):
            raise ValueError(f"AI returned invalid hand index: {choice}")
        target = player.hand.pop(choice)
        target.is_tomorrow = False
        target.markers.clear()
        game.stats.seen_suitcase_ids.add(target.instance_id)
        game.return_cards_to_trunk_top(player_index, (target,))


LAST_YEARS_SHIRT = CardDefinition(
    slug="last-years-shirt",
    title="Last Year's Shirt",
    tags=frozenset({"Item"}),
    cost=0,
    base_fun=0,
    behavior=LastYearsShirtBehavior(),
)
