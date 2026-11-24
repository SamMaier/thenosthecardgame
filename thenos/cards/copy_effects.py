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
    """Return this card to its owner's hand when more than 7 Fun was scored today."""

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
    base_fun=1,
    behavior=NosShirtBehavior(),
)


class LastYearsShirtBehavior(CardBehavior):
    """Pick any number of Suitcase cards and replenish taken slots from hand."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        if not player.hand or not game.suitcase:
            return
        player_index = game.players.index(player)
        hand = tuple(player.hand)
        raw_choices = tuple(
            game.ais[player_index].choose_cards_to_discard(
                game, player_index, hand
            )
        )
        if len(set(raw_choices)) != len(raw_choices):
            raise ValueError("Cannot select the same hand card more than once")
        if any(index < 0 or index >= len(hand) for index in raw_choices):
            raise ValueError("Invalid hand index in selection")
        if len(raw_choices) > len(game.suitcase):
            raw_choices = tuple(
                sorted(
                    raw_choices,
                    key=lambda index: (
                        hand[index].effective_base_fun,
                        -hand[index].effective_cost,
                        index,
                    ),
                )[: len(game.suitcase)]
            )
        choices = raw_choices
        if not choices:
            return
        original_suitcase = list(game.suitcase)
        game.stats.suitcase_offers.update(c.title for c in original_suitcase)
        game.stats.seen_suitcase_ids.update(
            c.instance_id for c in original_suitcase
        )
        replacements = [hand[index] for index in choices]
        for replacement in replacements:
            player.hand.remove(replacement)
            replacement.is_tomorrow = False
            replacement.markers.clear()
            game.stats.seen_suitcase_ids.add(replacement.instance_id)
        taken_slot_to_replacement: dict[int, CardInstance] = {}
        try:
            for replacement in replacements:
                chosen_index = game.ais[player_index].choose_suitcase_card(
                    game, player_index, tuple(game.suitcase)
                )
                if chosen_index < 0 or chosen_index >= len(game.suitcase):
                    raise ValueError(
                        f"AI returned invalid Suitcase index: {chosen_index}"
                    )
                picked = game.suitcase.pop(chosen_index)
                picked.markers.clear()
                slot_index = next(
                    index
                    for index, candidate in enumerate(original_suitcase)
                    if candidate is picked
                )
                taken_slot_to_replacement[slot_index] = replacement
                game.give_card(player_index, picked)
                game.record_card_pick_or_draw(player_index, picked)
                player.picked_cards[picked.title] += 1
                game.stats.suitcase_picks[picked.title] += 1
        finally:
            game.suitcase = [
                taken_slot_to_replacement.get(index, original_card)
                for index, original_card in enumerate(original_suitcase)
            ]


LAST_YEARS_SHIRT = CardDefinition(
    slug="last-years-shirt",
    title="Last Year's Shirt",
    tags=frozenset({"Item"}),
    cost=0,
    base_fun=0,
    behavior=LastYearsShirtBehavior(),
)
