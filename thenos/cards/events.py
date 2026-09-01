"""Event card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class SingSongBehavior(CardBehavior):
    """Cost one less for each opponent who played an Event today."""

    def modify_own_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
        current_cost: int,
    ) -> int:
        event_opponents = sum(
            any("Event" in played_card.tags for played_card in opponent.played_today)
            for opponent in game.players
            if opponent is not player
        )
        return current_cost - event_opponents


class ChristmasNameDrawBehavior(CardBehavior):
    """Match this player's visible cards to a selected opponent card's tags."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        eligible_cards = tuple(
            played_card
            for opponent_index, opponent in enumerate(game.players)
            if opponent_index != player_index
            for played_card in opponent.played_today
        )
        if not eligible_cards:
            return

        target = game.choose_card_target(player_index, eligible_cards)
        target.markers["energy_cube"] = True
        card.markers["target_card"] = target

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        target = card.markers.get("target_card")
        if not isinstance(target, CardInstance):
            return card.effective_base_fun

        return card.effective_base_fun + sum(
            bool(target.tags.intersection(visible_card.tags))
            for visible_card in player.visible_cards
        )


class StayUpLateBehavior(CardBehavior):
    """Gain Energy now, then reduce the next day's starting Energy."""

    has_tomorrow_action = True

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 2, card)

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy -= 2


class PhotoShootBehavior(CardBehavior):
    """While active Tomorrow, pick a Suitcase card after each play."""

    has_tomorrow_action = True

    def on_tomorrow_card_play(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        played_card: CardInstance,
    ) -> None:
        game.pick_from_suitcase(game.players.index(player))


SING_SONG = CardDefinition(
    slug="sing-song",
    title="Sing Song",
    tags=frozenset({"Event"}),
    cost=4,
    base_fun=4,
    behavior=SingSongBehavior(),
)

CHRISTMAS_NAME_DRAW = CardDefinition(
    slug="christmas-name-draw",
    title="Christmas Name Draw",
    tags=frozenset({"Event"}),
    cost=2,
    behavior=ChristmasNameDrawBehavior(),
)

STAY_UP_LATE = CardDefinition(
    slug="stay-up-late",
    title="Stay Up Late",
    tags=frozenset({"Event", "Indoors"}),
    cost=0,
    behavior=StayUpLateBehavior(),
)

PHOTO_SHOOT = CardDefinition(
    slug="photo-shoot",
    title="Photo Shoot",
    tags=frozenset({"Event", "Outdoors"}),
    cost=3,
    base_fun=-1,
    behavior=PhotoShootBehavior(),
)
