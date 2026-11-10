"""Definitions for several Social cards with bespoke behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class SingSongBehavior(CardBehavior):
    """Cost one less for each opponent who played a Social card today."""

    def modify_own_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
        current_cost: int,
    ) -> int:
        social_opponents = sum(
            any("Social" in played_card.tags for played_card in opponent.played_today)
            for opponent in game.players
            if opponent is not player
        )
        return current_cost - social_opponents


class ChristmasNameDrawBehavior(CardBehavior):
    """Reward cards played after this that share a tag with a marked opponent card."""

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

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if (
            target is not source
            and source in player.played_today
            and target in player.played_today
            and player.played_today.index(target) > player.played_today.index(source)
        ):
            marked = source.markers.get("target_card")
            if (
                isinstance(marked, CardInstance)
                and marked.markers.get("energy_cube")
                and any(
                    marked in opponent.played_today
                    for opponent in game.players
                    if opponent is not player
                )
                and bool(marked.tags.intersection(target.tags))
            ):
                return current_fun + 2
        return current_fun


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
    """While active Tomorrow, pick a Suitcase card after each Social or Relax play."""

    has_tomorrow_action = True
    tags = frozenset({"Social", "Relax"})

    def on_tomorrow_card_play(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        played_card: CardInstance,
    ) -> None:
        if bool(self.tags.intersection(played_card.tags)):
            game.pick_from_suitcase(game.players.index(player))


SING_SONG = CardDefinition(
    slug="sing-song",
    title="Sing Song",
    tags=frozenset({"Social"}),
    cost=4,
    base_fun=4,
    behavior=SingSongBehavior(),
)

CHRISTMAS_NAME_DRAW = CardDefinition(
    slug="christmas-name-draw",
    title="Christmas Name Draw",
    tags=frozenset({"Social"}),
    cost=3,
    behavior=ChristmasNameDrawBehavior(),
)

STAY_UP_LATE = CardDefinition(
    slug="stay-up-late",
    title="Stay Up Late",
    tags=frozenset({"Social", "Indoors"}),
    cost=0,
    behavior=StayUpLateBehavior(),
)

PHOTO_SHOOT = CardDefinition(
    slug="photo-shoot",
    title="Photo Shoot",
    tags=frozenset({"Social", "Outdoors"}),
    cost=3,
    base_fun=-1,
    behavior=PhotoShootBehavior(),
)
