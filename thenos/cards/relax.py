"""Relax card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class EarlyBedtimeBehavior(CardBehavior):
    """Start the next day with three additional Energy; play before the fourth card."""

    has_tomorrow_action = True

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return len(player.played_today) < 3

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 3, card)


class CheesyPhoneGameBehavior(CardBehavior):
    """Remember whether enough opponents played Relax cards before this."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        opponents = [opponent for opponent in game.players if opponent is not player]
        relax_opponents = sum(
            any("Relax" in played_card.tags for played_card in opponent.played_today)
            for opponent in opponents
        )
        if relax_opponents * 2 >= len(opponents):
            card.markers["energy_cube"] = True
            card.markers["_cheesy_phone_game_energy_cube"] = True

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + (
            3 if card.markers.get("_cheesy_phone_game_energy_cube") else 0
        )


class FancyCraftBehavior(CardBehavior):
    """Perform Unpack for +1 Fun, with an optional second Unpack."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        game.unpack(player_index, fun_delta=1)
        if game.choose_optional_action(player_index, "unpack"):
            game.unpack(player_index, fun_delta=1)


class ClassicBookBehavior(CardBehavior):
    """Reduce this card's cost for each earlier Relax card played today."""

    def modify_own_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
        current_cost: int,
    ) -> int:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        previous_cards = (
            player.played_today
            if card_position is None
            else player.played_today[:card_position]
        )

        relax_cards = sum(
            "Relax" in played_card.tags for played_card in previous_cards
        )
        return current_cost - relax_cards


EARLY_BEDTIME = CardDefinition(
    slug="early-bedtime",
    title="Early Bedtime",
    tags=frozenset({"Relax"}),
    cost=1,
    behavior=EarlyBedtimeBehavior(),
)

CHEESY_PHONE_GAME = CardDefinition(
    slug="cheesy-phone-game",
    title="Cheesy Phone Game",
    tags=frozenset({"Relax"}),
    cost=1,
    base_fun=1,
    behavior=CheesyPhoneGameBehavior(),
)

FANCY_CRAFT = CardDefinition(
    slug="fancy-craft",
    title="Fancy Craft",
    tags=frozenset({"Relax"}),
    cost=2,
    behavior=FancyCraftBehavior(),
)

CLASSIC_BOOK = CardDefinition(
    slug="classic-book",
    title="Classic Book",
    tags=frozenset({"Relax"}),
    cost=2,
    base_fun=2,
    behavior=ClassicBookBehavior(),
)
