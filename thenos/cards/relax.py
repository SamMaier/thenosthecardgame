"""Relax card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.fun_effects import FunForTagAfterBehavior

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
        return len(game.cards_played_before(player, card)) < 3

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


class SunriseBehavior(CardBehavior):
    """Score four Fun; this must be the first card played today."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not game.cards_played_before(player, card)


class FishingMorningBehavior(CardBehavior):
    """Score five Fun; this must be the first card played today."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not game.cards_played_before(player, card)


class FishingEveningBehavior(CardBehavior):
    """Score five Fun; this must be the fourth or later card played today."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return len(game.cards_played_before(player, card)) >= 3


class TanningBehavior(FunForTagAfterBehavior):
    """Score five Fun, then penalize later Outdoors cards today."""

    def __init__(self) -> None:
        super().__init__("Outdoors", -1)

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not any(
            "Outdoors" in played_card.tags
            for played_card in game.cards_played_before(player, card)
        )


class SleepInBehavior(CardBehavior):
    """Gain Energy as the first play, then prevent other Energy gains today."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not game.cards_played_before(player, card)

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 2, card)

    def allows_energy_gain(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
        source: CardInstance | None,
    ) -> bool:
        return source is card


class FloatingBehavior(CardBehavior):
    """Gain Energy now, then reduce the next day's starting Energy."""

    has_tomorrow_action = True

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 3, card)

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy -= 1


class ColouringBehavior(CardBehavior):
    """Draw one card from the Trunk into the player's hand."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        drawn_card = game.draw_from_trunk(player_index, 1)[0]
        game.give_card(player_index, drawn_card)


class PaintBehavior(CardBehavior):
    """Score four Fun and skip the player's next playing-phase turn."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.skip_next_turn(player)


class PaintRocksBehavior(CardBehavior):
    """Score an additional two Fun after two earlier Outdoors cards today."""

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
        if card_position is None:
            return card.effective_base_fun

        previous_outdoors = sum(
            "Outdoors" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        return card.effective_base_fun + (2 if previous_outdoors >= 2 else 0)


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

SUNRISE = CardDefinition(
    slug="sunrise",
    title="Sunrise",
    tags=frozenset({"Relax"}),
    cost=2,
    base_fun=4,
    behavior=SunriseBehavior(),
)

FISHING_MORNING = CardDefinition(
    slug="fishing-morning",
    title="Fishing Morning",
    tags=frozenset({"Relax", "Outdoors"}),
    cost=3,
    base_fun=5,
    behavior=FishingMorningBehavior(),
)

FISHING_EVENING = CardDefinition(
    slug="fishing-evening",
    title="Fishing Evening",
    tags=frozenset({"Relax", "Outdoors"}),
    cost=3,
    base_fun=5,
    behavior=FishingEveningBehavior(),
)

TANNING = CardDefinition(
    slug="tanning",
    title="Tanning",
    tags=frozenset({"Relax", "Outdoors"}),
    cost=3,
    base_fun=5,
    behavior=TanningBehavior(),
)

SLEEP_IN = CardDefinition(
    slug="sleep-in",
    title="Sleep In",
    tags=frozenset({"Relax", "Indoors"}),
    cost=0,
    behavior=SleepInBehavior(),
)

FLOATING = CardDefinition(
    slug="floating",
    title="Floating",
    tags=frozenset({"Relax", "Outdoors"}),
    cost=1,
    behavior=FloatingBehavior(),
)

COLOURING = CardDefinition(
    slug="colouring",
    title="Colouring",
    tags=frozenset({"Relax"}),
    cost=3,
    base_fun=2,
    behavior=ColouringBehavior(),
)

PAINT = CardDefinition(
    slug="paint",
    title="Paint",
    tags=frozenset({"Relax"}),
    cost=3,
    base_fun=4,
    behavior=PaintBehavior(),
)

PAINT_ROCKS = CardDefinition(
    slug="paint-rocks",
    title="Paint Rocks",
    tags=frozenset({"Relax", "Outdoors"}),
    cost=2,
    base_fun=2,
    behavior=PaintRocksBehavior(),
)
