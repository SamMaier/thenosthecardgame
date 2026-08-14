"""Food cards with immediate effects based on cards played today."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.pure_energy import GainOneEnergyBehavior

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class DoritosBehavior(CardBehavior):
    """Gain Energy after playing at least two earlier Food cards today."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return

        food_cards_played = sum(
            "Food" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        if food_cards_played >= 2:
            game.gain_energy(player, 2, card)


DORITOS = CardDefinition(
    slug="doritos",
    title="Doritos",
    tags=frozenset({"Food"}),
    cost=0,
    behavior=DoritosBehavior(),
)


class WeirdChipFlavorBehavior(CardBehavior):
    """Gain two Energy immediately when this card is played."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 2, card)


WEIRD_CHIP_FLAVOR = CardDefinition(
    slug="weird-chip-flavor",
    title="Weird Chip Flavor",
    tags=frozenset({"Food"}),
    cost=0,
    base_fun=-2,
    behavior=WeirdChipFlavorBehavior(),
)


class CharcuterieBehavior(CardBehavior):
    """Gain one Energy immediately when this card is played."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 1, card)


CHARCUTERIE = CardDefinition(
    slug="charcuterie",
    title="Charcuterie",
    tags=frozenset({"Food"}),
    cost=2,
    base_fun=2,
    behavior=CharcuterieBehavior(),
)


class SteakBehavior(CardBehavior):
    """Gain Energy when an earlier card today had a high written cost."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return

        has_high_cost_card = any(
            played_card.effective_cost >= 5
            for played_card in player.played_today[:card_position]
        )
        if has_high_cost_card:
            game.gain_energy(player, 3, card)


STEAK = CardDefinition(
    slug="steak",
    title="Steak",
    tags=frozenset({"Food"}),
    cost=2,
    base_fun=1,
    behavior=SteakBehavior(),
)


class IceCreamSandwichBehavior(CardBehavior):
    """Gain Energy after playing an earlier Food card today."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return

        has_previous_food = any(
            "Food" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        if has_previous_food:
            game.gain_energy(player, 2, card)


ICE_CREAM_SANDWICH = CardDefinition(
    slug="ice-cream-sandwich",
    title="Ice Cream Sandwich",
    tags=frozenset({"Food"}),
    cost=2,
    base_fun=1,
    behavior=IceCreamSandwichBehavior(),
)


class LeftoversBehavior(CardBehavior):
    """Copy an earlier Food card's effects without copying its cost or tags."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        # A Leftovers effect copied from an otherwise unfulfilled Leftovers
        # must terminate rather than recursively selecting itself.
        if card.markers.get("_copying_effect"):
            return

        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return

        player_index = game.players.index(player)
        eligible_cards = [
            candidate
            for candidate in player.played_today[:card_position]
            if "Food" in candidate.tags
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


LEFTOVERS = CardDefinition(
    slug="leftovers",
    title="Leftovers",
    tags=frozenset({"Food"}),
    cost=2,
    behavior=LeftoversBehavior(),
)


class BublyBehavior(CardBehavior):
    """Reward this card when an earlier card gave its player Energy today."""

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

        bonus = any(
            played_card.markers.get("_gave_energy")
            for played_card in player.played_today[:card_position]
        )
        return card.effective_base_fun + (2 if bonus else 0)


BUBLY = CardDefinition(
    slug="bubly",
    title="Bubly",
    tags=frozenset({"Food"}),
    cost=0,
    behavior=BublyBehavior(),
)


class CampCrossroadsDessertBehavior(CardBehavior):
    """Gain one Energy for each earlier Exercise card played today."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return

        exercise_cards_played = sum(
            "Exercise" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        game.gain_energy(player, exercise_cards_played, card)


CAMP_CROSSROADS_DESSERT = CardDefinition(
    slug="camp-crossroads-dessert",
    title="Camp Crossroads Dessert",
    tags=frozenset({"Food"}),
    cost=1,
    behavior=CampCrossroadsDessertBehavior(),
)


class PuddingChomeurBehavior(CardBehavior):
    """Gain Energy, then discard one remaining card from hand if possible."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 3, card)

        if not player.hand:
            return

        player_index = game.players.index(player)
        hand = tuple(player.hand)
        choice = game.ais[player_index].choose_card_to_discard(
            game, player_index, hand
        )
        if choice < 0 or choice >= len(hand):
            raise ValueError(f"AI returned invalid hand discard index: {choice}")

        game.discard_from_hand(player_index, choice)


PUDDING_CHOMEUR = CardDefinition(
    slug="pudding-chomeur",
    title="Pudding Chômeur",
    tags=frozenset({"Food"}),
    cost=1,
    behavior=PuddingChomeurBehavior(),
)


class MorningCoffeeBehavior(CardBehavior):
    """Gain three Energy; this must be the first card played today."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return not player.played_today

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 3, card)


MORNING_COFFEE = CardDefinition(
    slug="morning-coffee",
    title="Morning Coffee",
    tags=frozenset({"Food"}),
    cost=1,
    behavior=MorningCoffeeBehavior(),
)


class AfternoonCoffeeBehavior(CardBehavior):
    """Gain three Energy; this must not be one of the first two cards today."""

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return len(player.played_today) >= 2

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.gain_energy(player, 3, card)


AFTERNOON_COFFEE = CardDefinition(
    slug="afternoon-coffee",
    title="Afternoon Coffee",
    tags=frozenset({"Food"}),
    cost=1,
    behavior=AfternoonCoffeeBehavior(),
)


class DecafBehavior(CardBehavior):
    """Score two Fun for each Energy remaining at the end of the day."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + 2 * player.energy


DECAF = CardDefinition(
    slug="decaf",
    title="Decaf",
    tags=frozenset({"Food"}),
    cost=3,
    behavior=DecafBehavior(),
)


KEEPER = CardDefinition(
    slug="keeper",
    title="Keeper",
    tags=frozenset({"Food"}),
    cost=2,
    base_fun=2,
    behavior=GainOneEnergyBehavior(),
)


FOOD_CARDS = (
    DORITOS,
    WEIRD_CHIP_FLAVOR,
    CHARCUTERIE,
    STEAK,
    BUBLY,
    CAMP_CROSSROADS_DESSERT,
    PUDDING_CHOMEUR,
    MORNING_COFFEE,
    AFTERNOON_COFFEE,
    ICE_CREAM_SANDWICH,
    LEFTOVERS,
    DECAF,
    KEEPER,
)
