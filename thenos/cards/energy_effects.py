"""Cards whose rules text changes the Energy cost of later cards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.fun_effects import _is_after_for_cost, _today_position

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


TagPredicate = Callable[[CardInstance], bool]


def _is_before(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
) -> bool:
    source_position = _today_position(player, source)
    target_position = _today_position(player, target)
    return (
        source_position is not None
        and target_position is not None
        and target_position < source_position
    )


def _is_next_matching_for_cost(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
    predicate: TagPredicate,
) -> bool:
    """Whether ``target`` is the next matching card, including a hand card."""
    source_position = _today_position(player, source)
    if source_position is None:
        return False

    target_position = _today_position(player, target)
    matching_after = [
        card
        for card in player.played_today[source_position + 1 :]
        if predicate(card)
    ]
    if target_position is None:
        return not matching_after and predicate(target)
    return bool(matching_after) and matching_after[0] is target


class EnergyForNextTagBehavior(CardBehavior):
    """Change the cost of the next matching card played today."""

    def __init__(self, tag: str, energy_delta: int) -> None:
        self.tag = tag
        self.energy_delta = energy_delta

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if _is_next_matching_for_cost(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
        ):
            return current_cost + self.energy_delta
        return current_cost


class SetEnergyForNextTagBehavior(CardBehavior):
    """Set the cost of the next matching card played today."""

    def __init__(self, tag: str, energy_cost: int) -> None:
        self.tag = tag
        self.energy_cost = energy_cost

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if _is_next_matching_for_cost(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
        ):
            return self.energy_cost
        return current_cost


class HalfEnergyForNextTagBehavior(CardBehavior):
    """Halve the cost of the next matching card, rounding down."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if _is_next_matching_for_cost(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
        ):
            return current_cost // 2
        return current_cost


class EnergyForTagAfterBehavior(CardBehavior):
    """Change the cost of all matching cards played after this card today."""

    def __init__(self, tag: str, energy_delta: int) -> None:
        self.tag = tag
        self.energy_delta = energy_delta

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if _is_after_for_cost(player, source, target) and self.tag in target.tags:
            return current_cost + self.energy_delta
        return current_cost


class EnergyForAllCardsAfterBehavior(CardBehavior):
    """Change the cost of every card played after this card today."""

    def __init__(self, energy_delta: int) -> None:
        self.energy_delta = energy_delta

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if _is_after_for_cost(player, source, target):
            return current_cost + self.energy_delta
        return current_cost


class TomorrowEnergyForTagBehavior(CardBehavior):
    """Change the cost of matching cards while this card is active Tomorrow."""

    has_tomorrow_action = True

    def __init__(self, tag: str, energy_delta: int) -> None:
        self.tag = tag
        self.energy_delta = energy_delta

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if source.is_tomorrow and self.tag in target.tags:
            return current_cost + self.energy_delta
        return current_cost


class MedicalAdviceBehavior(CardBehavior):
    """Pick three cards, penalize previous cards, and tax later cards."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        for _ in range(3):
            game.pick_from_suitcase(player_index)

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_before(player, source, target):
            return current_fun - 1
        return current_fun

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        if _is_after_for_cost(player, source, target):
            return current_cost + 1
        return current_cost


class AfterDinnerEntertainmentBehavior(EnergyForNextTagBehavior):
    """Discount the next Social card and enable Tomorrow's extra-pick option."""

    has_tomorrow_action = True

    def __init__(self) -> None:
        super().__init__("Social", -1)

    def allows_extra_suitcase_pick(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return True


FORCED_FAMILY_FUN = CardDefinition(
    slug="forced-family-fun",
    title="Forced Family Fun",
    tags=frozenset({"Event", "Indoors"}),
    cost=2,
    behavior=HalfEnergyForNextTagBehavior("Board Game"),
)

BOAT_RIDE = CardDefinition(
    slug="boat-ride",
    title="Boat Ride",
    tags=frozenset({"Event", "Outdoors"}),
    cost=3,
    base_fun=1,
    behavior=SetEnergyForNextTagBehavior("Item", 0),
)

TREAT_CEREAL = CardDefinition(
    slug="treat-cereal",
    title="Treat Cereal",
    tags=frozenset({"Food"}),
    cost=2,
    behavior=EnergyForNextTagBehavior("Exercise", -4),
)

BEAVER_BURGER = CardDefinition(
    slug="beaver-burger",
    title="Beaver Burger",
    tags=frozenset({"Food"}),
    cost=3,
    base_fun=5,
    behavior=EnergyForAllCardsAfterBehavior(1),
)

ROULADEN = CardDefinition(
    slug="rouladen",
    title="Rouladen",
    tags=frozenset({"Food"}),
    cost=4,
    behavior=EnergyForAllCardsAfterBehavior(-1),
)

ZERO_GRAVITY_CHAIR = CardDefinition(
    slug="zero-gravity-chair",
    title="Zero Gravity Chair",
    tags=frozenset({"Item"}),
    cost=2,
    behavior=EnergyForTagAfterBehavior("Relax", -1),
)

SUNSCREEN = CardDefinition(
    slug="sunscreen",
    title="Sunscreen",
    tags=frozenset({"Item"}),
    cost=4,
    behavior=EnergyForTagAfterBehavior("Outdoors", -1),
)

SHADY_SPOT = CardDefinition(
    slug="shady-spot",
    title="Shady Spot",
    tags=frozenset({"Relax"}),
    cost=4,
    behavior=EnergyForTagAfterBehavior("Outdoors", -1),
)

BEND_THE_RULES = CardDefinition(
    slug="bend-the-rules",
    title="Bend the Rules",
    tags=frozenset({"Social"}),
    cost=2,
    behavior=EnergyForTagAfterBehavior("Board Game", -1),
)

MEDICAL_ADVICE = CardDefinition(
    slug="medical-advice",
    title="Medical Advice",
    tags=frozenset({"Social"}),
    cost=2,
    behavior=MedicalAdviceBehavior(),
)

AFTER_DINNER_ENTERTAINMENT = CardDefinition(
    slug="after-dinner-entertainment",
    title="After Dinner Entertainment",
    tags=frozenset({"Social"}),
    cost=2,
    behavior=AfterDinnerEntertainmentBehavior(),
)

DISHWASHING = CardDefinition(
    slug="dishwashing",
    title="Dishwashing",
    tags=frozenset({"Event"}),
    cost=2,
    behavior=TomorrowEnergyForTagBehavior("Food", -1),
)


ENERGY_EFFECT_CARDS = (
    FORCED_FAMILY_FUN,
    BOAT_RIDE,
    TREAT_CEREAL,
    BEAVER_BURGER,
    ROULADEN,
    ZERO_GRAVITY_CHAIR,
    SUNSCREEN,
    SHADY_SPOT,
    BEND_THE_RULES,
    MEDICAL_ADVICE,
    AFTER_DINNER_ENTERTAINMENT,
    DISHWASHING,
)
