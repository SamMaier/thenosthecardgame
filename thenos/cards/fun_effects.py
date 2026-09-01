"""Cards whose effects modify the Fun scored by other cards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


TagPredicate = Callable[[CardInstance], bool]


def _today_position(player: PlayerState, card: CardInstance) -> int | None:
    for position, played_card in enumerate(player.played_today):
        if played_card is card:
            return position
    return None


def _is_after(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
) -> bool:
    source_position = _today_position(player, source)
    target_position = _today_position(player, target)
    return (
        source_position is not None
        and target_position is not None
        and target_position > source_position
    )


def _is_after_for_cost(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
) -> bool:
    """A hand card is a future card while its cost is being evaluated."""
    source_position = _today_position(player, source)
    target_position = _today_position(player, target)
    return (
        source_position is not None
        and (target_position is None or target_position > source_position)
    )


def _is_today_and_matches(
    player: PlayerState,
    target: CardInstance,
    predicate: TagPredicate,
) -> bool:
    return _today_position(player, target) is not None and predicate(target)


def _is_next_matching(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
    predicate: TagPredicate,
) -> bool:
    source_position = _today_position(player, source)
    target_position = _today_position(player, target)
    if source_position is None or target_position is None:
        return False
    matching_after = [
        card
        for card in player.played_today[source_position + 1 :]
        if predicate(card)
    ]
    return bool(matching_after) and matching_after[0] is target


def _is_next_card(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
) -> bool:
    source_position = _today_position(player, source)
    target_position = _today_position(player, target)
    return (
        source_position is not None
        and target_position == source_position + 1
    )


def _is_nth_matching(
    player: PlayerState,
    source: CardInstance,
    target: CardInstance,
    predicate: TagPredicate,
    ordinal: int,
) -> bool:
    source_position = _today_position(player, source)
    target_position = _today_position(player, target)
    if source_position is None or target_position is None:
        return False
    matching_after = [
        card
        for card in player.played_today[source_position + 1 :]
        if predicate(card)
    ]
    return (
        len(matching_after) >= ordinal
        and matching_after[ordinal - 1] is target
    )


class FunForTagTodayBehavior(CardBehavior):
    """Add Fun to every matching card played today, including this card."""

    def __init__(self, tag: str, bonus: int) -> None:
        self.tag = tag
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_today_and_matches(player, target, lambda card: self.tag in card.tags):
            return current_fun + self.bonus
        return current_fun


class ZeroFunForTagTodayBehavior(CardBehavior):
    """Set every matching card played today to zero Fun."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_today_and_matches(player, target, lambda card: self.tag in card.tags):
            return 0
        return current_fun


class TomorrowFunForTagBehavior(CardBehavior):
    """Add Fun to matching cards while this card is active Tomorrow."""

    has_tomorrow_action = True

    def __init__(self, tag: str, bonus: int) -> None:
        self.tag = tag
        self.bonus = bonus

    def modify_tomorrow_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if self.tag in target.tags:
            return current_fun + self.bonus
        return current_fun


class FunForTagAfterBehavior(CardBehavior):
    """Add Fun to each matching card played after this card today."""

    def __init__(self, tag: str, bonus: int) -> None:
        self.tag = tag
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_after(player, source, target) and self.tag in target.tags:
            return current_fun + self.bonus
        return current_fun


class FunForTagBeforeBehavior(CardBehavior):
    """Add Fun to each matching card played before this card today."""

    def __init__(self, tag: str, bonus: int) -> None:
        self.tag = tag
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        source_position = _today_position(player, source)
        target_position = _today_position(player, target)
        if (
            source_position is not None
            and target_position is not None
            and target_position < source_position
            and self.tag in target.tags
        ):
            return current_fun + self.bonus
        return current_fun


class FunForAllCardsBeforeBehavior(CardBehavior):
    """Add Fun to every card played before this card today."""

    def __init__(self, bonus: int) -> None:
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        source_position = _today_position(player, source)
        target_position = _today_position(player, target)
        if (
            source_position is not None
            and target_position is not None
            and target_position < source_position
        ):
            return current_fun + self.bonus
        return current_fun


class FunForTagsBeforeAndAfterBehavior(CardBehavior):
    """Add Fun to matching cards played before and after this card today."""

    def __init__(self, tags: frozenset[str], bonus: int) -> None:
        self.tags = tags
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        source_position = _today_position(player, source)
        target_position = _today_position(player, target)
        if (
            source_position is not None
            and target_position is not None
            and target is not source
            and self.tags.intersection(target.tags)
        ):
            return current_fun + self.bonus
        return current_fun


class FunFromOtherCardsWrittenCostBehavior(CardBehavior):
    """Adjust this card's Fun by the written costs of other cards today."""

    def __init__(self, multiplier: int = 1) -> None:
        self.multiplier = multiplier

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        written_cost_total = sum(
            other_card.effective_cost
            for other_card in player.played_today
            if other_card is not card
        )
        return card.effective_base_fun + self.multiplier * written_cost_total


class WakeboardBehavior(FunFromOtherCardsWrittenCostBehavior):
    """Reduce this card's Fun by other cards' written Energy costs."""

    def __init__(self) -> None:
        super().__init__(multiplier=-1)


class FunForNextTagBehavior(CardBehavior):
    """Add Fun to the next matching card played after this card today."""

    def __init__(self, tag: str, bonus: int) -> None:
        self.tag = tag
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_next_matching(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
        ):
            return current_fun + self.bonus
        return current_fun


class FunForNextCardBehavior(CardBehavior):
    """Apply a Fun multiplier to the immediately following card today."""

    def __init__(self, multiplier: int) -> None:
        self.multiplier = multiplier

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_next_card(player, source, target):
            return current_fun * self.multiplier
        return current_fun


class FunForNextTagWrittenCostBehavior(CardBehavior):
    """Add a matching card's printed Energy cost to its Fun."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_next_matching(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
        ):
            return current_fun + target.effective_cost
        return current_fun


class FunEqualToNextTagWrittenCostBehavior(CardBehavior):
    """Set the next matching card's Fun to twice its printed Energy cost."""

    def __init__(self, tag: str, multiplier: int = 2) -> None:
        self.tag = tag
        self.multiplier = multiplier

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_next_matching(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
        ):
            return self.multiplier * target.definition.cost
        return current_fun


class FunForNthTagBehavior(CardBehavior):
    """Add Fun to a particular matching card after this card today."""

    def __init__(self, tag: str, ordinal: int, bonus: int) -> None:
        self.tag = tag
        self.ordinal = ordinal
        self.bonus = bonus

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_nth_matching(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
            self.ordinal,
        ):
            return current_fun + self.bonus
        return current_fun


class FunForNthTagDoubleBehavior(CardBehavior):
    """Double Fun on a particular matching card after this card today."""

    def __init__(self, tag: str, ordinal: int) -> None:
        self.tag = tag
        self.ordinal = ordinal

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_nth_matching(
            player,
            source,
            target,
            lambda card: self.tag in card.tags,
            self.ordinal,
        ):
            return current_fun * 2
        return current_fun


class FunAndEnergyForTagAfterBehavior(CardBehavior):
    """Modify cost and Fun for matching cards played after this card today."""

    def __init__(self, tag: str, energy_delta: int, fun_bonus: int) -> None:
        self.tag = tag
        self.energy_delta = energy_delta
        self.fun_bonus = fun_bonus

    def _matches(self, player: PlayerState, source: CardInstance, target: CardInstance) -> bool:
        return _is_after(player, source, target) and self.tag in target.tags

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

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if self._matches(player, source, target):
            return current_fun + self.fun_bonus
        return current_fun


class FunAndEnergyForAllCardsAfterBehavior(CardBehavior):
    """Modify cost and Fun for every card played after this card today."""

    def __init__(self, energy_delta: int, fun_bonus: int) -> None:
        self.energy_delta = energy_delta
        self.fun_bonus = fun_bonus

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

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_after(player, source, target):
            return current_fun + self.fun_bonus
        return current_fun


class FunForAllCardsAfterDoubleBehavior(CardBehavior):
    """Double Fun on every card played after this card today."""

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_after(player, source, target):
            return current_fun * 2
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
            return current_cost * 2
        return current_cost


class PickThreeAndFunForSocialAfterBehavior(FunForTagAfterBehavior):
    """Pick three Suitcase cards, then reward Social cards played after this."""

    def __init__(self) -> None:
        super().__init__("Social", 1)

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player_index = game.players.index(player)
        for _ in range(3):
            game.pick_from_suitcase(player_index)


TREKKING_THROUGH_HISTORY = CardDefinition(
    slug="trekking-through-history",
    title="Trekking Through History",
    tags=frozenset({"Board Game"}),
    cost=2,
    behavior=FunForTagsBeforeAndAfterBehavior(frozenset({"Board Game"}), 1),
)

FAMILY_BASEBALL_GAME = CardDefinition(
    slug="family-baseball-game",
    title="Family Baseball Game",
    tags=frozenset({"Exercise", "Event", "Outdoors"}),
    cost=3,
    base_fun=5,
    behavior=ZeroFunForTagTodayBehavior("Board Game"),
)

EUCHRE = CardDefinition(
    slug="euchre",
    title="Euchre",
    tags=frozenset({"Board Game", "Indoors"}),
    cost=2,
    base_fun=2,
    behavior=FunForTagAfterBehavior("Social", 1),
)

WORK_CALL = CardDefinition(
    slug="work-call",
    title="Work Call",
    tags=frozenset({"Event", "Indoors"}),
    cost=1,
    base_fun=-4,
    behavior=FunForNextCardBehavior(2),
)

STRETCH = CardDefinition(
    slug="stretch",
    title="Stretch",
    tags=frozenset({"Exercise"}),
    cost=1,
    behavior=FunForTagAfterBehavior("Exercise", 2),
)

CANOE = CardDefinition(
    slug="canoe",
    title="Canoe",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=3,
    base_fun=2,
    behavior=FunForNextTagWrittenCostBehavior("Item"),
)

class WaterTrampolineBehavior(FunForNextCardBehavior):
    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if _is_next_card(player, source, target) and "Relax" in target.tags:
            return current_fun * 2
        return current_fun


WATER_TRAMPOLINE = CardDefinition(
    slug="water-trampoline",
    title="Water Trampoline",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    base_fun=1,
    behavior=WaterTrampolineBehavior(2),
)

WATER_VOLLEYBALL = CardDefinition(
    slug="water-volleyball",
    title="Water Volleyball",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    base_fun=2,
    behavior=FunForNextTagBehavior("Exercise", 4),
)

CHEAP_RED = CardDefinition(
    slug="cheap-red",
    title="Cheap Red",
    tags=frozenset({"Food"}),
    cost=2,
    base_fun=2,
    behavior=FunForNextTagBehavior("Food", 2),
)

SCHWANK = CardDefinition(
    slug="schwank",
    title="Schwank",
    tags=frozenset({"Food"}),
    cost=2,
    behavior=FunForNthTagDoubleBehavior("Exercise", 3),
)

HIGH_END_RED = CardDefinition(
    slug="high-end-red",
    title="High-End Red",
    tags=frozenset({"Food"}),
    cost=3,
    behavior=FunForTagAfterBehavior("Food", 2),
)

HIGH_END_WHITE = CardDefinition(
    slug="high-end-white",
    title="High-End White",
    tags=frozenset({"Food"}),
    cost=3,
    base_fun=2,
    behavior=FunForTagAfterBehavior("Food", 1),
)

NOS_SHIRT = CardDefinition(
    slug="nos-shirt",
    title="Nos Shirt",
    tags=frozenset({"Item"}),
    cost=1,
    behavior=FunForTagAfterBehavior("Event", 2),
)

EPIC_PLAYLIST = CardDefinition(
    slug="epic-playlist",
    title="Epic Playlist",
    tags=frozenset({"Item"}),
    cost=1,
    behavior=FunForTagAfterBehavior("Social", 2),
)

BUG_SPRAY = CardDefinition(
    slug="bug-spray",
    title="Bug Spray",
    tags=frozenset({"Item"}),
    cost=3,
    behavior=FunForTagAfterBehavior("Outdoors", 1),
)

PRIME_PICNIC_TABLE = CardDefinition(
    slug="prime-picnic-table",
    title="Prime Picnic Table",
    tags=frozenset({"Item"}),
    cost=3,
    behavior=FunAndEnergyForTagAfterBehavior("Event", -1, 1),
)

NOS_BOOK = CardDefinition(
    slug="nos-book",
    title="Nos Book",
    tags=frozenset({"Item"}),
    cost=2,
    base_fun=1,
    behavior=FunForTagsBeforeAndAfterBehavior(frozenset({"Social", "Event"}), 1),
)

SWEET_LAWN_CHAIR = CardDefinition(
    slug="sweet-lawn-chair",
    title="Sweet Lawn Chair",
    tags=frozenset({"Item"}),
    cost=3,
    behavior=FunForTagAfterBehavior("Relax", 1),
)

BRACELET_MAKING = CardDefinition(
    slug="bracelet-making",
    title="Bracelet Making",
    tags=frozenset({"Relax"}),
    cost=3,
    base_fun=1,
    behavior=FunForNextTagBehavior("Social", 4),
)

MOVIE = CardDefinition(
    slug="movie",
    title="Movie",
    tags=frozenset({"Relax", "Indoors"}),
    cost=1,
    base_fun=1,
    behavior=FunForTagAfterBehavior("Indoors", 2),
)

JOHNNY_APPLESEED = CardDefinition(
    slug="johnny-appleseed",
    title="Johnny Appleseed",
    tags=frozenset({"Social"}),
    cost=1,
    base_fun=1,
    behavior=FunForNextTagBehavior("Event", 2),
)

AUNTERVIEW = CardDefinition(
    slug="aunterview",
    title="Aunterview",
    tags=frozenset({"Social"}),
    cost=2,
    base_fun=1,
    behavior=TomorrowFunForTagBehavior("Social", 2),
)

BRING_A_FRIEND = CardDefinition(
    slug="bring-a-friend",
    title="Bring a Friend",
    tags=frozenset({"Social"}),
    cost=1,
    behavior=FunForAllCardsAfterDoubleBehavior(),
)

DOXOLOGY = CardDefinition(
    slug="doxology",
    title="Doxology",
    tags=frozenset({"Social"}),
    cost=1,
    behavior=FunForTagAfterBehavior("Social", 1),
)

LONG_DISTANCE_VISITORS = CardDefinition(
    slug="long-distance-visitors",
    title="Long Distance Visitors",
    tags=frozenset({"Social"}),
    cost=4,
    behavior=PickThreeAndFunForSocialAfterBehavior(),
)

HOLD_THE_BABY = CardDefinition(
    slug="hold-the-baby",
    title="Hold the Baby",
    tags=frozenset({"Social"}),
    cost=2,
    behavior=FunAndEnergyForAllCardsAfterBehavior(1, 2),
)

OUTDOOR_MOVIE = CardDefinition(
    slug="outdoor-movie",
    title="Outdoor Movie",
    tags=frozenset({"Event", "Outdoors"}),
    cost=2,
    base_fun=2,
    behavior=FunForTagBeforeBehavior("Relax", 1),
)

CLIFF_CLIMBING = CardDefinition(
    slug="cliff-climbing",
    title="Cliff Climbing",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=5,
    base_fun=3,
    behavior=FunForTagBeforeBehavior("Outdoors", 2),
)

ICE_WINE = CardDefinition(
    slug="ice-wine",
    title="Ice Wine",
    tags=frozenset({"Food"}),
    cost=5,
    behavior=FunForTagBeforeBehavior("Food", 3),
)

EVENING_ON_THE_DOCK = CardDefinition(
    slug="evening-on-the-dock",
    title="Evening on the Dock",
    tags=frozenset({"Relax", "Social"}),
    cost=4,
    behavior=FunForAllCardsBeforeBehavior(1),
)

COUCH_TUBE = CardDefinition(
    slug="couch-tube",
    title="Couch Tube",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    behavior=FunFromOtherCardsWrittenCostBehavior(),
)

WAKEBOARD = CardDefinition(
    slug="wakeboard",
    title="Wakeboard",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=8,
    base_fun=10,
)

TEACH_KID_TO_SKI = CardDefinition(
    slug="teach-kid-to-ski",
    title="Teach Kid to Ski",
    tags=frozenset({"Event", "Outdoors"}),
    cost=2,
    behavior=TomorrowFunForTagBehavior("Exercise", 1),
)


FUN_EFFECT_CARDS = (
    TREKKING_THROUGH_HISTORY,
    FAMILY_BASEBALL_GAME,
    EUCHRE,
    WORK_CALL,
    STRETCH,
    CANOE,
    WATER_TRAMPOLINE,
    WATER_VOLLEYBALL,
    CHEAP_RED,
    SCHWANK,
    HIGH_END_RED,
    HIGH_END_WHITE,
    NOS_SHIRT,
    EPIC_PLAYLIST,
    BUG_SPRAY,
    PRIME_PICNIC_TABLE,
    NOS_BOOK,
    SWEET_LAWN_CHAIR,
    BRACELET_MAKING,
    MOVIE,
    JOHNNY_APPLESEED,
    AUNTERVIEW,
    BRING_A_FRIEND,
    DOXOLOGY,
    LONG_DISTANCE_VISITORS,
    HOLD_THE_BABY,
    OUTDOOR_MOVIE,
    CLIFF_CLIMBING,
    ICE_WINE,
    EVENING_ON_THE_DOCK,
    COUCH_TUBE,
    WAKEBOARD,
    TEACH_KID_TO_SKI,
)
