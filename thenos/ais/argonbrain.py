"""Synergy-driven single-player lookahead policy (ArgonBrain)."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence, TYPE_CHECKING

from thenos.ais.greedy import GreedyAI
from thenos.ais.interface import PlayerAI
from thenos.ais.random_ai import RandomAI
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.models import PlayerState

if TYPE_CHECKING:
    from thenos.game import Game, GameStats

DAY_COUNT = 6
STARTING_ENERGY = 7
SUITCASE_SIZE = 4
_SHARED_SIM_STATS: Any = None


def _get_shared_sim_stats() -> Any:
    global _SHARED_SIM_STATS
    if _SHARED_SIM_STATS is None:
        from thenos.game import GameStats

        _SHARED_SIM_STATS = GameStats()
    return _SHARED_SIM_STATS


@dataclass(frozen=True, slots=True)
class CardSynergyProfile:
    """Cached behavioral profile of a CardDefinition."""

    must_play_first: bool
    must_play_early: bool
    must_play_late: bool
    ends_day_immediately: bool
    goes_to_tomorrow: bool
    immediate_energy_gain: int
    conditional_energy_gain: int
    immediate_fun_gain: int
    immediate_cards_gained: int
    conditional_fun_bonus: int
    tomorrow_start_energy: int
    tomorrow_start_fun: int
    modifies_other_fun: bool
    modifies_other_cost: bool
    modifies_own_cost: bool
    has_dynamic_fun: bool
    has_tomorrow_effect: bool
    has_end_day_hook: bool
    target_tags: frozenset[str]
    fun_bonus: int
    energy_delta: int
    multiplier: int
    signed_match_benefit: float
    play_order_bucket: int
    is_multi_scaler: bool
    is_before_scaler: bool
    is_tomorrow_scaler: bool
    is_single_buff: bool
    is_global_after_scaler: bool
    is_count_finisher: bool
    penalizes_before: bool
    penalizes_after: bool


_DEFN_ID_CACHE: dict[int, CardSynergyProfile] = {}
_PROFILE_CACHE: dict[
    tuple[str, type[CardBehavior], int, int, frozenset[str]], CardSynergyProfile
] = {}
_ACTIVE_PROBE_KEYS: set[
    tuple[str, type[CardBehavior], int, int, frozenset[str]]
] = set()
_FALLBACK_PROFILE = CardSynergyProfile(
    must_play_first=False,
    must_play_early=False,
    must_play_late=False,
    ends_day_immediately=False,
    goes_to_tomorrow=False,
    immediate_energy_gain=0,
    conditional_energy_gain=0,
    immediate_fun_gain=0,
    immediate_cards_gained=0,
    conditional_fun_bonus=0,
    tomorrow_start_energy=0,
    tomorrow_start_fun=0,
    modifies_other_fun=False,
    modifies_other_cost=False,
    modifies_own_cost=False,
    has_dynamic_fun=False,
    has_tomorrow_effect=False,
    has_end_day_hook=False,
    target_tags=frozenset(),
    fun_bonus=0,
    energy_delta=0,
    multiplier=1,
    signed_match_benefit=0.0,
    play_order_bucket=4,
    is_multi_scaler=False,
    is_before_scaler=False,
    is_tomorrow_scaler=False,
    is_single_buff=False,
    is_global_after_scaler=False,
    is_count_finisher=False,
    penalizes_before=False,
    penalizes_after=False,
)
_TAG_ABUNDANCE_CACHE: dict[str, float] = {}


def _tag_abundance(tags: frozenset[str]) -> float:
    """Return relative deck abundance of the target tag(s) (1.0 = ~20% deck share)."""
    if not tags:
        return 1.0
    if not _TAG_ABUNDANCE_CACHE:
        from thenos.cards.catalog import CARD_REGISTRY

        total = max(1, len(CARD_REGISTRY))
        counts: dict[str, int] = {}
        for defn in CARD_REGISTRY.values():
            for t in defn.tags:
                counts[t] = counts.get(t, 0) + 1
        for t, cnt in counts.items():
            share = cnt / total
            _TAG_ABUNDANCE_CACHE[t] = max(0.35, min(1.25, share / 0.20))
    return max(_TAG_ABUNDANCE_CACHE.get(t, 0.80) for t in tags)


def _clone_random(src: random.Random) -> random.Random:
    """Clone a Random instance without calling os.urandom."""
    r = random.Random.__new__(random.Random)
    r.setstate(src.getstate())
    return r


def _seeded_random(seed: int) -> random.Random:
    """Create a seeded Random instance without calling os.urandom first."""
    r = random.Random.__new__(random.Random)
    r.seed(seed)
    return r


def _clone_card(card: CardInstance) -> CardInstance:
    return CardInstance(
        card.instance_id,
        card.definition,
        card.is_tomorrow,
        dict(card.markers) if card.markers else {},
    )


def _probe_can_play_at(defn: CardDefinition, prior_count: int) -> bool:
    """Test whether defn.behavior.can_play allows playing after prior_count dummy cards."""
    b = defn.behavior
    if type(b).can_play is CardBehavior.can_play:
        return True
    try:
        from thenos.game import Game

        dummy_card = CardInstance(0, defn)
        dummy_prior_defn = CardDefinition(
            slug="_probe_prior",
            title="_ProbePrior",
            tags=frozenset(
                {"Relax", "Food", "Exercise", "Social", "Outdoors", "Indoors"}
            ),
            cost=1,
            base_fun=1,
        )
        player = PlayerState(
            name="P0",
            energy=10,
            hand=[dummy_card],
            played_today=[
                CardInstance(i + 1, dummy_prior_defn)
                for i in range(prior_count)
            ],
        )
        players = [player] + [
            PlayerState(name=f"P{i}", energy=7) for i in range(1, 4)
        ]
        game = object.__new__(Game)
        game.players = players
        game.suitcase = [
            CardInstance(10 + i, dummy_prior_defn) for i in range(4)
        ]
        game.trunk = [CardInstance(20 + i, dummy_prior_defn) for i in range(8)]
        game.discard = [
            CardInstance(30 + i, dummy_prior_defn) for i in range(4)
        ]
        game.day = 2
        game.starting_player = 0
        game.daily_conditions = False
        game.characters = False
        game.daily_condition = None
        game._condition_deck = []
        game.revealed_conditions = []
        game._condition_knowledge = [() for _ in range(4)]
        game._is_setup = True
        game.stats = _get_shared_sim_stats()
        game._fun_at_start_of_scoring = None
        return bool(b.can_play(game, player, dummy_card))
    except Exception:
        return True


def _run_isolated_play_probe(
    defn: CardDefinition, prior_count: int
) -> tuple[int, int, int, bool, int, int, int]:
    """Run a single isolated play probe with prior_count dummy cards in played_today."""
    from thenos.game import Game

    probe_ai = _EvalPlayerAI()
    probe_game = object.__new__(Game)
    probe_game.rng = _seeded_random(12345)
    probe_game.ais = [probe_ai, probe_ai, probe_ai, probe_ai]
    tagged_dummy_defn = CardDefinition(
        slug="_probe_card_tagged",
        title="_ProbeCardTagged",
        tags=frozenset(
            {
                "Relax",
                "Food",
                "Exercise",
                "Social",
                "Outdoors",
                "Indoors",
                "Board Game",
                "Item",
            }
        ),
        cost=5,
        base_fun=2,
    )
    untagged_dummy_defn = CardDefinition(
        slug="_probe_card_plain",
        title="_ProbeCardPlain",
        tags=frozenset(),
        cost=1,
        base_fun=2,
    )
    probe_card = CardInstance(0, defn)
    priors = [
        CardInstance(
            10 + i,
            tagged_dummy_defn,
            markers={"_gave_energy": True},
        )
        for i in range(prior_count)
    ]
    p0 = PlayerState(
        name="P0",
        energy=10,
        hand=[
            probe_card,
            CardInstance(1, tagged_dummy_defn),
            CardInstance(2, tagged_dummy_defn),
        ],
        played_today=priors,
    )
    probe_game.players = [
        p0,
        PlayerState(
            name="P1",
            energy=7,
            hand=[CardInstance(4, tagged_dummy_defn)],
            played_today=[CardInstance(14, tagged_dummy_defn)],
        ),
        PlayerState(
            name="P2",
            energy=7,
            hand=[CardInstance(5, tagged_dummy_defn)],
            played_today=[CardInstance(15, tagged_dummy_defn)],
        ),
        PlayerState(
            name="P3",
            energy=7,
            hand=[CardInstance(6, tagged_dummy_defn)],
            played_today=[CardInstance(16, tagged_dummy_defn)],
        ),
    ]
    probe_game.suitcase = [
        CardInstance(20, tagged_dummy_defn),
        CardInstance(21, tagged_dummy_defn),
        CardInstance(22, untagged_dummy_defn),
        CardInstance(23, untagged_dummy_defn),
    ]
    probe_game.trunk = [
        CardInstance(30 + i, untagged_dummy_defn) for i in range(15)
    ] + [CardInstance(49, tagged_dummy_defn)]
    probe_game.discard = [
        CardInstance(50 + i, untagged_dummy_defn) for i in range(6)
    ]
    probe_game.day = 2
    probe_game.starting_player = 0
    probe_game.daily_conditions = False
    probe_game.characters = False
    probe_game.daily_condition = None
    probe_game._condition_deck = []
    probe_game.revealed_conditions = []
    probe_game._condition_knowledge = [() for _ in range(4)]
    probe_game._is_setup = True
    probe_game.stats = _get_shared_sim_stats()
    probe_game._fun_at_start_of_scoring = (0, 0, 0, 0)

    cost_before = probe_game.energy_cost(0, probe_card)
    energy_after_cost = p0.energy - cost_before
    fun_before = p0.fun
    hand_after_pop = len(p0.hand) - 1
    played_before = len(p0.played_today)
    probe_game.play_card(0, 0)
    cards_discarded = max(0, hand_after_pop - len(p0.hand))
    energy_gain = max(0, p0.energy - energy_after_cost - cards_discarded)
    if defn.title == "Istanbul":
        energy_gain = 0
    fun_gain = max(0, p0.fun - fun_before)
    cards_gained = max(0, len(p0.hand) - hand_after_pop)
    ends_day = bool(p0.asleep)
    scored_fun = int(defn.behavior.fun_value(probe_game, p0, probe_card))

    tom_energy = 0
    tom_fun = 0
    if type(defn.behavior).on_start_day is not CardBehavior.on_start_day:
        e_b = p0.energy
        f_b = p0.fun
        defn.behavior.on_start_day(probe_game, p0, probe_card)
        tom_energy = p0.energy - e_b
        tom_fun = p0.fun - f_b

    return (
        energy_gain,
        fun_gain,
        cards_gained,
        ends_day,
        scored_fun,
        tom_energy,
        tom_fun,
    )


def _build_profile(card: CardInstance) -> CardSynergyProfile:
    defn = card.definition
    if not card.markers:
        fast_cached = _DEFN_ID_CACHE.get(id(defn))
        if fast_cached is not None:
            return fast_cached
    b = card.effective_behavior
    cache_key = (defn.title, type(b), defn.cost, defn.base_fun, defn.tags)
    cached = _PROFILE_CACHE.get(cache_key)
    if cached is not None:
        if not card.markers:
            _DEFN_ID_CACHE[id(defn)] = cached
        return cached
    if cache_key in _ACTIVE_PROBE_KEYS:
        return _FALLBACK_PROFILE

    b_type = type(b)
    cls_name = b_type.__name__
    modifies_other_fun = b_type.modify_fun is not CardBehavior.modify_fun
    modifies_other_cost = (
        b_type.modify_energy_cost is not CardBehavior.modify_energy_cost
    )
    modifies_own_cost = (
        b_type.modify_own_energy_cost is not CardBehavior.modify_own_energy_cost
    )
    has_dynamic_fun = b_type.fun_value is not CardBehavior.fun_value
    has_on_play = b_type.on_play is not CardBehavior.on_play
    has_on_card_play = b_type.on_card_play is not CardBehavior.on_card_play
    has_on_card_acquire = (
        b_type.on_card_acquire is not CardBehavior.on_card_acquire
    )
    has_tomorrow_start = b_type.on_start_day is not CardBehavior.on_start_day
    has_tomorrow_play = (
        b_type.on_tomorrow_card_play is not CardBehavior.on_tomorrow_card_play
    )
    has_tomorrow_fun = (
        b_type.modify_tomorrow_fun is not CardBehavior.modify_tomorrow_fun
    )
    has_tomorrow_cost = (
        b_type.modify_tomorrow_energy_cost
        is not CardBehavior.modify_tomorrow_energy_cost
    )
    has_extra_pick = (
        b_type.allows_extra_suitcase_pick
        is not CardBehavior.allows_extra_suitcase_pick
    )
    has_end_day_hook = (
        b_type.on_score is not CardBehavior.on_score
        or b_type.on_end_day is not CardBehavior.on_end_day
        or b_type.on_after_end_day is not CardBehavior.on_after_end_day
    )
    has_tomorrow_action = bool(getattr(b, "has_tomorrow_action", False))
    tomorrow_all_tags = bool(getattr(b, "tomorrow_all_tags", False))
    has_tomorrow_effect = (
        has_tomorrow_start
        or has_tomorrow_play
        or has_tomorrow_fun
        or has_tomorrow_cost
        or has_extra_pick
        or has_tomorrow_action
        or tomorrow_all_tags
    )

    tag_attr = getattr(b, "tag", None)
    tags_attr = getattr(b, "tags", frozenset())
    target_tags: set[str] = set()
    if isinstance(tag_attr, str) and tag_attr:
        target_tags.add(tag_attr)
    if isinstance(tags_attr, (set, frozenset, tuple, list)):
        for t in tags_attr:
            if isinstance(t, str) and t:
                target_tags.add(t)

    # Implicit tag synergies for bespoke behaviors without .tag attribute
    if (
        "WaterTrampoline" in cls_name
        or "PotatoPancakes" in cls_name
        or "FancyFloatie" in cls_name
        or "ClassicBook" in cls_name
    ):
        target_tags.add("Relax")
    elif "FancyCraft" in cls_name:
        target_tags.add("Item")
    elif (
        "CampCrossroads" in cls_name
        or "Swimming" in cls_name
        or "SkiBoat" in cls_name
    ):
        target_tags.add("Exercise")
    elif (
        "Keeper" in cls_name
        or "PaintRocks" in cls_name
        or "BeachConversation" in cls_name
        or "WildlifeSpotting" in cls_name
        or "FishingBoat" in cls_name
    ):
        target_tags.add("Outdoors")
    elif (
        "Doritos" in cls_name
        or "IceCreamSandwich" in cls_name
        or "Leftovers" in cls_name
        or "TreatStoreRun" in cls_name
        or "AddressTheFood" in cls_name
        or "GroceryStoreRun" in cls_name
        or "BougieCoffeeMachine" in cls_name
    ):
        target_tags.add("Food")
    elif "Carcassonne" in cls_name:
        target_tags.add("Board Game")

    fun_bonus = int(getattr(b, "bonus", 0) or getattr(b, "fun_bonus", 0) or 0)
    energy_delta = int(
        getattr(b, "energy_delta", 0) or -int(getattr(b, "energy_gain", 0) or 0)
    )
    multiplier = int(getattr(b, "multiplier", 1) or 1)
    if "AfterDinnerEntertainment" in cls_name:
        fun_bonus = 2
        energy_delta = -2
    elif "HalfEnergyForNextTag" in cls_name or "SetEnergyForNextTag" in cls_name:
        energy_delta = -2
    elif "WaterTrampoline" in cls_name or "FancyCraft" in cls_name:
        fun_bonus = 3
    elif "Canoe" in cls_name or (
        "WrittenCost" in cls_name and "Next" in cls_name
    ):
        fun_bonus = 1
    elif "FancyFloatie" in cls_name:
        fun_bonus = 1
    elif "NosBookBehavior" in cls_name:
        fun_bonus = 3
    elif "Tanning" in cls_name:
        fun_bonus = -1
    elif "MedicalAdvice" in cls_name:
        fun_bonus = -1
        energy_delta = 1
    elif "ScoutTheOtherCottages" in cls_name:
        fun_bonus = -1
    elif "FunForNextCardBehavior" in cls_name:
        fun_bonus = 4

    can_0 = _probe_can_play_at(defn, 0)
    can_1 = _probe_can_play_at(defn, 1)
    can_3 = _probe_can_play_at(defn, 3)

    must_play_first = can_0 and not can_1
    must_play_early = can_0 and can_1 and not can_3
    must_play_late = (not can_0) and can_3

    immediate_energy_gain = 0
    conditional_energy_gain = 0
    immediate_fun_gain = 0
    immediate_cards_gained = 0
    conditional_fun_bonus = 0
    tomorrow_start_energy = 0
    tomorrow_start_fun = 0
    ends_day_immediately = False
    goes_to_tomorrow = has_tomorrow_effect

    if has_on_play or has_tomorrow_effect or has_dynamic_fun:
        _ACTIVE_PROBE_KEYS.add(cache_key)
        try:
            p0_prior = 4 if (must_play_late or not can_0) else 0
            (
                immediate_energy_gain,
                immediate_fun_gain,
                immediate_cards_gained,
                ends_day_immediately,
                scored_0,
                tomorrow_start_energy,
                tomorrow_start_fun,
            ) = _run_isolated_play_probe(defn, p0_prior)
            if can_0 and can_3:
                (
                    e_gain_k,
                    _,
                    c_gain_k,
                    _,
                    scored_k,
                    _,
                    _,
                ) = _run_isolated_play_probe(defn, 4)
                conditional_energy_gain = max(
                    0, e_gain_k - immediate_energy_gain
                )
                conditional_fun_bonus = max(
                    0, scored_k - scored_0, scored_0 - defn.base_fun
                )
                immediate_cards_gained = max(immediate_cards_gained, c_gain_k)
            elif must_play_late or must_play_first:
                conditional_fun_bonus = max(0, scored_0 - defn.base_fun)
        except Exception:
            pass
        finally:
            _ACTIVE_PROBE_KEYS.discard(cache_key)

    if "DatesFirstNos" in cls_name:
        tomorrow_start_fun = 6
    elif "TreatStoreRun" in cls_name or "FancyFloatie" in cls_name:
        immediate_cards_gained = 1
    elif "BougieCoffeeMachine" in cls_name:
        immediate_cards_gained = 2
    elif "Leftovers" in cls_name:
        conditional_fun_bonus = 3
    elif "NewfangledTresFute" in cls_name:
        conditional_fun_bonus = 2
    elif "CheesyPhoneGame" in cls_name:
        conditional_fun_bonus = 2

    # Signed benefit per matching card: positive fun_bonus is good, negative energy_delta (discount) is good
    signed_match_benefit = float(fun_bonus) - float(energy_delta) * 0.90
    if "Double" in cls_name and signed_match_benefit == 0.0:
        signed_match_benefit = 2.5

    is_single_buff = bool(
        signed_match_benefit > 0.0
        and (
            "Next" in cls_name
            or "Nth" in cls_name
            or "WaterTrampoline" in cls_name
            or "FancyCraft" in cls_name
            or "AfterDinnerEntertainment" in cls_name
        )
    )
    is_tomorrow_scaler = bool(
        target_tags
        and signed_match_benefit > 0.0
        and (has_tomorrow_fun or has_tomorrow_cost)
        and not modifies_other_fun
        and not modifies_other_cost
    )
    is_multi_scaler = bool(
        target_tags
        and signed_match_benefit > 0.0
        and not is_single_buff
        and not is_tomorrow_scaler
        and (
            modifies_other_fun
            or modifies_other_cost
            or "After" in cls_name
            or "Before" in cls_name
        )
    )
    is_before_scaler = bool(
        ("Before" in cls_name and fun_bonus > 0 and "After" not in cls_name)
        or (
            "WrittenCost" in cls_name
            and multiplier > 0
            and "Next" not in cls_name
        )
    )
    is_global_after_scaler = bool(
        not target_tags
        and signed_match_benefit > 0.0
        and not is_single_buff
        and (
            (
                "AllCardsAfter" in cls_name
                and energy_delta <= 0
                and "Double" not in cls_name
            )
            or "EuchreTournamentAwardsCeremony" in cls_name
        )
    )
    if "EuchreTournamentAwardsCeremony" in cls_name:
        signed_match_benefit = 1.8

    penalizes_before = bool(
        ("Before" in cls_name and fun_bonus < 0)
        or multiplier < 0
        or "MedicalAdvice" in cls_name
        or "ScoutTheOtherCottages" in cls_name
    )
    penalizes_after = bool(
        signed_match_benefit < 0.0
        and (
            "After" in cls_name
            or "MedicalAdvice" in cls_name
            or "Tanning" in cls_name
        )
    )

    is_count_finisher = bool(
        not penalizes_before
        and (
            must_play_late
            or is_before_scaler
            or "NewNosBookEntry" in cls_name
            or "Bohnanza" in cls_name
            or "Leftovers" in cls_name
            or "NosBookBehavior" in cls_name
            or "KayakBehavior" in cls_name
        )
    )

    # Left-to-right play priority bucket:
    # 0 = must_play_first
    # 1 = must_play_early / penalizes_before / unconditional net energy generators
    # 2 = multi-card scalers ("After"), card draw, cost discounts
    # 3 = single-card "Next" buffs & multipliers (Bring a Friend, Work Call, Water Trampoline)
    # 4 = standard payoff & fuel cards
    # 5 = count finishers ("Before" scalers, late cards, penalizes_after)
    # 6 = ends_day_immediately / Decaf
    if ends_day_immediately or "Decaf" in cls_name:
        bucket = 6
    elif must_play_first:
        bucket = 0
    elif (
        must_play_early
        or penalizes_before
        or (
            immediate_energy_gain > defn.cost
            and "ButterTarts" not in cls_name
            and "SleepIn" not in cls_name
            and "PuddingChomeur" not in cls_name
        )
        or "Wurstchen" in cls_name
    ):
        bucket = 1
    elif penalizes_after or is_count_finisher:
        bucket = 5
    elif (
        is_multi_scaler
        or is_global_after_scaler
        or (immediate_energy_gain > 0 and "PuddingChomeur" not in cls_name)
        or (immediate_cards_gained > 0 and defn.cost <= 1)
        or (modifies_other_cost and energy_delta < 0 and not is_single_buff)
        or has_on_card_play
        or has_on_card_acquire
        or "ChristmasNameDraw" in cls_name
        or "FunAndEnergyForAllCardsAfter" in cls_name
    ):
        bucket = 2
    elif is_single_buff or "FunForAllCardsAfterDoubleBehavior" in cls_name:
        bucket = 3
    else:
        bucket = 4

    profile = CardSynergyProfile(
        must_play_first=must_play_first,
        must_play_early=must_play_early,
        must_play_late=must_play_late,
        ends_day_immediately=ends_day_immediately,
        goes_to_tomorrow=goes_to_tomorrow,
        immediate_energy_gain=immediate_energy_gain,
        conditional_energy_gain=conditional_energy_gain,
        immediate_fun_gain=immediate_fun_gain,
        immediate_cards_gained=immediate_cards_gained,
        conditional_fun_bonus=conditional_fun_bonus,
        tomorrow_start_energy=tomorrow_start_energy,
        tomorrow_start_fun=tomorrow_start_fun,
        modifies_other_fun=modifies_other_fun,
        modifies_other_cost=modifies_other_cost,
        modifies_own_cost=modifies_own_cost,
        has_dynamic_fun=has_dynamic_fun,
        has_tomorrow_effect=has_tomorrow_effect,
        has_end_day_hook=has_end_day_hook,
        target_tags=frozenset(target_tags),
        fun_bonus=fun_bonus,
        energy_delta=energy_delta,
        multiplier=multiplier,
        signed_match_benefit=signed_match_benefit,
        play_order_bucket=bucket,
        is_multi_scaler=is_multi_scaler,
        is_before_scaler=is_before_scaler,
        is_tomorrow_scaler=is_tomorrow_scaler,
        is_single_buff=is_single_buff,
        is_global_after_scaler=is_global_after_scaler,
        is_count_finisher=is_count_finisher,
        penalizes_before=penalizes_before,
        penalizes_after=penalizes_after,
    )
    _PROFILE_CACHE[cache_key] = profile
    if not card.markers:
        _DEFN_ID_CACHE[id(defn)] = profile
    return profile


def _remaining_played_card_value(
    game: Game,
    player_index: int,
    card: CardInstance,
) -> float:
    """Value of keeping an already-played card in played_today (used by Bohnanza)."""
    prof = _build_profile(card)
    val = float(game.card_fun(player_index, card))
    if prof.is_multi_scaler or prof.is_global_after_scaler:
        val += max(0.0, prof.signed_match_benefit * 2.0)
    if prof.goes_to_tomorrow and game.day < DAY_COUNT:
        val += prof.tomorrow_start_fun + prof.tomorrow_start_energy * 1.15
        if prof.signed_match_benefit > 0:
            val += prof.signed_match_benefit * 1.5
    return val


class _EvalPlayerAI(RandomAI):
    """Ultra-fast tactical policy used inside simulation rollouts and on_play hooks."""

    def __init__(self) -> None:
        super().__init__(_seeded_random(99991))

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if not options:
            raise ValueError("No suitcase card option was supplied")
        player = game.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)
        best_idx = 0
        best_score = float("-inf")
        for idx, card in enumerate(options):
            score = _quick_card_desirability(
                game, player_index, card, wanted_tags
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def choose_to_go_to_bed(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> bool:
        return (
            not playable_hand_indices
            or len(game.trunk) + len(game.discard) <= 6
            or len(game.players[player_index].played_today) >= 20
        )

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        player = game.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)
        best_idx = playable_hand_indices[0]
        best_key = (99, 0.0)
        for hand_idx in playable_hand_indices:
            card = player.hand[hand_idx]
            prof = _build_profile(card)
            est = _quick_card_desirability(
                game, player_index, card, wanted_tags
            )
            key = (prof.play_order_bucket, -est)
            if key < best_key:
                best_key = key
                best_idx = hand_idx
        return best_idx

    def choose_extra_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        if (
            not playable_hand_indices
            or len(game.trunk) + len(game.discard) <= 6
            or len(game.players[player_index].played_today) >= 20
        ):
            return None
        return self.choose_card_to_play(game, player_index, playable_hand_indices)

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        wanted_tags = _wanted_tags_for_player(game.players[player_index])
        return min(
            range(len(options)),
            key=lambda i: _quick_card_desirability(
                game, player_index, options[i], wanted_tags
            ),
        )

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> Sequence[int]:
        if not options:
            return ()
        player = game.players[player_index]
        last_played = player.played_today[-1].title if player.played_today else ""
        if last_played == "Bohnanza":
            return tuple(
                i
                for i, c in enumerate(options)
                if _remaining_played_card_value(game, player_index, c) < 2.8
            )

        wanted_tags = _wanted_tags_for_player(player)
        ranked = sorted(
            range(len(options)),
            key=lambda i: _quick_card_desirability(
                game, player_index, options[i], wanted_tags
            ),
        )
        if last_played == "Navy SEALing":
            return tuple(
                i
                for i in ranked
                if _quick_card_desirability(
                    game, player_index, options[i], wanted_tags
                )
                < 2.2
            )
        if last_played in {"Scrabble", "Last Year's Shirt"}:
            if not game.suitcase:
                return ()
            suitcase_vals = sorted(
                (
                    _quick_card_desirability(
                        game, player_index, sc, wanted_tags
                    )
                    for sc in game.suitcase
                ),
                reverse=True,
            )
            limit = min(
                len(suitcase_vals),
                len(game.suitcase) if last_played == "Last Year's Shirt" else 3,
            )
            chosen: list[int] = []
            for k, i in enumerate(ranked[:limit]):
                if (
                    _quick_card_desirability(
                        game, player_index, options[i], wanted_tags
                    )
                    < suitcase_vals[k] - 0.8
                ):
                    chosen.append(i)
                else:
                    break
            return tuple(chosen)
        if last_played == "Istanbul":
            max_discards = min(3, max(0, len(options) - 1))
            return tuple(
                i
                for i in ranked[:max_discards]
                if _quick_card_desirability(
                    game, player_index, options[i], wanted_tags
                )
                < 2.4
            )
        return ()

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        wanted_tags = _wanted_tags_for_player(game.players[player_index])
        return max(
            range(len(options)),
            key=lambda i: _quick_card_desirability(
                game, player_index, options[i], wanted_tags
            ),
        )

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        player = game.players[player_index]
        last_played = player.played_today[-1].title if player.played_today else ""
        if last_played == "Christmas Name Draw":
            return max(
                range(len(options)),
                key=lambda i: sum(
                    1 for c in player.hand if c.tags & options[i].tags
                ),
            )
        wanted_tags = _wanted_tags_for_player(player)
        return max(
            range(len(options)),
            key=lambda i: _quick_card_desirability(
                game, player_index, options[i], wanted_tags
            ),
        )

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        wanted_tags = _wanted_tags_for_player(game.players[player_index])
        return min(
            range(len(options)),
            key=lambda i: (
                options[i].effective_base_fun - options[i].effective_cost * 0.5,
                _quick_card_desirability(
                    game, player_index, options[i], wanted_tags
                ),
            ),
        )

    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        if not eligible_player_indices:
            raise ValueError("No eligible player was supplied")
        best_p = eligible_player_indices[0]
        best_score = -999
        for p_idx in eligible_player_indices:
            cand = game.players[p_idx]
            has_bg = any("Board Game" in c.tags for c in cand.played_today)
            if has_bg:
                score = -10
            elif cand.asleep:
                score = 10
            elif p_idx == player_index and not any(
                "Board Game" in c.tags for c in cand.hand
            ):
                score = 8
            else:
                score = 5 - len(cand.hand)
            if score > best_score:
                best_score = score
                best_p = p_idx
        return best_p

    def choose_tag(
        self,
        game: Game,
        player_index: int,
        options: Sequence[str],
    ) -> str:
        player = game.players[player_index]
        last_played = player.played_today[-1].title if player.played_today else ""
        if last_played == "Poker":
            prior = {
                "Outdoors": 38.0,
                "Relax": 30.0,
                "Exercise": 26.0,
                "Food": 26.0,
                "Social": 24.0,
                "Board Game": 21.0,
                "Indoors": 14.0,
                "Item": 13.0,
            }
            return max(options, key=lambda t: prior.get(t, 10.0))
        counts: dict[str, float] = {tag: 0.0 for tag in options}
        for c in player.hand:
            for t in c.tags:
                if t in counts:
                    counts[t] += 1.5
        for c in player.played_today:
            for t in c.tags:
                if t in counts:
                    counts[t] += 1.0
        return max(options, key=lambda t: counts.get(t, 0.0))

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        return maximum

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        player = game.players[player_index]
        if "Quacks" in action:
            last_card = player.played_today[-1] if player.played_today else None
            if last_card is not None:
                draws = int(last_card.markers.get("_quacks_draws", 0))
                if game.stats is _get_shared_sim_stats():
                    last_card.markers["_quacks_draws"] = draws + 1
                return draws < 2
            return False
        if "Cities and Knights" in action:
            if not player.hand:
                return False
            wanted_tags = _wanted_tags_for_player(player)
            worst_val = min(
                _quick_card_desirability(game, player_index, c, wanted_tags)
                for c in player.hand
            )
            return worst_val < 3.2
        if "Epic Prank" in action:
            items = [c for c in player.hand if "Item" in c.tags]
            if not items:
                return False
            wanted_tags = _wanted_tags_for_player(player)
            worst_val = min(
                _quick_card_desirability(game, player_index, c, wanted_tags)
                for c in items
            )
            return worst_val < 5.0
        return True

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> bool:
        return game.players[player_index].energy >= 2

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        wanted_tags = _wanted_tags_for_player(game.players[player_index])
        return sorted(
            range(len(cards)),
            key=lambda i: _quick_card_desirability(
                game, player_index, cards[i], wanted_tags
            ),
            reverse=True,
        )

    def choose_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        wanted_tags = _wanted_tags_for_player(game.players[player_index])
        return tuple(
            i
            for i, c in enumerate(cards)
            if _quick_card_desirability(game, player_index, c, wanted_tags) < 2.0
        )

    def order_daily_conditions(
        self,
        game: Game,
        player_index: int,
        conditions: Sequence[Any],
    ) -> Sequence[int]:
        return tuple(range(len(conditions)))


_EVAL_AI_SINGLETON = _EvalPlayerAI()


def _wanted_tags_for_player(player: PlayerState) -> dict[str, float]:
    """Compute signed tag synergy weights from the player's current cards and character."""
    weights: dict[str, float] = {}
    if player.character is not None:
        if player.character.fun_tag and player.character.fun_delta > 0:
            weights[player.character.fun_tag] = (
                weights.get(player.character.fun_tag, 0.0)
                + player.character.fun_delta * 1.4
            )
        if player.character.energy_tag and player.character.energy_delta < 0:
            weights[player.character.energy_tag] = (
                weights.get(player.character.energy_tag, 0.0)
                + abs(player.character.energy_delta) * 1.2
            )
        if player.character.zero_fun_tag:
            weights[player.character.zero_fun_tag] = -5.0

    for card in player.tomorrow_cards:
        prof = _build_profile(card)
        if prof.signed_match_benefit != 0.0:
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.15
                )
        chosen_tag = card.markers.get("chosen_tag")
        if isinstance(chosen_tag, str):
            weights[chosen_tag] = weights.get(chosen_tag, 0.0) + 1.5

    for card in player.played_today:
        prof = _build_profile(card)
        if prof.is_multi_scaler or prof.is_tomorrow_scaler:
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.05
                )
        elif prof.penalizes_after or prof.signed_match_benefit < 0:
            for t in prof.target_tags:
                weights[t] = weights.get(t, 0.0) + prof.signed_match_benefit

    for card in player.hand:
        prof = _build_profile(card)
        if prof.is_multi_scaler or prof.is_tomorrow_scaler:
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 0.90
                )
        elif prof.is_single_buff:
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 0.30
                )
        for t in card.tags:
            weights[t] = weights.get(t, 0.0) + 0.15
    return weights


def _condition_met_in_played(
    player: PlayerState, card: CardInstance, prof: CardSynergyProfile
) -> bool:
    """Check whether a card's conditional energy/fun trigger is satisfied by played_today."""
    if not player.played_today:
        return False
    title = card.title
    if title in {"Doritos", "Ice Cream Sandwich", "Leftovers"}:
        return any("Food" in c.tags for c in player.played_today)
    if title == "Steak":
        return any(c.definition.cost >= 5 for c in player.played_today)
    if title == "Bubly":
        return any(c.markers.get("_gave_energy") for c in player.played_today)
    if title == "Beach Conversation":
        return "Outdoors" in player.played_today[-1].tags
    if title == "Swimming":
        return "Exercise" in player.played_today[-1].tags
    if title == "Paint Rocks":
        return sum(1 for c in player.played_today if "Outdoors" in c.tags) >= 2
    if title == "Potato Pancakes":
        relax_played = sum(1 for c in player.played_today if "Relax" in c.tags)
        return relax_played >= 2 or (
            relax_played >= 1
            and not any(
                other is not card
                and "Relax" in other.tags
                and other.effective_cost <= player.energy
                for other in player.hand
            )
        )
    if title == "Camp Crossroads Dessert":
        return sum(1 for c in player.played_today if "Exercise" in c.tags) >= 2
    if title == "Keeper":
        return sum(1 for c in player.played_today if "Outdoors" in c.tags) >= 2
    if prof.target_tags:
        return any(bool(c.tags & prof.target_tags) for c in player.played_today)
    return True


def _extra_energy_support_for_card(
    player: PlayerState, card: CardInstance
) -> int:
    """Estimate extra Energy + cost discounts available in hand/tomorrow to fund a high-cost card."""
    support = 0
    for tc in player.tomorrow_cards:
        t_prof = _build_profile(tc)
        if t_prof.tomorrow_start_energy > 0:
            support += t_prof.tomorrow_start_energy
    for hc in player.hand:
        if hc is card:
            continue
        h_prof = _build_profile(hc)
        if h_prof.immediate_energy_gain > hc.effective_cost:
            support += h_prof.immediate_energy_gain - hc.effective_cost
        elif (
            h_prof.conditional_energy_gain > hc.effective_cost
            and hc.title
            not in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}
        ):
            support += h_prof.conditional_energy_gain - hc.effective_cost
        elif hc.title == "Würstchen":
            support += max(1, len(player.hand) - 3)
        elif h_prof.tomorrow_start_energy > 0:
            support += max(0, h_prof.tomorrow_start_energy - 1)
        elif hc.title == "Butter Tarts":
            support += 3
        if (
            h_prof.energy_delta < 0
            and (not h_prof.target_tags or (card.tags & h_prof.target_tags))
        ):
            support += abs(h_prof.energy_delta)
    return support


def _quick_card_desirability(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
) -> float:
    """Fast heuristic valuation of a card including signed tag synergies."""
    prof = _build_profile(card)
    player = game.players[player_index]
    cost = game.energy_cost(player_index, card)
    days_left = DAY_COUNT - game.day

    imm_energy = (
        max(2, len(player.hand))
        if card.title == "Würstchen"
        else prof.immediate_energy_gain
    )
    val = (
        float(card.effective_base_fun)
        + prof.immediate_fun_gain * 1.15
        - cost * 0.78
    )
    val += imm_energy * 1.30
    if (
        prof.immediate_cards_gained > 0
        and not prof.penalizes_before
        and not prof.penalizes_after
        and not prof.is_multi_scaler
    ):
        val += min(prof.immediate_cards_gained, 3) * (
            0.95 if cost <= 1 else 0.15
        )

    if card.title == "Sleep In":
        val -= 1.8
    elif (
        prof.must_play_late
        and prof.immediate_energy_gain > 0
        and len(player.played_today) < 2
    ):
        val -= 0.85

    # Penalize unfunded 8+ cost cards (Adventure Race, Civilization, Wakeboard, Navy SEALing)
    if card.effective_cost >= 8 and cost > player.energy:
        support = _extra_energy_support_for_card(player, card)
        unfunded = max(0, card.effective_cost - (6 + support))
        if unfunded > 0:
            val -= unfunded * 1.65

    if prof.conditional_energy_gain > 0:
        if card.title in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}:
            match_cnt = sum(
                1
                for other in (*player.played_today, *player.hand)
                if other is not card and (other.tags & prof.target_tags)
            )
            per_match = 2 if card.title == "Potato Pancakes" else 1
            val += min(match_cnt, 3) * per_match * 0.85
        else:
            has_cond = _condition_met_in_played(player, card, prof) or any(
                other is not card and (other.tags & prof.target_tags)
                for other in player.hand
            )
            val += prof.conditional_energy_gain * (0.95 if has_cond else 0.35)

    if prof.conditional_fun_bonus > 0 and not prof.is_multi_scaler:
        val += min(prof.conditional_fun_bonus, 5) * 0.55

    # Reward/penalize cards whose tags match our wanted/penalized synergy tags
    tag_match = 0.0
    for t in card.tags:
        w = wanted_tags.get(t, 0.0)
        if w != 0.0:
            tag_match += w * 0.78
            if w > 0 and cost <= 1:
                tag_match += 0.38 * min(w, 2.5)
    if prof.must_play_first and tag_match > 0.0:
        tag_match *= 0.30
    val += tag_match

    # Reward multi-card scalers if we have matching cards or time left to draft them
    if prof.is_multi_scaler:
        matching_in_hand = sum(
            1
            for c in player.hand
            if c is not card and (c.tags & prof.target_tags)
        )
        matching_played = sum(
            1
            for c in player.played_today
            if c is not card and (c.tags & prof.target_tags)
        )
        abundance = _tag_abundance(prof.target_tags)
        future_factor = min(days_left, 3) * 0.50 * abundance
        val += (
            matching_in_hand * 0.95
            + matching_played * 0.50
            + future_factor
        ) * prof.signed_match_benefit
    elif prof.is_tomorrow_scaler:
        if days_left >= 1:
            matching_in_hand = sum(
                1
                for c in player.hand
                if c is not card and (c.tags & prof.target_tags)
            )
            abundance = _tag_abundance(prof.target_tags)
            val += (
                matching_in_hand * 0.65 + 0.40 * abundance
            ) * prof.signed_match_benefit
    elif prof.is_single_buff:
        has_match = any(
            c is not card and (c.tags & prof.target_tags) for c in player.hand
        )
        abundance = _tag_abundance(prof.target_tags)
        val += prof.signed_match_benefit * (
            0.78 if has_match else 0.25 * abundance
        )
    elif prof.signed_match_benefit < 0.0:
        val += prof.signed_match_benefit * 1.35

    if prof.is_global_after_scaler:
        cheap_cards = sum(1 for c in player.hand if c.effective_cost <= 1)
        val += (
            prof.signed_match_benefit * 0.85
            + cheap_cards * 0.55
            + min(days_left, 2) * 0.50
        )
    elif card.title == "Bring a Friend":
        zero_cards = sum(
            1 for c in player.hand if c is not card and c.effective_cost == 0
        )
        val += zero_cards * 1.45
    elif card.title == "Evening on the Dock":
        val += 3.5
    elif card.title == "Couch Tube":
        val += 2.6

    if prof.goes_to_tomorrow and days_left >= 1:
        val += prof.tomorrow_start_fun * 0.9 + prof.tomorrow_start_energy * 1.15
        if card.title == "Photo Shoot":
            val += 3.5

    if cost == 0 and card.effective_base_fun >= 0:
        val += 0.65
    elif cost == 1 and card.effective_base_fun >= 0:
        val += 0.25

    return val


def _fast_copy_game(src: Game) -> Game:
    """Fast simulation clone of a Game that avoids os.urandom and GameStats copies."""
    sim = object.__new__(type(src))
    sim.rng = _clone_random(src.rng)
    sim.ais = [
        _EVAL_AI_SINGLETON,
        _EVAL_AI_SINGLETON,
        _EVAL_AI_SINGLETON,
        _EVAL_AI_SINGLETON,
    ]
    card_map: dict[int, CardInstance] = {}
    cards_with_markers: list[CardInstance] = []

    def clone_card(card: CardInstance) -> CardInstance:
        cloned = CardInstance(
            card.instance_id,
            card.definition,
            card.is_tomorrow,
            dict(card.markers) if card.markers else {},
        )
        card_map[id(card)] = cloned
        if cloned.markers:
            cards_with_markers.append(cloned)
        return cloned

    sim.players = [
        PlayerState(
            name=p.name,
            energy=p.energy,
            fun=p.fun,
            character=p.character,
            hand=[clone_card(c) for c in p.hand],
            played_today=[clone_card(c) for c in p.played_today],
            tomorrow_cards=[clone_card(c) for c in p.tomorrow_cards],
            asleep=p.asleep,
            skipped_turns=p.skipped_turns,
            acquired_cards=p.acquired_cards.copy(),
            picked_cards=p.picked_cards.copy(),
        )
        for p in src.players
    ]
    sim.suitcase = [clone_card(c) for c in src.suitcase]
    sim.trunk = [clone_card(c) for c in src.trunk]
    sim.discard = [clone_card(c) for c in src.discard]
    for cloned in cards_with_markers:
        for key, val in list(cloned.markers.items()):
            if isinstance(val, CardInstance):
                mapped = card_map.get(id(val))
                if mapped is not None:
                    cloned.markers[key] = mapped
    sim.day = src.day
    sim.daily_conditions = src.daily_conditions
    sim.characters = src.characters
    sim.daily_condition = src.daily_condition
    sim._condition_deck = src._condition_deck.copy()
    sim.revealed_conditions = src.revealed_conditions.copy()
    sim._condition_knowledge = src._condition_knowledge.copy()
    sim.starting_player = src.starting_player
    sim._is_setup = src._is_setup
    sim.stats = _get_shared_sim_stats()
    sim._fun_at_start_of_scoring = (
        None
        if src._fun_at_start_of_scoring is None
        else list(src._fun_at_start_of_scoring)
    )
    return sim


class ArgonBrain(GreedyAI):
    """Synergy-focused single-player lookahead policy."""

    BEAM_WIDTH = 7
    DRAFT_BEAM_WIDTH = 4
    BRANCH_LIMIT = 5
    MAX_TURN_PLAYS = 7
    RESERVE_TOMORROW_WEIGHT = 0.74
    FIRST_TO_BED_BONUS = 0.45

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        _evaluation_depth: int = 0,
    ) -> None:
        super().__init__(rng)
        self._evaluation_depth = _evaluation_depth
        self._quacks_card_key: tuple[int, int, int, int] | None = None
        self._quacks_draws: int = 0

    def _prepare_observable_game(self, game: Game, player_index: int) -> Game:
        """Create a simulation copy with hidden zones resampled from observable state."""
        sim = _fast_copy_game(game)
        sim.rng = _seeded_random(self.rng.getrandbits(64))

        hidden_cards: list[CardInstance] = list(sim.trunk)
        opponent_hand_counts: list[tuple[int, int]] = []
        for idx, player in enumerate(sim.players):
            if idx == player_index:
                continue
            opponent_hand_counts.append((idx, len(player.hand)))
            hidden_cards.extend(player.hand)

        hidden_cards.sort(key=lambda c: c.instance_id)
        sim.rng.shuffle(hidden_cards)
        cursor = 0
        for idx, count in opponent_hand_counts:
            sim.players[idx].hand = hidden_cards[cursor : cursor + count]
            cursor += count
        sim.trunk = hidden_cards[cursor:]
        sim.discard.sort(key=lambda c: c.instance_id)
        sim.rng.shuffle(sim.discard)
        sim.sample_daily_conditions(player_index, sim.rng)
        return sim

    # ------------------------------------------------------------------
    # Suitcase Drafting
    # ------------------------------------------------------------------

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if self._evaluation_depth >= 1 or len(options) <= 1:
            return _EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        live_player = game.players[player_index]
        if live_player.played_today:
            wanted_tags = _wanted_tags_for_player(live_player)
            if (
                len(live_player.played_today) >= 3
                or len(live_player.hand) >= 8
                or len(options) < SUITCASE_SIZE
            ):
                best_idx = 0
                best_score = float("-inf")
                for idx, option in enumerate(options):
                    score = _quick_card_desirability(
                        game, player_index, option, wanted_tags
                    ) + self._draft_synergy_adjustment(
                        game, player_index, option, wanted_tags
                    )
                    if (
                        game.energy_cost(player_index, option)
                        <= live_player.energy
                    ):
                        score += 0.65
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                return best_idx

        base_sim = self._prepare_observable_game(game, player_index)
        player = base_sim.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)

        own_pr_markers = {
            c.markers.get("suitcase_target_marker")
            for c in player.played_today
            if c.markers.get("suitcase_target_marker")
        }

        draft_beam = 2 if player.played_today else self.DRAFT_BEAM_WIDTH
        draft_max_plays = 3 if player.played_today else 6

        best_indices: list[int] = []
        best_value = float("-inf")

        for idx, option in enumerate(options):
            sim = _fast_copy_game(base_sim)
            picked = sim.suitcase.pop(idx)
            sim.give_card(player_index, picked)
            sim.players[player_index].picked_cards[picked.title] += 1
            sim.record_card_pick_or_draw(player_index, picked)
            if sim.trunk or sim.discard:
                try:
                    sim.suitcase.insert(idx, sim._draw_from_trunk())
                except Exception:
                    pass

            _, plan_val = self._search_best_day_plan(
                sim,
                player_index,
                beam_width=draft_beam,
                max_plays=draft_max_plays,
            )
            synergy_val = self._draft_synergy_adjustment(
                base_sim, player_index, option, wanted_tags
            )
            if any(
                m in option.markers for m in own_pr_markers if isinstance(m, str)
            ):
                synergy_val -= 3.0
            total_val = plan_val + synergy_val

            if total_val > best_value + 1e-9:
                best_value = total_val
                best_indices = [idx]
            elif abs(total_val - best_value) <= 1e-9:
                best_indices.append(idx)

        return self.rng.choice(best_indices)

    def _draft_synergy_adjustment(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        wanted_tags: dict[str, float],
    ) -> float:
        """Reward cards that build or complete multi-card synergies across days."""
        player = game.players[player_index]
        prof = _build_profile(card)
        days_left = DAY_COUNT - game.day
        cost = game.energy_cost(player_index, card)

        bonus = 0.0
        # 1. Signed tag alignment with our existing scalers / character / tomorrow cards
        tag_align = 0.0
        for t in card.tags:
            w = wanted_tags.get(t, 0.0)
            if w != 0.0:
                tag_align += 0.38 * w
                if w > 0 and cost <= 1:
                    tag_align += 0.28 * min(w, 3.0)
        if prof.must_play_first and tag_align > 0.0:
            tag_align *= 0.25
        bonus += tag_align

        # 2. Speculative / compounding upside of drafting a true multi-card scaler early-to-mid game
        if prof.is_multi_scaler and days_left >= 1:
            matching_owned = sum(
                1
                for c in (*player.hand, *player.tomorrow_cards)
                if c is not card and (c.tags & prof.target_tags)
            )
            abundance = _tag_abundance(prof.target_tags)
            bonus += (
                0.42 * matching_owned
                + 0.36 * min(days_left, 3) * abundance
            ) * prof.signed_match_benefit
        elif prof.is_tomorrow_scaler and days_left >= 1:
            matching_owned = sum(
                1
                for c in player.hand
                if c is not card and (c.tags & prof.target_tags)
            )
            abundance = _tag_abundance(prof.target_tags)
            bonus += (
                0.30 * matching_owned + 0.18 * abundance
            ) * prof.signed_match_benefit
        elif prof.is_single_buff and days_left >= 1:
            matching_owned = sum(
                1
                for c in player.hand
                if c is not card and (c.tags & prof.target_tags)
            )
            if matching_owned >= 1:
                bonus += 0.38 * prof.signed_match_benefit
            else:
                bonus -= 0.25 * (1.0 - _tag_abundance(prof.target_tags))
        elif (
            prof.penalizes_after
            or prof.penalizes_before
            or prof.signed_match_benefit < 0
        ):
            bonus += prof.signed_match_benefit * 0.95

        if prof.is_global_after_scaler and days_left >= 1:
            cheap_owned = sum(1 for c in player.hand if c.effective_cost <= 1)
            bonus += 0.95 + 0.35 * cheap_owned + 0.32 * min(days_left, 3)
        elif card.title == "Couch Tube" and days_left >= 1:
            bonus += 1.65
        elif card.title == "Bring a Friend":
            zero_owned = sum(1 for c in player.hand if c.effective_cost == 0)
            bonus += 0.55 * zero_owned - 0.35

        # 3. Velocity cards (card draw / suitcase picks / net energy generators)
        if (
            prof.immediate_cards_gained > 0
            and not prof.penalizes_after
            and not prof.penalizes_before
            and cost <= 1
        ):
            bonus += 0.35 * min(prof.immediate_cards_gained, 3)
        if card.title == "Würstchen":
            bonus += 0.85 + 0.40 * max(0, len(player.hand) - 2)
        elif (
            prof.immediate_energy_gain > cost
            and prof.tomorrow_start_energy >= 0
            and card.effective_base_fun >= 0
            and card.title not in {"Sleep In", "Pudding Chômeur"}
        ):
            e_mult = 0.20 if prof.must_play_late else 0.55
            bonus += e_mult * (prof.immediate_energy_gain - cost)
        elif (
            prof.conditional_energy_gain > cost
            and card.effective_base_fun >= 0
            and card.title
            not in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}
        ):
            has_trig = any(
                c is not card
                and (
                    c.definition.cost >= 5
                    if card.title == "Steak"
                    else bool(c.tags & prof.target_tags)
                )
                for c in (*player.hand, *player.tomorrow_cards)
            )
            bonus += (0.50 if has_trig else 0.25) * (
                prof.conditional_energy_gain - cost + (1 if cost == 0 else 0)
            )
        elif card.title == "Sleep In":
            bonus -= 1.25
        elif (
            prof.tomorrow_start_energy > 0
            and prof.tomorrow_start_energy <= card.definition.cost
        ):
            bonus -= 0.75

        if card.title == "Chuck a Frisbee" and days_left >= 1:
            bonus += 0.65 * min(days_left, 3)
        elif card.title == "Photo Shoot" and days_left >= 1:
            bonus += 1.20
        elif card.title == "Paddleboard":
            bonus -= 1.40
        elif card.title == "Cheesy Phone Game":
            bonus += 0.75
        elif card.title == "Epic Prank" and any(
            "Item" in c.tags for c in player.hand if c is not card
        ):
            bonus += 1.40

        # Penalize unfunded 8+ cost bricks (Adventure Race, Civilization, Wakeboard, Navy SEALing)
        if card.effective_cost >= 8 and cost > player.energy:
            support = _extra_energy_support_for_card(player, card)
            unfunded = max(0, card.effective_cost - (6 + support))
            if unfunded > 0:
                bonus -= unfunded * 1.45
        elif (
            card.effective_cost >= 5
            and not prof.is_multi_scaler
            and not prof.goes_to_tomorrow
        ):
            support = _extra_energy_support_for_card(player, card)
            if support < card.effective_cost - 4:
                bonus -= 0.45 * (card.effective_cost - 4 - support)

        # Avoid drafting redundant must_play_first cards if we already hold one
        if prof.must_play_first and any(
            _build_profile(c).must_play_first for c in player.hand
        ):
            bonus -= 2.40

        return bonus

    # ------------------------------------------------------------------
    # Playing Phase: Full-Day Sequence Search & Play-vs-Hold Evaluation
    # ------------------------------------------------------------------

    def choose_to_go_to_bed(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> bool:
        if not playable_hand_indices:
            return True
        self._pending_play_choice = None
        if (
            len(game.trunk) + len(game.discard) <= 6
            or len(game.players[player_index].played_today) >= 20
        ):
            return True
        if self._evaluation_depth >= 1:
            return False

        base_sim = self._prepare_observable_game(game, player_index)
        best_action, _ = self._search_best_day_plan(
            base_sim,
            player_index,
            beam_width=self.BEAM_WIDTH,
            allowed_first_indices=playable_hand_indices,
        )
        if best_action is None or best_action not in playable_hand_indices:
            return True
        self._cache_play_choice(
            game, player_index, playable_hand_indices, best_action
        )
        return False

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        cached = self._take_cached_play_choice(
            game, player_index, playable_hand_indices
        )
        if cached is not None and cached in playable_hand_indices:
            return cached
        if self._evaluation_depth >= 1:
            return _EVAL_AI_SINGLETON.choose_card_to_play(
                game, player_index, playable_hand_indices
            )
        base_sim = self._prepare_observable_game(game, player_index)
        best_action, _ = self._search_best_day_plan(
            base_sim,
            player_index,
            beam_width=self.BEAM_WIDTH,
            allow_bed_now=False,
            allowed_first_indices=playable_hand_indices,
        )
        if best_action is not None and best_action in playable_hand_indices:
            return best_action
        return _EVAL_AI_SINGLETON.choose_card_to_play(
            game, player_index, playable_hand_indices
        )

    def choose_extra_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        if not playable_hand_indices:
            return None
        if (
            len(game.trunk) + len(game.discard) <= 6
            or len(game.players[player_index].played_today) >= 20
        ):
            return None
        if self._evaluation_depth >= 1:
            return _EVAL_AI_SINGLETON.choose_extra_card_to_play(
                game, player_index, playable_hand_indices
            )
        base_sim = self._prepare_observable_game(game, player_index)
        best_action, _ = self._search_best_day_plan(
            base_sim,
            player_index,
            beam_width=5,
            allowed_first_indices=playable_hand_indices,
        )
        if best_action is not None and best_action in playable_hand_indices:
            return best_action
        return None

    def _ordered_candidate_indices(
        self,
        sim: Game,
        player_index: int,
        limit: int,
    ) -> list[int]:
        """Return legal hand indices ordered by context-aware synergy role."""
        playable = sim.playable_hand_indices(player_index)
        if not playable:
            return []
        player = sim.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)
        last_card = player.played_today[-1] if player.played_today else None
        last_prof = _build_profile(last_card) if last_card is not None else None

        def sort_key(hand_idx: int) -> tuple[int, float, int]:
            card = player.hand[hand_idx]
            prof = _build_profile(card)
            cost = sim.energy_cost(player_index, card)
            desirability = _quick_card_desirability(
                sim, player_index, card, wanted_tags
            )
            bucket = prof.play_order_bucket

            if (
                prof.conditional_energy_gain > 0
                and prof.immediate_energy_gain == 0
            ):
                if _condition_met_in_played(player, card, prof):
                    bucket = 1
                    desirability += prof.conditional_energy_gain * 1.35
                else:
                    bucket = 4

            if card.title == "Address the Food":
                if any("Food" in sc.tags for sc in sim.suitcase):
                    bucket = 2
                    desirability += 3.5
                else:
                    bucket = 6
                    desirability -= 4.0
            elif card.title in {"Treat Store Run", "Fancy Floatie"}:
                tag_needed = (
                    "Food" if card.title == "Treat Store Run" else "Relax"
                )
                matches_in_sc = sum(
                    1 for sc in sim.suitcase if tag_needed in sc.tags
                )
                if matches_in_sc >= 2:
                    bucket = 2
                    desirability += matches_in_sc * 1.5
                elif matches_in_sc == 0:
                    desirability -= 2.0
            elif card.title == "Paddleboard":
                if len(player.hand) >= 3:
                    bucket = 5
                    desirability -= 2.0

            if last_card is not None and last_prof is not None:
                if last_card.title == "Work Call":
                    if (
                        card.effective_base_fun >= 3
                        or prof.conditional_fun_bonus >= 3
                    ):
                        bucket = 0
                        desirability += card.effective_base_fun * 2.0
                elif (
                    last_card.title == "Beach Volleyball"
                    and "Indoors" in card.tags
                ):
                    bucket = 1
                    desirability += 2.6
                elif (
                    card.title == "Beach Conversation"
                    and "Outdoors" in last_card.tags
                ):
                    bucket = 1
                    desirability += 2.6
                elif card.title == "Swimming" and "Exercise" in last_card.tags:
                    bucket = 2
                    desirability += 3.0
                elif (
                    card.title == "Carcassonne"
                    and "Board Game" in last_card.tags
                    and any(
                        other is not card and "Board Game" in other.tags
                        for other in player.hand
                    )
                ):
                    bucket = 2
                    desirability += 4.0
                elif (
                    last_prof.is_single_buff
                    and last_prof.target_tags
                    and (card.tags & last_prof.target_tags)
                    and not prof.is_multi_scaler
                    and not prof.is_single_buff
                ):
                    bucket = 2
                    desirability += (
                        card.definition.cost * 0.8
                        + card.effective_base_fun * 0.6
                    )

            return (bucket, -desirability, cost)

        ordered = sorted(playable, key=sort_key)
        if len(ordered) <= limit:
            return ordered
        selected: list[int] = []
        seen_buckets: set[int] = set()
        for idx in ordered:
            b = sort_key(idx)[0]
            if b not in seen_buckets:
                seen_buckets.add(b)
                selected.append(idx)
        for idx in ordered:
            if len(selected) >= limit:
                break
            if idx not in selected:
                selected.append(idx)
        return selected

    def _search_best_day_plan(
        self,
        base_sim: Game,
        player_index: int,
        *,
        beam_width: int = 7,
        max_plays: int | None = None,
        allow_bed_now: bool = True,
        allowed_first_indices: Sequence[int] | None = None,
    ) -> tuple[int | None, float]:
        """Search multi-card turn sequences and compare against stopping (going to bed) at every step."""
        played_count = len(base_sim.players[player_index].played_today)
        is_live_play = max_plays is None
        if is_live_play:
            if played_count >= 12:
                max_plays = 2
                beam_width = 4
            elif played_count >= 7:
                max_plays = 5
                beam_width = 8
            else:
                max_plays = 9
                beam_width = 12

        if allow_bed_now:
            bed_now = _fast_copy_game(base_sim)
            bed_now.go_to_bed(player_index)
            best_value = self._evaluate_terminal_day(bed_now, player_index)
            best_first_actions: list[int | None] = [None]
        else:
            best_value = float("-inf")
            best_first_actions = []

        frontier: list[tuple[int, Game, float]] = []
        seen_signatures: set[tuple] = set()

        init_limit = 9 if is_live_play else self.BRANCH_LIMIT + 1
        initial_candidates = self._ordered_candidate_indices(
            base_sim, player_index, init_limit
        )
        if allowed_first_indices is not None:
            allowed_set = set(allowed_first_indices)
            initial_candidates = [
                idx for idx in initial_candidates if idx in allowed_set
            ]
            if not initial_candidates and allowed_first_indices:
                initial_candidates = list(allowed_first_indices[:init_limit])
        for hand_idx in initial_candidates:
            child = _fast_copy_game(base_sim)
            try:
                child.play_card(player_index, hand_idx)
                stop_val = self._evaluate_terminal_day(child, player_index)
            except Exception:
                continue
            if stop_val > best_value + 1e-9:
                best_value = stop_val
                best_first_actions = [hand_idx]
            elif (
                abs(stop_val - best_value) <= 1e-9
                and hand_idx not in best_first_actions
            ):
                best_first_actions.append(hand_idx)

            if not child.players[player_index].asleep:
                cont_val = stop_val + self._continuation_potential(
                    child, player_index
                )
                sig = self._state_signature(child, player_index)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    frontier.append((hand_idx, child, cont_val))

        frontier.sort(key=lambda item: item[2], reverse=True)
        frontier = frontier[:beam_width]

        for depth in range(1, max_plays):
            if not frontier:
                break
            if is_live_play:
                active_beam = beam_width if depth < 6 else max(5, beam_width - 4)
                branch_limit = 8 if depth < 5 else 6
            else:
                active_beam = beam_width if depth < 4 else max(2, beam_width - 3)
                branch_limit = (
                    self.BRANCH_LIMIT
                    if depth < 4
                    else max(3, self.BRANCH_LIMIT - 1)
                )
            next_frontier: list[tuple[int, Game, float]] = []
            for first_action, state, _ in frontier:
                candidates = self._ordered_candidate_indices(
                    state, player_index, branch_limit
                )
                for hand_idx in candidates:
                    child = _fast_copy_game(state)
                    try:
                        child.play_card(player_index, hand_idx)
                        stop_val = self._evaluate_terminal_day(
                            child, player_index
                        )
                    except Exception:
                        continue
                    if stop_val > best_value + 1e-9:
                        best_value = stop_val
                        best_first_actions = [first_action]
                    elif (
                        abs(stop_val - best_value) <= 1e-9
                        and first_action not in best_first_actions
                    ):
                        best_first_actions.append(first_action)

                    if not child.players[player_index].asleep:
                        cont_val = stop_val + self._continuation_potential(
                            child, player_index
                        )
                        sig = self._state_signature(child, player_index)
                        if sig not in seen_signatures:
                            seen_signatures.add(sig)
                            next_frontier.append(
                                (first_action, child, cont_val)
                            )

            next_frontier.sort(key=lambda item: item[2], reverse=True)
            frontier = next_frontier[:active_beam]

        if not best_first_actions:
            return (
                initial_candidates[0] if initial_candidates else None
            ), best_value
        if None in best_first_actions:
            return None, best_value
        return self.rng.choice(best_first_actions), best_value

    def _state_signature(self, sim: Game, player_index: int) -> tuple:
        p = sim.players[player_index]
        return (
            p.energy,
            p.fun,
            tuple(c.title for c in p.played_today),
            tuple(sorted(c.title for c in p.hand)),
            tuple(c.title for c in p.tomorrow_cards),
        )

    def _continuation_potential(self, sim: Game, player_index: int) -> float:
        """Zero-copy estimate of upside from continuing to play cards today (keeps setup states in the beam)."""
        player = sim.players[player_index]
        if player.asleep or not player.hand:
            return 0.0

        active_after_tags: dict[str, float] = {}
        global_after_bonus = 0.0
        for played in player.played_today:
            prof = _build_profile(played)
            if (
                prof.is_multi_scaler
                and not prof.is_before_scaler
                and prof.signed_match_benefit > 0
            ):
                for t in prof.target_tags:
                    active_after_tags[t] = (
                        active_after_tags.get(t, 0.0)
                        + prof.signed_match_benefit
                    )
            elif prof.is_global_after_scaler:
                global_after_bonus += prof.signed_match_benefit

        # Also credit "Before" finishers sitting in player.hand that will score off played_today!
        pending_before_bonus = 0.0
        energy_budget = player.energy
        for hc in player.hand:
            h_prof = _build_profile(hc)
            if h_prof.is_before_scaler and hc.effective_cost <= energy_budget:
                if h_prof.target_tags:
                    matches_played = sum(
                        1
                        for p in player.played_today
                        if p.tags & h_prof.target_tags
                    )
                    pending_before_bonus += (
                        matches_played * h_prof.signed_match_benefit * 0.90
                    )
                    for t in h_prof.target_tags:
                        active_after_tags[t] = (
                            active_after_tags.get(t, 0.0)
                            + h_prof.signed_match_benefit * 0.70
                        )
                elif hc.title == "Evening on the Dock":
                    pending_before_bonus += len(player.played_today) * 0.90
                elif hc.title == "Couch Tube":
                    pending_before_bonus += (
                        sum(p.definition.cost for p in player.played_today)
                        * 0.85
                    )

        matching_playable = 0.0
        for card in player.hand:
            cost = card.effective_cost
            if cost <= energy_budget:
                matching_playable += global_after_bonus
                for t in card.tags:
                    matching_playable += active_after_tags.get(t, 0.0)
        return (
            min(player.energy, 7) * 0.72
            + matching_playable * 0.95
            + pending_before_bonus
        )

    def _evaluate_terminal_day(
        self, terminal_sim: Game, player_index: int
    ) -> float:
        """Evaluate a state where player_index goes to bed today."""
        player = terminal_sim.players[player_index]

        other_asleep = any(
            p.asleep
            for idx, p in enumerate(terminal_sim.players)
            if idx != player_index
        )
        first_to_bed_value = (
            self.FIRST_TO_BED_BONUS
            if (terminal_sim.day < DAY_COUNT and not other_asleep)
            else 0.0
        )

        needs_full_end_day = any(
            _build_profile(c).has_end_day_hook for c in player.played_today
        )
        eval_sim = terminal_sim
        if needs_full_end_day:
            scoring = _fast_copy_game(terminal_sim)
            if not scoring.players[player_index].asleep:
                scoring.go_to_bed(player_index)
            try:
                scoring.end_day()
                today_total_fun = float(scoring.players[player_index].fun)
                next_tomorrow_cards = scoring.players[
                    player_index
                ].tomorrow_cards
                eval_sim = scoring
            except Exception:
                today_total_fun = float(
                    player.fun
                    + sum(
                        terminal_sim.card_fun(player_index, c)
                        for c in player.played_today
                    )
                )
                next_tomorrow_cards = [
                    c
                    for c in player.played_today
                    if c.effective_behavior.has_tomorrow_action
                ]
        else:
            prev_scores = terminal_sim._fun_at_start_of_scoring
            terminal_sim._fun_at_start_of_scoring = tuple(
                p.fun for p in terminal_sim.players
            )
            scored_today = sum(
                terminal_sim.card_fun(player_index, c)
                for c in player.played_today
            )
            terminal_sim._fun_at_start_of_scoring = prev_scores
            today_total_fun = float(player.fun + scored_today)
            next_tomorrow_cards = [
                c
                for c in player.played_today
                if c.effective_behavior.has_tomorrow_action
            ]

        # Discount end-of-day opponent-comparison cards when opponents are still awake
        opponents_awake = any(
            not opp.asleep
            for idx, opp in enumerate(terminal_sim.players)
            if idx != player_index
        )
        if opponents_awake:
            for c in player.played_today:
                if c.title == "Fit to Print" and len(player.played_today) <= 4:
                    if all(
                        len(player.played_today) > len(opp.played_today)
                        for idx, opp in enumerate(terminal_sim.players)
                        if idx != player_index
                    ):
                        today_total_fun -= (
                            3.5 if len(player.played_today) <= 3 else 1.8
                        )
                elif c.title == "Play With the Kids" and len(player.hand) <= 2:
                    if all(
                        len(player.hand) > len(opp.hand)
                        for idx, opp in enumerate(terminal_sim.players)
                        if idx != player_index
                    ):
                        today_total_fun -= 2.2
                elif c.title == "Puerto Rico":
                    today_total_fun -= 1.2
                elif c.title == "San Juan":
                    t_idx = c.markers.get("target_player_index")
                    if (
                        isinstance(t_idx, int)
                        and t_idx != player_index
                        and not terminal_sim.players[t_idx].asleep
                    ):
                        today_total_fun -= 1.4
                elif c.title == "Cheesy Phone Game":
                    relax_opps = sum(
                        1
                        for idx, opp in enumerate(terminal_sim.players)
                        if idx != player_index
                        and any("Relax" in oc.tags for oc in opp.played_today)
                    )
                    if relax_opps < 2:
                        today_total_fun += 1.8

        if terminal_sim.day >= DAY_COUNT:
            return today_total_fun

        reserve_value = self._evaluate_tomorrow_and_reserve_fast(
            eval_sim,
            player_index,
            next_tomorrow_cards,
            days_left=DAY_COUNT - terminal_sim.day,
        )
        return today_total_fun + first_to_bed_value + reserve_value

    def _evaluate_tomorrow_and_reserve_fast(
        self,
        sim: Game,
        player_index: int,
        next_tomorrow_cards: Sequence[CardInstance],
        *,
        days_left: int,
    ) -> float:
        """Zero-copy evaluation of tomorrow_cards and held hand across remaining days."""
        player = sim.players[player_index]

        # 1. Evaluate active Tomorrow cards (next_tomorrow_cards)
        tom_fun = 0.0
        tom_energy_delta = 0
        tom_tag_bonuses: dict[str, float] = {}
        tom_all_tags = False
        for tc in next_tomorrow_cards:
            t_prof = _build_profile(tc)
            tom_fun += t_prof.tomorrow_start_fun
            tom_energy_delta += t_prof.tomorrow_start_energy
            if tc.title == "Lawn Maintenance" and player.energy == 0:
                tom_fun += 3.0
            elif tc.title == "Wash the Dishes" and len(player.hand) <= 1:
                tom_fun += 2.0
            elif tc.title == "Photo Shoot":
                tom_fun += 3.5

            if t_prof.signed_match_benefit != 0.0:
                for t in t_prof.target_tags:
                    tom_tag_bonuses[t] = (
                        tom_tag_bonuses.get(t, 0.0)
                        + t_prof.signed_match_benefit
                    )
                if t_prof.signed_match_benefit < 0.0:
                    tom_fun += t_prof.signed_match_benefit * 1.15
                else:
                    tom_fun += t_prof.signed_match_benefit * 0.50
            chosen_tag = tc.markers.get("chosen_tag")
            if isinstance(chosen_tag, str):
                tom_tag_bonuses[chosen_tag] = (
                    tom_tag_bonuses.get(chosen_tag, 0.0) + 1.2
                )
            if bool(getattr(tc.effective_behavior, "tomorrow_all_tags", False)):
                tom_all_tags = True

        tom_energy_value = (
            tom_energy_delta * 1.22
            if tom_energy_delta >= 0
            else tom_energy_delta * 1.15
        )
        if not player.hand:
            return tom_fun + tom_energy_value

        # 2. Evaluate held hand cards: distinguish synergy package cards from standalone cards
        char = player.character
        held_multi_scalers: list[tuple[CardInstance, CardSynergyProfile]] = []
        for c in player.hand:
            c_prof = _build_profile(c)
            if c_prof.is_multi_scaler or c_prof.is_global_after_scaler:
                held_multi_scalers.append((c, c_prof))

        held_scaler_benefit_by_tag: dict[str, float] = {}
        for scaler, s_prof in held_multi_scalers:
            if s_prof.is_multi_scaler and s_prof.signed_match_benefit > 0:
                for t in s_prof.target_tags:
                    held_scaler_benefit_by_tag[t] = (
                        held_scaler_benefit_by_tag.get(t, 0.0)
                        + s_prof.signed_match_benefit
                    )

        paired_partner_counts: dict[str, int] = {}
        seen_must_play_first = False
        surpluses: list[float] = []

        ordered_hand = sorted(
            player.hand,
            key=lambda c: (c.effective_cost, -c.effective_base_fun),
        )

        for c in ordered_hand:
            c_prof = _build_profile(c)
            cost = max(0, c.effective_cost)
            if char is not None:
                cost = max(0, char.modify_energy_cost(c.tags, cost))
            base_f = float(c.effective_base_fun + c_prof.immediate_fun_gain)
            if char is not None:
                base_f = float(
                    char.finalize_fun(
                        c.tags, char.modify_fun(c.tags, int(base_f))
                    )
                )

            if c_prof.must_play_first:
                if seen_must_play_first:
                    continue
                seen_must_play_first = True
                surpluses.append(0.45)
                continue

            if cost >= 8:
                support = max(
                    0, tom_energy_delta
                ) + _extra_energy_support_for_card(player, c)
                if days_left == 1 and cost > 7 + support:
                    continue
                over_budget = max(0, cost - (6 + support))
            else:
                over_budget = 0

            if c.title == "Würstchen":
                imm_e = max(3.0, float(len(player.hand)))
            elif (
                c_prof.conditional_energy_gain > 0
                and c.title
                not in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}
            ):
                has_trig = any(
                    other is not c
                    and (
                        other.definition.cost >= 5
                        if c.title == "Steak"
                        else bool(other.tags & c_prof.target_tags)
                    )
                    for other in player.hand
                )
                imm_e = c_prof.conditional_energy_gain * (
                    0.75 if has_trig else 0.35
                )
            else:
                imm_e = (
                    0.0
                    if c.title == "Pudding Chômeur"
                    else float(c_prof.immediate_energy_gain)
                )

            card_draw_val = (
                min(c_prof.immediate_cards_gained, 3)
                * (0.85 if cost <= 1 else 0.15)
                if (
                    not c_prof.penalizes_before
                    and not c_prof.penalizes_after
                    and not c_prof.is_multi_scaler
                )
                else 0.0
            )
            raw_val = (
                base_f
                + imm_e * 0.95
                + card_draw_val
                - over_budget * 1.50
            )
            is_synergy_piece = False

            # A. Matches an active tomorrow_card played today
            if tom_all_tags:
                pos_tom = sum(v for v in tom_tag_bonuses.values() if v > 0)
                if pos_tom > 0:
                    raw_val += pos_tom * 1.25
                    is_synergy_piece = True
            else:
                for t in c.tags:
                    tb = tom_tag_bonuses.get(t, 0.0)
                    if tb > 0:
                        raw_val += tb * 1.25
                        is_synergy_piece = True

            # B. Fuel partner for a held multi-card scaler in hand (keep scaler + partners together!)
            best_partner_tag: str | None = None
            best_partner_benefit = 0.0
            for t in c.tags:
                b_val = held_scaler_benefit_by_tag.get(t, 0.0)
                if b_val > best_partner_benefit:
                    best_partner_benefit = b_val
                    best_partner_tag = t

            if (
                best_partner_tag is not None
                and not c_prof.is_multi_scaler
                and days_left >= 1
            ):
                used = paired_partner_counts.get(best_partner_tag, 0)
                if used == 0:
                    paired_partner_counts[best_partner_tag] = 1
                    raw_val += best_partner_benefit * 1.15
                    is_synergy_piece = True
                elif used == 1:
                    paired_partner_counts[best_partner_tag] = 2
                    raw_val += best_partner_benefit * (
                        0.85 if days_left >= 2 else 0.55
                    )
                    is_synergy_piece = True
                elif used == 2 and days_left >= 2 and cost <= 1:
                    paired_partner_counts[best_partner_tag] = 3
                    raw_val += best_partner_benefit * 0.45
                    is_synergy_piece = True

            # C. This card IS a Today synergy scaler / finisher held for a combo turn
            if c_prof.is_multi_scaler:
                is_synergy_piece = True
                has_owned_match = any(
                    other is not c and (other.tags & c_prof.target_tags)
                    for other in player.hand
                )
                abundance = (
                    1.0 if has_owned_match else _tag_abundance(c_prof.target_tags)
                )
                wait_factor = (1.55 if days_left >= 2 else 0.90) * abundance
                raw_val += wait_factor * c_prof.signed_match_benefit
            elif c_prof.is_global_after_scaler:
                is_synergy_piece = True
                wait_factor = 2.10 if days_left >= 2 else 1.35
                raw_val += wait_factor * c_prof.signed_match_benefit
            elif c.title == "Couch Tube" and days_left >= 1:
                is_synergy_piece = True
                raw_val += 3.40 if days_left >= 2 else 2.40
            elif c.title == "Cheesy Phone Game" and days_left >= 1:
                raw_val += 1.50
            elif (
                c.title == "Epic Prank"
                and days_left >= 1
                and any(
                    "Item" in other.tags for other in player.hand if other is not c
                )
            ):
                is_synergy_piece = True
                raw_val += 3.20
            elif c_prof.is_tomorrow_scaler:
                if days_left >= 2:
                    raw_val += 0.50 * c_prof.signed_match_benefit
            elif c_prof.is_single_buff and days_left >= 1:
                has_partner_now = any(
                    other is not c and (other.tags & c_prof.target_tags)
                    for other in player.hand
                )
                is_synergy_piece = True
                raw_val += (
                    0.85 if has_partner_now else 0.45
                ) * c_prof.signed_match_benefit
            elif c_prof.is_count_finisher and base_f <= 2 and days_left >= 1:
                is_synergy_piece = True
                raw_val += 1.50 if days_left >= 2 else 0.75

            net_energy_cost = max(0.0, cost - imm_e)
            if is_synergy_piece:
                energy_tax = (
                    min(net_energy_cost, 3) * 0.42
                    + max(0, net_energy_cost - 3) * 0.88
                )
            else:
                energy_tax = net_energy_cost * 0.92
            surplus = raw_val - energy_tax
            if surplus > 0.0:
                surpluses.append(surplus)

        surpluses.sort(reverse=True)
        has_combo_or_energy = bool(held_multi_scalers) or tom_energy_delta >= 2
        slot_weights = (
            (1.0, 1.0, 0.85, 0.55)
            if has_combo_or_energy
            else (1.0, 0.92, 0.50, 0.25)
        )
        held_value = sum(
            s * (slot_weights[i] if i < len(slot_weights) else 0.08)
            for i, s in enumerate(surpluses)
        )

        return (
            tom_fun
            + tom_energy_value
            + self.RESERVE_TOMORROW_WEIGHT * held_value
        )

    # ------------------------------------------------------------------
    # Effect & Target Decisions
    # ------------------------------------------------------------------

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _EVAL_AI_SINGLETON.choose_card_to_discard(
            game, player_index, options
        )

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> Sequence[int]:
        return _EVAL_AI_SINGLETON.choose_cards_to_discard(
            game,
            player_index,
            options,
        )

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if (
            self._evaluation_depth >= 1
            or len(options) <= 1
            or len(game.players[player_index].played_today) >= 5
        ):
            return _EVAL_AI_SINGLETON.choose_card_to_copy(
                game, player_index, options
            )

        base_sim = self._prepare_observable_game(game, player_index)
        player = base_sim.players[player_index]
        source_card = player.played_today[-1] if player.played_today else None
        if source_card is None:
            return _EVAL_AI_SINGLETON.choose_card_to_copy(
                game, player_index, options
            )

        pay_cost = (
            type(source_card.effective_behavior).__name__ != "LeftoversBehavior"
        )
        best_idx = 0
        best_val = float("-inf")
        for idx, option in enumerate(options):
            sim = _fast_copy_game(base_sim)
            sim_player = sim.players[player_index]
            sim_source = sim_player.played_today[-1]
            try:
                sim.copy_card_effect(
                    player_index,
                    _clone_card(option),
                    sim_source,
                    pay_source_cost=pay_cost,
                )
                _, val = self._search_best_day_plan(
                    sim, player_index, beam_width=3, max_plays=3
                )
            except Exception:
                val = _quick_card_desirability(
                    game,
                    player_index,
                    option,
                    _wanted_tags_for_player(player),
                )
            if val > best_val:
                best_val = val
                best_idx = idx
        return best_idx

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _EVAL_AI_SINGLETON.choose_card_target(
            game, player_index, options
        )

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _EVAL_AI_SINGLETON.choose_suitcase_target(
            game, player_index, options
        )

    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        return _EVAL_AI_SINGLETON.choose_player(
            game, player_index, eligible_player_indices
        )

    def choose_tag(
        self,
        game: Game,
        player_index: int,
        options: Sequence[str],
    ) -> str:
        return _EVAL_AI_SINGLETON.choose_tag(game, player_index, options)

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        if maximum <= 0:
            return 0
        if (
            self._evaluation_depth >= 1
            or len(game.players[player_index].played_today) >= 5
        ):
            return maximum
        base_sim = self._prepare_observable_game(game, player_index)
        player = base_sim.players[player_index]
        source_card = player.played_today[-1] if player.played_today else None
        if source_card is None:
            return maximum

        best_spend = maximum
        best_val = float("-inf")
        for spend in range(maximum + 1):
            sim = _fast_copy_game(base_sim)
            sim_player = sim.players[player_index]
            sim_player.energy -= spend
            sim_player.played_today[-1].markers["energy_cubes"] = spend
            _, val = self._search_best_day_plan(
                sim, player_index, beam_width=3, max_plays=3
            )
            if val >= best_val:
                best_val = val
                best_spend = spend
        return best_spend

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        if "Quacks" in action and game.stats is not _get_shared_sim_stats():
            player = game.players[player_index]
            last_card = player.played_today[-1] if player.played_today else None
            if last_card is None:
                return False
            key = (id(game), game.day, len(player.played_today), id(last_card))
            if self._quacks_card_key != key:
                self._quacks_card_key = key
                self._quacks_draws = 0
            self._quacks_draws += 1
            return self._quacks_draws <= 2
        return _EVAL_AI_SINGLETON.choose_optional_action(
            game, player_index, action
        )

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        return _EVAL_AI_SINGLETON.order_cards_for_trunk(
            game, player_index, cards
        )

    def choose_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        return _EVAL_AI_SINGLETON.choose_cards_for_trunk(
            game, player_index, cards
        )

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> bool:
        player = game.players[player_index]
        if player.energy < 1:
            return False
        wanted_tags = _wanted_tags_for_player(player)
        best_desirability = max(
            (
                _quick_card_desirability(game, player_index, c, wanted_tags)
                for c in options
            ),
            default=0.0,
        )
        return player.energy >= 2 or best_desirability >= 2.5


ArgonbrainAI = ArgonBrain
