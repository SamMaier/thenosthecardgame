"""Aggressive, multi-big-day opponent-aware lookahead policy (HydrogenAI)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from operator import attrgetter
from typing import Any, Sequence, TYPE_CHECKING
import random

from thenos.ais.argonbrain import (
    CardSynergyProfile,
    DAY_COUNT,
    SUITCASE_SIZE,
    _TRUNK_REORDER_TITLES,
    _clone_card,
    _clone_random,
    _condition_met_in_played,
    _get_shared_sim_stats,
    _get_sim_known_trunk_top,
    _tag_abundance,
    _unseen_tag_counts_and_total,
)
from thenos.ais.beryllium import (
    _ACQUIRER_TITLES,
    _HIGH_COPY_FOOD_TITLES,
    _NEUTRAL_DRAFT_DEFNS,
    _NOOP_COUNTER,
    _STOCHASTIC_PLAY_TITLES,
    _TAGGED_NEUTRAL_3_BY_TAG,
    _allows_energy_gain_today,
    _beryllium_quick_card_desirability,
    _contextual_copy_value,
    _effective_source_definition,
    _evaluate_istanbul_discards,
    _istanbul_energy_per_discard,
    _istanbul_extra_energy_support,
    _select_bohnanza_discards,
)
from thenos.ais.carbon import (
    _ALL_TAG_BONUS_KEYS,
    _CARBON_PROFILE_BY_DEFN_ID,
    _QUICK_BRANCH_TITLES,
    _QUICK_PENALTY_BY_TITLE,
    _RESERVE_DYNAMIC_EXTRA_1,
    _RESERVE_DYNAMIC_EXTRA_2,
    _RESERVE_EMPIRICAL_FLAT,
    _RESERVE_FLAT_EXTRA,
    _TERMINAL_AWAKE_OPP_TITLES,
    CarbonAI,
    _CarbonEvalAI,
    _beryllium_build_profile,
    _carbon_extra_energy_support,
    _carbon_free_play_value,
    _carbon_own_suitcase_markers,
    _effective_card_title,
    _ensure_carbon_caches,
    _get_carbon_sim_game_cls,
    _get_known_trunk_top_any,
    _opponent_suitcase_markers,
    _suitcase_denial_value,
    _wanted_tags_for_player,
)
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.energy_effects import (
    AfterDinnerEntertainmentBehavior,
    EnergyForAllCardsAfterBehavior,
    EnergyForNextTagBehavior,
    EnergyForTagAfterBehavior,
    HalfEnergyForNextTagBehavior,
    MedicalAdviceBehavior,
    SetEnergyForNextTagBehavior,
    TomorrowEnergyForTagBehavior,
)
from thenos.cards.fun_effects import (
    FunAndEnergyForAllCardsAfterBehavior,
    FunAndEnergyForTagAfterBehavior,
    FunForAllCardsAfterDoubleBehavior,
    FunForAllCardsBeforeBehavior,
    FunForNextCardBehavior,
    FunForNextTagBehavior,
    FunForNextTagWrittenCostBehavior,
    FunForNthTagDoubleBehavior,
    FunForTagAfterBehavior,
    FunForTagBeforeBehavior,
    FunForTagsBeforeAndAfterBehavior,
    NosBookBehavior,
    PickTwoAndFunForSocialAfterBehavior,
    TomorrowFunForTagBehavior,
    WaterTrampolineBehavior,
)
from thenos.cards.items import FancyFloatieBehavior
from thenos.cards.relax import (
    FancyCraftBehavior,
    TanningBehavior,
    WildlifeSpottingBehavior,
)
from thenos.cards.social import (
    DatesFirstNosBehavior,
    ScoutTheOtherCottagesBehavior,
)
from thenos.models import PlayerState

if TYPE_CHECKING:
    from thenos.game import Game


_HYDROGEN_RESERVE_EVAL_CACHE: dict[tuple, float] = {}
_HYDROGEN_ISTANBUL_CACHE: dict[tuple, list[tuple[float, tuple[int, ...]]]] = {}
_HYDROGEN_SIM_GAME_CLS: type | None = None
_ALL_TAGS_CACHED: frozenset[str] | None = None

# Morning Coffee requires being the first card played today, so Leftovers (played
# second or later) can never legally copy it.
_HYDROGEN_COPY_FOOD_TITLES: frozenset[str] = (
    _HIGH_COPY_FOOD_TITLES - frozenset({"Morning Coffee"})
) | frozenset({"Grocery Store Run", "Address the Food"})

_EARLY_COPYABLE_FOOD_TITLES: frozenset[str] = frozenset(
    {
        "Rouladen",
        "High-End Red",
        "High-End White",
        "Afternoon Coffee",
        "Fajitas",
        "Butter Tarts",
        "Würstchen",
        "Calzones",
        "Ice Cream Sandwich",
        "Doritos",
        "Potato Pancakes",
    }
)

_TAG_NAMES: tuple[str, ...] = (
    "Board Game",
    "Exercise",
    "Food",
    "Indoors",
    "Item",
    "Outdoors",
    "Relax",
    "Social",
)
_TAG_BIT: dict[str, int] = {t: i for i, t in enumerate(_TAG_NAMES)}
_TAG_MASK: dict[str, int] = {t: 1 << i for i, t in enumerate(_TAG_NAMES)}
_MASK_BOARD_GAME = _TAG_MASK["Board Game"]
_MASK_EXERCISE = _TAG_MASK["Exercise"]
_MASK_FOOD = _TAG_MASK["Food"]
_MASK_INDOORS = _TAG_MASK["Indoors"]
_MASK_ITEM = _TAG_MASK["Item"]
_MASK_OUTDOORS = _TAG_MASK["Outdoors"]
_MASK_RELAX = _TAG_MASK["Relax"]
_MASK_SOCIAL = _TAG_MASK["Social"]
_ALL_TAGS_BITMASK = (1 << len(_TAG_NAMES)) - 1

_FTB_ZERO_E_STAY: tuple[float, ...] = (
    0.0,
    1.0 - 0.85,
    1.0 - (0.85**2),
    1.0 - (0.85**3),
    1.0 - (0.85**4),
)

# Per-player board capability flags on _HydrogenSimGame._p_flags
_PF_ALL_TAGS = 1 << 0
_PF_HAS_TOM_PLAY = 1 << 1
_PF_HAS_CARD_PLAY = 1 << 2
_PF_HAS_ACQUIRE = 1 << 3
_PF_HAS_PICK = 1 << 4
_PF_HAS_ALLOWS_E = 1 << 5
_PF_HAS_TOM_COST_MOD = 1 << 6
_PF_HAS_PLAYED_COST_MOD = 1 << 7
_PF_HAS_COST_MOD = _PF_HAS_TOM_COST_MOD | _PF_HAS_PLAYED_COST_MOD
_PF_HAS_TOM_FUN_MOD = 1 << 8
_PF_HAS_END_DAY = 1 << 9
_PF_HAS_LOOP2 = 1 << 10
_PF_HAS_AWAKE_OPP_TERM = 1 << 11
_PF_HAS_FUN_MOD = 1 << 12
_PF_HAS_NONADD_FUN_MOD = 1 << 13
_PF_HAS_CONT_AFTER = 1 << 14
_PF_HAS_TOM_ACTION = 1 << 15
_PF_HAS_COPIED = 1 << 16
_PF_HAS_SIG_MARKER = 1 << 17
_PF_HAS_CUSTOM_TOM_FUN_MOD = 1 << 18

_MASK_TO_PACKED_CNT: tuple[int, ...] = tuple(
    sum(1 << (b << 3) for b in range(8) if (m & (1 << b))) for m in range(256)
)
_ALL_TAGS_PACKED_CNT: int = _MASK_TO_PACKED_CNT[_ALL_TAGS_BITMASK]

# Compiled modify_tomorrow_fun opcodes
_TMF_NONE = 0
_TMF_GENERIC = 1
_TMF_TAG_ADD = 2
_TMF_FIRST_OUTDOORS = 3
_TMF_ALL_ADD = 4

# Compiled modify_energy_cost / modify_tomorrow_energy_cost opcodes
_CM_NONE = 0
_CM_GENERIC = 1
_CM_TAG_ADD = 2
_CM_ALL_ADD = 3
_CM_ALL_MUL2 = 4
_CM_NEXT_TAG_ADD = 5
_CM_NEXT_TAG_SET = 6
_CM_NEXT_TAG_HALF = 7

# Reserve immediate energy modes
_RIE_STATIC = 0
_RIE_WURSTCHEN = 1
_RIE_ISTANBUL = 2
_RIE_STEAK = 3
_RIE_COND_MASK = 4

# Compiled modify_fun opcodes for score_played_today_fast
_MF_NONE = 0
_MF_GENERIC = 1
_MF_AFTER_TAG_ADD = 2
_MF_AFTER_ALL_ADD = 3
_MF_AFTER_ALL_MUL2 = 4
_MF_BEFORE_TAG_ADD = 5
_MF_BEFORE_ALL_ADD = 6
_MF_OTHER_TAGS_ADD = 7
_MF_NEXT_CARD_MUL = 8
_MF_NEXT_TAG_ADD = 9
_MF_NEXT_TAG_COST_ADD = 10
_MF_NEXT_TAG_MUL2 = 11
_MF_NTH_TAG_MUL2 = 12
_MF_NOS_BOOK = 13

# Resolved active today_mod opcodes during score_played_today_fast
_RMF_GENERIC = 1
_RMF_AFTER_TAG_ADD = 2
_RMF_AFTER_ALL_ADD = 3
_RMF_AFTER_ALL_MUL2 = 4
_RMF_BEFORE_TAG_ADD = 5
_RMF_BEFORE_ALL_ADD = 6
_RMF_OTHER_TAGS_ADD = 7
_RMF_EXACT_POS_ADD = 8
_RMF_EXACT_POS_MUL = 9

# Carbon quick desirability title dispatch codes
_CQ_NONE = 0
_CQ_NOS_BOOK = 1
_CQ_GROUP_DINNER = 2
_CQ_TELL_A_STORY = 3
_CQ_ASSORTED_CUTLERY = 4
_CQ_BOUGIE_COFFEE_MACHINE = 5
_CQ_MEDICAL_ADVICE = 6
_CQ_SCOUT_OTHER_COTTAGES = 7
_CQ_BOHNANZA = 8
_CQ_SKI_ON_SHOULDERS = 9
_CQ_CHEESY_PHONE_GAME = 10
_CQ_WAKESURF = 11
_CQ_SLALOM_START = 12

_CQ_BY_TITLE: dict[str, int] = {
    "Nos Book": _CQ_NOS_BOOK,
    "Group Dinner": _CQ_GROUP_DINNER,
    "Tell a Story": _CQ_TELL_A_STORY,
    "Assorted Cutlery": _CQ_ASSORTED_CUTLERY,
    "Bougie Coffee Machine": _CQ_BOUGIE_COFFEE_MACHINE,
    "Medical Advice": _CQ_MEDICAL_ADVICE,
    "Scout Other Cottages": _CQ_SCOUT_OTHER_COTTAGES,
    "Bohnanza": _CQ_BOHNANZA,
    "Ski on Shoulders": _CQ_SKI_ON_SHOULDERS,
    "Cheesy Phone Game": _CQ_CHEESY_PHONE_GAME,
    "Wakesurf": _CQ_WAKESURF,
    "Drop a Ski": _CQ_SLALOM_START,
}

# Hydrogen quick desirability title dispatch codes
_HQ_NONE = 0
_HQ_CHUCK_A_FRISBEE = 1
_HQ_CITIES_AND_KNIGHTS = 2
_HQ_WILDLIFE_SPOTTING = 3
_HQ_WORK_CALL = 4
_HQ_FLOATING = 5
_HQ_MAKE_PLANS = 6
_HQ_COUCH_TUBE = 7
_HQ_HEAVY_TRIO = 8
_HQ_KEEPER = 9
_HQ_FORCED_FAMILY_FUN = 10
_HQ_HIGH_END_RED = 11
_HQ_JOHNNY_APPLESEED = 12
_HQ_STRETCH = 13
_HQ_DOXOLOGY = 14
_HQ_BUG_SPRAY = 15
_HQ_SCRABBLE = 16
_HQ_SUNRISE = 17
_HQ_EVENING_CHAT = 18
_HQ_PAINT = 19
_HQ_CLIFF_CLIMBING = 20
_HQ_WATER_VOLLEYBALL = 21
_HQ_BRACELET_MAKING = 22

_HQ_BY_TITLE: dict[str, int] = {
    "Chuck a Frisbee": _HQ_CHUCK_A_FRISBEE,
    "Cities and Knights": _HQ_CITIES_AND_KNIGHTS,
    "Wildlife Spotting": _HQ_WILDLIFE_SPOTTING,
    "Work Call": _HQ_WORK_CALL,
    "Floating": _HQ_FLOATING,
    "Make Plans": _HQ_MAKE_PLANS,
    "Couch Tube": _HQ_COUCH_TUBE,
    "Wakeboard": _HQ_HEAVY_TRIO,
    "Risk": _HQ_HEAVY_TRIO,
    "Screamer Battle": _HQ_HEAVY_TRIO,
    "Keeper": _HQ_KEEPER,
    "Forced Family Fun": _HQ_FORCED_FAMILY_FUN,
    "High-End Red": _HQ_HIGH_END_RED,
    "Johnny Appleseed": _HQ_JOHNNY_APPLESEED,
    "Stretch": _HQ_STRETCH,
    "Doxology": _HQ_DOXOLOGY,
    "Bug Spray": _HQ_BUG_SPRAY,
    "Scrabble": _HQ_SCRABBLE,
    "Sunrise": _HQ_SUNRISE,
    "Evening Chat": _HQ_EVENING_CHAT,
    "Paint": _HQ_PAINT,
    "Cliff Climbing": _HQ_CLIFF_CLIMBING,
    "Water Volleyball": _HQ_WATER_VOLLEYBALL,
    "Bracelet Making": _HQ_BRACELET_MAKING,
}

# Reserve dynamic adjustment codes
_RD1_NONE = 0
_RD1_LEFTOVERS = 1
_RD1_PADDLEBOAT = 2
_RD1_POTATO_PANCAKES = 3
_RD1_KEEPER = 4
_RD1_CAMP_CROSSROADS = 5
_RD1_PUDDING_CHOMEUR = 6

_RD1_BY_TITLE: dict[str, int] = {
    "Leftovers": _RD1_LEFTOVERS,
    "Paddleboat": _RD1_PADDLEBOAT,
    "Potato Pancakes": _RD1_POTATO_PANCAKES,
    "Keeper": _RD1_KEEPER,
    "Camp Crossroads Dessert": _RD1_CAMP_CROSSROADS,
    "Pudding Chômeur": _RD1_PUDDING_CHOMEUR,
}

_RD2_NONE = 0
_RD2_ASSORTED_CUTLERY = 1
_RD2_SKI_ON_SHOULDERS = 2
_RD2_ADVENTURE_RACE = 3
_RD2_WURSTCHEN = 4
_RD2_MEDICAL_ADVICE = 5
_RD2_EMPTY_HAND_PAIR = 6

_RD2_BY_TITLE: dict[str, int] = {
    "Assorted Cutlery": _RD2_ASSORTED_CUTLERY,
    "Ski on Shoulders": _RD2_SKI_ON_SHOULDERS,
    "Adventure Race": _RD2_ADVENTURE_RACE,
    "Würstchen": _RD2_WURSTCHEN,
    "Medical Advice": _RD2_MEDICAL_ADVICE,
    "Wakesurf": _RD2_EMPTY_HAND_PAIR,
    "Drop a Ski": _RD2_EMPTY_HAND_PAIR,
}

_CUTLERY_SYNERGY_TITLES: frozenset[str] = _TRUNK_REORDER_TITLES | frozenset(
    {"Boat Ride", "Fancy Craft", "Canoe"}
)
_CUTLERY_REAL_COMBO_TITLES: frozenset[str] = frozenset(
    {
        "Read the Radar",
        "Fishing Boat",
        "Ski Boat",
        "Grocery Store Run",
        "Boat Ride",
        "Fancy Craft",
    }
)
_DUMP_HELPER_TITLES: frozenset[str] = frozenset(
    {"Paddleboard", "Navy SEALing", "Istanbul", "Scrabble"}
)
_ISTANBUL_SINK_HAND_TITLES: frozenset[str] = frozenset(
    {"Kayak", "Decaf", "Wakesurf", "Adventure Race", "Wakeboard"}
)
_ISTANBUL_SIMPLE_MARKER_KEYS: frozenset[str] = frozenset(
    {"_gave_energy", "_all_tags"}
)
_ISTANBUL_OPP_SENSITIVE_SLUGS: frozenset[str] = frozenset(
    {
        "cheesy-phone-game",
        "christmas-name-draw",
        "fit-to-print",
        "group-dinner",
        "play-with-the-kids",
        "san-juan",
        "sing-song",
        "ski-on-shoulders",
    }
)
_SUITCASE_TOUCH_TITLES: frozenset[str] = frozenset(
    {"Scout Other Cottages", "Last Year's Shirt", "Puerto Rico"}
)
_BLIND_TRUNK_STEP_TITLES: frozenset[str] = frozenset(
    {"Assorted Cutlery", "Poker", "Quacks"}
)
_TRUNK_CONSUMER_TITLES: frozenset[str] = frozenset(
    {
        "Assorted Cutlery",
        "Quacks",
        "Poker",
        "Bougie Coffee Machine",
        "Booby Prize",
    }
)

# Candidate ordering special title dispatch codes
_OQ_NONE = 0
_OQ_COLOURING = 1
_OQ_NOS_BOOK = 2
_OQ_EPIC_DUELS = 3
_OQ_GROUP_DINNER = 4
_OQ_SING_SONG = 5
_OQ_THE_CREW = 6
_OQ_WEDDING_ANNIVERSARY = 7
_OQ_MEDICAL_ADVICE = 8
_OQ_TELL_A_STORY = 9
_OQ_TRUNK_REORDER_4 = 10
_OQ_ASSORTED_CUTLERY = 11
_OQ_QUACKS = 12
_OQ_POKER = 13
_OQ_SHIRT_SCRABBLE = 14
_OQ_SOLO = 15
_OQ_SAN_JUAN = 16
_OQ_CITIES_AND_KNIGHTS = 17
_OQ_ADDRESS_THE_FOOD = 18
_OQ_SUITCASE_SWEEPER_2 = 19
_OQ_PADDLEBOARD = 20
_OQ_ISTANBUL = 21
_OQ_SHADY_SPOT = 22

_OQ_BY_TITLE: dict[str, int] = {
    "Colouring": _OQ_COLOURING,
    "Nos Book": _OQ_NOS_BOOK,
    "Epic Duels": _OQ_EPIC_DUELS,
    "Group Dinner": _OQ_GROUP_DINNER,
    "Sing Song": _OQ_SING_SONG,
    "The Crew": _OQ_THE_CREW,
    "Wedding Anniversary": _OQ_WEDDING_ANNIVERSARY,
    "Medical Advice": _OQ_MEDICAL_ADVICE,
    "Tell a Story": _OQ_TELL_A_STORY,
    "Read the Radar": _OQ_TRUNK_REORDER_4,
    "Grocery Store Run": _OQ_TRUNK_REORDER_4,
    "Fishing Boat": _OQ_TRUNK_REORDER_4,
    "Ski Boat": _OQ_TRUNK_REORDER_4,
    "Assorted Cutlery": _OQ_ASSORTED_CUTLERY,
    "Quacks": _OQ_QUACKS,
    "Poker": _OQ_POKER,
    "Last Year's Shirt": _OQ_SHIRT_SCRABBLE,
    "Scrabble": _OQ_SHIRT_SCRABBLE,
    "Solo": _OQ_SOLO,
    "San Juan": _OQ_SAN_JUAN,
    "Cities and Knights": _OQ_CITIES_AND_KNIGHTS,
    "Address the Food": _OQ_ADDRESS_THE_FOOD,
    "Treat Store Run": _OQ_SUITCASE_SWEEPER_2,
    "Fancy Floatie": _OQ_SUITCASE_SWEEPER_2,
    "Paddleboard": _OQ_PADDLEBOARD,
    "Istanbul": _OQ_ISTANBUL,
    "Shady Spot": _OQ_SHADY_SPOT,
}

# Quick desirability synergy / global mode codes
_QSYN_NONE = 0
_QSYN_MULTI = 1
_QSYN_TOMORROW = 2
_QSYN_SINGLE = 3
_QSYN_NEG_SMB = 4

_QGLOB_NONE = 0
_QGLOB_AFTER = 1
_QGLOB_BRING_A_FRIEND = 2
_QGLOB_DOCK = 3
_QGLOB_COUCH_TUBE = 4

_RESERVE_RANK_GETTER = attrgetter("reserve_rank")
_CHAR_FAST_CACHE: dict[int, tuple[Any, int, int, int, int, int, int, bool]] = {}
_COND_FAST_CACHE: dict[int, tuple[Any, int, int, bool, bool]] = {}


def _get_char_fast(
    char: Any,
) -> tuple[Any, int, int, int, int, int, int, bool]:
    cid = id(char)
    entry = _CHAR_FAST_CACHE.get(cid)
    if entry is not None and entry[0] is char:
        return entry
    ft = char.fun_tag
    zt = char.zero_fun_tag
    et = char.energy_tag
    f_mask = _TAG_MASK.get(ft, 0) if ft else 0
    z_mask = _TAG_MASK.get(zt, 0) if zt else 0
    z_bit = _TAG_BIT.get(zt, -2) if zt else -2
    e_mask = _TAG_MASK.get(et, 0) if et else 0
    is_custom = (
        (bool(ft) and not f_mask)
        or (bool(zt) and not z_mask)
        or (bool(et) and not e_mask)
    )
    entry = (
        char,
        f_mask,
        char.fun_delta,
        z_mask,
        z_bit,
        e_mask,
        char.energy_delta,
        is_custom,
    )
    if len(_CHAR_FAST_CACHE) > 256:
        _CHAR_FAST_CACHE.clear()
    _CHAR_FAST_CACHE[cid] = entry
    return entry


def _get_cond_fast(cond: Any) -> tuple[Any, int, int, bool, bool]:
    cid = id(cond)
    entry = _COND_FAST_CACHE.get(cid)
    if entry is not None and entry[0] is cond:
        return entry
    ft = cond.fun_tag
    f_mask = _TAG_MASK.get(ft, 0) if ft else 0
    is_custom = bool(ft) and not f_mask
    entry = (cond, f_mask, cond.fun_delta, bool(cond.hot_day), is_custom)
    if len(_COND_FAST_CACHE) > 256:
        _COND_FAST_CACHE.clear()
    _COND_FAST_CACHE[cid] = entry
    return entry


def _is_plain_hand_markers(m: dict[str, object]) -> bool:
    return not m or (
        "_copied_definition" not in m
        and not ("_all_tags" in m and m["_all_tags"])
    )


def _tags_to_mask(tags: frozenset[str]) -> tuple[int, bool]:
    mask = 0
    all_ok = True
    for t in tags:
        b = _TAG_MASK.get(t)
        if b is None:
            all_ok = False
        else:
            mask |= b
    return mask, all_ok


@dataclass(frozen=True, slots=True)
class _HydrogenDefnMeta:
    defn: CardDefinition
    prof: CardSynergyProfile
    beh: CardBehavior
    slug: str
    title: str
    cost: int
    base_fun: int
    tags: frozenset[str]
    sorted_tags: tuple[str, ...]
    tag_mask: int
    tags_canonical: bool
    target_tags: frozenset[str]
    sorted_target_tags: tuple[str, ...]
    target_mask: int
    target_canonical: bool
    target_single_bit: int
    tag_abundance_target: float
    play_order_bucket: int
    immediate_energy_gain: int
    conditional_energy_gain: int
    immediate_fun_gain: int
    conditional_fun_bonus: int
    immediate_cards_gained: int
    energy_delta: int
    signed_match_benefit: float
    tomorrow_start_fun: int
    tomorrow_start_energy: int
    must_play_first: bool
    must_play_late: bool
    goes_to_tomorrow: bool
    is_multi_scaler: bool
    is_before_scaler: bool
    is_tomorrow_scaler: bool
    is_single_buff: bool
    is_global_after_scaler: bool
    is_count_finisher: bool
    penalizes_before: bool
    penalizes_after: bool
    ends_day_immediately: bool
    has_can_play: bool
    has_own_cost_mod: bool
    has_cost_mod: bool
    has_tom_cost_mod: bool
    has_fun_val: bool
    has_fun_mod: bool
    has_tom_fun_mod: bool
    has_on_play: bool
    has_on_card_play: bool
    has_on_tom_card_play: bool
    has_on_acquire: bool
    has_on_pick: bool
    has_end_day_any: bool
    has_allows_extra: bool
    has_allows_energy_gain: bool
    tomorrow_all_tags: bool
    after_all_tags: bool
    has_tomorrow_action: bool
    touches_suitcase_on_play: bool
    played_board_flags: int
    tom_board_flags: int
    mod_fun_kind: int
    mod_fun_tag: str
    mod_fun_tags: frozenset[str]
    mod_fun_mask: int
    mod_fun_param: int
    mod_fun_ord: int
    tom_mod_fun_kind: int
    tom_mod_fun_mask: int
    tom_mod_fun_param: int
    cost_mod_kind: int
    cost_mod_mask: int
    cost_mod_param: int
    tom_cost_mod_kind: int
    tom_cost_mod_mask: int
    tom_cost_mod_param: int
    tag_packed_cnt: int
    sorted_tag_bits: tuple[int, ...]
    wanted_tom_deltas: tuple[tuple[str, float], ...]
    wanted_played_deltas: tuple[tuple[str, float], ...]
    wanted_hand_deltas: tuple[tuple[str, float], ...]
    wanted_idx_tom_deltas: tuple[tuple[int, float], ...]
    wanted_idx_played_deltas: tuple[tuple[int, float], ...]
    wanted_idx_hand_deltas: tuple[tuple[int, float], ...]
    c_qcode: int
    h_qcode: int
    order_qcode: int
    quick_penalty: float
    has_quick_draw: bool
    quick_draw_cheap: float
    quick_draw_exp: float
    quick_sleep_or_late: int
    quick_syn_kind: int
    quick_neg_smb_pen: float
    quick_global_kind: int
    quick_tom_base: float
    quick_tom_extra: float
    is_fodder_plain: bool
    is_sink_hand: bool
    is_sink_played: bool
    is_dump_helper: bool
    is_baf_sink: bool
    reserve_sort_key: tuple[int, int, str]
    reserve_rank: int
    reserve_base_f: float
    reserve_base_f_int: int
    reserve_imm_e_mode: int
    reserve_static_imm_e: float
    reserve_draw_cheap: float
    reserve_draw_exp: float
    reserve_has_syn_branch: bool
    reserve_flat1: float | None
    reserve_flat2: float | None
    reserve_dyn1_code: int
    reserve_dyn2_code: int
    has_reserve_extra: bool
    reserve_colouring_extra: float
    reserve_adv_race_sup: int
    istanbul_e_per: int
    is_copy_food: bool
    is_acquirer: bool
    is_cutlery_combo: bool
    is_cutlery_real_combo: bool
    can_give_esup_hand: bool
    played_tom_esup: int
    q_fun_term: float
    q_imm_e_term: float
    q_early_flags: bool
    q_late_flags: bool


_HYDROGEN_META_BY_DEFN_ID: dict[int, _HydrogenDefnMeta] = {}
_HYDROGEN_META_BY_SLUG: dict[str, _HydrogenDefnMeta] = {}
_HYDROGEN_COPIED_META_CACHE: dict[tuple, _HydrogenDefnMeta] = {}


def _compile_mod_fun(
    beh: CardBehavior, has_fun_mod: bool
) -> tuple[int, str, frozenset[str], int, int]:
    if not has_fun_mod:
        return (_MF_NONE, "", frozenset(), 0, 0)
    bt = type(beh)
    if (
        bt is FunForTagAfterBehavior
        or bt is PickTwoAndFunForSocialAfterBehavior
        or bt is TanningBehavior
    ):
        return (_MF_AFTER_TAG_ADD, beh.tag, frozenset(), beh.bonus, 0)  # type: ignore[attr-defined]
    if bt is FunAndEnergyForTagAfterBehavior:
        return (_MF_AFTER_TAG_ADD, beh.tag, frozenset(), beh.fun_bonus, 0)  # type: ignore[attr-defined]
    if bt is FunAndEnergyForAllCardsAfterBehavior:
        return (_MF_AFTER_ALL_ADD, "", frozenset(), beh.fun_bonus, 0)  # type: ignore[attr-defined]
    if bt is FunForAllCardsAfterDoubleBehavior:
        return (_MF_AFTER_ALL_MUL2, "", frozenset(), 2, 0)
    if bt is FunForTagBeforeBehavior:
        return (_MF_BEFORE_TAG_ADD, beh.tag, frozenset(), beh.bonus, 0)  # type: ignore[attr-defined]
    if bt is FunForAllCardsBeforeBehavior or bt is ScoutTheOtherCottagesBehavior:
        return (_MF_BEFORE_ALL_ADD, "", frozenset(), beh.bonus, 0)  # type: ignore[attr-defined]
    if bt is MedicalAdviceBehavior:
        return (_MF_BEFORE_ALL_ADD, "", frozenset(), -1, 0)
    if bt is FunForTagsBeforeAndAfterBehavior:
        return (_MF_OTHER_TAGS_ADD, "", beh.tags, beh.bonus, 0)  # type: ignore[attr-defined]
    if bt is FunForNextCardBehavior:
        return (_MF_NEXT_CARD_MUL, "", frozenset(), beh.multiplier, 0)  # type: ignore[attr-defined]
    if bt is FunForNextTagBehavior:
        return (_MF_NEXT_TAG_ADD, beh.tag, frozenset(), beh.bonus, 0)  # type: ignore[attr-defined]
    if bt is AfterDinnerEntertainmentBehavior:
        return (_MF_NEXT_TAG_ADD, "Social", frozenset(), 2, 0)
    if bt is FunForNextTagWrittenCostBehavior or bt is FancyCraftBehavior:
        return (_MF_NEXT_TAG_COST_ADD, beh.tag, frozenset(), beh.multiplier, 0)  # type: ignore[attr-defined]
    if bt is WaterTrampolineBehavior:
        return (_MF_NEXT_TAG_MUL2, "Relax", frozenset(), 2, 0)
    if bt is FunForNthTagDoubleBehavior:
        return (_MF_NTH_TAG_MUL2, beh.tag, frozenset(), 2, beh.ordinal)  # type: ignore[attr-defined]
    if bt is NosBookBehavior:
        return (_MF_NOS_BOOK, "", frozenset(), 3, 0)
    return (_MF_GENERIC, "", frozenset(), 0, 0)


def _compile_tom_mod_fun(
    beh: CardBehavior, has_tom_fun_mod: bool
) -> tuple[int, int, int]:
    if not has_tom_fun_mod:
        return (_TMF_NONE, 0, 0)
    bt = type(beh)
    if bt is TomorrowFunForTagBehavior:
        m = _TAG_MASK.get(beh.tag, 0)  # type: ignore[attr-defined]
        if m:
            return (_TMF_TAG_ADD, m, beh.bonus)  # type: ignore[attr-defined]
        return (_TMF_GENERIC, 0, 0)
    if bt is FancyFloatieBehavior:
        return (_TMF_TAG_ADD, _MASK_RELAX, 1)
    if bt is TanningBehavior:
        return (_TMF_TAG_ADD, _MASK_OUTDOORS, -1)
    if bt is WildlifeSpottingBehavior:
        return (_TMF_FIRST_OUTDOORS, _MASK_OUTDOORS, 2)
    if bt is DatesFirstNosBehavior:
        return (_TMF_ALL_ADD, 0, 2)
    return (_TMF_GENERIC, 0, 0)


def _compile_cost_mod(
    beh: CardBehavior, has_cost_mod: bool
) -> tuple[int, int, int]:
    if not has_cost_mod:
        return (_CM_NONE, 0, 0)
    bt = type(beh)
    if bt is EnergyForTagAfterBehavior or bt is FunAndEnergyForTagAfterBehavior:
        m = _TAG_MASK.get(beh.tag, 0)  # type: ignore[attr-defined]
        if m:
            return (_CM_TAG_ADD, m, beh.energy_delta)  # type: ignore[attr-defined]
        return (_CM_GENERIC, 0, 0)
    if bt is EnergyForAllCardsAfterBehavior or bt is FunAndEnergyForAllCardsAfterBehavior:
        return (_CM_ALL_ADD, 0, beh.energy_delta)  # type: ignore[attr-defined]
    if bt is MedicalAdviceBehavior:
        return (_CM_ALL_ADD, 0, 1)
    if bt is FunForAllCardsAfterDoubleBehavior:
        return (_CM_ALL_MUL2, 0, 2)
    if bt is EnergyForNextTagBehavior or bt is AfterDinnerEntertainmentBehavior:
        m = _TAG_MASK.get(beh.tag, 0)  # type: ignore[attr-defined]
        if m:
            return (_CM_NEXT_TAG_ADD, m, beh.energy_delta)  # type: ignore[attr-defined]
        return (_CM_GENERIC, 0, 0)
    if bt is SetEnergyForNextTagBehavior:
        m = _TAG_MASK.get(beh.tag, 0)  # type: ignore[attr-defined]
        if m:
            return (_CM_NEXT_TAG_SET, m, beh.energy_cost)  # type: ignore[attr-defined]
        return (_CM_GENERIC, 0, 0)
    if bt is HalfEnergyForNextTagBehavior:
        m = _TAG_MASK.get(beh.tag, 0)  # type: ignore[attr-defined]
        if m:
            return (_CM_NEXT_TAG_HALF, m, 0)
        return (_CM_GENERIC, 0, 0)
    return (_CM_GENERIC, 0, 0)


def _compile_tom_cost_mod(
    beh: CardBehavior, has_tom_cost_mod: bool
) -> tuple[int, int, int]:
    if not has_tom_cost_mod:
        return (_CM_NONE, 0, 0)
    if isinstance(beh, TomorrowEnergyForTagBehavior) and (
        type(beh).modify_tomorrow_energy_cost
        is TomorrowEnergyForTagBehavior.modify_tomorrow_energy_cost
    ):
        m = _TAG_MASK.get(beh.tag, 0)
        if m:
            return (_CM_TAG_ADD, m, beh.energy_delta)
    return (_CM_GENERIC, 0, 0)


def _build_defn_meta(
    defn: CardDefinition,
    prof: CardSynergyProfile,
    eff_title: str | None = None,
) -> _HydrogenDefnMeta:
    beh = defn.behavior
    bt = type(beh)
    title = defn.title
    eff_t = eff_title if eff_title is not None else title
    tags = defn.tags
    sorted_tags = tuple(tags) if len(tags) <= 1 else tuple(sorted(tags))
    tag_mask, tags_canonical = _tags_to_mask(tags)

    t_tags = prof.target_tags
    sorted_t_tags = tuple(t_tags) if len(t_tags) <= 1 else tuple(sorted(t_tags))
    target_mask, target_canonical = _tags_to_mask(t_tags)
    if len(t_tags) == 1 and target_canonical:
        only_tag = next(iter(t_tags))
        target_single_bit = _TAG_BIT[only_tag]
    else:
        target_single_bit = -1
    abundance = _tag_abundance(t_tags) if t_tags else 1.0

    has_can_play = bt.can_play is not CardBehavior.can_play
    has_own_cost_mod = (
        bt.modify_own_energy_cost is not CardBehavior.modify_own_energy_cost
    )
    has_cost_mod = bt.modify_energy_cost is not CardBehavior.modify_energy_cost
    has_tom_cost_mod = (
        bt.modify_tomorrow_energy_cost
        is not CardBehavior.modify_tomorrow_energy_cost
    )
    has_fun_val = bt.fun_value is not CardBehavior.fun_value
    has_fun_mod = bt.modify_fun is not CardBehavior.modify_fun
    has_tom_fun_mod = (
        bt.modify_tomorrow_fun is not CardBehavior.modify_tomorrow_fun
    )
    has_on_play = bt.on_play is not CardBehavior.on_play
    has_on_card_play = bt.on_card_play is not CardBehavior.on_card_play
    has_on_tom_card_play = (
        bt.on_tomorrow_card_play is not CardBehavior.on_tomorrow_card_play
    )
    has_on_acquire = bt.on_card_acquire is not CardBehavior.on_card_acquire
    has_on_pick = (
        bt.on_card_pick_or_draw is not CardBehavior.on_card_pick_or_draw
    )
    has_end_day_any = (
        bt.on_score is not CardBehavior.on_score
        or bt.on_end_day is not CardBehavior.on_end_day
        or bt.on_after_end_day is not CardBehavior.on_after_end_day
    )
    has_allows_extra = (
        bt.allows_extra_card_plays is not CardBehavior.allows_extra_card_plays
    )
    has_allows_energy_gain = (
        bt.allows_energy_gain is not CardBehavior.allows_energy_gain
    )
    tomorrow_all_tags = bool(beh.tomorrow_all_tags)
    after_all_tags = bool(beh.after_all_tags)
    has_tomorrow_action = bool(beh.has_tomorrow_action)

    mf_kind, mf_tag, mf_tags, mf_param, mf_ord = _compile_mod_fun(
        beh, has_fun_mod
    )
    if mf_tag:
        mf_mask = _TAG_MASK.get(mf_tag, 0)
        mf_mask_ok = mf_tag in _TAG_MASK
    elif mf_tags:
        mf_mask, mf_mask_ok = _tags_to_mask(mf_tags)
    else:
        mf_mask = 0
        mf_mask_ok = True

    played_flags = 0
    if after_all_tags:
        played_flags |= _PF_ALL_TAGS
    if has_on_card_play:
        played_flags |= _PF_HAS_CARD_PLAY
    if has_on_acquire:
        played_flags |= _PF_HAS_ACQUIRE | _PF_HAS_SIG_MARKER
    if has_tomorrow_action:
        played_flags |= _PF_HAS_TOM_ACTION
    if has_on_pick:
        played_flags |= _PF_HAS_PICK
    if has_allows_energy_gain:
        played_flags |= _PF_HAS_ALLOWS_E
    if has_cost_mod:
        played_flags |= _PF_HAS_PLAYED_COST_MOD
    if has_end_day_any:
        played_flags |= _PF_HAS_END_DAY
    if eff_t == "Group Dinner":
        played_flags |= _PF_HAS_LOOP2
    if eff_t in _TERMINAL_AWAKE_OPP_TITLES:
        played_flags |= _PF_HAS_AWAKE_OPP_TERM
    if has_fun_mod:
        played_flags |= _PF_HAS_FUN_MOD
        if (
            mf_kind
            in (
                _MF_GENERIC,
                _MF_AFTER_ALL_MUL2,
                _MF_NEXT_CARD_MUL,
                _MF_NEXT_TAG_MUL2,
                _MF_NTH_TAG_MUL2,
            )
            or not mf_mask_ok
        ):
            played_flags |= _PF_HAS_NONADD_FUN_MOD
    if (
        after_all_tags
        or (
            prof.is_multi_scaler
            and not prof.is_before_scaler
            and prof.signed_match_benefit > 0.0
        )
        or prof.is_global_after_scaler
    ):
        played_flags |= _PF_HAS_CONT_AFTER

    tom_flags = 0
    if tomorrow_all_tags:
        tom_flags |= _PF_ALL_TAGS
    if has_on_tom_card_play:
        tom_flags |= _PF_HAS_TOM_PLAY
    if has_tom_cost_mod:
        tom_flags |= _PF_HAS_TOM_COST_MOD
    tmf_kind, tmf_mask, tmf_param = _compile_tom_mod_fun(beh, has_tom_fun_mod)
    if has_tom_fun_mod:
        tom_flags |= _PF_HAS_TOM_FUN_MOD
        if tmf_kind == _TMF_GENERIC:
            tom_flags |= _PF_HAS_CUSTOM_TOM_FUN_MOD
    cm_kind, cm_mask, cm_param = _compile_cost_mod(beh, has_cost_mod)
    tcm_kind, tcm_mask, tcm_param = _compile_tom_cost_mod(beh, has_tom_cost_mod)
    tag_packed_cnt = _MASK_TO_PACKED_CNT[tag_mask]
    sorted_tag_bits = tuple(_TAG_BIT[t] for t in sorted_tags if t in _TAG_BIT)

    tom_deltas: list[tuple[str, float]] = []
    if prof.signed_match_benefit != 0.0:
        d_tom = prof.signed_match_benefit * 1.15
        for t in t_tags:
            tom_deltas.append((t, d_tom))
    if tomorrow_all_tags:
        for t in _ALL_TAG_BONUS_KEYS:
            tom_deltas.append((t, 0.35))

    played_deltas: list[tuple[str, float]] = []
    if (
        prof.is_multi_scaler and not prof.is_before_scaler
    ) or prof.is_tomorrow_scaler:
        d_pl = prof.signed_match_benefit * 1.05
        for t in t_tags:
            played_deltas.append((t, d_pl))
    elif prof.penalizes_after or prof.signed_match_benefit < 0:
        d_pl = prof.signed_match_benefit
        for t in t_tags:
            played_deltas.append((t, d_pl))

    hand_deltas: list[tuple[str, float]] = []
    if prof.is_multi_scaler or prof.is_tomorrow_scaler:
        d_hd = prof.signed_match_benefit * 0.90
        for t in t_tags:
            hand_deltas.append((t, d_hd))
    elif prof.is_single_buff:
        d_hd = prof.signed_match_benefit * 0.30
        for t in t_tags:
            hand_deltas.append((t, d_hd))
    for t in tags:
        hand_deltas.append((t, 0.15))

    if prof.is_multi_scaler or prof.is_before_scaler:
        col_extra = max(0.0, prof.signed_match_benefit * 2.2)
    elif title == "Newfangled Tres Fute":
        col_extra = 6.0
    elif title in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}:
        col_extra = 3.0
    else:
        col_extra = 0.0

    if title in {"Shady Spot", "Butter Tarts", "Istanbul"}:
        adv_sup = 4
    elif title == "Würstchen":
        adv_sup = 1
    elif prof.immediate_energy_gain > defn.cost:
        adv_sup = prof.immediate_energy_gain - defn.cost
    else:
        adv_sup = 0

    res_base_int = (
        defn.base_fun
        + prof.immediate_fun_gain
        + (3 if title == "Nos Book" else 0)
    )
    d1_code = _RD1_BY_TITLE.get(title, _RD1_NONE)
    if title == "Würstchen":
        rie_mode = _RIE_WURSTCHEN
        rie_static = 0.0
    elif title == "Istanbul":
        rie_mode = _RIE_ISTANBUL
        rie_static = 0.0
    elif prof.conditional_energy_gain > 0 and d1_code not in (
        _RD1_POTATO_PANCAKES,
        _RD1_KEEPER,
        _RD1_CAMP_CROSSROADS,
    ):
        rie_mode = _RIE_STEAK if title == "Steak" else _RIE_COND_MASK
        rie_static = 0.0
    else:
        rie_mode = _RIE_STATIC
        rie_static = (
            0.0
            if title == "Pudding Chômeur"
            else float(prof.immediate_energy_gain)
        )
    if (
        not prof.penalizes_before
        and not prof.penalizes_after
        and not prof.is_multi_scaler
        and prof.immediate_cards_gained > 0
    ):
        has_q_draw = True
        icg_c = (
            prof.immediate_cards_gained
            if prof.immediate_cards_gained < 3
            else 3
        )
        r_draw_cheap = icg_c * 0.85
        r_draw_exp = icg_c * 0.15
        q_draw_cheap = icg_c * 1.05
        q_draw_exp = icg_c * 0.25
    else:
        has_q_draw = False
        r_draw_cheap = 0.0
        r_draw_exp = 0.0
        q_draw_cheap = 0.0
        q_draw_exp = 0.0
    if title == "Sleep In":
        q_sleep_late = 1
    elif prof.must_play_late and prof.immediate_energy_gain > 0:
        q_sleep_late = 2
    else:
        q_sleep_late = 0
    if prof.is_multi_scaler:
        q_syn = _QSYN_MULTI
        q_neg_smb = 0.0
    elif prof.is_tomorrow_scaler:
        q_syn = _QSYN_TOMORROW
        q_neg_smb = 0.0
    elif prof.is_single_buff:
        q_syn = _QSYN_SINGLE
        q_neg_smb = 0.0
    elif prof.signed_match_benefit < 0.0 and title not in {
        "Medical Advice",
        "Scout Other Cottages",
    }:
        q_syn = _QSYN_NEG_SMB
        q_neg_smb = prof.signed_match_benefit * 1.35
    else:
        q_syn = _QSYN_NONE
        q_neg_smb = 0.0
    if prof.is_global_after_scaler:
        q_glob = _QGLOB_AFTER
    elif title == "Bring a Friend":
        q_glob = _QGLOB_BRING_A_FRIEND
    elif title == "Evening on the Dock":
        q_glob = _QGLOB_DOCK
    elif title == "Couch Tube":
        q_glob = _QGLOB_COUCH_TUBE
    else:
        q_glob = _QGLOB_NONE
    if prof.goes_to_tomorrow:
        q_tom_base = (
            prof.tomorrow_start_fun * 0.95 + prof.tomorrow_start_energy * 1.18
        )
        if title == "Photo Shoot":
            q_tom_extra = 3.7
        elif title == "Date's First Nos":
            q_tom_extra = 3.2
        elif title == "Floating":
            q_tom_extra = 0.65
        elif title == "Stay Up Late":
            q_tom_extra = 0.45
        else:
            q_tom_extra = 0.0
    else:
        q_tom_base = 0.0
        q_tom_extra = 0.0
    r_flat1 = _RESERVE_FLAT_EXTRA.get(title)
    r_flat2 = _RESERVE_EMPIRICAL_FLAT.get(title)
    d2_code = _RD2_BY_TITLE.get(title, _RD2_NONE)
    has_r_extra = (
        r_flat1 is not None
        or d1_code != _RD1_NONE
        or r_flat2 is not None
        or d2_code != _RD2_NONE
    )
    ist_e_per = (
        _istanbul_energy_per_discard(CardInstance(-1, defn))
        if title == "Istanbul"
        else 0
    )
    r_has_syn = bool(
        prof.is_multi_scaler
        or prof.is_global_after_scaler
        or prof.is_tomorrow_scaler
        or prof.is_single_buff
        or prof.is_count_finisher
        or title
        in {
            "Couch Tube",
            "Cheesy Phone Game",
            "Assorted Cutlery",
            "Ski on Shoulders",
            "Epic Prank",
        }
    )
    can_esup_h = (
        title
        in {
            "Shady Spot",
            "Forced Family Fun",
            "Boat Ride",
            "Treat Cereal",
            "Pudding Chômeur",
            "Potato Pancakes",
            "Würstchen",
            "Butter Tarts",
            "Istanbul",
        }
        or (
            prof.immediate_energy_gain > defn.cost
            and title not in {"Sleep In", "Pudding Chômeur"}
        )
        or (
            prof.conditional_energy_gain > defn.cost
            and title
            not in {
                "Potato Pancakes",
                "Keeper",
                "Camp Crossroads Dessert",
                "Würstchen",
            }
        )
        or prof.tomorrow_start_energy >= 2
        or (
            prof.energy_delta < 0
            and not prof.is_single_buff
            and abs(prof.energy_delta) > max(0, defn.cost - 1)
        )
    )
    pl_tom_esup = (
        prof.tomorrow_start_energy
        if (prof.goes_to_tomorrow and prof.tomorrow_start_energy > 0)
        else 0
    )

    return _HydrogenDefnMeta(
        defn=defn,
        prof=prof,
        beh=beh,
        slug=defn.slug,
        title=title,
        cost=defn.cost,
        base_fun=defn.base_fun,
        tags=tags,
        sorted_tags=sorted_tags,
        tag_mask=tag_mask,
        tags_canonical=tags_canonical,
        target_tags=t_tags,
        sorted_target_tags=sorted_t_tags,
        target_mask=target_mask,
        target_canonical=target_canonical,
        target_single_bit=target_single_bit,
        tag_abundance_target=abundance,
        play_order_bucket=prof.play_order_bucket,
        immediate_energy_gain=prof.immediate_energy_gain,
        conditional_energy_gain=prof.conditional_energy_gain,
        immediate_fun_gain=prof.immediate_fun_gain,
        conditional_fun_bonus=prof.conditional_fun_bonus,
        immediate_cards_gained=prof.immediate_cards_gained,
        energy_delta=prof.energy_delta,
        signed_match_benefit=prof.signed_match_benefit,
        tomorrow_start_fun=prof.tomorrow_start_fun,
        tomorrow_start_energy=prof.tomorrow_start_energy,
        must_play_first=prof.must_play_first,
        must_play_late=prof.must_play_late,
        goes_to_tomorrow=prof.goes_to_tomorrow,
        is_multi_scaler=prof.is_multi_scaler,
        is_before_scaler=prof.is_before_scaler,
        is_tomorrow_scaler=prof.is_tomorrow_scaler,
        is_single_buff=prof.is_single_buff,
        is_global_after_scaler=prof.is_global_after_scaler,
        is_count_finisher=prof.is_count_finisher,
        penalizes_before=prof.penalizes_before,
        penalizes_after=prof.penalizes_after,
        ends_day_immediately=prof.ends_day_immediately,
        has_can_play=has_can_play,
        has_own_cost_mod=has_own_cost_mod,
        has_cost_mod=has_cost_mod,
        has_tom_cost_mod=has_tom_cost_mod,
        has_fun_val=has_fun_val,
        has_fun_mod=has_fun_mod,
        has_tom_fun_mod=has_tom_fun_mod,
        has_on_play=has_on_play,
        has_on_card_play=has_on_card_play,
        has_on_tom_card_play=has_on_tom_card_play,
        has_on_acquire=has_on_acquire,
        has_on_pick=has_on_pick,
        has_end_day_any=has_end_day_any,
        has_allows_extra=has_allows_extra,
        has_allows_energy_gain=has_allows_energy_gain,
        tomorrow_all_tags=tomorrow_all_tags,
        after_all_tags=after_all_tags,
        has_tomorrow_action=has_tomorrow_action,
        touches_suitcase_on_play=eff_t in _SUITCASE_TOUCH_TITLES,
        played_board_flags=played_flags,
        tom_board_flags=tom_flags,
        mod_fun_kind=mf_kind,
        mod_fun_tag=mf_tag,
        mod_fun_tags=mf_tags,
        mod_fun_mask=mf_mask,
        mod_fun_param=mf_param,
        mod_fun_ord=mf_ord,
        tom_mod_fun_kind=tmf_kind,
        tom_mod_fun_mask=tmf_mask,
        tom_mod_fun_param=tmf_param,
        cost_mod_kind=cm_kind,
        cost_mod_mask=cm_mask,
        cost_mod_param=cm_param,
        tom_cost_mod_kind=tcm_kind,
        tom_cost_mod_mask=tcm_mask,
        tom_cost_mod_param=tcm_param,
        tag_packed_cnt=tag_packed_cnt,
        sorted_tag_bits=sorted_tag_bits,
        wanted_tom_deltas=tuple(tom_deltas),
        wanted_played_deltas=tuple(played_deltas),
        wanted_hand_deltas=tuple(hand_deltas),
        wanted_idx_tom_deltas=tuple(
            (_TAG_BIT[t], d) for t, d in tom_deltas if t in _TAG_BIT
        ),
        wanted_idx_played_deltas=tuple(
            (_TAG_BIT[t], d) for t, d in played_deltas if t in _TAG_BIT
        ),
        wanted_idx_hand_deltas=tuple(
            (_TAG_BIT[t], d) for t, d in hand_deltas if t in _TAG_BIT
        ),
        c_qcode=_CQ_BY_TITLE.get(title, _CQ_NONE),
        h_qcode=_HQ_BY_TITLE.get(title, _HQ_NONE),
        order_qcode=_OQ_BY_TITLE.get(title, _OQ_NONE),
        quick_penalty=_QUICK_PENALTY_BY_TITLE.get(title, 0.0),
        has_quick_draw=has_q_draw,
        quick_draw_cheap=q_draw_cheap,
        quick_draw_exp=q_draw_exp,
        quick_sleep_or_late=q_sleep_late,
        quick_syn_kind=q_syn,
        quick_neg_smb_pen=q_neg_smb,
        quick_global_kind=q_glob,
        quick_tom_base=q_tom_base,
        quick_tom_extra=q_tom_extra,
        is_fodder_plain=(defn.base_fun <= 0 and not prof.goes_to_tomorrow),
        is_sink_hand=(title in _ISTANBUL_SINK_HAND_TITLES),
        is_sink_played=(title in {"Decaf", "Wakesurf"}),
        is_dump_helper=(title in _DUMP_HELPER_TITLES),
        is_baf_sink=(
            defn.cost == 0
            or title
            in {
                "Assorted Cutlery",
                "Address the Food",
                "Bougie Coffee Machine",
            }
        ),
        reserve_sort_key=(defn.cost, -defn.base_fun, defn.slug),
        reserve_rank=-1,
        reserve_base_f=float(res_base_int),
        reserve_base_f_int=res_base_int,
        reserve_imm_e_mode=rie_mode,
        reserve_static_imm_e=rie_static,
        reserve_draw_cheap=r_draw_cheap,
        reserve_draw_exp=r_draw_exp,
        reserve_has_syn_branch=r_has_syn,
        reserve_flat1=r_flat1,
        reserve_flat2=r_flat2,
        reserve_dyn1_code=d1_code,
        reserve_dyn2_code=d2_code,
        has_reserve_extra=has_r_extra,
        reserve_colouring_extra=col_extra,
        reserve_adv_race_sup=adv_sup,
        istanbul_e_per=ist_e_per,
        is_copy_food=(title in _HYDROGEN_COPY_FOOD_TITLES),
        is_acquirer=(title in _ACQUIRER_TITLES),
        is_cutlery_combo=(
            title in _CUTLERY_SYNERGY_TITLES or defn.cost >= 6
        ),
        is_cutlery_real_combo=(title in _CUTLERY_REAL_COMBO_TITLES),
        can_give_esup_hand=can_esup_h,
        played_tom_esup=pl_tom_esup,
        q_fun_term=float(defn.base_fun) + prof.immediate_fun_gain * 1.15,
        q_imm_e_term=prof.immediate_energy_gain * 1.30,
        q_early_flags=bool(
            title == "Istanbul"
            or has_q_draw
            or q_sleep_late
            or defn.cost >= 8
            or prof.conditional_energy_gain > 0
            or _CQ_BY_TITLE.get(title, _CQ_NONE) != _CQ_NONE
            or (prof.conditional_fun_bonus > 0 and not prof.is_multi_scaler)
            or _QUICK_PENALTY_BY_TITLE.get(title, 0.0) != 0.0
        ),
        q_late_flags=bool(
            q_syn != _QSYN_NONE
            or q_glob != _QGLOB_NONE
            or prof.goes_to_tomorrow
        ),
    )


_ALL_CANONICAL_META_TAGS_OK: bool = True


def _ensure_hydrogen_caches() -> None:
    global _ALL_TAGS_CACHED, _ALL_CANONICAL_META_TAGS_OK
    if _HYDROGEN_META_BY_SLUG:
        return
    _ensure_carbon_caches()
    from thenos.cards.catalog import CARD_REGISTRY

    _ALL_TAGS_CACHED = frozenset(
        tag
        for definition in CARD_REGISTRY.values()
        for tag in definition.tags
    )
    all_ok = True
    metas_list: list[_HydrogenDefnMeta] = []
    for defn in (
        *CARD_REGISTRY.values(),
        *_NEUTRAL_DRAFT_DEFNS,
        *_TAGGED_NEUTRAL_3_BY_TAG.values(),
    ):
        did = id(defn)
        if did not in _HYDROGEN_META_BY_DEFN_ID:
            prof = _CARBON_PROFILE_BY_DEFN_ID[did]
            meta = _build_defn_meta(defn, prof)
            if not meta.tags_canonical or not meta.target_canonical:
                all_ok = False
            _HYDROGEN_META_BY_DEFN_ID[did] = meta
            _HYDROGEN_META_BY_SLUG[defn.slug] = meta
            metas_list.append(meta)
    metas_list.sort(key=lambda m: m.reserve_sort_key)
    for rank_i, m_item in enumerate(metas_list):
        object.__setattr__(m_item, "reserve_rank", rank_i)
    _ALL_CANONICAL_META_TAGS_OK = all_ok


def _get_card_meta(card: CardInstance) -> _HydrogenDefnMeta:
    m = card.markers
    if not m or "_copied_definition" not in m:
        d = card.definition
        meta = _HYDROGEN_META_BY_SLUG.get(d.slug)
        if meta is not None and meta.defn is d:
            return meta
        if not _HYDROGEN_META_BY_SLUG:
            _ensure_hydrogen_caches()
            meta = _HYDROGEN_META_BY_SLUG.get(d.slug)
            if meta is not None and meta.defn is d:
                return meta
        return _build_defn_meta(d, _beryllium_build_profile(card))
    eff_defn = m["_copied_definition"]
    if not isinstance(eff_defn, CardDefinition):
        eff_defn = card.effective_definition
    eff_title = _effective_card_title(card)
    meta = _HYDROGEN_META_BY_SLUG.get(eff_defn.slug)
    if meta is not None and meta.defn is eff_defn and meta.title == eff_title:
        return meta
    ckey = (
        eff_defn.slug,
        eff_defn.title,
        eff_title,
        eff_defn.cost,
        eff_defn.base_fun,
        eff_defn.tags,
        id(eff_defn.behavior),
    )
    meta = _HYDROGEN_COPIED_META_CACHE.get(ckey)
    if meta is not None and meta.beh is eff_defn.behavior:
        return meta
    meta = _build_defn_meta(
        eff_defn, _beryllium_build_profile(card), eff_title=eff_title
    )
    if len(_HYDROGEN_COPIED_META_CACHE) > 4096:
        _HYDROGEN_COPIED_META_CACHE.clear()
    _HYDROGEN_COPIED_META_CACHE[ckey] = meta
    return meta


def _wanted_tags_for_player_fast(player: PlayerState) -> dict[str, float]:
    """Exact fast equivalent of `_wanted_tags_for_player` using precomputed delta tuples."""
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    if not meta_by_slug:
        _ensure_hydrogen_caches()
    weights: dict[str, float] = {}
    char = player.character
    if char is not None:
        if char.fun_tag and char.fun_delta > 0:
            ft = char.fun_tag
            weights[ft] = (
                weights[ft] + char.fun_delta * 1.4
                if ft in weights
                else char.fun_delta * 1.4
            )
        if char.energy_tag and char.energy_delta < 0:
            et = char.energy_tag
            ed = abs(char.energy_delta) * 1.2
            weights[et] = weights[et] + ed if et in weights else ed
        if char.zero_fun_tag:
            weights[char.zero_fun_tag] = -5.0

    for card in player.tomorrow_cards:
        m = card.markers
        d_c = card.definition
        if not m:
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c:
                for t, d in meta.wanted_tom_deltas:
                    weights[t] = weights[t] + d if t in weights else d
                continue
        un_copied = not m or "_copied_definition" not in m
        meta = meta_by_slug.get(d_c.slug) if un_copied else None
        if meta is not None and meta.defn is not d_c:
            meta = None
        prof = meta.prof if meta is not None else _beryllium_build_profile(card)
        if prof.signed_match_benefit != 0.0:
            delta = prof.signed_match_benefit * 1.15
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta
        if m and "chosen_tag" in m:
            chosen_tag = m["chosen_tag"]
            if isinstance(chosen_tag, str):
                weights[chosen_tag] = (
                    weights[chosen_tag] + 1.5
                    if chosen_tag in weights
                    else 1.5
                )
        c_beh = d_c.behavior if un_copied else card.effective_behavior
        if c_beh.tomorrow_all_tags:
            for t in _ALL_TAG_BONUS_KEYS:
                weights[t] = weights[t] + 0.35 if t in weights else 0.35

    for card in player.played_today:
        m = card.markers
        d_c = card.definition
        if not m or "_copied_definition" not in m:
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c:
                for t, d in meta.wanted_played_deltas:
                    weights[t] = weights[t] + d if t in weights else d
                continue
        prof = _beryllium_build_profile(card)
        if (
            prof.is_multi_scaler and not prof.is_before_scaler
        ) or prof.is_tomorrow_scaler:
            delta = prof.signed_match_benefit * 1.05
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta
        elif prof.penalizes_after or prof.signed_match_benefit < 0:
            delta = prof.signed_match_benefit
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta

    for card in player.hand:
        m = card.markers
        d_c = card.definition
        if _is_plain_hand_markers(m):
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c:
                for t, d in meta.wanted_hand_deltas:
                    weights[t] = weights[t] + d if t in weights else d
                continue
        meta = (
            meta_by_slug.get(d_c.slug)
            if "_copied_definition" not in m
            else None
        )
        if meta is not None and meta.defn is not d_c:
            meta = None
        prof = meta.prof if meta is not None else _beryllium_build_profile(card)
        if prof.is_multi_scaler or prof.is_tomorrow_scaler:
            delta = prof.signed_match_benefit * 0.90
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta
        elif prof.is_single_buff:
            delta = prof.signed_match_benefit * 0.30
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta
        for t in card.tags:
            weights[t] = weights[t] + 0.15 if t in weights else 0.15
    return weights


@dataclass(slots=True)
class _QuickZoneCtx:
    all_canonical: bool
    _player: PlayerState
    hand_mask_1: int
    hand_mask_2: int
    played_mask_1: int
    tom_mask_1: int
    cheap_hand_cnt: int
    has_dump_helper_hand: bool
    hand_tag_cnts: int = 0
    played_tag_cnts: int = 0
    w_arr: list[float] | None = None
    w_mask: int = 0
    wanted_ref: dict[str, float] | None = None
    has_hand_esup: bool = True
    played_future_tom_e: int = -1
    _hand_masks: list[int] | None = None
    _played_masks: list[int] | None = None

    def get_hand_masks(self) -> list[int]:
        hm = self._hand_masks
        if hm is not None:
            return hm
        meta_by_slug = _HYDROGEN_META_BY_SLUG
        hm = []
        for c in self._player.hand:
            cm = c.markers
            d_c = c.definition
            if not cm or _is_plain_hand_markers(cm):
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c and meta.tags_canonical:
                    hm.append(meta.tag_mask)
                    continue
            hm.append(
                _tags_to_mask(
                    d_c.tags if (not cm or _is_plain_hand_markers(cm)) else c.tags
                )[0]
            )
        self._hand_masks = hm
        return hm

    def get_played_masks(self) -> list[int]:
        pm = self._played_masks
        if pm is not None:
            return pm
        meta_by_slug = _HYDROGEN_META_BY_SLUG
        pm = []
        for c in self._player.played_today:
            cm = c.markers
            d_c = c.definition
            if not cm:
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c and meta.tags_canonical:
                    pm.append(meta.tag_mask)
                    continue
            elif "_all_tags" in cm and cm["_all_tags"]:
                pm.append(_ALL_TAGS_BITMASK)
                continue
            elif "_copied_definition" not in cm:
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c and meta.tags_canonical:
                    pm.append(meta.tag_mask)
                    continue
            pm.append(_tags_to_mask(c.tags)[0])
        self._played_masks = pm
        return pm


def _build_wanted_and_ctx(
    player: PlayerState,
    is_canon: bool = False,
) -> tuple[dict[str, float], _QuickZoneCtx]:
    """Fused single-pass `_wanted_tags_for_player_fast` + `_build_quick_zone_ctx`."""
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    if not meta_by_slug:
        _ensure_hydrogen_caches()
    char = player.character
    fast_canon = is_canon and _ALL_CANONICAL_META_TAGS_OK
    hand = player.hand
    played_today = player.played_today
    tom_cards = player.tomorrow_cards

    # Ultra-fast 8-float array path for standard canonical states (no custom tags/characters)
    if (
        fast_canon
        and char is None
        and len(hand) < 250
        and len(played_today) < 250
    ):
        w0 = w1 = w2 = w3 = w4 = w5 = w6 = w7 = 0.0
        w_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w_mask = 0
        arr_ok = True
        t_m1 = 0
        if tom_cards:
            for tc in tom_cards:
                if tc.markers:
                    arr_ok = False
                    break
                meta = meta_by_slug[tc.definition.slug]
                for b, d in meta.wanted_idx_tom_deltas:
                    w_arr[b] += d
                    w_mask |= 1 << b
                t_m1 |= meta.tag_mask
        p_m1 = 0
        p_cnts = 0
        p_tom_e = 0
        if arr_ok and played_today:
            for pc in played_today:
                pm = pc.markers
                if not pm:
                    meta = meta_by_slug[pc.definition.slug]
                    for b, d in meta.wanted_idx_played_deltas:
                        w_arr[b] += d
                        w_mask |= 1 << b
                    p_m1 |= meta.tag_mask
                    p_cnts += meta.tag_packed_cnt
                    if not pc.is_tomorrow and meta.played_tom_esup:
                        p_tom_e += meta.played_tom_esup
                elif "_copied_definition" not in pm:
                    meta = meta_by_slug[pc.definition.slug]
                    for b, d in meta.wanted_idx_played_deltas:
                        w_arr[b] += d
                        w_mask |= 1 << b
                    if "_all_tags" in pm and pm["_all_tags"]:
                        p_m1 |= _ALL_TAGS_BITMASK
                        p_cnts += _ALL_TAGS_PACKED_CNT
                    else:
                        p_m1 |= meta.tag_mask
                        p_cnts += meta.tag_packed_cnt
                    if not pc.is_tomorrow and meta.played_tom_esup:
                        p_tom_e += meta.played_tom_esup
                else:
                    arr_ok = False
                    break
        h_m1 = 0
        h_m2 = 0
        h_cnts = 0
        cheap_cnt = 0
        has_dump = False
        has_h_esup = False
        if arr_ok and hand:
            for hc in hand:
                hm = hc.markers
                if not hm or _is_plain_hand_markers(hm):
                    meta = meta_by_slug[hc.definition.slug]
                    for b, d in meta.wanted_idx_hand_deltas:
                        w_arr[b] += d
                        w_mask |= 1 << b
                    cmask = meta.tag_mask
                    h_m2 |= h_m1 & cmask
                    h_m1 |= cmask
                    h_cnts += meta.tag_packed_cnt
                    if meta.cost <= 1:
                        cheap_cnt += 1
                    if meta.is_dump_helper:
                        has_dump = True
                    if meta.can_give_esup_hand:
                        has_h_esup = True
                else:
                    arr_ok = False
                    break
        if arr_ok:
            if w_mask == 0:
                weights_fast: dict[str, float] = {}
            else:
                weights_fast = {}
                if w_mask & 1:
                    weights_fast["Board Game"] = w_arr[0]
                if w_mask & 2:
                    weights_fast["Exercise"] = w_arr[1]
                if w_mask & 4:
                    weights_fast["Food"] = w_arr[2]
                if w_mask & 8:
                    weights_fast["Indoors"] = w_arr[3]
                if w_mask & 16:
                    weights_fast["Item"] = w_arr[4]
                if w_mask & 32:
                    weights_fast["Outdoors"] = w_arr[5]
                if w_mask & 64:
                    weights_fast["Relax"] = w_arr[6]
                if w_mask & 128:
                    weights_fast["Social"] = w_arr[7]
            return weights_fast, _QuickZoneCtx(
                all_canonical=True,
                _player=player,
                hand_mask_1=h_m1,
                hand_mask_2=h_m2,
                played_mask_1=p_m1,
                tom_mask_1=t_m1,
                cheap_hand_cnt=cheap_cnt,
                has_dump_helper_hand=has_dump,
                hand_tag_cnts=h_cnts,
                played_tag_cnts=p_cnts,
                w_arr=w_arr,
                w_mask=w_mask,
                wanted_ref=weights_fast,
                has_hand_esup=has_h_esup,
                played_future_tom_e=p_tom_e,
            )

    weights: dict[str, float] = {}
    if char is not None:
        if char.fun_tag and char.fun_delta > 0:
            ft = char.fun_tag
            weights[ft] = (
                weights[ft] + char.fun_delta * 1.4
                if ft in weights
                else char.fun_delta * 1.4
            )
        if char.energy_tag and char.energy_delta < 0:
            et = char.energy_tag
            ed = abs(char.energy_delta) * 1.2
            weights[et] = weights[et] + ed if et in weights else ed
        if char.zero_fun_tag:
            weights[char.zero_fun_tag] = -5.0

    all_can = len(hand) < 250 and len(played_today) < 250
    t_m1 = 0
    if tom_cards:
        for card in tom_cards:
            m = card.markers
            d_c = card.definition
            if not m:
                if fast_canon:
                    meta = meta_by_slug[d_c.slug]
                    for t, d in meta.wanted_tom_deltas:
                        weights[t] = weights[t] + d if t in weights else d
                    t_m1 |= meta.tag_mask
                    continue
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c:
                    for t, d in meta.wanted_tom_deltas:
                        weights[t] = weights[t] + d if t in weights else d
                    if not meta.tags_canonical:
                        all_can = False
                    t_m1 |= meta.tag_mask
                    continue
            un_copied = not m or "_copied_definition" not in m
            meta = meta_by_slug.get(d_c.slug) if un_copied else None
            if meta is not None and meta.defn is not d_c:
                meta = None
            prof = (
                meta.prof if meta is not None else _beryllium_build_profile(card)
            )
            if prof.signed_match_benefit != 0.0:
                delta = prof.signed_match_benefit * 1.15
                for t in prof.target_tags:
                    weights[t] = weights[t] + delta if t in weights else delta
            if m and "chosen_tag" in m:
                chosen_tag = m["chosen_tag"]
                if isinstance(chosen_tag, str):
                    weights[chosen_tag] = (
                        weights[chosen_tag] + 1.5
                        if chosen_tag in weights
                        else 1.5
                    )
            c_beh = d_c.behavior if un_copied else card.effective_behavior
            if c_beh.tomorrow_all_tags:
                for t in _ALL_TAG_BONUS_KEYS:
                    weights[t] = weights[t] + 0.35 if t in weights else 0.35
            cmask, ok = _tags_to_mask(card.tags)
            if not ok:
                all_can = False
            t_m1 |= cmask

    p_m1 = 0
    p_cnts = 0
    if fast_canon:
        for card in played_today:
            m = card.markers
            if not m:
                meta = meta_by_slug[card.definition.slug]
                for t, d in meta.wanted_played_deltas:
                    weights[t] = weights[t] + d if t in weights else d
                p_m1 |= meta.tag_mask
                p_cnts += meta.tag_packed_cnt
            elif "_copied_definition" not in m:
                meta = meta_by_slug[card.definition.slug]
                for t, d in meta.wanted_played_deltas:
                    weights[t] = weights[t] + d if t in weights else d
                if "_all_tags" in m and m["_all_tags"]:
                    p_m1 |= _ALL_TAGS_BITMASK
                    p_cnts += _ALL_TAGS_PACKED_CNT
                else:
                    p_m1 |= meta.tag_mask
                    p_cnts += meta.tag_packed_cnt
            else:
                prof = _beryllium_build_profile(card)
                if (
                    prof.is_multi_scaler and not prof.is_before_scaler
                ) or prof.is_tomorrow_scaler:
                    delta = prof.signed_match_benefit * 1.05
                    for t in prof.target_tags:
                        weights[t] = (
                            weights[t] + delta if t in weights else delta
                        )
                elif prof.penalizes_after or prof.signed_match_benefit < 0:
                    delta = prof.signed_match_benefit
                    for t in prof.target_tags:
                        weights[t] = (
                            weights[t] + delta if t in weights else delta
                        )
                if "_all_tags" in m and m["_all_tags"]:
                    p_m1 |= _ALL_TAGS_BITMASK
                    p_cnts += _ALL_TAGS_PACKED_CNT
                else:
                    cmask, ok = _tags_to_mask(card.tags)
                    if not ok:
                        all_can = False
                    p_m1 |= cmask
                    p_cnts += _MASK_TO_PACKED_CNT[cmask]
    else:
        for card in played_today:
            m = card.markers
            d_c = card.definition
            if not m:
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c:
                    for t, d in meta.wanted_played_deltas:
                        weights[t] = weights[t] + d if t in weights else d
                    if meta.tags_canonical:
                        p_m1 |= meta.tag_mask
                        p_cnts += meta.tag_packed_cnt
                        continue
            elif "_copied_definition" not in m:
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c:
                    for t, d in meta.wanted_played_deltas:
                        weights[t] = weights[t] + d if t in weights else d
                    if "_all_tags" in m and m["_all_tags"]:
                        p_m1 |= _ALL_TAGS_BITMASK
                        p_cnts += _ALL_TAGS_PACKED_CNT
                        continue
                    if meta.tags_canonical:
                        p_m1 |= meta.tag_mask
                        p_cnts += meta.tag_packed_cnt
                        continue
                    cmask, ok = _tags_to_mask(card.tags)
                    if not ok:
                        all_can = False
                    p_m1 |= cmask
                    p_cnts += _MASK_TO_PACKED_CNT[cmask]
                    continue
            prof = _beryllium_build_profile(card)
            if (
                prof.is_multi_scaler and not prof.is_before_scaler
            ) or prof.is_tomorrow_scaler:
                delta = prof.signed_match_benefit * 1.05
                for t in prof.target_tags:
                    weights[t] = weights[t] + delta if t in weights else delta
            elif prof.penalizes_after or prof.signed_match_benefit < 0:
                delta = prof.signed_match_benefit
                for t in prof.target_tags:
                    weights[t] = weights[t] + delta if t in weights else delta
            if m and "_all_tags" in m and m["_all_tags"]:
                p_m1 |= _ALL_TAGS_BITMASK
                p_cnts += _ALL_TAGS_PACKED_CNT
            else:
                cmask, ok = _tags_to_mask(card.tags)
                if not ok:
                    all_can = False
                p_m1 |= cmask
                p_cnts += _MASK_TO_PACKED_CNT[cmask]

    h_m1 = 0
    h_m2 = 0
    h_cnts = 0
    cheap_cnt = 0
    has_dump = False
    if fast_canon:
        for card in hand:
            m = card.markers
            d_c = card.definition
            if not m or _is_plain_hand_markers(m):
                meta = meta_by_slug[d_c.slug]
                for t, d in meta.wanted_hand_deltas:
                    weights[t] = weights[t] + d if t in weights else d
                cmask = meta.tag_mask
                h_m2 |= h_m1 & cmask
                h_m1 |= cmask
                h_cnts += meta.tag_packed_cnt
                if meta.cost <= 1:
                    cheap_cnt += 1
                if meta.is_dump_helper:
                    has_dump = True
                continue
            all_can = False
            un_copied = "_copied_definition" not in m
            meta = meta_by_slug[d_c.slug] if un_copied else None
            prof = (
                meta.prof if meta is not None else _beryllium_build_profile(card)
            )
            if prof.is_multi_scaler or prof.is_tomorrow_scaler:
                delta = prof.signed_match_benefit * 0.90
                for t in prof.target_tags:
                    weights[t] = weights[t] + delta if t in weights else delta
            elif prof.is_single_buff:
                delta = prof.signed_match_benefit * 0.30
                for t in prof.target_tags:
                    weights[t] = weights[t] + delta if t in weights else delta
            for t in card.tags:
                weights[t] = weights[t] + 0.15 if t in weights else 0.15
            cmask, _ = _tags_to_mask(card.tags)
            h_m2 |= h_m1 & cmask
            h_m1 |= cmask
            h_cnts += _MASK_TO_PACKED_CNT[cmask]
            if (d_c.cost if un_copied else card.effective_cost) <= 1:
                cheap_cnt += 1
            if d_c.title in _DUMP_HELPER_TITLES:
                has_dump = True
    else:
        for card in hand:
            m = card.markers
            d_c = card.definition
            if not m or _is_plain_hand_markers(m):
                meta = meta_by_slug.get(d_c.slug)
                if meta is not None and meta.defn is d_c and meta.tags_canonical:
                    for t, d in meta.wanted_hand_deltas:
                        weights[t] = weights[t] + d if t in weights else d
                    cmask = meta.tag_mask
                    h_m2 |= h_m1 & cmask
                    h_m1 |= cmask
                    h_cnts += meta.tag_packed_cnt
                    if meta.cost <= 1:
                        cheap_cnt += 1
                    if meta.is_dump_helper:
                        has_dump = True
                    continue
            all_can = False
            un_copied = not m or "_copied_definition" not in m
            meta = meta_by_slug.get(d_c.slug) if un_copied else None
            if meta is not None and meta.defn is not d_c:
                meta = None
            if (not m or _is_plain_hand_markers(m)) and meta is not None:
                for t, d in meta.wanted_hand_deltas:
                    weights[t] = weights[t] + d if t in weights else d
            else:
                prof = (
                    meta.prof
                    if meta is not None
                    else _beryllium_build_profile(card)
                )
                if prof.is_multi_scaler or prof.is_tomorrow_scaler:
                    delta = prof.signed_match_benefit * 0.90
                    for t in prof.target_tags:
                        weights[t] = (
                            weights[t] + delta if t in weights else delta
                        )
                elif prof.is_single_buff:
                    delta = prof.signed_match_benefit * 0.30
                    for t in prof.target_tags:
                        weights[t] = (
                            weights[t] + delta if t in weights else delta
                        )
                for t in card.tags:
                    weights[t] = weights[t] + 0.15 if t in weights else 0.15
            cmask, _ = _tags_to_mask(
                d_c.tags if (not m or _is_plain_hand_markers(m)) else card.tags
            )
            h_m2 |= h_m1 & cmask
            h_m1 |= cmask
            h_cnts += _MASK_TO_PACKED_CNT[cmask]
            if (d_c.cost if un_copied else card.effective_cost) <= 1:
                cheap_cnt += 1
            if d_c.title in _DUMP_HELPER_TITLES:
                has_dump = True

    return weights, _QuickZoneCtx(
        all_canonical=all_can,
        _player=player,
        hand_mask_1=h_m1,
        hand_mask_2=h_m2,
        played_mask_1=p_m1,
        tom_mask_1=t_m1,
        cheap_hand_cnt=cheap_cnt,
        has_dump_helper_hand=has_dump,
        hand_tag_cnts=h_cnts,
        played_tag_cnts=p_cnts,
    )


def _build_quick_zone_ctx(
    player: PlayerState,
    is_canon: bool = False,
) -> _QuickZoneCtx:
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    if not meta_by_slug:
        _ensure_hydrogen_caches()
        meta_by_slug = _HYDROGEN_META_BY_SLUG
    hand = player.hand
    played_today = player.played_today
    all_can = len(hand) < 250 and len(played_today) < 250
    h_m1 = 0
    h_m2 = 0
    h_cnts = 0
    cheap_cnt = 0
    has_dump = False
    if is_canon and _ALL_CANONICAL_META_TAGS_OK:
        has_h_esup = False
        for c in hand:
            cm = c.markers
            d_c = c.definition
            if not cm or _is_plain_hand_markers(cm):
                meta = meta_by_slug[d_c.slug]
                cmask = meta.tag_mask
                h_m2 |= h_m1 & cmask
                h_m1 |= cmask
                h_cnts += meta.tag_packed_cnt
                if meta.cost <= 1:
                    cheap_cnt += 1
                if meta.is_dump_helper:
                    has_dump = True
                if meta.can_give_esup_hand:
                    has_h_esup = True
                continue
            all_can = False
            has_h_esup = True
            cmask, _ = _tags_to_mask(c.tags)
            h_m2 |= h_m1 & cmask
            h_m1 |= cmask
            h_cnts += _MASK_TO_PACKED_CNT[cmask]
            if (
                d_c.cost
                if "_copied_definition" not in cm
                else c.effective_cost
            ) <= 1:
                cheap_cnt += 1
            if d_c.title in _DUMP_HELPER_TITLES:
                has_dump = True
        p_m1 = 0
        p_cnts = 0
        p_tom_e = 0
        for c in played_today:
            cm = c.markers
            d_c = c.definition
            if not cm:
                meta = meta_by_slug[d_c.slug]
                p_m1 |= meta.tag_mask
                p_cnts += meta.tag_packed_cnt
                if not c.is_tomorrow and meta.played_tom_esup:
                    p_tom_e += meta.played_tom_esup
                continue
            if "_all_tags" in cm and cm["_all_tags"]:
                p_m1 |= _ALL_TAGS_BITMASK
                p_cnts += _ALL_TAGS_PACKED_CNT
                if "_copied_definition" not in cm:
                    meta = meta_by_slug[d_c.slug]
                    if not c.is_tomorrow and meta.played_tom_esup and p_tom_e >= 0:
                        p_tom_e += meta.played_tom_esup
                else:
                    p_tom_e = -1
                continue
            if "_copied_definition" not in cm:
                meta = meta_by_slug[d_c.slug]
                p_m1 |= meta.tag_mask
                p_cnts += meta.tag_packed_cnt
                if not c.is_tomorrow and meta.played_tom_esup and p_tom_e >= 0:
                    p_tom_e += meta.played_tom_esup
                continue
            p_tom_e = -1
            cmask, ok = _tags_to_mask(c.tags)
            if not ok:
                all_can = False
            p_m1 |= cmask
            p_cnts += _MASK_TO_PACKED_CNT[cmask]
        t_m1 = 0
        for c in player.tomorrow_cards:
            cm = c.markers
            d_c = c.definition
            if not cm or _is_plain_hand_markers(cm):
                t_m1 |= meta_by_slug[d_c.slug].tag_mask
                continue
            cmask, ok = _tags_to_mask(c.tags)
            if not ok:
                all_can = False
            t_m1 |= cmask
        return _QuickZoneCtx(
            all_canonical=all_can,
            _player=player,
            hand_mask_1=h_m1,
            hand_mask_2=h_m2,
            played_mask_1=p_m1,
            tom_mask_1=t_m1,
            cheap_hand_cnt=cheap_cnt,
            has_dump_helper_hand=has_dump,
            hand_tag_cnts=h_cnts,
            played_tag_cnts=p_cnts,
            has_hand_esup=has_h_esup,
            played_future_tom_e=p_tom_e,
        )
    for c in hand:
        cm = c.markers
        d_c = c.definition
        if not cm or _is_plain_hand_markers(cm):
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c and meta.tags_canonical:
                cmask = meta.tag_mask
                h_m2 |= h_m1 & cmask
                h_m1 |= cmask
                h_cnts += meta.tag_packed_cnt
                if meta.cost <= 1:
                    cheap_cnt += 1
                if meta.is_dump_helper:
                    has_dump = True
                continue
        all_can = False
        cmask, _ = _tags_to_mask(
            d_c.tags if (not cm or _is_plain_hand_markers(cm)) else c.tags
        )
        h_m2 |= h_m1 & cmask
        h_m1 |= cmask
        h_cnts += _MASK_TO_PACKED_CNT[cmask]
        if (
            d_c.cost
            if (not cm or "_copied_definition" not in cm)
            else c.effective_cost
        ) <= 1:
            cheap_cnt += 1
        if d_c.title in _DUMP_HELPER_TITLES:
            has_dump = True

    p_m1 = 0
    p_cnts = 0
    for c in played_today:
        cm = c.markers
        d_c = c.definition
        if not cm:
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c and meta.tags_canonical:
                p_m1 |= meta.tag_mask
                p_cnts += meta.tag_packed_cnt
                continue
        elif "_all_tags" in cm and cm["_all_tags"]:
            p_m1 |= _ALL_TAGS_BITMASK
            p_cnts += _ALL_TAGS_PACKED_CNT
            continue
        elif "_copied_definition" not in cm:
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c and meta.tags_canonical:
                p_m1 |= meta.tag_mask
                p_cnts += meta.tag_packed_cnt
                continue
        cmask, ok = _tags_to_mask(c.tags)
        if not ok:
            all_can = False
        p_m1 |= cmask
        p_cnts += _MASK_TO_PACKED_CNT[cmask]

    t_m1 = 0
    for c in player.tomorrow_cards:
        cm = c.markers
        d_c = c.definition
        if not cm or _is_plain_hand_markers(cm):
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is d_c and meta.tags_canonical:
                t_m1 |= meta.tag_mask
                continue
        cmask, ok = _tags_to_mask(c.tags)
        if not ok:
            all_can = False
        t_m1 |= cmask

    return _QuickZoneCtx(
        all_canonical=all_can,
        _player=player,
        hand_mask_1=h_m1,
        hand_mask_2=h_m2,
        played_mask_1=p_m1,
        tom_mask_1=t_m1,
        cheap_hand_cnt=cheap_cnt,
        has_dump_helper_hand=has_dump,
        hand_tag_cnts=h_cnts,
        played_tag_cnts=p_cnts,
    )


def _fast_extra_energy_support(
    game: Game,
    player_index: int,
    card: CardInstance,
    *,
    extra_fodder: int = 0,
    ignore_today_block: bool = False,
    include_future_tomorrow: bool = False,
    _ctx: _QuickZoneCtx | None = None,
    _in_hand: bool | None = None,
) -> int:
    """Bit-exact fast replacement for `_carbon_extra_energy_support`."""
    if _ctx is not None and not _ctx.has_hand_esup:
        if not include_future_tomorrow:
            return 0
        pf_tom = _ctx.played_future_tom_e
        if pf_tom == 0:
            return 0
        if pf_tom > 0 and (
            _in_hand is True
            or not any(
                pc is card for pc in game.players[player_index].played_today
            )
        ):
            return pf_tom

    player = game.players[player_index]
    hand = player.hand
    played_today = player.played_today
    cm = card.markers
    eff_cost = card.definition.cost if not cm else card.effective_cost
    c_tags = card.definition.tags if not cm else card.tags
    ignore_or_future = ignore_today_block or include_future_tomorrow
    can_gain_e = ignore_or_future or (
        (
            type(game) is _HYDROGEN_SIM_GAME_CLS
            and not (game._p_flags[player_index] & _PF_HAS_ALLOWS_E)  # type: ignore[attr-defined]
        )
        or _allows_energy_gain_today(game, player)
    )
    support = 0
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    is_can_sim = getattr(game, "_all_canonical_defs", False)

    if include_future_tomorrow:
        if (
            _ctx is not None
            and _ctx.played_future_tom_e >= 0
            and (
                _in_hand is True
                or not any(pc is card for pc in played_today)
            )
        ):
            support += _ctx.played_future_tom_e
        else:
            for pc in played_today:
                if pc is card or pc.is_tomorrow:
                    continue
                pm = pc.markers
                if not pm or "_copied_definition" not in pm:
                    d_p = pc.definition
                    p_m = (
                        meta_by_slug[d_p.slug]
                        if is_can_sim
                        else meta_by_slug.get(d_p.slug)
                    )
                    if p_m is not None and (is_can_sim or p_m.defn is d_p):
                        support += p_m.played_tom_esup
                        continue
                p_prof = _beryllium_build_profile(pc)
                if p_prof.goes_to_tomorrow and p_prof.tomorrow_start_energy > 0:
                    support += p_prof.tomorrow_start_energy

    has_sunscreen: bool | None = None
    has_istanbul = False
    for hc in hand:
        if hc is card:
            continue
        hm = hc.markers
        d_h = hc.definition
        if not hm or "_copied_definition" not in hm:
            h_meta = (
                meta_by_slug[d_h.slug]
                if is_can_sim
                else meta_by_slug.get(d_h.slug)
            )
            if h_meta is not None and (is_can_sim or h_meta.defn is d_h):
                if h_meta.title == "Istanbul":
                    has_istanbul = True
                    continue
                if not h_meta.can_give_esup_hand:
                    continue
                h_title = h_meta.title
                h_prof = h_meta.prof
                h_cost = h_meta.cost
            else:
                h_meta = None
        else:
            h_meta = None
        if h_meta is None:
            h_title = _effective_card_title(hc)
            if h_title == "Istanbul":
                has_istanbul = True
            h_prof = _beryllium_build_profile(hc)
            h_cost = d_h.cost if not hm else hc.effective_cost

        if h_title == "Shady Spot" and "Outdoors" in c_tags:
            if has_sunscreen is None:
                has_sunscreen = False
                for sc_c in hand:
                    if sc_c is not card:
                        sm = sc_c.markers
                        st = (
                            sc_c.definition.title
                            if (not sm or "_copied_definition" not in sm)
                            else _effective_card_title(sc_c)
                        )
                        if st == "Sunscreen":
                            has_sunscreen = True
                            break
                if not has_sunscreen and not ignore_or_future:
                    for sc_c in played_today:
                        if sc_c is not card:
                            sm = sc_c.markers
                            st = (
                                sc_c.definition.title
                                if (not sm or "_copied_definition" not in sm)
                                else _effective_card_title(sc_c)
                            )
                            if st == "Sunscreen":
                                has_sunscreen = True
                                break
            shady_cost = h_cost - (1 if has_sunscreen else 0)
            if shady_cost < 0:
                shady_cost = 0
            disc_gain = (eff_cost - (eff_cost // 2)) - shady_cost
            if disc_gain > 0:
                support += disc_gain
            continue
        if h_title == "Forced Family Fun" and "Board Game" in c_tags:
            disc_gain = (eff_cost - (eff_cost // 2)) - h_cost
            if disc_gain > 0:
                support += disc_gain
            continue
        if h_title == "Boat Ride" and "Item" in c_tags:
            disc_gain = eff_cost - h_cost
            if disc_gain > 0:
                support += disc_gain
            continue
        if h_title == "Treat Cereal" and "Exercise" in c_tags:
            disc_gain = 4 - h_cost
            if disc_gain > 0:
                support += disc_gain
            continue

        if can_gain_e:
            if h_prof.immediate_energy_gain > h_cost and h_title not in {
                "Sleep In",
                "Pudding Chômeur",
            }:
                support += h_prof.immediate_energy_gain - h_cost
            elif h_title == "Pudding Chômeur" and len(hand) >= 2:
                support += 2
            elif (
                h_prof.conditional_energy_gain > h_cost
                and h_title
                not in {
                    "Potato Pancakes",
                    "Keeper",
                    "Camp Crossroads Dessert",
                    "Würstchen",
                }
            ):
                support += h_prof.conditional_energy_gain - h_cost
            elif h_title == "Potato Pancakes":
                relax_cnt = 0
                if not ignore_or_future:
                    for oc in played_today:
                        if oc is not hc and oc is not card and "Relax" in (
                            oc.definition.tags if not oc.markers else oc.tags
                        ):
                            relax_cnt += 1
                for oc in hand:
                    if oc is not hc and oc is not card and "Relax" in (
                        oc.definition.tags if not oc.markers else oc.tags
                    ):
                        relax_cnt += 1
                pp_gain = 2 * relax_cnt - h_cost
                if pp_gain > 0:
                    support += pp_gain
            elif h_title == "Würstchen":
                prior_cnt = 1 if extra_fodder >= 2 else 0
                if not ignore_or_future:
                    for oc in played_today:
                        if oc is not hc and oc is not card:
                            prior_cnt += 1
                for oc in hand:
                    if oc is not hc and oc is not card and (
                        oc.definition.cost
                        if not oc.markers
                        else oc.effective_cost
                    ) <= 2:
                        prior_cnt += 1
                w_gain = prior_cnt - h_cost
                if w_gain > 0:
                    support += w_gain
            elif h_title == "Butter Tarts":
                support += 4
            elif (
                include_future_tomorrow
                and h_prof.tomorrow_start_energy > 0
                and game.day < DAY_COUNT
            ):
                if h_prof.tomorrow_start_energy > 1:
                    support += h_prof.tomorrow_start_energy - 1
        elif (
            include_future_tomorrow
            and h_prof.tomorrow_start_energy > 0
            and game.day < DAY_COUNT
        ):
            if h_prof.tomorrow_start_energy > 1:
                support += h_prof.tomorrow_start_energy - 1

        if (
            h_prof.energy_delta < 0
            and not h_prof.is_single_buff
            and (not h_prof.target_tags or (c_tags & h_prof.target_tags))
        ):
            ed_gain = (-h_prof.energy_delta) - (h_cost - 1 if h_cost > 1 else 0)
            if ed_gain > 0:
                support += ed_gain

    if has_istanbul:
        support += _istanbul_extra_energy_support(
            game,
            player_index,
            card,
            extra_fodder=extra_fodder,
            ignore_today_block=ignore_or_future,
        )
    return support


def _hydrogen_quick_card_desirability(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
    *,
    cost: int | None = None,
    prof: CardSynergyProfile | None = None,
    remaining_draw_picks: int = 0,
    _ctx: _QuickZoneCtx | None = None,
    _in_hand: bool | None = None,
    _meta: _HydrogenDefnMeta | None = None,
) -> float:
    """Data-calibrated heuristic card desirability for Hydrogen."""
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    if not meta_by_slug:
        _ensure_hydrogen_caches()

    m = card.markers
    uncopied = not m or "_copied_definition" not in m
    meta = _meta
    if meta is None:
        if uncopied:
            d_c = card.definition
            meta = meta_by_slug.get(d_c.slug)
            if meta is not None and meta.defn is not d_c:
                meta = None
        else:
            meta = _get_card_meta(card)

    player = game.players[player_index]
    if cost is None:
        cost = game.energy_cost(player_index, card)

    # Ultra-fast path for plain canonical cards with no special early/late/HQ branches
    if (
        meta is not None
        and (prof is None or prof is meta.prof)
        and uncopied
        and not (m and "_all_tags" in m and m["_all_tags"])
        and player.character is None
        and game.daily_condition is None
        and not meta.q_early_flags
        and not meta.q_late_flags
        and meta.h_qcode == _HQ_NONE
    ):
        val_q = (meta.q_fun_term - cost * 0.78) + meta.q_imm_e_term
        if _ctx is not None and wanted_tags is _ctx.wanted_ref:
            if _ctx.w_mask & meta.tag_mask:
                w_arr_q = _ctx.w_arr  # type: ignore[assignment]
                tm_q = 0.0
                if cost <= 1:
                    for b_q in meta.sorted_tag_bits:
                        wq = w_arr_q[b_q]  # type: ignore[index]
                        if wq != 0.0:
                            tm_q += wq * 0.78
                            if wq > 0.0:
                                tm_q += 0.38 * (wq if wq < 2.5 else 2.5)
                else:
                    for b_q in meta.sorted_tag_bits:
                        wq = w_arr_q[b_q]  # type: ignore[index]
                        if wq != 0.0:
                            tm_q += wq * 0.78
                if meta.must_play_first and tm_q > 0.0:
                    tm_q *= 0.30
                val_q += tm_q
        elif wanted_tags:
            tm_q = 0.0
            if cost <= 1:
                for t_q in meta.sorted_tags:
                    if t_q in wanted_tags:
                        wq = wanted_tags[t_q]
                        if wq != 0.0:
                            tm_q += wq * 0.78
                            if wq > 0.0:
                                tm_q += 0.38 * (wq if wq < 2.5 else 2.5)
            else:
                for t_q in meta.sorted_tags:
                    if t_q in wanted_tags:
                        wq = wanted_tags[t_q]
                        if wq != 0.0:
                            tm_q += wq * 0.78
            if meta.must_play_first and tm_q > 0.0:
                tm_q *= 0.30
            val_q += tm_q
        if cost <= 1 and meta.base_fun >= 0:
            val_q += 0.65 if cost == 0 else 0.25
        return val_q

    hand = player.hand
    played_today = player.played_today
    hand_len = len(hand)
    played_len = len(played_today)
    days_left = DAY_COUNT - game.day

    if meta is not None and (prof is None or prof is meta.prof):
        prof = meta.prof
        c_defn = meta.defn
        title = meta.title
        eff_cost = meta.cost
        eff_base_fun = meta.base_fun
        c_qcode = meta.c_qcode
        h_qcode = meta.h_qcode
        quick_pen = meta.quick_penalty
    else:
        if prof is None:
            prof = _beryllium_build_profile(card)
        c_defn = card.definition if uncopied else card.effective_definition
        title = c_defn.title
        eff_cost = c_defn.cost
        eff_base_fun = c_defn.base_fun
        c_qcode = _CQ_BY_TITLE.get(title, _CQ_NONE)
        h_qcode = _HQ_BY_TITLE.get(title, _HQ_NONE)
        quick_pen = _QUICK_PENALTY_BY_TITLE.get(title, 0.0)
        meta = None

    has_all_tags = bool(m and "_all_tags" in m and m["_all_tags"])
    if not has_all_tags:
        c_tags = card.definition.tags
        cmask = (
            meta.tag_mask
            if (meta is not None and uncopied)
            else _tags_to_mask(c_tags)[0]
        )
    else:
        c_tags = _ALL_TAGS_CACHED if _ALL_TAGS_CACHED is not None else card.tags
        cmask = _ALL_TAGS_BITMASK

    char = player.character
    if char is not None:
        _, _, _, z_mask_q, _, _, _, char_cust_q = _get_char_fast(char)
        zero_tag = char.zero_fun_tag
        if not zero_tag:
            is_zeroed = False
        elif not char_cust_q and meta is not None and uncopied:
            is_zeroed = True if has_all_tags else bool(cmask & z_mask_q)
        else:
            is_zeroed = zero_tag in c_tags
    else:
        zero_tag = ""
        is_zeroed = False

    in_hand: bool | None = _in_hand
    imm_energy = prof.immediate_energy_gain

    scored_base_fun = 0.0 if is_zeroed else float(eff_base_fun)
    cond = game.daily_condition
    if not is_zeroed and cond is not None and cond.fun_tag:
        _, cf_mask_q, cf_delta_q, _, cf_cust_q = _get_cond_fast(cond)
        has_cf = (
            (True if has_all_tags else bool(cmask & cf_mask_q))
            if (not cf_cust_q and meta is not None and uncopied)
            else (cond.fun_tag in c_tags)
        )
        if has_cf:
            cond_weight = 1.0 if (days_left == 0 or played_len > 0) else 0.55
            scored_base_fun += cf_delta_q * cond_weight

    imm_fun_gain = 0.0 if is_zeroed else prof.immediate_fun_gain

    val = (
        scored_base_fun
        + imm_fun_gain * 1.15
        - cost * 0.78
        + imm_energy * 1.30
    )

    if title == "Istanbul" and _allows_energy_gain_today(game, player, card):
        e_per = _istanbul_energy_per_discard(card)
        if in_hand is None:
            in_hand = any(hc is card for hc in hand)
        other_cnt = (hand_len - 1 if in_hand else hand_len) + remaining_draw_picks
        has_e_or_empty_sink = False
        for hc in hand:
            if hc is not card and _effective_card_title(hc) in _ISTANBUL_SINK_HAND_TITLES:
                has_e_or_empty_sink = True
                break
        if not has_e_or_empty_sink:
            for pc in played_today:
                if _effective_card_title(pc) in {"Decaf", "Wakesurf"}:
                    has_e_or_empty_sink = True
                    break
        disc_est = (
            other_cnt
            if has_e_or_empty_sink
            else (other_cnt - 1 if other_cnt > 1 else 0)
        )
        if disc_est > 3:
            disc_est = 3
        if disc_est > 0:
            other_cost = 0
            for hc in hand:
                if hc is not card and hc.effective_behavior.can_play(game, player, hc):
                    other_cost += game.energy_cost(player_index, hc)
            rem_e = player.energy - cost
            has_sink = has_e_or_empty_sink or other_cost > (3 if rem_e < 3 else rem_e)
            sac_cost = 0.70 if days_left == 0 else 1.45
            raw_net = e_per * (1.20 if has_sink else 0.75) - sac_cost
            if raw_net > 0.0:
                val += disc_est * raw_net

    if meta is not None:
        if meta.has_quick_draw:
            val += meta.quick_draw_cheap if cost <= 2 else meta.quick_draw_exp
        qsl = meta.quick_sleep_or_late
        if qsl:
            if qsl == 1:
                val -= 1.6
            elif played_len < 2:
                val -= 0.75
    else:
        if (
            prof.immediate_cards_gained > 0
            and not prof.penalizes_before
            and not prof.penalizes_after
            and not prof.is_multi_scaler
        ):
            icg = prof.immediate_cards_gained
            val += (icg if icg < 3 else 3) * (1.05 if cost <= 2 else 0.25)

        if title == "Sleep In":
            val -= 1.6
        elif (
            prof.must_play_late
            and prof.immediate_energy_gain > 0
            and played_len < 2
        ):
            val -= 0.75

    if eff_cost >= 8 and cost > player.energy:
        today_sup = _fast_extra_energy_support(
            game,
            player_index,
            card,
            extra_fodder=remaining_draw_picks,
            _ctx=_ctx,
            _in_hand=in_hand,
        )
        today_unfunded = max(0, cost - (player.energy + today_sup))
        if days_left >= 1:
            if in_hand is None:
                in_hand = card in hand
            extra_fodder = remaining_draw_picks + (0 if in_hand else 2)
            future_sup = _fast_extra_energy_support(
                game,
                player_index,
                card,
                extra_fodder=extra_fodder,
                include_future_tomorrow=True,
                _ctx=_ctx,
                _in_hand=in_hand,
            )
            char_cost = (
                max(0, char.modify_energy_cost(c_tags, eff_cost))
                if char is not None
                else eff_cost
            )
            future_unfunded = max(0, char_cost - (7 + future_sup))
            unfunded = (
                today_unfunded
                if today_unfunded < future_unfunded
                else future_unfunded
            )
            support_for_slack = (
                today_sup if today_sup > future_sup else future_sup
            )
        else:
            unfunded = today_unfunded
            support_for_slack = today_sup

        if unfunded > 0:
            slack = (
                1
                if (
                    title == "Adventure Race"
                    and support_for_slack >= 3
                    and days_left >= 2
                )
                else 0
            )
            eff_unfunded = unfunded - slack
            if eff_unfunded > 0:
                val -= eff_unfunded * (1.50 if days_left >= 1 else 1.95)

    if prof.conditional_energy_gain > 0:
        t_tags = prof.target_tags
        if title in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}:
            match_cnt = 0
            if (
                _ctx is not None
                and _ctx.all_canonical
                and meta is not None
                and meta.target_canonical
            ):
                tmask = meta.target_mask
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                sb = meta.target_single_bit
                if sb >= 0:
                    sh = sb << 3
                    match_cnt = (
                        ((_ctx.played_tag_cnts >> sh) & 0xFF)
                        + ((_ctx.hand_tag_cnts >> sh) & 0xFF)
                    )
                    if cmask & tmask:
                        if in_hand or any(pc is card for pc in played_today):
                            match_cnt -= 1
                else:
                    for pm in _ctx.get_played_masks():
                        if pm & tmask:
                            match_cnt += 1
                    if not in_hand and (cmask & tmask) and any(pc is card for pc in played_today):
                        match_cnt -= 1
                    for hm in _ctx.get_hand_masks():
                        if hm & tmask:
                            match_cnt += 1
                    if in_hand and (cmask & tmask):
                        match_cnt -= 1
            else:
                for other in played_today:
                    if other is not card and (
                        (other.definition.tags if not other.markers else other.tags)
                        & t_tags
                    ):
                        match_cnt += 1
                for other in hand:
                    if other is not card and (
                        (other.definition.tags if not other.markers else other.tags)
                        & t_tags
                    ):
                        match_cnt += 1
            per_match = 2 if title == "Potato Pancakes" else 1
            future_matches = (
                0.8 if (days_left >= 1 and title == "Potato Pancakes") else 0.3
            )
            val += (
                (match_cnt if match_cnt < 3 else 3) + future_matches
            ) * per_match * 0.95
        elif title == "Würstchen":
            prior_cnt = 0
            for other in played_today:
                if other is not card:
                    prior_cnt += 1
            for other in hand:
                if other is not card and (
                    other.definition.cost
                    if not other.markers
                    else other.effective_cost
                ) <= 2:
                    prior_cnt += 1
            future_prior = 1.5 if days_left >= 1 else 0.0
            val += ((prior_cnt if prior_cnt < 5 else 5) + future_prior) * 0.95
        else:
            has_cond = _condition_met_in_played(player, card, prof)
            if not has_cond:
                if (
                    _ctx is not None
                    and _ctx.all_canonical
                    and meta is not None
                    and meta.target_canonical
                ):
                    tmask = meta.target_mask
                    if in_hand is None:
                        in_hand = any(hc is card for hc in hand)
                    has_cond = (
                        bool(tmask & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                        if in_hand
                        else bool(tmask & _ctx.hand_mask_1)
                    )
                else:
                    for other in hand:
                        if other is not card and (
                            (
                                other.definition.tags
                                if not other.markers
                                else other.tags
                            )
                            & t_tags
                        ):
                            has_cond = True
                            break
            val += prof.conditional_energy_gain * (0.95 if has_cond else 0.35)

    # Carbon calibrated card-specific adjustments
    if c_qcode != _CQ_NONE:
        if c_qcode == _CQ_NOS_BOOK:
            if played_len > 0 or hand_len >= 2 or game.day < DAY_COUNT:
                val += 2.45
        elif c_qcode == _CQ_GROUP_DINNER:
            val -= 1.85
        elif c_qcode == _CQ_TELL_A_STORY:
            has_payoff = False
            has_ed_hand = False
            for other in hand:
                ot = other.definition.title
                if ot == "Assorted Cutlery" or ot == "Bougie Coffee Machine":
                    has_payoff = True
                elif ot == "Epic Duels":
                    has_ed_hand = True
            has_same_turn_combo = False
            if has_payoff:
                if has_ed_hand:
                    has_same_turn_combo = True
                else:
                    for pc in played_today:
                        if _effective_card_title(pc) == "Epic Duels":
                            has_same_turn_combo = True
                            break
                    if not has_same_turn_combo:
                        players = game.players
                        all_asleep = True
                        for idx in range(4):
                            if idx != player_index and not players[idx].asleep:
                                all_asleep = False
                                break
                        has_same_turn_combo = all_asleep
            if not has_same_turn_combo:
                val -= 1.45
        elif c_qcode == _CQ_ASSORTED_CUTLERY:
            has_combo = bool(_get_known_trunk_top_any(game, player_index))
            if not has_combo:
                for other in hand:
                    if other is not card and other.definition.title in _CUTLERY_REAL_COMBO_TITLES:
                        has_combo = True
                        break
            val += 2.6 if has_combo else 0.55
        elif c_qcode == _CQ_BOUGIE_COFFEE_MACHINE:
            val += 1.35
        elif c_qcode == _CQ_MEDICAL_ADVICE:
            if days_left >= 1 and played_len <= 1:
                pb_bonus = 0.0
                for c in played_today:
                    if _effective_card_title(c) == "Paddleboat":
                        pb_bonus = 4.5
                        break
                if pb_bonus == 0.0:
                    for c in hand:
                        if _effective_card_title(c) == "Paddleboat":
                            pb_bonus = 4.5
                            break
                val += 4.6 + pb_bonus - 0.9 * played_len
        elif c_qcode == _CQ_SCOUT_OTHER_COTTAGES:
            if days_left >= 1 and played_len <= 1:
                val += 3.8 - 0.9 * played_len
        elif c_qcode == _CQ_BOHNANZA:
            fodder_cnt = 0
            for zone in (played_today, hand):
                for c in zone:
                    if c is card:
                        continue
                    cm = c.markers
                    if not cm:
                        d_o = c.definition
                        om = meta_by_slug.get(d_o.slug)
                        if om is not None and om.defn is d_o:
                            if om.is_fodder_plain:
                                fodder_cnt += 1
                            continue
                    if (
                        c.definition.base_fun if not cm else c.effective_base_fun
                    ) <= 0 and not _beryllium_build_profile(c).goes_to_tomorrow:
                        fodder_cnt += 1
            val += 1.15 + 0.65 * (fodder_cnt if fodder_cnt < 3 else 3)
        elif c_qcode == _CQ_SKI_ON_SHOULDERS:
            if game.day >= 2:
                p_fun = player.fun
                players = game.players
                opps_strictly_ahead = 0
                for idx in range(4):
                    if idx != player_index and players[idx].fun > p_fun:
                        opps_strictly_ahead += 1
                if opps_strictly_ahead >= 2:
                    val += 2.1
                elif days_left >= 1 and opps_strictly_ahead == 1:
                    val += 0.4
                else:
                    val -= 0.5
            else:
                val += 0.3
        elif c_qcode == _CQ_CHEESY_PHONE_GAME:
            val += 1.05
        elif c_qcode == _CQ_WAKESURF or c_qcode == _CQ_SLALOM_START:
            if not is_zeroed:
                val += 2.20 if c_qcode == _CQ_WAKESURF else 2.00
            if hand_len >= 4:
                if _ctx is not None:
                    has_helper = _ctx.has_dump_helper_hand
                else:
                    has_helper = False
                    for other in hand:
                        if other.definition.title in _DUMP_HELPER_TITLES:
                            has_helper = True
                            break
                if not has_helper:
                    val -= 0.65 * (hand_len - 3)
    elif (
        prof.conditional_fun_bonus > 0
        and not prof.is_multi_scaler
        and not is_zeroed
    ):
        cfb = prof.conditional_fun_bonus
        val += (cfb if cfb < 5 else 5) * 0.55

    if quick_pen != 0.0:
        val -= quick_pen

    if (
        _ctx is not None
        and wanted_tags is _ctx.wanted_ref
        and meta is not None
        and uncopied
        and not has_all_tags
        and not zero_tag
    ):
        if _ctx.w_mask & cmask:
            w_arr = _ctx.w_arr  # type: ignore[assignment]
            tag_match = 0.0
            if cost <= 1:
                for b in meta.sorted_tag_bits:
                    w = w_arr[b]  # type: ignore[index]
                    if w != 0.0:
                        tag_match += w * 0.78
                        if w > 0.0:
                            tag_match += 0.38 * (w if w < 2.5 else 2.5)
            else:
                for b in meta.sorted_tag_bits:
                    w = w_arr[b]  # type: ignore[index]
                    if w != 0.0:
                        tag_match += w * 0.78
            if prof.must_play_first and tag_match > 0.0:
                tag_match *= 0.30
            val += tag_match
    elif wanted_tags:
        tag_match = 0.0
        iter_tags = (
            meta.sorted_tags
            if (meta is not None and uncopied and not has_all_tags)
            else (c_tags if len(c_tags) <= 1 else sorted(c_tags))
        )
        if not zero_tag:
            if cost <= 1:
                for t in iter_tags:
                    if t in wanted_tags:
                        w = wanted_tags[t]
                        if w != 0.0:
                            tag_match += w * 0.78
                            if w > 0.0:
                                tag_match += 0.38 * (w if w < 2.5 else 2.5)
            else:
                for t in iter_tags:
                    if t in wanted_tags:
                        w = wanted_tags[t]
                        if w != 0.0:
                            tag_match += w * 0.78
        else:
            for t in iter_tags:
                if t == zero_tag:
                    if eff_base_fun > 0 or prof.conditional_fun_bonus > 0:
                        tag_match -= 0.75
                    continue
                if t in wanted_tags:
                    w = wanted_tags[t]
                    if w != 0.0:
                        tag_match += w * 0.78
                        if w > 0.0 and cost <= 1:
                            tag_match += 0.38 * (w if w < 2.5 else 2.5)
        if prof.must_play_first and tag_match > 0.0:
            tag_match *= 0.30
        val += tag_match

    if meta is None or meta.quick_syn_kind != _QSYN_NONE:
        pure_zero_tag_fun_buff = bool(
            zero_tag
            and len(prof.target_tags) == 1
            and zero_tag in prof.target_tags
            and prof.energy_delta >= 0
        )

        if prof.is_multi_scaler and not pure_zero_tag_fun_buff:
            t_tags = prof.target_tags
            matching_in_hand = 0
            matching_played = 0
            if (
                _ctx is not None
                and _ctx.all_canonical
                and meta is not None
                and meta.target_canonical
            ):
                tmask = meta.target_mask
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                sb = meta.target_single_bit
                if sb >= 0:
                    sh = sb << 3
                    matching_in_hand = (_ctx.hand_tag_cnts >> sh) & 0xFF
                    matching_played = (_ctx.played_tag_cnts >> sh) & 0xFF
                    if cmask & tmask:
                        if in_hand:
                            matching_in_hand -= 1
                        elif any(pc is card for pc in played_today):
                            matching_played -= 1
                else:
                    for hm in _ctx.get_hand_masks():
                        if hm & tmask:
                            matching_in_hand += 1
                    if in_hand and (cmask & tmask):
                        matching_in_hand -= 1
                    for pm in _ctx.get_played_masks():
                        if pm & tmask:
                            matching_played += 1
                    if not in_hand and (cmask & tmask) and any(pc is card for pc in played_today):
                        matching_played -= 1
            else:
                for c in hand:
                    if c is not card and (
                        (c.definition.tags if not c.markers else c.tags) & t_tags
                    ):
                        matching_in_hand += 1
                for c in played_today:
                    if c is not card and (
                        (c.definition.tags if not c.markers else c.tags) & t_tags
                    ):
                        matching_played += 1
            abundance = (
                meta.tag_abundance_target
                if meta is not None
                else _tag_abundance(t_tags)
            )
            future_factor = (days_left if days_left < 3 else 3) * 0.50 * abundance
            val += (
                matching_in_hand * 0.95
                + matching_played * 0.50
                + future_factor
            ) * prof.signed_match_benefit
        elif prof.is_tomorrow_scaler and not pure_zero_tag_fun_buff:
            if days_left >= 1:
                t_tags = prof.target_tags
                matching_in_hand = 0
                if (
                    _ctx is not None
                    and _ctx.all_canonical
                    and meta is not None
                    and meta.target_canonical
                ):
                    tmask = meta.target_mask
                    if in_hand is None:
                        in_hand = any(hc is card for hc in hand)
                    sb = meta.target_single_bit
                    if sb >= 0:
                        matching_in_hand = (_ctx.hand_tag_cnts >> (sb << 3)) & 0xFF
                        if in_hand and (cmask & tmask):
                            matching_in_hand -= 1
                    else:
                        for hm in _ctx.get_hand_masks():
                            if hm & tmask:
                                matching_in_hand += 1
                        if in_hand and (cmask & tmask):
                            matching_in_hand -= 1
                else:
                    for c in hand:
                        if c is not card and (
                            (c.definition.tags if not c.markers else c.tags) & t_tags
                        ):
                            matching_in_hand += 1
                abundance = (
                    meta.tag_abundance_target
                    if meta is not None
                    else _tag_abundance(t_tags)
                )
                val += (
                    matching_in_hand * 0.65 + 0.45 * abundance
                ) * prof.signed_match_benefit
        elif prof.is_single_buff and not pure_zero_tag_fun_buff:
            t_tags = prof.target_tags
            if (
                _ctx is not None
                and _ctx.all_canonical
                and meta is not None
                and meta.target_canonical
                and t_tags
            ):
                tmask = meta.target_mask
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                has_match = (
                    bool(tmask & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                    if in_hand
                    else bool(tmask & _ctx.hand_mask_1)
                )
            else:
                has_match = False
                for c in hand:
                    if c is not card and (
                        not t_tags
                        or ((c.definition.tags if not c.markers else c.tags) & t_tags)
                    ):
                        has_match = True
                        break
            abundance = (
                meta.tag_abundance_target
                if meta is not None
                else (_tag_abundance(t_tags) if t_tags else 1.0)
            )
            val += prof.signed_match_benefit * (
                0.78 if has_match else 0.25 * abundance
            )
        elif prof.signed_match_benefit < 0.0 and title not in {
            "Medical Advice",
            "Scout Other Cottages",
        }:
            val += (
                meta.quick_neg_smb_pen
                if meta is not None
                else prof.signed_match_benefit * 1.35
            )

    if meta is None or meta.quick_global_kind != _QGLOB_NONE:
        if prof.is_global_after_scaler:
            if zero_tag and title == "Colouring":
                val -= 8.0
            else:
                if _ctx is not None:
                    cheap_cards = _ctx.cheap_hand_cnt
                else:
                    cheap_cards = 0
                    for c in hand:
                        cm = c.markers
                        if (
                            c.definition.cost
                            if (not cm or "_copied_definition" not in cm)
                            else c.effective_cost
                        ) <= 1:
                            cheap_cards += 1
                val += (
                    prof.signed_match_benefit * 0.85
                    + cheap_cards * 0.55
                    + (days_left if days_left < 2 else 2) * 0.50
                )
        elif title == "Bring a Friend":
            zero_or_sink = 0
            for c in hand:
                if c is card:
                    continue
                cm = c.markers
                if _is_plain_hand_markers(cm):
                    d_o = c.definition
                    om = meta_by_slug.get(d_o.slug)
                    if om is not None and om.defn is d_o:
                        if om.is_baf_sink:
                            zero_or_sink += 1
                        continue
                if (
                    c.definition.cost if not cm else c.effective_cost
                ) == 0 or c.definition.title in {
                    "Assorted Cutlery",
                    "Address the Food",
                    "Bougie Coffee Machine",
                }:
                    zero_or_sink += 1
            val += zero_or_sink * 1.65
        elif title == "Evening on the Dock":
            val += 3.5
        elif title == "Couch Tube":
            val += 3.1

    if prof.goes_to_tomorrow and days_left >= 1:
        if meta is not None:
            val += meta.quick_tom_base
            if meta.quick_tom_extra != 0.0:
                val += meta.quick_tom_extra
        else:
            val += (
                prof.tomorrow_start_fun * 0.95 + prof.tomorrow_start_energy * 1.18
            )
            if title == "Photo Shoot":
                val += 3.7
            elif title == "Date's First Nos":
                val += 3.2
            elif title == "Floating":
                val += 0.65
            elif title == "Stay Up Late":
                val += 0.45

    if cost == 0 and eff_base_fun >= 0 and not is_zeroed:
        val += 0.65
    elif cost == 1 and eff_base_fun >= 0 and not is_zeroed:
        val += 0.25

    # Hydrogen-specific calibrated adjustments (applied in identical order after Carbon base)
    if h_qcode != _HQ_NONE:
        if h_qcode == _HQ_CHUCK_A_FRISBEE:
            if days_left >= 1 and not is_zeroed:
                val += 0.80 + 0.20 * (days_left if days_left < 3 else 3)
                if wanted_tags:
                    iter_tags_h = (
                        meta.sorted_tags
                        if (meta is not None and not has_all_tags)
                        else (c_tags if len(c_tags) <= 1 else sorted(c_tags))
                    )
                    for t in iter_tags_h:
                        w = wanted_tags.get(t, 0.0)
                        if w > 0.0:
                            val += 0.20 * (w if w < 3.0 else 3.0)
        elif h_qcode == _HQ_CITIES_AND_KNIGHTS:
            has_fodder_or_empty = False
            for other in hand:
                if other is card:
                    continue
                od = other.definition
                ot = od.title
                if ot == "Wakesurf" or ot == "Drop a Ski":
                    has_fodder_or_empty = True
                    break
                if od.base_fun <= 0:
                    if not other.markers:
                        om = meta_by_slug.get(od.slug)
                        if om is not None and om.defn is od:
                            if om.is_fodder_plain:
                                has_fodder_or_empty = True
                                break
                            continue
                    if not _beryllium_build_profile(other).goes_to_tomorrow:
                        has_fodder_or_empty = True
                        break
            val -= 0.55 if has_fodder_or_empty else 1.55
        elif h_qcode == _HQ_WILDLIFE_SPOTTING:
            if days_left >= 1 and not is_zeroed:
                if _ctx is not None and _ctx.all_canonical:
                    if in_hand is None:
                        in_hand = any(hc is card for hc in hand)
                    has_outdoors = (
                        bool(_MASK_OUTDOORS & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                        if in_hand
                        else bool(_MASK_OUTDOORS & _ctx.hand_mask_1)
                    )
                else:
                    has_outdoors = False
                    for other in hand:
                        if other is not card and "Outdoors" in (
                            other.definition.tags if not other.markers else other.tags
                        ):
                            has_outdoors = True
                            break
                val += 1.10 if has_outdoors else 0.75
        elif h_qcode == _HQ_WORK_CALL:
            if not is_zeroed:
                base_b = 7 if days_left >= 1 else 0
                budget_cap = (player.energy if player.energy > base_b else base_b) + 2
                best_hit = 0
                for other in hand:
                    if other is card:
                        continue
                    om = other.markers
                    o_uncopied = not om or "_copied_definition" not in om
                    o_defn = (
                        other.definition
                        if o_uncopied
                        else other.effective_definition
                    )
                    o_tags = o_defn.tags if not om else other.tags
                    if zero_tag and zero_tag in o_tags:
                        continue
                    if o_defn.base_fun >= 5 and o_defn.base_fun > best_hit:
                        o_prof = (
                            _CARBON_PROFILE_BY_DEFN_ID.get(id(o_defn))
                            if o_uncopied
                            else None
                        ) or _beryllium_build_profile(other)
                        if (
                            not o_prof.must_play_first
                            and cost + game.energy_cost(player_index, other)
                            <= budget_cap
                        ):
                            best_hit = o_defn.base_fun
                if best_hit >= 5:
                    hit_b = (best_hit - 4) * 0.35 + 0.55
                    val += 2.2 if hit_b > 2.2 else hit_b
        elif h_qcode == _HQ_FLOATING:
            if days_left >= 1:
                val += 0.60
        elif h_qcode == _HQ_MAKE_PLANS:
            if days_left >= 1 or hand_len <= 3:
                val += 0.55
        elif h_qcode == _HQ_COUCH_TUBE:
            if not is_zeroed:
                has_heavy = False
                for other in hand:
                    if other is not card and (
                        other.definition.cost
                        if not other.markers
                        else other.effective_cost
                    ) >= 4:
                        has_heavy = True
                        break
                val += 0.50 + (0.35 if has_heavy else 0.0)
        elif h_qcode == _HQ_HEAVY_TRIO:
            if not is_zeroed and (days_left >= 2 or cost <= player.energy + 3):
                val += 0.35
        elif h_qcode == _HQ_KEEPER:
            if _ctx is not None and _ctx.all_canonical:
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                has_k_out = bool(_MASK_OUTDOORS & _ctx.tom_mask_1) or (
                    bool(_MASK_OUTDOORS & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                    if in_hand
                    else bool(_MASK_OUTDOORS & _ctx.hand_mask_1)
                )
            else:
                has_k_out = False
                for other in hand:
                    if other is not card and "Outdoors" in (
                        other.definition.tags if not other.markers else other.tags
                    ):
                        has_k_out = True
                        break
                if not has_k_out:
                    for other in player.tomorrow_cards:
                        if other is not card and "Outdoors" in (
                            other.definition.tags if not other.markers else other.tags
                        ):
                            has_k_out = True
                            break
            if has_k_out:
                val += 0.40
        elif h_qcode == _HQ_FORCED_FAMILY_FUN:
            if days_left >= 2:
                val += 0.35
            else:
                if _ctx is not None and _ctx.all_canonical:
                    if in_hand is None:
                        in_hand = any(hc is card for hc in hand)
                    has_bg = (
                        bool(_MASK_BOARD_GAME & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                        if in_hand
                        else bool(_MASK_BOARD_GAME & _ctx.hand_mask_1)
                    )
                else:
                    has_bg = False
                    for other in hand:
                        if other is not card and "Board Game" in (
                            other.definition.tags if not other.markers else other.tags
                        ):
                            has_bg = True
                            break
                if has_bg:
                    val += 0.35
        elif h_qcode == _HQ_HIGH_END_RED:
            other_food = 0
            for zone in (hand, played_today):
                for c in zone:
                    if c is card:
                        continue
                    cm = c.markers
                    if not cm:
                        d_o = c.definition
                        om = meta_by_slug.get(d_o.slug)
                        if om is not None and om.defn is d_o:
                            if (om.tag_mask & _MASK_FOOD) and not om.must_play_first:
                                other_food += 1
                            continue
                    if "Food" in (
                        c.definition.tags if not cm else c.tags
                    ) and not _beryllium_build_profile(c).must_play_first:
                        other_food += 1
            if other_food < 2:
                val -= 0.55
        elif h_qcode == _HQ_JOHNNY_APPLESEED:
            val -= 0.55
        elif h_qcode == _HQ_STRETCH:
            has_later_ex = False
            for other in hand:
                if other is card:
                    continue
                om = other.markers
                if not om:
                    d_o = other.definition
                    ometa = meta_by_slug.get(d_o.slug)
                    if ometa is not None and ometa.defn is d_o:
                        if (
                            (ometa.tag_mask & _MASK_EXERCISE)
                            and ometa.cost <= 5
                            and not ometa.must_play_first
                        ):
                            has_later_ex = True
                            break
                        continue
                if (
                    "Exercise"
                    in (other.definition.tags if not om else other.tags)
                    and (other.definition.cost if not om else other.effective_cost)
                    <= 5
                    and not _beryllium_build_profile(other).must_play_first
                ):
                    has_later_ex = True
                    break
            if not has_later_ex:
                val -= 0.50
        elif h_qcode == _HQ_DOXOLOGY:
            val -= 0.45
        elif h_qcode == _HQ_BUG_SPRAY:
            out_cnt = 0
            for zone in (hand, player.tomorrow_cards):
                for c in zone:
                    if c is not card and "Outdoors" in (
                        c.definition.tags if not c.markers else c.tags
                    ):
                        out_cnt += 1
            if out_cnt < 2:
                val -= 0.40
        elif h_qcode == _HQ_SCRABBLE:
            has_pb = False
            for c in hand:
                if _effective_card_title(c) == "Paddleboat":
                    has_pb = True
                    break
            if not has_pb:
                for c in played_today:
                    if _effective_card_title(c) == "Paddleboat":
                        has_pb = True
                        break
            if not has_pb:
                val -= 0.50
        elif h_qcode == _HQ_SUNRISE or h_qcode == _HQ_EVENING_CHAT:
            val -= 0.45
        elif h_qcode == _HQ_PAINT:
            val -= 0.40
        elif h_qcode == _HQ_CLIFF_CLIMBING:
            if _ctx is not None and _ctx.all_canonical:
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                has_out = (
                    bool(_MASK_OUTDOORS & (_ctx.tom_mask_1 | _ctx.played_mask_1))
                    if in_hand
                    else bool(
                        _MASK_OUTDOORS
                        & (_ctx.hand_mask_1 | _ctx.tom_mask_1)
                    )
                )
                if not has_out:
                    if in_hand:
                        has_out = bool(
                            _MASK_OUTDOORS
                            & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2)
                        )
                    else:
                        for pc in played_today:
                            if pc is not card and "Outdoors" in (
                                pc.definition.tags if not pc.markers else pc.tags
                            ):
                                has_out = True
                                break
            else:
                has_out = False
                for zone in (hand, player.tomorrow_cards, played_today):
                    for c in zone:
                        if c is not card and "Outdoors" in (
                            c.definition.tags if not c.markers else c.tags
                        ):
                            has_out = True
                            break
                    if has_out:
                        break
            if not has_out:
                val -= 0.55
        elif h_qcode == _HQ_WATER_VOLLEYBALL:
            if _ctx is not None and _ctx.all_canonical:
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                has_ex = bool(_MASK_EXERCISE & _ctx.tom_mask_1) or (
                    bool(_MASK_EXERCISE & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                    if in_hand
                    else bool(_MASK_EXERCISE & _ctx.hand_mask_1)
                )
            else:
                has_ex = False
                for zone in (hand, player.tomorrow_cards):
                    for c in zone:
                        if c is not card and "Exercise" in (
                            c.definition.tags if not c.markers else c.tags
                        ):
                            has_ex = True
                            break
                    if has_ex:
                        break
            if not has_ex:
                val -= 0.40
        elif h_qcode == _HQ_BRACELET_MAKING:
            if _ctx is not None and _ctx.all_canonical:
                if in_hand is None:
                    in_hand = any(hc is card for hc in hand)
                has_soc = bool(_MASK_SOCIAL & _ctx.tom_mask_1) or (
                    bool(_MASK_SOCIAL & ((_ctx.hand_mask_1 & ~cmask) | _ctx.hand_mask_2))
                    if in_hand
                    else bool(_MASK_SOCIAL & _ctx.hand_mask_1)
                )
            else:
                has_soc = False
                for zone in (hand, player.tomorrow_cards):
                    for c in zone:
                        if c is not card and "Social" in (
                            c.definition.tags if not c.markers else c.tags
                        ):
                            has_soc = True
                            break
                    if has_soc:
                        break
            if not has_soc:
                val -= 0.35

    return val


def _hydrogen_hand_keep_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
    *,
    cost: int | None = None,
    _ctx: _QuickZoneCtx | None = None,
    _in_hand: bool | None = None,
    _meta: _HydrogenDefnMeta | None = None,
    _has_eod_hand: bool | None = None,
) -> float:
    """Value of keeping `card` in hand right now using Hydrogen calibrations."""
    player = game.players[player_index]
    meta = _meta if _meta is not None else _get_card_meta(card)
    can_play_now = (
        meta.beh.can_play(game, player, card) if meta.has_can_play else True
    )
    if cost is None:
        cost = game.energy_cost(player_index, card)
    if game.day >= DAY_COUNT:
        if not can_play_now:
            return -5.0
        extra_e = _fast_extra_energy_support(
            game, player_index, card, _ctx=_ctx, _in_hand=_in_hand
        )
        if cost > player.energy + extra_e:
            if _has_eod_hand is None:
                has_eod_hand = False
                for zone in (player.played_today, player.hand):
                    for c in zone:
                        cm = c.markers
                        ct = (
                            c.definition.title
                            if (not cm or "_copied_definition" not in cm)
                            else _effective_card_title(c)
                        )
                        if ct == "Wrestle the Kids" or ct == "Play With the Kids":
                            has_eod_hand = True
                            break
                    if has_eod_hand:
                        break
            else:
                has_eod_hand = _has_eod_hand
            if not has_eod_hand:
                return -4.0
    val = _hydrogen_quick_card_desirability(
        game,
        player_index,
        card,
        wanted_tags,
        cost=cost,
        prof=meta.prof,
        _ctx=_ctx,
        _in_hand=_in_hand,
        _meta=meta,
    )
    if game.day < DAY_COUNT and meta.title == "Chuck a Frisbee":
        val += 1.25
    if not can_play_now:
        val -= 1.25
    elif cost <= player.energy:
        val += 0.45
    return val


def _hydrogen_free_play_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float] | None = None,
) -> float:
    """Deterministic value of playing `card` for 0 Energy for Hydrogen."""
    if wanted_tags is None:
        wanted_tags = _wanted_tags_for_player_fast(game.players[player_index])
    val = _carbon_free_play_value(game, player_index, card, wanted_tags)
    player = game.players[player_index]
    zero_tag = (
        player.character.zero_fun_tag
        if player.character is not None
        else ""
    )
    title = _effective_card_title(card)
    if zero_tag and zero_tag in card.tags:
        return val
    if title == "Chuck a Frisbee" and game.day < DAY_COUNT:
        val += 1.10
    elif title == "Date's First Nos" and game.day < DAY_COUNT:
        val += 0.50
    elif title == "Wildlife Spotting" and game.day < DAY_COUNT:
        val += 1.00
    elif title == "Make Plans":
        val += 0.50
    return val


def _select_bohnanza_discards_fast(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> tuple[int, ...]:
    """Fast CoW equivalent of `_select_bohnanza_discards`."""
    if not options:
        return ()
    player = game.players[player_index]
    if len(player.played_today) <= 1:
        return ()
    sim = _hydrogen_copy_game_for_play(game, player_index)
    if getattr(sim, "_played_items_shared", False):
        sim._unshare_played_items()  # type: ignore[attr-defined]
    sim_player = sim.players[player_index]
    bohnanza_card = (
        sim_player.played_today[-1] if sim_player.played_today else None
    )
    if bohnanza_card is None or len(sim_player.played_today) <= 1:
        return ()

    days_left = DAY_COUNT - game.day
    hand = player.hand
    hand_tags: set[str] = set()
    has_acquirer_hand = False
    for hc in hand:
        hand_tags.update(hc.tags)
        if hc.title in _ACQUIRER_TITLES:
            has_acquirer_hand = True

    def _compute_affordable_hand() -> list[tuple[CardInstance, str]]:
        p_e = player.energy
        out_h: list[tuple[CardInstance, str]] = []
        for hc in hand:
            if sim.energy_cost(player_index, hc) <= p_e:
                out_h.append((hc, hc.title))
        return out_h

    affordable_hand = _compute_affordable_hand()

    def _board_fun() -> int:
        if type(sim) is _HYDROGEN_SIM_GAME_CLS:
            has_on_score = False
            for pc in sim_player.played_today:
                if _get_card_meta(pc).has_end_day_any and (
                    type(pc.effective_behavior).on_score
                    is not CardBehavior.on_score
                ):
                    has_on_score = True
                    break
            if not has_on_score:
                prev_s = sim._fun_at_start_of_scoring
                ps = sim.players
                sim._fun_at_start_of_scoring = (
                    ps[0].fun,
                    ps[1].fun,
                    ps[2].fun,
                    ps[3].fun,
                )
                try:
                    return sim.score_played_today_fast(player_index)[0]  # type: ignore[attr-defined]
                finally:
                    sim._fun_at_start_of_scoring = prev_s
        prev_s = sim._fun_at_start_of_scoring
        sim._fun_at_start_of_scoring = tuple(p.fun for p in sim.players)
        try:
            return sum(
                sim.card_fun(player_index, c) for c in sim_player.played_today
            )
        finally:
            sim._fun_at_start_of_scoring = prev_s

    def future_aura_value(card: CardInstance) -> float:
        meta = _get_card_meta(card)
        prof = meta.prof
        c_title = card.title
        aura = 0.0
        if prof.goes_to_tomorrow and days_left >= 1:
            t_val = prof.tomorrow_start_fun + prof.tomorrow_start_energy * 1.20
            if c_title == "Photo Shoot":
                t_val += 5.5
            elif c_title == "Date's First Nos":
                t_val += 6.0
            elif prof.signed_match_benefit != 0.0:
                t_val += prof.signed_match_benefit * 1.4
            if t_val > 0.0:
                aura += t_val
        if c_title == "Chuck a Frisbee" and days_left >= 1:
            aura += 2.2
        elif c_title == "Scout Other Cottages" and days_left >= 1:
            aura += 3.2
        if hand and player.energy > 0:
            if prof.is_multi_scaler and not prof.is_before_scaler:
                if prof.target_tags & hand_tags:
                    smb = prof.signed_match_benefit * 1.5
                    if smb > 0.0:
                        aura += smb
            elif prof.is_global_after_scaler:
                smb = prof.signed_match_benefit * 1.8
                if smb > 0.0:
                    aura += smb
            elif c_title == "Christmas Name Draw":
                aura += 2.5
            elif c_title == "Paddleboat" and has_acquirer_hand:
                aura += 3.0
        c_tags = card.tags
        played_len = len(sim_player.played_today)
        for hc, h_title in affordable_hand:
            if h_title == "Couch Tube":
                aura += card.definition.cost * 0.9
            elif h_title == "Evening on the Dock":
                aura += 0.9
            elif h_title == "New Nos Book Entry" and played_len <= 5:
                aura += 0.8
            elif h_title == "Cliff Climbing" and "Outdoors" in c_tags:
                aura += 1.8
            elif h_title == "Ice Wine" and "Food" in c_tags:
                aura += 1.8
            elif h_title == "Outdoor Movie" and "Relax" in c_tags:
                aura += 1.8
            elif h_title == "Bubly" and card.markers.get("_gave_energy"):
                aura += 1.8
        return aura

    chosen: list[int] = []
    option_clones = list(
        sim_player.played_today[
            : min(len(options), len(sim_player.played_today) - 1)
        ]
    )
    base_board_fun = _board_fun()

    for idx, clone in enumerate(option_clones):
        if clone not in sim_player.played_today:
            continue
        aura = future_aura_value(clone)
        if aura >= 3.0:
            continue
        pos = sim_player.played_today.index(clone)
        sim_player.played_today.pop(pos)
        bohnanza_card.markers["energy_cubes"] = (
            int(bohnanza_card.markers.get("energy_cubes", 0)) + 1
        )
        new_board_fun = _board_fun()
        net_gain = (new_board_fun - base_board_fun) - aura
        if net_gain > 0.25:
            chosen.append(idx)
            base_board_fun = new_board_fun
            affordable_hand = _compute_affordable_hand()
        else:
            sim_player.played_today.insert(pos, clone)
            bohnanza_card.markers["energy_cubes"] = (
                int(bohnanza_card.markers.get("energy_cubes", 0)) - 1
            )

    return tuple(chosen)


def _evaluate_istanbul_discards_fast(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> list[tuple[float, tuple[int, ...]]]:
    """Exact, fast CoW + cached implementation of `_evaluate_istanbul_discards`."""
    if not options:
        return [(0.0, ())]

    meta_by_slug = _HYDROGEN_META_BY_SLUG
    if not meta_by_slug:
        _ensure_hydrogen_caches()

    live_player = game.players[player_index]
    n = len(options)
    can_cache = len(live_player.hand) == n
    has_opp_sensitive = False
    is_can_sim = getattr(game, "_all_canonical_defs", False)
    if can_cache:
        for hc, c in zip(live_player.hand, options):
            d_c = c.definition
            if c.markers or hc is not c or (
                not is_can_sim
                and (
                    (m_c := meta_by_slug.get(d_c.slug)) is None
                    or m_c.defn is not d_c
                )
            ):
                can_cache = False
                break
            if d_c.slug in _ISTANBUL_OPP_SENSITIVE_SLUGS:
                has_opp_sensitive = True
    if can_cache:
        for tc in live_player.tomorrow_cards:
            d_t = tc.definition
            if tc.markers or (
                not is_can_sim
                and (
                    (m_t := meta_by_slug.get(d_t.slug)) is None
                    or m_t.defn is not d_t
                )
            ):
                can_cache = False
                break
    if can_cache:
        for pc in live_player.played_today:
            pm = pc.markers
            d_p = pc.definition
            if (
                pm and not (pm.keys() <= _ISTANBUL_SIMPLE_MARKER_KEYS)
            ) or (
                not is_can_sim
                and (
                    (m_p := meta_by_slug.get(d_p.slug)) is None
                    or m_p.defn is not d_p
                )
            ):
                can_cache = False
                break
            if d_p.slug in _ISTANBUL_OPP_SENSITIVE_SLUGS:
                has_opp_sensitive = True

    cache_key: tuple | None = None
    if can_cache:
        if has_opp_sensitive and not is_can_sim:
            meta_by_id = _HYDROGEN_META_BY_DEFN_ID
            for idx_o, opp in enumerate(game.players):
                if idx_o == player_index:
                    continue
                for oc in opp.played_today:
                    if id(oc.definition) not in meta_by_id or (
                        oc.markers
                        and "_copied_definition" in oc.markers
                        and id(_effective_source_definition(oc)) not in meta_by_id
                    ):
                        can_cache = False
                        break
                if not can_cache:
                    break
        if can_cache:
            if has_opp_sensitive:
                p_fun = live_player.fun
                opp_key: tuple = tuple(
                    (
                        opp.fun > p_fun,
                        opp.asleep,
                        len(opp.hand),
                        tuple(
                            (
                                oc.definition.slug,
                                _effective_card_title(oc),
                                bool(oc.markers.get("_all_tags")),
                            )
                            if oc.markers
                            else oc.definition.slug
                            for oc in opp.played_today
                        ),
                    )
                    for idx_o, opp in enumerate(game.players)
                    if idx_o != player_index
                )
            else:
                opp_key = ()
            cache_key = (
                game.day,
                game.daily_condition,
                live_player.character,
                live_player.energy,
                live_player.asleep,
                opp_key,
                tuple(tc.definition.slug for tc in live_player.tomorrow_cards),
                tuple(
                    (pc.definition.slug, bool(pc.markers.get("_gave_energy")))
                    if pc.markers
                    else pc.definition.slug
                    for pc in live_player.played_today
                ),
                tuple(c.definition.slug for c in options),
            )
            cached = _HYDROGEN_ISTANBUL_CACHE.get(cache_key)
            if cached is not None:
                return cached

    if not live_player.played_today:
        return [(0.0, ())]

    sim = _hydrogen_copy_game_for_play(game, player_index)
    if getattr(sim, "_played_items_shared", False):
        sim._unshare_played_items()  # type: ignore[attr-defined]
    player = sim.players[player_index]
    source_card = player.played_today[-1] if player.played_today else None
    if source_card is None:
        return [(0.0, ())]

    raw_e_per = _istanbul_energy_per_discard(source_card)
    allows_e = raw_e_per > 0 and _allows_energy_gain_today(
        sim, player, source_card
    )
    e_per = raw_e_per if allows_e else 0
    orig_energy = player.energy
    days_left = DAY_COUNT - sim.day
    wanted_tags = _wanted_tags_for_player_fast(player)

    sim_opts = [
        CardInstance(
            c.instance_id,
            c.definition,
            c.is_tomorrow,
            dict(c.markers) if c.markers else {},
        )
        for c in options
    ]

    sim._sync_played_today_tags(player)
    orig_played = tuple(player.played_today)
    pending_post_e = (
        2
        if (
            allows_e
            and len(orig_played) >= 2
            and _effective_card_title(orig_played[-2]) == "Beach Volleyball"
            and "Indoors" in source_card.tags
        )
        else 0
    )
    eff_start_energy = orig_energy + pending_post_e

    max_opp_hand = 0
    for idx_o, opp in enumerate(sim.players):
        if idx_o != player_index:
            oh = len(opp.hand)
            if oh > max_opp_hand:
                max_opp_hand = oh

    baf_positions = [
        idx_p
        for idx_p, pc in enumerate(orig_played)
        if _effective_card_title(pc) == "Bring a Friend"
    ]
    char = player.character

    def _prior_card_mult(pos: int, pc: CardInstance) -> float:
        if (
            char is not None
            and char.zero_fun_tag
            and char.zero_fun_tag in pc.tags
        ):
            return 0.0
        mult = 1.0
        for bp in baf_positions:
            if bp < pos:
                mult *= 2.0
        if pos >= 1 and _effective_card_title(orig_played[pos - 1]) == "Work Call":
            mult *= 2.0
        return mult

    future_mult = float(2 ** len(baf_positions))
    decaf_played = 0.0
    wrestle_played = 0.0
    pwtk_played = 0.0
    wakesurf_played = 0.0
    slalom_played = 0.0
    paddleboat_played = 0.0
    prior_gave_energy = pending_post_e > 0
    for pos, pc in enumerate(orig_played):
        if not prior_gave_energy and pc.markers.get("_gave_energy"):
            prior_gave_energy = True
        pt = _effective_card_title(pc)
        if pt == "Decaf":
            decaf_played += _prior_card_mult(pos, pc)
        elif pt == "Wrestle the Kids":
            wrestle_played += _prior_card_mult(pos, pc)
        elif pt == "Play With the Kids":
            pwtk_played += _prior_card_mult(pos, pc)
        elif pt == "Wakesurf":
            wakesurf_played += _prior_card_mult(pos, pc)
        elif pt == "Drop a Ski":
            slalom_played += _prior_card_mult(pos, pc)
        elif pt == "Paddleboat":
            paddleboat_played += _prior_card_mult(pos, pc)

    post_draws = 0
    s_tags = source_card.tags
    if s_tags & {"Food", "Board Game"}:
        for pc in orig_played[:-1]:
            if _effective_card_title(pc) == "Euchre Awards Ceremony":
                post_draws += 1
    if s_tags & {"Social", "Relax"}:
        for tc in player.tomorrow_cards:
            if _effective_card_title(tc) == "Photo Shoot":
                post_draws += 1

    neutral_pair = [
        CardInstance(-951, _NEUTRAL_DRAFT_DEFNS[0]),
        CardInstance(-952, _NEUTRAL_DRAFT_DEFNS[1]),
    ]

    profs: list[CardSynergyProfile] = []
    costs: list[int] = []
    can_plays: list[bool] = []
    titles: list[str] = []
    card_mults: list[float] = []
    funded_des: list[float] = []
    gross_play_vals: list[float] = []
    hold_surpluses: list[float] = []

    orig_hand = player.hand
    sim_hand = list(sim_opts)
    is_hsim = type(sim) is _HYDROGEN_SIM_GAME_CLS
    if is_hsim:
        sim._recompute_player_flags(player_index)  # type: ignore[attr-defined]
    try:
        player.energy = 0
        player.hand = neutral_pair
        if is_hsim:
            base_prior_fun = sim.score_played_today_fast(player_index)[0]  # type: ignore[attr-defined]
        else:
            base_prior_fun = sum(
                sim.card_fun(player_index, pc) for pc in orig_played
            )

        player.hand = sim_hand
        player.energy = eff_start_energy
        batch_costs = (
            sim.cards_energy_costs(player_index, sim_opts)  # type: ignore[attr-defined]
            if is_hsim
            else None
        )
        no_hand_size_played = (
            wrestle_played == 0.0
            and pwtk_played == 0.0
            and wakesurf_played == 0.0
            and slalom_played == 0.0
        )

        for idx_opt, c in enumerate(sim_opts):
            c_meta = _get_card_meta(c)
            prof = c_meta.prof
            title = (
                _effective_card_title(c)
                if (c.markers and "_copied_definition" in c.markers)
                else c_meta.title
            )
            player.hand = sim_hand
            player.energy = eff_start_energy
            can_p = (
                c_meta.beh.can_play(sim, player, c)
                if c_meta.has_can_play
                else True
            )
            cost_c = (
                batch_costs[idx_opt]
                if batch_costs is not None
                else sim.energy_cost(player_index, c)
            )

            player.energy = eff_start_energy if eff_start_energy > 30 else 30
            f_des = _beryllium_quick_card_desirability(
                sim, player_index, c, wanted_tags, cost=cost_c, prof=prof
            )

            had_all_tags = "_all_tags" in c.markers
            orig_pf = sim._p_flags[player_index] if is_hsim else 0  # type: ignore[attr-defined]
            player.played_today.append(c)
            if is_hsim:
                extra_pf = c_meta.played_board_flags
                if c.markers and "_copied_definition" in c.markers:
                    extra_pf |= _PF_HAS_COPIED
                sim._p_flags[player_index] = orig_pf | extra_pf  # type: ignore[attr-defined]
            own_fun = 0
            try:
                player.energy = 0
                player.hand = neutral_pair
                if is_hsim and title not in {"Wakesurf", "Drop a Ski"}:
                    if no_hand_size_played:
                        player.hand = []
                        total_with_c = sim.score_played_today_fast(player_index)[0]  # type: ignore[attr-defined]
                        if title == "Wingspan":
                            own_fun = sim.card_fun(player_index, c)
                        base_board_fun = float(total_with_c - base_prior_fun)
                    else:
                        total_np = sim.score_played_today_fast(player_index)[0]  # type: ignore[attr-defined]
                        own_np = sim.card_fun(player_index, c)
                        prior_delta = (total_np - own_np) - base_prior_fun
                        player.hand = []
                        own_fun = sim.card_fun(player_index, c)
                        base_board_fun = float(own_fun + prior_delta)
                    c_mult = (
                        0.0
                        if (
                            char is not None
                            and char.zero_fun_tag
                            and char.zero_fun_tag in c.tags
                        )
                        else future_mult
                    )
                else:
                    prior_delta = (
                        sum(sim.card_fun(player_index, pc) for pc in orig_played)
                        - base_prior_fun
                    )
                    c_mult = (
                        0.0
                        if (
                            char is not None
                            and char.zero_fun_tag
                            and char.zero_fun_tag in c.tags
                        )
                        else future_mult
                    )
                    if title not in {"Wakesurf", "Drop a Ski"}:
                        player.hand = []
                    own_fun = sim.card_fun(player_index, c)
                    base_board_fun = float(own_fun + prior_delta)
            finally:
                player.played_today.pop()
                if is_hsim:
                    sim._p_flags[player_index] = orig_pf  # type: ignore[attr-defined]
                if not had_all_tags:
                    c.markers.pop("_all_tags", None)
                if (
                    not is_hsim
                    or (orig_pf & _PF_ALL_TAGS)
                    or c_meta.after_all_tags
                    or had_all_tags
                ):
                    sim._sync_played_today_tags(player)

            imm_fun_bonus = (
                float(own_fun)
                if title == "Wingspan"
                else prof.immediate_fun_gain * c_mult
            )
            g_val = base_board_fun + imm_fun_bonus
            if title == "Nos Book":
                g_val += 3.0
            elif title == "San Juan":
                if any(
                    not opp.asleep
                    and len(opp.played_today) <= 2
                    and not any(
                        "Board Game" in oc.tags for oc in opp.played_today
                    )
                    for idx_o, opp in enumerate(sim.players)
                    if idx_o != player_index
                ):
                    g_val += 2.2 * c_mult
            elif title == "Quacks" or title == "Poker":
                g_val += 1.8 * c_mult
            elif title == "Bohnanza":
                if any(
                    sim.card_fun(player_index, pc) <= 1 for pc in orig_played
                ):
                    g_val += 2.2 * c_mult

            if days_left >= 1:
                if prof.goes_to_tomorrow:
                    g_val += (
                        prof.tomorrow_start_fun * 0.95
                        + prof.tomorrow_start_energy * 1.20
                        + max(0.0, prof.signed_match_benefit) * 0.90
                    )
                    if title == "Photo Shoot":
                        g_val += 3.5
                    elif title == "Date's First Nos":
                        g_val += 2.5
                elif title == "Chuck a Frisbee":
                    g_val += 2.0
                elif title == "Scout Other Cottages":
                    g_val += 2.8

            if (
                prof.immediate_cards_gained > 0
                and not prof.penalizes_before
                and not prof.penalizes_after
            ):
                g_val += min(prof.immediate_cards_gained, 3) * (
                    1.10 + 2.0 * paddleboat_played
                )
            elif (
                title == "Medical Advice"
                and days_left >= 1
                and len(orig_played) <= 2
            ):
                g_val += (
                    3.0 - 0.9 * len(orig_played) + 6.0 * paddleboat_played
                )

            if (
                (prof.is_multi_scaler and not prof.is_before_scaler)
                or prof.is_global_after_scaler
                or prof.is_single_buff
            ) and prof.signed_match_benefit > 0:
                g_val += prof.signed_match_benefit * 0.85

            h_surp = 0.0 if days_left <= 0 else (max(0.0, f_des) + 0.30)

            profs.append(prof)
            costs.append(cost_c)
            can_plays.append(can_p)
            titles.append(title)
            card_mults.append(c_mult)
            funded_des.append(f_des)
            gross_play_vals.append(g_val)
            hold_surpluses.append(h_surp)
    finally:
        player.hand = orig_hand
        player.energy = orig_energy

    relax_played_cnt = sum(1 for pc in orig_played if "Relax" in pc.tags)
    outdoors_played_cnt = sum(1 for pc in orig_played if "Outdoors" in pc.tags)
    exercise_played_cnt = sum(1 for pc in orig_played if "Exercise" in pc.tags)
    static_ramp: list[int] = []
    ramp_mode: list[int] = []  # 0=static, 2=butter_tarts
    for idx_i in range(n):
        if not allows_e:
            static_ramp.append(0)
            ramp_mode.append(0)
            continue
        t = titles[idx_i]
        p = profs[idx_i]
        if t == "Würstchen":
            static_ramp.append(len(orig_played))
            ramp_mode.append(0)
        elif t == "Butter Tarts":
            static_ramp.append(0)
            ramp_mode.append(2)
        elif t == "Pudding Chômeur":
            static_ramp.append(3)
            ramp_mode.append(0)
        elif p.conditional_energy_gain > 0 and p.immediate_energy_gain == 0:
            if t == "Potato Pancakes":
                static_ramp.append(2 * relax_played_cnt)
            elif t == "Keeper":
                static_ramp.append(outdoors_played_cnt)
            elif t == "Camp Crossroads Dessert":
                static_ramp.append(exercise_played_cnt)
            else:
                static_ramp.append(
                    p.conditional_energy_gain
                    if _condition_met_in_played(player, sim_opts[idx_i], p)
                    else 0
                )
            ramp_mode.append(0)
        else:
            static_ramp.append(p.immediate_energy_gain)
            ramp_mode.append(0)

    def ramp_gain_for(idx_i: int, cur_e: int, rem_cards: int) -> int:
        if ramp_mode[idx_i] == 0:
            return static_ramp[idx_i]
        ce = cur_e - costs[idx_i]
        return ce if ce > 0 else 0

    # Integer codes: 0=other, 1=Kayak, 2=Decaf, 3=Bubly, 4=Wrestle, 5=PWTK, 6=Wakesurf, 7=DropASki, 8=Pudding
    ist_kinds: list[int] = []
    has_any_kdb = False
    for t in titles:
        if t == "Kayak":
            ist_kinds.append(1)
            has_any_kdb = True
        elif t == "Decaf":
            ist_kinds.append(2)
            has_any_kdb = True
        elif t == "Bubly":
            ist_kinds.append(3)
            has_any_kdb = True
        elif t == "Wrestle the Kids":
            ist_kinds.append(4)
        elif t == "Play With the Kids":
            ist_kinds.append(5)
        elif t == "Wakesurf":
            ist_kinds.append(6)
        elif t == "Drop a Ski":
            ist_kinds.append(7)
        elif t == "Pudding Chômeur":
            ist_kinds.append(8)
        else:
            ist_kinds.append(0)

    candidate_tuples: list[tuple[int, ...]] = []
    if n <= 6:
        for k in range(n + 1):
            candidate_tuples.extend(combinations(range(n), k))
    else:
        seen_tuples: set[tuple[int, ...]] = set()
        hold_75 = [hs * 0.75 for hs in hold_surpluses]
        unreach_vals = (
            hold_75
            if days_left >= 1
            else [-10.0 + 0.01 * fd for fd in funded_des]
        )
        n_range = tuple(range(n))

        for i in range(n):
            t1 = (i,)
            seen_tuples.add(t1)
            candidate_tuples.append(t1)
        if n <= 8:
            for cand in combinations(n_range, 2):
                if cand not in seen_tuples:
                    seen_tuples.add(cand)
                    candidate_tuples.append(cand)

        for k in range(n + 1):
            e_k = eff_start_energy + k * e_per
            m_kept = n - k
            extra_ramp = 0
            for i in range(n):
                if can_plays[i] and costs[i] <= e_k:
                    rg = ramp_gain_for(i, e_k, m_kept)
                    if rg > costs[i]:
                        extra_ramp += rg - costs[i]
            reach_e = e_k + extra_ramp
            bubly_k = not prior_gave_energy and k > 0 and e_per > 0

            base_p_k: list[float] = [0.0] * n
            reach_ok: list[bool] = [False] * n
            for i in range(n):
                if can_plays[i] and costs[i] <= reach_e:
                    reach_ok[i] = True
                    rg = ramp_gain_for(i, e_k, m_kept)
                    c_i = costs[i]
                    net_e_bonus = (rg - c_i) * 1.15 if rg > c_i else 0.0
                    ik = ist_kinds[i]
                    extra_sink = 0.0
                    if ik == 1:
                        rem_k = e_k - c_i
                        if rem_k > 0:
                            extra_sink = (rem_k if rem_k < 8 else 8) * 0.85 * card_mults[i]
                    elif ik == 2:
                        rem_k = e_k - c_i
                        if rem_k > 0:
                            extra_sink = (rem_k if rem_k < 8 else 8) * 0.80 * card_mults[i]
                    elif ik == 3 and bubly_k:
                        extra_sink = 2.0 * card_mults[i]
                    base_p_k[i] = gross_play_vals[i] + extra_sink + net_e_bonus

            for cost_w in (0.18, 0.52, 1.05):
                r_keys = [
                    (
                        hold_75[i]
                        if hold_75[i] > (pv := base_p_k[i] - costs[i] * cost_w)
                        else pv
                    )
                    if reach_ok[i]
                    else unreach_vals[i]
                    for i in range(n)
                ]
                ordered = sorted(n_range, key=r_keys.__getitem__)
                cand_k = tuple(sorted(ordered[:k]))
                if cand_k not in seen_tuples:
                    seen_tuples.add(cand_k)
                    candidate_tuples.append(cand_k)

    slot_weights = (1.0, 0.92, 0.50, 0.25)
    is_kd = [ik == 1 or ik == 2 for ik in ist_kinds]
    imm_draws_cap2 = [
        p.immediate_cards_gained if p.immediate_cards_gained < 2 else 2
        for p in profs
    ]
    hold_55 = [hs * 0.55 for hs in hold_surpluses]
    inv_costs = [1.0 / (c if c > 1 else 1) for c in costs]
    surplus_desc_indices = sorted(
        range(n), key=hold_surpluses.__getitem__, reverse=True
    )
    ramp_mask = 0
    for idx_i in range(n):
        if can_plays[idx_i] and (
            ramp_mode[idx_i] != 0 or static_ramp[idx_i] > costs[idx_i]
        ):
            ramp_mask |= 1 << idx_i
    all_n_mask = (1 << n) - 1

    all_playable = [j for j in range(n) if can_plays[j]]
    base_net_vals = [gross_play_vals[j] - hold_55[j] for j in range(n)]
    base_eff_vals = [base_net_vals[j] * inv_costs[j] for j in range(n)]
    sorted_eff_by_k: list[list[int]] = []
    sorted_peak_by_k: list[list[int]] = []
    if len(all_playable) <= 1:
        for _ in range(n + 1):
            sorted_eff_by_k.append(all_playable)
            sorted_peak_by_k.append(all_playable)
    elif not has_any_kdb:
        s_eff = sorted(all_playable, key=base_eff_vals.__getitem__, reverse=True)
        s_peak = sorted(all_playable, key=base_net_vals.__getitem__, reverse=True)
        for _ in range(n + 1):
            sorted_eff_by_k.append(s_eff)
            sorted_peak_by_k.append(s_peak)
    else:
        for k_idx in range(n + 1):
            se_k = eff_start_energy + k_idx * e_per
            b_act_k = (prior_gave_energy or (k_idx > 0 and e_per > 0)) and not prior_gave_energy
            pk_vals = base_net_vals.copy()
            ef_vals = base_eff_vals.copy()
            for j in all_playable:
                ik = ist_kinds[j]
                if ik == 3:
                    if b_act_k:
                        adj = 2.0 * card_mults[j]
                        pk_vals[j] = (gross_play_vals[j] + adj) - hold_55[j]
                        ef_vals[j] = pk_vals[j] * inv_costs[j]
                elif ik == 1:
                    rem = se_k - costs[j]
                    if rem > 0:
                        adj = (rem if rem < 10 else 10) * 0.85 * card_mults[j]
                        pk_vals[j] = (gross_play_vals[j] + adj) - hold_55[j]
                        ef_vals[j] = pk_vals[j] * inv_costs[j]
                elif ik == 2:
                    rem = se_k - costs[j]
                    if rem > 0:
                        adj = (rem if rem < 10 else 10) * 0.80 * card_mults[j]
                        pk_vals[j] = (gross_play_vals[j] + adj) - hold_55[j]
                        ef_vals[j] = pk_vals[j] * inv_costs[j]
            sorted_eff_by_k.append(
                sorted(all_playable, key=ef_vals.__getitem__, reverse=True)
            )
            sorted_peak_by_k.append(
                sorted(all_playable, key=pk_vals.__getitem__, reverse=True)
            )

    def _simulate_playout(
        order: list[int],
        cur_e: int,
        cur_gave_e: bool,
        today_val: float,
        extra_draws: int,
        unplayed_mask: int,
        m_rem: int,
        k: int,
    ) -> float:
        decaf_w = decaf_played
        wrestle_w = wrestle_played
        pwtk_w = pwtk_played
        wakesurf_w = wakesurf_played
        slalom_w = slalom_played
        kayak_w = 0.0
        for idx_u in order:
            c_cost = costs[idx_u]
            if c_cost > cur_e:
                continue
            ik = ist_kinds[idx_u]
            cm = card_mults[idx_u]
            c_val = gross_play_vals[idx_u]
            proj_sink = 0.0
            if ik:
                if ik == 3:
                    if cur_gave_e and not prior_gave_energy:
                        c_val += 2.0 * cm
                elif ik == 1 or ik == 2:
                    rem = cur_e - c_cost
                    if rem > 0:
                        proj_sink = rem * cm
                elif ik == 4:
                    if m_rem > 1:
                        proj_sink = (m_rem - 1) * cm
                elif ik == 5:
                    if (m_rem - 1) > max_opp_hand:
                        proj_sink = 3.0 * cm
                elif ik == 6:
                    if m_rem == 1:
                        proj_sink = 6.0 * cm
                elif ik == 7:
                    if m_rem == 2:
                        proj_sink = 5.0 * cm

            eod_hand_cost = wrestle_w
            eod_decaf_cost = c_cost * decaf_w if decaf_w > 0.0 else 0.0
            hold_opp = hold_55[idx_u] if days_left >= 1 else 0.0
            if (
                c_val + proj_sink - eod_hand_cost - eod_decaf_cost
                <= hold_opp - 1e-6
            ):
                continue

            cur_e -= c_cost
            today_val += c_val
            unplayed_mask &= ~(1 << idx_u)
            m_rem -= 1
            extra_draws += imm_draws_cap2[idx_u]
            if ik:
                if ik == 1:
                    if cm > kayak_w:
                        kayak_w = cm
                elif ik == 2:
                    decaf_w += cm
                elif ik == 4:
                    wrestle_w += cm
                elif ik == 5:
                    pwtk_w += cm
                elif ik == 6:
                    wakesurf_w += cm
                elif ik == 7:
                    slalom_w += cm

        rem_sink_e = cur_e
        if extra_draws > 0 and rem_sink_e > 0:
            cap_d = 2 * extra_draws
            used_draw_e = rem_sink_e if rem_sink_e < cap_d else cap_d
            today_val += used_draw_e * 1.05
            rem_sink_e -= used_draw_e
        if rem_sink_e > 0 and (kayak_w > 0.0 or decaf_w > 0.0):
            per_e = kayak_w if kayak_w > decaf_w else decaf_w
            today_val += rem_sink_e * per_e

        if wrestle_w > 0.0:
            today_val += wrestle_w * m_rem
        if pwtk_w > 0.0 and m_rem > max_opp_hand:
            today_val += pwtk_w * 3.0
        if wakesurf_w > 0.0 and m_rem == 0:
            today_val += wakesurf_w * 6.0
        if slalom_w > 0.0 and m_rem == 1:
            today_val += slalom_w * 5.0

        reserve_val = 0.0
        if days_left >= 1 and m_rem > 0:
            acc_res = 0.0
            pos_r = 0
            for j in surplus_desc_indices:
                if unplayed_mask & (1 << j):
                    w_r = slot_weights[pos_r] if pos_r < 4 else 0.08
                    acc_res += hold_surpluses[j] * w_r
                    pos_r += 1
            reserve_val = 0.75 * acc_res

        return today_val + reserve_val - 0.02 * k

    def score_discard_tuple(disc_tuple: tuple[int, ...]) -> float:
        k = len(disc_tuple)
        disc_mask = 0
        for d_i in disc_tuple:
            disc_mask |= 1 << d_i
        kept_mask = all_n_mask ^ disc_mask
        start_e = eff_start_energy + k * e_per
        start_gave_e = prior_gave_energy or (k > 0 and e_per > 0)

        if ramp_mask & kept_mask:
            unplayed = [i for i in range(n) if (kept_mask & (1 << i))]
            cur_e = start_e
            cur_gave_e = start_gave_e
            today_val = 0.0
            extra_draws = post_draws
            progressed = True
            while progressed and unplayed:
                progressed = False
                other_cost_sum = 0
                has_kd = False
                for j in unplayed:
                    if can_plays[j]:
                        other_cost_sum += costs[j]
                    if is_kd[j]:
                        has_kd = True
                has_sink_card = decaf_played > 0.0 or has_kd or extra_draws > 0
                u_len = len(unplayed)
                for idx_u in unplayed:
                    if not can_plays[idx_u] or costs[idx_u] > cur_e:
                        continue
                    rg = ramp_gain_for(idx_u, cur_e, u_len)
                    if rg <= costs[idx_u]:
                        continue
                    need_e = (other_cost_sum > cur_e) or has_sink_card
                    h_keep = hold_55[idx_u]
                    if not need_e and gross_play_vals[idx_u] < h_keep:
                        continue
                    if ist_kinds[idx_u] == 8:
                        if u_len > 1:
                            fodder = -1
                            best_f_hs = float("inf")
                            for j in unplayed:
                                if (
                                    j != idx_u
                                    and hold_surpluses[j] < best_f_hs
                                ):
                                    best_f_hs = hold_surpluses[j]
                                    fodder = j
                            if (
                                not need_e
                                and gross_play_vals[idx_u]
                                < h_keep + hold_55[fodder]
                            ):
                                continue
                            unplayed.remove(fodder)
                    cur_e = cur_e - costs[idx_u] + rg
                    cur_gave_e = True
                    today_val += gross_play_vals[idx_u]
                    extra_draws += imm_draws_cap2[idx_u]
                    unplayed.remove(idx_u)
                    progressed = True
                    break
            playable_rem = [j for j in unplayed if can_plays[j]]
            unplayed_mask = 0
            for j in unplayed:
                unplayed_mask |= 1 << j
            m_rem = len(unplayed)
            if len(playable_rem) <= 1:
                return _simulate_playout(
                    playable_rem,
                    cur_e,
                    cur_gave_e,
                    today_val,
                    extra_draws,
                    unplayed_mask,
                    m_rem,
                    k,
                )
            if not has_any_kdb:
                order_eff = sorted(
                    playable_rem, key=base_eff_vals.__getitem__, reverse=True
                )
                order_peak = sorted(
                    playable_rem, key=base_net_vals.__getitem__, reverse=True
                )
            else:
                bubly_active = cur_gave_e and not prior_gave_energy
                pk_r: dict[int, float] = {}
                ef_r: dict[int, float] = {}
                for j in playable_rem:
                    ik = ist_kinds[j]
                    v = gross_play_vals[j]
                    if ik == 3:
                        if bubly_active:
                            v += 2.0 * card_mults[j]
                    elif ik == 1:
                        rem = cur_e - costs[j]
                        if rem > 0:
                            v += (rem if rem < 10 else 10) * 0.85 * card_mults[j]
                    elif ik == 2:
                        rem = cur_e - costs[j]
                        if rem > 0:
                            v += (rem if rem < 10 else 10) * 0.80 * card_mults[j]
                    nv = v - hold_55[j]
                    pk_r[j] = nv
                    ef_r[j] = nv * inv_costs[j]
                order_eff = sorted(playable_rem, key=ef_r.__getitem__, reverse=True)
                order_peak = sorted(playable_rem, key=pk_r.__getitem__, reverse=True)
            s0 = _simulate_playout(
                order_eff,
                cur_e,
                cur_gave_e,
                today_val,
                extra_draws,
                unplayed_mask,
                m_rem,
                k,
            )
            if order_peak == order_eff:
                return s0
            s1 = _simulate_playout(
                order_peak,
                cur_e,
                cur_gave_e,
                today_val,
                extra_draws,
                unplayed_mask,
                m_rem,
                k,
            )
            return s1 if s1 > s0 else s0

        m_rem = n - k
        order_eff = [j for j in sorted_eff_by_k[k] if (kept_mask & (1 << j))]
        if len(order_eff) <= 1:
            return _simulate_playout(
                order_eff,
                start_e,
                start_gave_e,
                0.0,
                post_draws,
                kept_mask,
                m_rem,
                k,
            )
        order_peak = [j for j in sorted_peak_by_k[k] if (kept_mask & (1 << j))]
        s0 = _simulate_playout(
            order_eff,
            start_e,
            start_gave_e,
            0.0,
            post_draws,
            kept_mask,
            m_rem,
            k,
        )
        if order_peak == order_eff:
            return s0
        s1 = _simulate_playout(
            order_peak,
            start_e,
            start_gave_e,
            0.0,
            post_draws,
            kept_mask,
            m_rem,
            k,
        )
        return s1 if s1 > s0 else s0

    scored = [(score_discard_tuple(cand), cand) for cand in candidate_tuples]
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    if cache_key is not None:
        if len(_HYDROGEN_ISTANBUL_CACHE) > 131072:
            _HYDROGEN_ISTANBUL_CACHE.clear()
        _HYDROGEN_ISTANBUL_CACHE[cache_key] = scored
    return scored


def _select_istanbul_discards_hydrogen_fast(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> tuple[int, ...]:
    return _evaluate_istanbul_discards_fast(game, player_index, options)[0][1]


def _fast_morning_denial_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    meta: _HydrogenDefnMeta | None,
) -> float:
    """Exact fast equivalent of morning hate-draft logic in `_suitcase_denial_value`."""
    cm = card.markers
    c_title = card.definition.title
    next_opp = game.players[(player_index + 1) & 3]
    tom_cards = next_opp.tomorrow_cards
    if not tom_cards:
        return (
            0.25
            if c_title
            in {"Colouring", "Euchre Awards Ceremony", "Butter Tarts"}
            else 0.0
        )
    c_tags = card.definition.tags if not cm else card.tags
    c_cost = (
        (meta.cost if meta is not None else card.definition.cost)
        if not cm
        else card.effective_cost
    )
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    opp_threat = 0.0
    for tc in tom_cards:
        tm = tc.markers
        t_uncopied = not tm or "_copied_definition" not in tm
        t_defn = tc.definition
        t_meta = meta_by_slug.get(t_defn.slug) if t_uncopied else None
        if t_meta is not None and t_meta.defn is not t_defn:
            t_meta = None
        t_title = t_defn.title if t_uncopied else _effective_card_title(tc)
        if (
            t_title == "Photo Shoot"
            and ("Social" in c_tags or "Relax" in c_tags)
            and c_cost <= 2
        ):
            opp_threat += 0.45
        elif t_title == "Date's First Nos" and c_cost <= 1:
            opp_threat += 0.35
        else:
            tc_prof = (
                t_meta.prof
                if t_meta is not None
                else _beryllium_build_profile(tc)
            )
            if tc_prof.signed_match_benefit >= 1.5 and (
                c_tags & tc_prof.target_tags
            ):
                opp_threat += 0.25
    if c_title in {"Colouring", "Euchre Awards Ceremony", "Butter Tarts"}:
        opp_threat += 0.25
    return opp_threat if opp_threat < 0.75 else 0.75


class _HydrogenEvalAI(_CarbonEvalAI):
    """Tactical rollout policy used inside HydrogenAI searches and sub-decisions."""

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if not options:
            raise ValueError("No suitcase card option was supplied")
        player = game.players[player_index]
        wanted_tags, ctx = _build_wanted_and_ctx(
            player, getattr(game, "_all_canonical_defs", False)
        )
        has_any_markers = False
        for opt in options:
            if opt.markers:
                has_any_markers = True
                break
        if has_any_markers:
            own_markers: set[str] | None = _carbon_own_suitcase_markers(player)
            pr_markers, scout_markers = _opponent_suitcase_markers(
                game, player_index
            )
        else:
            own_markers = None
            pr_markers = None
            scout_markers = None

        is_hsim = type(game) is _HYDROGEN_SIM_GAME_CLS
        is_can_sim = is_hsim and game._all_canonical_defs  # type: ignore[attr-defined]
        costs = (
            game.cards_energy_costs(player_index, options)  # type: ignore[attr-defined]
            if is_hsim
            else None
        )
        has_played = bool(player.played_today)
        p_energy = player.energy
        meta_by_slug = _HYDROGEN_META_BY_SLUG
        best_idx = 0
        best_score = float("-inf")
        for idx, card in enumerate(options):
            cm = card.markers
            if not cm:
                d_c = card.definition
                if is_can_sim:
                    meta: _HydrogenDefnMeta | None = meta_by_slug[d_c.slug]
                else:
                    meta = meta_by_slug.get(d_c.slug)
                    if meta is not None and meta.defn is not d_c:
                        meta = None
            else:
                meta = None
            cost = (
                costs[idx]
                if costs is not None
                else game.energy_cost(player_index, card)
            )
            score = _hydrogen_quick_card_desirability(
                game,
                player_index,
                card,
                wanted_tags,
                cost=cost,
                prof=meta.prof if meta is not None else None,
                _ctx=ctx,
                _in_hand=False if not cm else None,
                _meta=meta,
            )
            if has_played and cost <= p_energy:
                can_p = (
                    (
                        meta.beh.can_play(game, player, card)
                        if meta.has_can_play
                        else True
                    )
                    if meta is not None
                    else card.effective_behavior.can_play(game, player, card)
                )
                if can_p:
                    score += 0.65
                elif game.day >= DAY_COUNT:
                    score -= 3.0
            if cm:
                if own_markers:
                    for om_k in own_markers:
                        if om_k in cm:
                            score -= 3.0
                            break
                score += _suitcase_denial_value(
                    game, player_index, card, pr_markers, scout_markers
                )
            elif not has_played:
                score += _fast_morning_denial_value(
                    game, player_index, card, meta
                )
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        player = game.players[player_index]
        wanted_tags, ctx = _build_wanted_and_ctx(
            player, getattr(game, "_all_canonical_defs", False)
        )
        best_idx = playable_hand_indices[0]
        best_key = (99, 0.0)
        for hand_idx in playable_hand_indices:
            card = player.hand[hand_idx]
            meta = _get_card_meta(card)
            est = _hydrogen_quick_card_desirability(
                game,
                player_index,
                card,
                wanted_tags,
                prof=meta.prof,
                _ctx=ctx,
                _in_hand=True,
                _meta=meta,
            )
            bucket = meta.play_order_bucket
            if card.definition.title == "Nos Book" and not player.played_today:
                bucket = 6
                est -= 6.0
            key = (bucket, -est)
            if key < best_key:
                best_key = key
                best_idx = hand_idx
        return best_idx

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if not options:
            raise ValueError("No discard option was supplied")
        player = game.players[player_index]
        wanted_tags, ctx = _build_wanted_and_ctx(
            player, getattr(game, "_all_canonical_defs", False)
        )
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played == "Tell a Story" and len(options) >= 2:
            has_epic_duels = any(
                _effective_card_title(pc) == "Epic Duels"
                for pc in player.played_today
            )
            awake_opps = sum(
                1
                for idx, opp in enumerate(game.players)
                if idx != player_index and not opp.asleep
            )
            safe_to_setup = has_epic_duels or awake_opps == 0

            cutlery_cards = [
                c for c in options if c.title == "Assorted Cutlery"
            ]
            if cutlery_cards and (safe_to_setup or game.day >= DAY_COUNT):
                cutlery = cutlery_cards[0]
                cutlery_cost = game.energy_cost(player_index, cutlery)
                if cutlery_cost <= player.energy:
                    rem_e = player.energy - cutlery_cost
                    best_i: int | None = None
                    best_net = float("-inf")
                    for i, cand in enumerate(options):
                        if cand is cutlery and len(cutlery_cards) == 1:
                            continue
                        fp_val = _hydrogen_free_play_value(
                            game, player_index, cand, wanted_tags
                        )
                        if not safe_to_setup:
                            fp_val *= 0.45
                        cand_cost = game.energy_cost(player_index, cand)
                        cand_des = _hydrogen_hand_keep_value(
                            game, player_index, cand, wanted_tags, _ctx=ctx
                        )
                        if (
                            cand_cost <= rem_e
                            and cand.effective_behavior.can_play(
                                game, player, cand
                            )
                        ):
                            opp_cost = max(0.0, cand_des)
                        elif game.day >= DAY_COUNT:
                            opp_cost = 0.0
                        else:
                            opp_cost = max(0.0, (cand_des - 0.5) * 0.55)
                        net = fp_val - opp_cost
                        if net > best_net:
                            best_net = net
                            best_i = i
                    if best_i is not None and best_net >= max(
                        1.8, cutlery_cost * 0.95
                    ):
                        return best_i

            bcm_cards = [
                c for c in options if c.title == "Bougie Coffee Machine"
            ]
            if bcm_cards and safe_to_setup:
                bcm = bcm_cards[0]
                bcm_cost = game.energy_cost(player_index, bcm)
                if bcm_cost <= player.energy:
                    rem_e = player.energy - bcm_cost
                    best_f_i: int | None = None
                    best_f_net = float("-inf")
                    for i, cand in enumerate(options):
                        if (
                            cand is bcm and len(bcm_cards) == 1
                        ) or "Food" not in cand.tags:
                            continue
                        fp_val = _hydrogen_free_play_value(
                            game, player_index, cand, wanted_tags
                        )
                        cand_cost = game.energy_cost(player_index, cand)
                        cand_des = _hydrogen_hand_keep_value(
                            game, player_index, cand, wanted_tags, _ctx=ctx
                        )
                        opp_cost = (
                            max(0.0, cand_des)
                            if (
                                cand_cost <= rem_e
                                and cand.effective_behavior.can_play(
                                    game, player, cand
                                )
                            )
                            else (
                                0.0
                                if game.day >= DAY_COUNT
                                else max(0.0, (cand_des - 0.5) * 0.55)
                            )
                        )
                        net = fp_val - opp_cost
                        if net > best_f_net:
                            best_f_net = net
                            best_f_i = i
                    if best_f_i is not None and best_f_net >= 1.8:
                        return best_f_i

        is_hsim_d = type(game) is _HYDROGEN_SIM_GAME_CLS
        opt_costs = (
            game.cards_energy_costs(player_index, options)  # type: ignore[attr-defined]
            if is_hsim_d
            else None
        )
        eod_h: bool | None = None
        if game.day >= DAY_COUNT:
            eod_h = False
            for zone in (player.played_today, player.hand):
                for c_z in zone:
                    cm_z = c_z.markers
                    ct_z = (
                        c_z.definition.title
                        if (not cm_z or "_copied_definition" not in cm_z)
                        else _effective_card_title(c_z)
                    )
                    if ct_z == "Wrestle the Kids" or ct_z == "Play With the Kids":
                        eod_h = True
                        break
                if eod_h:
                    break
        best_disc_i = 0
        best_disc_v = float("inf")
        for i, opt_c in enumerate(options):
            v = _hydrogen_hand_keep_value(
                game,
                player_index,
                opt_c,
                wanted_tags,
                cost=opt_costs[i] if opt_costs is not None else None,
                _ctx=ctx,
                _has_eod_hand=eod_h,
            )
            if v < best_disc_v:
                best_disc_v = v
                best_disc_i = i
        return best_disc_i

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> Sequence[int]:
        if not options:
            return ()
        player = game.players[player_index]
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played == "Bohnanza":
            return _select_bohnanza_discards_fast(game, player_index, options)
        if last_played == "Istanbul":
            return _select_istanbul_discards_hydrogen_fast(
                game, player_index, options
            )
        if last_played == "Navy SEALing":
            wanted_tags, ctx = _build_wanted_and_ctx(
                player, getattr(game, "_all_canonical_defs", False)
            )
            opt_costs_ns = (
                game.cards_energy_costs(player_index, options)  # type: ignore[attr-defined]
                if type(game) is _HYDROGEN_SIM_GAME_CLS
                else None
            )
            eod_ns: bool | None = None
            if game.day >= DAY_COUNT:
                eod_ns = False
                for zone in (player.played_today, player.hand):
                    for c_z in zone:
                        cm_z = c_z.markers
                        ct_z = (
                            c_z.definition.title
                            if (not cm_z or "_copied_definition" not in cm_z)
                            else _effective_card_title(c_z)
                        )
                        if ct_z == "Wrestle the Kids" or ct_z == "Play With the Kids":
                            eod_ns = True
                            break
                    if eod_ns:
                        break
            vals = [
                _hydrogen_hand_keep_value(
                    game,
                    player_index,
                    options[i],
                    wanted_tags,
                    cost=opt_costs_ns[i] if opt_costs_ns is not None else None,
                    _ctx=ctx,
                    _has_eod_hand=eod_ns,
                )
                for i in range(len(options))
            ]
            ranked = sorted(range(len(options)), key=lambda i: vals[i])
            is_doubled = any(
                _effective_card_title(c) == "Bring a Friend"
                for c in player.played_today[:-1]
            ) or (
                len(player.played_today) >= 2
                and _effective_card_title(player.played_today[-2]) == "Work Call"
            )
            thresh = 4.2 if is_doubled else 2.2
            return tuple(i for i in ranked if vals[i] < thresh)
        if last_played in {"Scrabble", "Last Year's Shirt"}:
            if not game.suitcase:
                return ()
            wanted_tags, ctx = _build_wanted_and_ctx(
            player, getattr(game, "_all_canonical_defs", False)
        )
            own_markers = _carbon_own_suitcase_markers(player)
            pr_markers, scout_markers = _opponent_suitcase_markers(
                game, player_index
            )
            paddleboat_bonus = 0.0
            want_empty_hand = False
            has_eod_hand = False
            for c in player.played_today:
                ct = _effective_card_title(c)
                if ct == "Paddleboat":
                    paddleboat_bonus += 2.0
                elif ct == "Wakesurf" or ct == "Drop a Ski":
                    want_empty_hand = True
                elif ct == "Wrestle the Kids" or ct == "Play With the Kids":
                    has_eod_hand = True
            if not has_eod_hand and game.day >= DAY_COUNT:
                for c in player.hand:
                    ct = _effective_card_title(c)
                    if ct == "Wrestle the Kids" or ct == "Play With the Kids":
                        has_eod_hand = True
                        break

            def sc_value(sc: CardInstance) -> float:
                sc_cost = game.energy_cost(player_index, sc)
                sc_meta = _get_card_meta(sc)
                v = _hydrogen_quick_card_desirability(
                    game,
                    player_index,
                    sc,
                    wanted_tags,
                    cost=sc_cost,
                    prof=sc_meta.prof,
                    _ctx=ctx,
                    _in_hand=False,
                    _meta=sc_meta,
                )
                if own_markers and sc.markers:
                    for m_k in own_markers:
                        if m_k in sc.markers:
                            v -= 3.0
                            break
                v += _suitcase_denial_value(
                    game, player_index, sc, pr_markers, scout_markers
                )
                can_play_now = sc_cost <= player.energy and (
                    sc_meta.beh.can_play(game, player, sc)
                    if sc_meta.has_can_play
                    else True
                )
                if can_play_now:
                    v += 0.55 + (1.8 if want_empty_hand else 0.0)
                elif game.day >= DAY_COUNT:
                    v = -1.5
                return v + paddleboat_bonus

            def hand_card_value(hc: CardInstance) -> float:
                hc_cost = game.energy_cost(player_index, hc)
                hc_meta = _get_card_meta(hc)
                v = _hydrogen_quick_card_desirability(
                    game,
                    player_index,
                    hc,
                    wanted_tags,
                    cost=hc_cost,
                    prof=hc_meta.prof,
                    _ctx=ctx,
                    _meta=hc_meta,
                )
                can_play_now = hc_cost <= player.energy and (
                    hc_meta.beh.can_play(game, player, hc)
                    if hc_meta.has_can_play
                    else True
                )
                if can_play_now:
                    v += 0.55 + (1.8 if want_empty_hand else 0.0)
                elif game.day >= DAY_COUNT and not has_eod_hand:
                    v = -1.8
                return v

            suitcase_vals = sorted(
                (sc_value(sc) for sc in game.suitcase),
                reverse=True,
            )
            h_vals = [hand_card_value(opt) for opt in options]
            hand_ranked = sorted(range(len(options)), key=lambda i: h_vals[i])
            limit = min(
                len(suitcase_vals),
                len(game.suitcase) if last_played == "Last Year's Shirt" else 3,
            )
            base_margin = (
                0.20
                if paddleboat_bonus > 0
                else (1.10 if last_played == "Last Year's Shirt" else 0.85)
            )
            chosen: list[int] = []
            for k, i in enumerate(hand_ranked[:limit]):
                h_val = h_vals[i]
                gift_tax = (
                    0.35 * max(0.0, h_val - 1.2)
                    if (
                        last_played == "Last Year's Shirt"
                        and paddleboat_bonus == 0.0
                        and game.day < DAY_COUNT
                    )
                    else 0.0
                )
                if h_val + gift_tax < suitcase_vals[k] - base_margin:
                    chosen.append(i)
                else:
                    break
            return tuple(chosen)
        return ()

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        player = game.players[player_index]
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played in {"Christmas Name Draw", "Nos Book"}:
            return super().choose_card_target(game, player_index, options)
        if last_played == "Address the Food":
            wanted_tags = _wanted_tags_for_player_fast(player)
            own_markers = _carbon_own_suitcase_markers(player)
            pr_markers, scout_markers = _opponent_suitcase_markers(
                game, player_index
            )

            def food_target_val(i: int) -> float:
                opt = options[i]
                denial = _suitcase_denial_value(
                    game, player_index, opt, pr_markers, scout_markers
                )
                own_pen = (
                    -3.0
                    if (own_markers and any(m in opt.markers for m in own_markers))
                    else 0.0
                )
                if opt.effective_behavior.can_play(game, player, opt):
                    return (
                        _hydrogen_free_play_value(
                            game, player_index, opt, wanted_tags
                        )
                        + denial
                        + own_pen
                    )
                return (
                    _hydrogen_quick_card_desirability(
                        game, player_index, opt, wanted_tags
                    )
                    - 2.0
                    + denial
                    + own_pen
                )

            return max(range(len(options)), key=food_target_val)

        wanted_tags = _wanted_tags_for_player_fast(player)
        return max(
            range(len(options)),
            key=lambda i: _hydrogen_quick_card_desirability(
                game, player_index, options[i], wanted_tags
            ),
        )

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        if "Cities and Knights" in action or "Epic Prank" in action:
            player = game.players[player_index]
            eligible = (
                [c for c in player.hand if "Item" in c.tags]
                if "Epic Prank" in action
                else list(player.hand)
            )
            if not eligible:
                return False
            if (
                any(
                    _effective_card_title(c) == "Drop a Ski"
                    for c in player.played_today
                )
                and len(player.hand) == 1
            ):
                return False
            wanted_tags = _wanted_tags_for_player_fast(player)
            worst_val = min(
                _hydrogen_hand_keep_value(game, player_index, c, wanted_tags)
                for c in eligible
            )
            is_doubled = any(
                _effective_card_title(c) == "Bring a Friend"
                for c in player.played_today[:-1]
            ) or (
                len(player.played_today) >= 2
                and _effective_card_title(player.played_today[-2]) == "Work Call"
            )
            base_thresh = 4.6 if "Epic Prank" in action else 3.6
            if is_doubled:
                base_thresh *= 1.85
            if any(
                _effective_card_title(c) == "Wakesurf"
                for c in player.played_today
            ) or (
                any(
                    _effective_card_title(c) == "Drop a Ski"
                    for c in player.played_today
                )
                and len(player.hand) == 2
            ):
                base_thresh += 2.6
            return worst_val < base_thresh
        return super().choose_optional_action(game, player_index, action)

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        if not cards:
            return ()
        player = game.players[player_index]
        wanted_tags = _wanted_tags_for_player_fast(player)

        has_cutlery = False
        has_bcm = False
        has_quacks = False
        for c in player.hand:
            ct = c.title
            if ct == "Assorted Cutlery" and not has_cutlery:
                if game.energy_cost(player_index, c) <= player.energy:
                    has_cutlery = True
            elif ct == "Bougie Coffee Machine" and not has_bcm:
                if game.energy_cost(player_index, c) <= player.energy:
                    has_bcm = True
            elif ct == "Quacks" and not has_quacks:
                if game.energy_cost(player_index, c) <= player.energy:
                    has_quacks = True

        indices = list(range(len(cards)))
        if has_cutlery:
            best_top = max(
                indices,
                key=lambda i: _hydrogen_free_play_value(
                    game, player_index, cards[i], wanted_tags
                ),
            )
            rest = [i for i in indices if i != best_top]
            if has_quacks:
                rest.sort(
                    key=lambda i: (
                        "Outdoors" in cards[i].tags,
                        _hydrogen_quick_card_desirability(
                            game, player_index, cards[i], wanted_tags
                        ),
                    )
                )
            else:
                rest.sort(
                    key=lambda i: _hydrogen_quick_card_desirability(
                        game, player_index, cards[i], wanted_tags
                    ),
                    reverse=True,
                )
            order = [best_top, *rest]
        elif has_bcm:
            order = sorted(
                indices,
                key=lambda i: (
                    _hydrogen_free_play_value(
                        game, player_index, cards[i], wanted_tags
                    )
                    + 2.5
                    if "Food" in cards[i].tags
                    else _hydrogen_quick_card_desirability(
                        game, player_index, cards[i], wanted_tags
                    )
                ),
                reverse=True,
            )
        elif has_quacks:
            order = sorted(
                indices,
                key=lambda i: (
                    "Outdoors" in cards[i].tags,
                    _hydrogen_quick_card_desirability(
                        game, player_index, cards[i], wanted_tags
                    ),
                ),
            )
        else:
            order = sorted(
                indices,
                key=lambda i: _hydrogen_quick_card_desirability(
                    game, player_index, cards[i], wanted_tags
                ),
                reverse=True,
            )

        if game.stats is _get_shared_sim_stats():
            prev_state = getattr(game, "_known_trunk_state", None)
            rem_prev: tuple[CardInstance, ...] = ()
            if prev_state is not None:
                prev_len, prev_cards = prev_state
                popped = prev_len - len(game.trunk)
                if 0 <= popped < len(prev_cards):
                    rem_prev = prev_cards[popped:]
            ordered_cards = tuple(cards[i] for i in order) + rem_prev
            game._known_trunk_state = (
                len(game.trunk) + len(cards),
                ordered_cards,
            )
        return order

    def choose_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        wanted_tags = _wanted_tags_for_player_fast(game.players[player_index])
        return tuple(
            i
            for i, c in enumerate(cards)
            if _hydrogen_quick_card_desirability(
                game, player_index, c, wanted_tags
            )
            < 2.0
        )


_HYDROGEN_EVAL_AI_SINGLETON = _HydrogenEvalAI()
_HYDROGEN_SIM_AIS: list[Any] = [
    _HYDROGEN_EVAL_AI_SINGLETON,
    _HYDROGEN_EVAL_AI_SINGLETON,
    _HYDROGEN_EVAL_AI_SINGLETON,
    _HYDROGEN_EVAL_AI_SINGLETON,
]


def _compute_player_board_flags(p: PlayerState) -> int:
    meta_by_slug = _HYDROGEN_META_BY_SLUG
    flags = 0
    for tc in p.tomorrow_cards:
        tm = tc.markers
        if not tm or "_copied_definition" not in tm:
            d = tc.definition
            tmeta = meta_by_slug.get(d.slug)
            if tmeta is not None and tmeta.defn is d:
                flags |= tmeta.tom_board_flags
                continue
        flags |= _get_card_meta(tc).tom_board_flags
    for pc in p.played_today:
        pm = pc.markers
        if not pm:
            d = pc.definition
            pmeta = meta_by_slug.get(d.slug)
            if pmeta is not None and pmeta.defn is d:
                flags |= pmeta.played_board_flags
                continue
        else:
            if "_blind_poker" in pm or "_blind_quacks" in pm:
                flags |= _PF_HAS_LOOP2
            if "_copied_definition" in pm:
                flags |= _PF_HAS_COPIED | _PF_HAS_SIG_MARKER
            else:
                if (
                    pm.get("energy_cubes")
                    or pm.get("target_card") is not None
                    or pm.get("target_player_index") is not None
                ):
                    flags |= _PF_HAS_SIG_MARKER
                d = pc.definition
                pmeta = meta_by_slug.get(d.slug)
                if pmeta is not None and pmeta.defn is d:
                    flags |= pmeta.played_board_flags
                    continue
        flags |= _get_card_meta(pc).played_board_flags
    return flags


def _get_hydrogen_sim_game_cls() -> type:
    global _HYDROGEN_SIM_GAME_CLS
    if _HYDROGEN_SIM_GAME_CLS is not None:
        return _HYDROGEN_SIM_GAME_CLS
    _ensure_hydrogen_caches()
    carbon_cls = _get_carbon_sim_game_cls()

    class _HydrogenSimGame(carbon_cls):  # type: ignore[valid-type, misc]
        """Hydrogen simulation Game with Suitcase/Played CoW, player board bitflags, and compiled scoring."""

        _has_nonplain_hand = False
        _played_items_shared = False
        _shared_played_len = 0

        def copy_for_simulation(self) -> Game:
            return _hydrogen_copy_game(self)

        def _unshare_suitcase(self) -> None:
            self._suitcase_shared = False
            self.suitcase = [
                CardInstance(
                    c.instance_id,
                    c.definition,
                    c.is_tomorrow,
                    dict(c.markers) if c.markers else {},
                )
                for c in self.suitcase
            ]

        def _unshare_played_items(self) -> None:
            n_sh = self._shared_played_len
            self._played_items_shared = False
            self._shared_played_len = 0
            idx = self._cow_acting_idx
            if n_sh > 0 and idx >= 0:
                played = self.players[idx].played_today
                limit = n_sh if n_sh <= len(played) else len(played)
                for i in range(limit):
                    c = played[i]
                    m = c.markers
                    played[i] = CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(m) if m else {},
                    )

        def _recompute_player_flags(self, player_index: int) -> None:
            self._p_flags[player_index] = _compute_player_board_flags(
                self.players[player_index]
            )

        def _recompute_all_flags(self) -> None:
            ps = self.players
            self._p_flags = [
                _compute_player_board_flags(ps[0]),
                _compute_player_board_flags(ps[1]),
                _compute_player_board_flags(ps[2]),
                _compute_player_board_flags(ps[3]),
            ]

        def pick_from_suitcase(self, player_index: int) -> CardInstance:
            if self._suitcase_shared:
                self._unshare_suitcase()
            if self.stats is _get_shared_sim_stats():
                sc = self.suitcase
                choice = self.ais[player_index].choose_suitcase_card(
                    self, player_index, tuple(sc)
                )
                if choice < 0 or choice >= len(sc):
                    raise ValueError(
                        f"AI returned invalid Suitcase index: {choice}"
                    )
                card = sc.pop(choice)
                self.give_card(player_index, card)
                if self._p_flags[player_index] & _PF_HAS_PICK:
                    self.record_card_pick_or_draw(player_index, card)
                sc.insert(choice, self._draw_from_trunk())
                return card
            return super().pick_from_suitcase(player_index)

        def pick_suitcase_cards(
            self,
            player_index: int,
            cards: Sequence[CardInstance],
        ) -> list[CardInstance]:
            if self._suitcase_shared:
                old_sc = self.suitcase
                self._unshare_suitcase()
                new_sc = self.suitcase
                cards = [
                    next(
                        (new_sc[i] for i, oc in enumerate(old_sc) if oc is tc),
                        tc,
                    )
                    for tc in cards
                ]
            if self.stats is _get_shared_sim_stats():
                if len(cards) > 1 and len({id(c) for c in cards}) != len(cards):
                    raise ValueError(
                        "Cannot pick the same Suitcase card more than once"
                    )
                sc = self.suitcase
                if any(c not in sc for c in cards):
                    raise ValueError(
                        "Every picked card must be currently visible in the Suitcase"
                    )
                picked_cards: list[CardInstance] = []
                for target in cards:
                    choice = sc.index(target)
                    card = sc.pop(choice)
                    self.give_card(player_index, card)
                    if self._p_flags[player_index] & _PF_HAS_PICK:
                        self.record_card_pick_or_draw(player_index, card)
                    sc.insert(choice, self._draw_from_trunk())
                    picked_cards.append(card)
                return picked_cards
            return super().pick_suitcase_cards(player_index, cards)

        def acquire_suitcase_cards(
            self,
            player_index: int,
            cards: Sequence[CardInstance],
        ) -> list[CardInstance]:
            if self._suitcase_shared:
                old_sc = self.suitcase
                self._unshare_suitcase()
                new_sc = self.suitcase
                cards = [
                    next(
                        (new_sc[i] for i, oc in enumerate(old_sc) if oc is tc),
                        tc,
                    )
                    for tc in cards
                ]
            return super().acquire_suitcase_cards(player_index, cards)

        def choose_suitcase_target(self, player_index: int) -> CardInstance:
            if self._suitcase_shared:
                self._unshare_suitcase()
            return super().choose_suitcase_target(player_index)

        def choose_card_to_copy(
            self,
            player_index: int,
            eligible_cards: Sequence[CardInstance],
        ) -> CardInstance:
            target = super().choose_card_to_copy(player_index, eligible_cards)
            if self._played_items_shared:
                act_idx = self._cow_acting_idx
                if act_idx >= 0:
                    played = self.players[act_idx].played_today
                    n_sh = self._shared_played_len
                    for pos in range(n_sh if n_sh <= len(played) else len(played)):
                        if played[pos] is target:
                            self._unshare_played_items()
                            return played[pos]
            return target

        def choose_card_target(
            self,
            player_index: int,
            eligible_cards: Sequence[CardInstance],
        ) -> CardInstance:
            target = super().choose_card_target(player_index, eligible_cards)
            if self._played_items_shared:
                act_idx = self._cow_acting_idx
                if act_idx >= 0:
                    played = self.players[act_idx].played_today
                    n_sh = self._shared_played_len
                    for pos in range(n_sh if n_sh <= len(played) else len(played)):
                        if played[pos] is target:
                            self._unshare_played_items()
                            return played[pos]
            return target

        def unpack(self, player_index: int, fun_delta: int = -1) -> None:
            if self._suitcase_shared:
                self._unshare_suitcase()
            super().unpack(player_index, fun_delta)

        def start_day(self) -> None:
            if self._suitcase_shared:
                self._unshare_suitcase()
            if self._played_items_shared:
                self._unshare_played_items()
            super().start_day()
            self._recompute_all_flags()

        def end_day(self) -> None:
            if self._suitcase_shared:
                self._unshare_suitcase()
            if self._played_items_shared:
                self._unshare_played_items()
            if self._cow_acting_idx >= 0:
                ap = self.players[self._cow_acting_idx]
                if ap.tomorrow_cards:
                    ap.tomorrow_cards = [
                        CardInstance(
                            c.instance_id,
                            c.definition,
                            c.is_tomorrow,
                            dict(c.markers) if c.markers else {},
                        )
                        for c in ap.tomorrow_cards
                    ]
            while self._shared_opp_mask:
                opp_i = (self._shared_opp_mask & -self._shared_opp_mask).bit_length() - 1
                self._unshare_opp(opp_i)
            super().end_day()
            self._recompute_all_flags()

        def card_fun(self, player_index: int, target: CardInstance) -> int:
            if self._played_items_shared and player_index == self._cow_acting_idx:
                if self._p_flags[player_index] & _PF_ALL_TAGS:
                    self._unshare_played_items()
            return super().card_fun(player_index, target)

        def give_card(self, player_index: int, card: CardInstance) -> None:
            cm = card.markers
            if cm or card.is_tomorrow:
                self._has_marked_hand = True
                if cm and not _is_plain_hand_markers(cm):
                    self._has_nonplain_hand = True
            player = self.players[player_index]
            player.hand.append(card)
            if self._p_flags[player_index] & _PF_HAS_ACQUIRE:
                if self._played_items_shared and player_index == self._cow_acting_idx:
                    self._unshare_played_items()
                for source in player.played_today:
                    sm = source.markers
                    s_beh = (
                        source.definition.behavior
                        if (not sm or "_copied_definition" not in sm)
                        else source.effective_behavior
                    )
                    if (
                        type(s_beh).on_card_acquire
                        is not CardBehavior.on_card_acquire
                    ):
                        s_beh.on_card_acquire(self, player, source, card)

        def record_card_pick_or_draw(
            self,
            player_index: int,
            card: CardInstance,
        ) -> None:
            if self._p_flags[player_index] & _PF_HAS_PICK:
                if self._played_items_shared and player_index == self._cow_acting_idx:
                    self._unshare_played_items()
                player = self.players[player_index]
                for source in player.played_today:
                    sm = source.markers
                    s_beh = (
                        source.definition.behavior
                        if (not sm or "_copied_definition" not in sm)
                        else source.effective_behavior
                    )
                    if (
                        type(s_beh).on_card_pick_or_draw
                        is not CardBehavior.on_card_pick_or_draw
                    ):
                        s_beh.on_card_pick_or_draw(self, player, source, card)

        def gain_energy(
            self,
            player: PlayerState,
            amount: int,
            source: CardInstance | None = None,
        ) -> None:
            if amount > 0:
                p_idx = -1
                ps = self.players
                for i in range(4):
                    if ps[i] is player:
                        p_idx = i
                        break
                if p_idx >= 0 and (self._p_flags[p_idx] & _PF_HAS_ALLOWS_E):
                    for card in player.played_today:
                        c_beh = (
                            card.definition.behavior
                            if not card.markers
                            else card.effective_behavior
                        )
                        if (
                            type(c_beh).allows_energy_gain
                            is not CardBehavior.allows_energy_gain
                            and not c_beh.allows_energy_gain(
                                self, player, card, source
                            )
                        ):
                            return
                if p_idx >= 0 and (self._shared_opp_mask & (1 << p_idx)):
                    player = self._unshare_opp(p_idx)
                player.energy += amount
                if source is not None:
                    if self._played_items_shared and p_idx == self._cow_acting_idx:
                        played = player.played_today
                        n_sh = self._shared_played_len
                        for i_s in range(n_sh if n_sh <= len(played) else len(played)):
                            if played[i_s] is source:
                                self._unshare_played_items()
                                source = played[i_s]
                                break
                    source.markers["_gave_energy"] = True
                return
            if amount < 0:
                if self._shared_opp_mask:
                    for opp_idx in range(4):
                        if (
                            self._shared_opp_mask & (1 << opp_idx)
                        ) and self.players[opp_idx] is player:
                            player = self._unshare_opp(opp_idx)
                            break
                player.energy += amount

        def _finish_play_card_idx(
            self,
            player_index: int,
            player: PlayerState,
            hand_index: int,
            raw_card: CardInstance,
            c_meta: _HydrogenDefnMeta,
            cost: int,
        ) -> CardInstance:
            player.energy -= cost
            player.hand.pop(hand_index)
            rm = raw_card.markers
            card = CardInstance(
                raw_card.instance_id,
                raw_card.definition,
                raw_card.is_tomorrow,
                dict(rm) if rm else {},
            )
            if c_meta.touches_suitcase_on_play and self._suitcase_shared:
                self._unshare_suitcase()
            is_lys = c_meta.title == "Last Year's Shirt" or (
                bool(rm)
                and "_copied_definition" in rm
                and _effective_card_title(raw_card) == "Last Year's Shirt"
            )
            if is_lys and player.hand:
                player.hand = [
                    CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(c.markers) if c.markers else {},
                    )
                    for c in player.hand
                ]
            pf = self._p_flags[player_index] | c_meta.played_board_flags
            self._p_flags[player_index] = pf
            is_bohnanza = type(c_meta.beh).__name__ == "BohnanzaBehavior"
            if self._played_items_shared and (
                (pf & (_PF_ALL_TAGS | _PF_HAS_CARD_PLAY | _PF_HAS_ACQUIRE | _PF_HAS_PICK))
                or is_bohnanza
            ):
                self._unshare_played_items()
            player.played_today.append(card)
            if pf & _PF_ALL_TAGS:
                self._sync_played_today_tags(player)
            if c_meta.has_on_play:
                c_meta.beh.on_play(self, player, card)
                if is_lys:
                    for hc in player.hand:
                        if hc.markers or hc.is_tomorrow:
                            self._has_marked_hand = True
                            if hc.markers and not _is_plain_hand_markers(
                                hc.markers
                            ):
                                self._has_nonplain_hand = True
                if is_bohnanza:
                    self._sync_played_today_tags(player)
                    self._recompute_player_flags(player_index)
                pf = self._p_flags[player_index]
            if (pf & _PF_HAS_TOM_PLAY) and player.tomorrow_cards:
                for source in tuple(player.tomorrow_cards):
                    sm = source.markers
                    s_beh = (
                        source.definition.behavior
                        if (not sm or "_copied_definition" not in sm)
                        else source.effective_behavior
                    )
                    if (
                        type(s_beh).on_tomorrow_card_play
                        is not CardBehavior.on_tomorrow_card_play
                    ):
                        s_beh.on_tomorrow_card_play(
                            self, player, source, card
                        )
                pf = self._p_flags[player_index]
            if pf & _PF_HAS_CARD_PLAY:
                if self._played_items_shared:
                    self._unshare_played_items()
                for source in tuple(player.played_today):
                    sm = source.markers
                    s_beh = (
                        source.definition.behavior
                        if (not sm or "_copied_definition" not in sm)
                        else source.effective_behavior
                    )
                    if type(s_beh).on_card_play is not CardBehavior.on_card_play:
                        s_beh.on_card_play(self, player, source, card)
            cm = card.markers
            if cm:
                if "target_card" in cm or "suitcase_target" in cm:
                    self._has_card_ref = True
                if "_copied_definition" in cm:
                    self._p_flags[player_index] |= (
                        _PF_HAS_COPIED | _PF_HAS_SIG_MARKER
                    )
                elif (
                    cm.get("energy_cubes")
                    or cm.get("target_card") is not None
                    or cm.get("target_player_index") is not None
                ):
                    self._p_flags[player_index] |= _PF_HAS_SIG_MARKER
            return card

        def play_card(self, player_index: int, hand_index: int) -> CardInstance:
            player = self.players[player_index]
            if player.asleep:
                raise ValueError(f"{player.name} has already gone to bed")
            if hand_index < 0 or hand_index >= len(player.hand):
                raise ValueError(f"Invalid hand index: {hand_index}")
            raw_card = player.hand[hand_index]
            c_meta = _get_card_meta(raw_card)
            if c_meta.has_can_play and not c_meta.beh.can_play(
                self, player, raw_card
            ):
                raise ValueError(f"{raw_card.title} cannot legally be played")
            cost = self.energy_cost(player_index, raw_card)
            if cost > player.energy:
                raise ValueError(
                    f"Not enough Energy to play {raw_card.title}"
                )
            return self._finish_play_card_idx(
                player_index, player, hand_index, raw_card, c_meta, cost
            )

        def play_card_known_cost(
            self, player_index: int, hand_index: int, cost: int
        ) -> CardInstance:
            player = self.players[player_index]
            raw_card = player.hand[hand_index]
            rm = raw_card.markers
            c_meta = (
                _HYDROGEN_META_BY_SLUG[raw_card.definition.slug]
                if (
                    (not rm or "_copied_definition" not in rm)
                    and self._all_canonical_defs
                )
                else _get_card_meta(raw_card)
            )
            return self._finish_play_card_idx(
                player_index, player, hand_index, raw_card, c_meta, cost
            )

        def play_card_for_effect(
            self,
            player_index: int,
            card: CardInstance,
            *,
            cost_adjustment: int = 0,
            pay_energy: bool = False,
            ignore_restrictions: bool = False,
        ) -> CardInstance:
            if self._suitcase_shared:
                self._unshare_suitcase()
            if self._played_items_shared:
                self._unshare_played_items()
            played = super().play_card_for_effect(
                player_index,
                card,
                cost_adjustment=cost_adjustment,
                pay_energy=pay_energy,
                ignore_restrictions=ignore_restrictions,
            )
            pm = played.markers
            if pm and ("target_card" in pm or "suitcase_target" in pm):
                self._has_card_ref = True
            self._recompute_player_flags(player_index)
            return played

        def copy_card_effect(
            self,
            player_index: int,
            source: CardInstance,
            destination: CardInstance,
            *,
            pay_source_cost: bool = True,
        ) -> None:
            if self._suitcase_shared:
                self._unshare_suitcase()
            if self._played_items_shared:
                self._unshare_played_items()
            if (
                _effective_card_title(source) == "Last Year's Shirt"
                and self.players[player_index].hand
            ):
                self.players[player_index].hand = [
                    CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(c.markers) if c.markers else {},
                    )
                    for c in self.players[player_index].hand
                ]
            super().copy_card_effect(
                player_index,
                source,
                destination,
                pay_source_cost=pay_source_cost,
            )
            dm = destination.markers
            if dm and ("target_card" in dm or "suitcase_target" in dm):
                self._has_card_ref = True
            for hc in self.players[player_index].hand:
                if hc.markers or hc.is_tomorrow:
                    self._has_marked_hand = True
                    if hc.markers and not _is_plain_hand_markers(hc.markers):
                        self._has_nonplain_hand = True
            self._recompute_player_flags(player_index)

        def energy_cost(self, player_index: int, card: CardInstance) -> int:
            player = self.players[player_index]
            pf = self._p_flags[player_index]
            if not (pf & _PF_ALL_TAGS):
                m = card.markers
                is_canon = self._all_canonical_defs
                c_meta = (
                    _HYDROGEN_META_BY_SLUG[card.definition.slug]
                    if ((not m or "_copied_definition" not in m) and is_canon)
                    else _get_card_meta(card)
                )
                cost = c_meta.cost
                cond = self.daily_condition
                char = player.character
                if cond is not None or char is not None:
                    if is_canon and not (m and "_all_tags" in m and m["_all_tags"]):
                        t_mask = (
                            c_meta.tag_mask
                            if (not m or "_copied_definition" not in m)
                            else _HYDROGEN_META_BY_SLUG[card.definition.slug].tag_mask
                        )
                        if cond is not None and cond.hot_day:
                            if t_mask & _MASK_OUTDOORS:
                                cost += 1
                            if t_mask & _MASK_INDOORS:
                                cost -= 1
                        if char is not None and char.energy_tag:
                            _, _, _, _, _, ce_m, ce_d, _ = _get_char_fast(char)
                            if ce_m:
                                if t_mask & ce_m:
                                    cost += ce_d
                            elif char.energy_tag in card.definition.tags:
                                cost += ce_d
                    else:
                        c_tags = (
                            _ALL_TAGS_CACHED
                            if (m and "_all_tags" in m and m["_all_tags"] and _ALL_TAGS_CACHED is not None)
                            else card.definition.tags
                        )
                        if cond is not None:
                            cost = cond.modify_energy_cost(c_tags, cost)
                        if char is not None:
                            cost = char.modify_energy_cost(c_tags, cost)
                if c_meta.has_own_cost_mod:
                    cost = c_meta.beh.modify_own_energy_cost(
                        self, player, card, cost
                    )
                if pf & _PF_HAS_TOM_COST_MOD:
                    for source in player.tomorrow_cards:
                        sm = source.markers
                        s_beh = (
                            source.definition.behavior
                            if (not sm or "_copied_definition" not in sm)
                            else source.effective_behavior
                        )
                        if (
                            type(s_beh).modify_tomorrow_energy_cost
                            is not CardBehavior.modify_tomorrow_energy_cost
                        ):
                            cost = s_beh.modify_tomorrow_energy_cost(
                                self, player, source, card, cost
                            )
                if pf & _PF_HAS_PLAYED_COST_MOD:
                    for source in player.played_today:
                        sm = source.markers
                        s_beh = (
                            source.definition.behavior
                            if (not sm or "_copied_definition" not in sm)
                            else source.effective_behavior
                        )
                        if (
                            type(s_beh).modify_energy_cost
                            is not CardBehavior.modify_energy_cost
                        ):
                            cost = s_beh.modify_energy_cost(
                                self, player, source, card, cost
                            )
                return cost if cost >= 0 else 0
            if (
                self._all_canonical_defs
                and self.daily_condition is None
                and player.character is None
                and not (pf & _PF_HAS_COST_MOD)
            ):
                m = card.markers
                c_meta = (
                    _HYDROGEN_META_BY_SLUG[card.definition.slug]
                    if (not m or "_copied_definition" not in m)
                    else _get_card_meta(card)
                )
                cost = c_meta.cost
                if c_meta.has_own_cost_mod:
                    return super().energy_cost(player_index, card)
                return cost if cost >= 0 else 0
            return super().energy_cost(player_index, card)

        def _resolve_board_cost_mods(
            self, player: PlayerState, pf: int
        ) -> list[tuple[int, int, int, Any, Any]]:
            mods: list[tuple[int, int, int, Any, Any]] = []
            meta_by_slug = _HYDROGEN_META_BY_SLUG
            if pf & _PF_HAS_TOM_COST_MOD:
                for s in player.tomorrow_cards:
                    sm = s.markers
                    s_meta = (
                        meta_by_slug[s.definition.slug]
                        if (not sm or "_copied_definition" not in sm)
                        else _get_card_meta(s)
                    )
                    if not s_meta.has_tom_cost_mod:
                        continue
                    tk = s_meta.tom_cost_mod_kind
                    if tk == _CM_TAG_ADD:
                        mods.append(
                            (
                                _CM_TAG_ADD,
                                s_meta.tom_cost_mod_mask,
                                s_meta.tom_cost_mod_param,
                                None,
                                None,
                            )
                        )
                    else:
                        mods.append(
                            (
                                _CM_GENERIC,
                                0,
                                0,
                                s,
                                s_meta.beh.modify_tomorrow_energy_cost,
                            )
                        )
            if pf & _PF_HAS_PLAYED_COST_MOD:
                played = player.played_today
                n_p = len(played)
                suffix_masks: list[int] | None = None
                for s_i in range(n_p):
                    s = played[s_i]
                    sm = s.markers
                    s_meta = (
                        meta_by_slug[s.definition.slug]
                        if (not sm or "_copied_definition" not in sm)
                        else _get_card_meta(s)
                    )
                    if not s_meta.has_cost_mod:
                        continue
                    ck = s_meta.cost_mod_kind
                    if (
                        ck == _CM_TAG_ADD
                        or ck == _CM_ALL_ADD
                        or ck == _CM_ALL_MUL2
                    ):
                        mods.append(
                            (
                                ck,
                                s_meta.cost_mod_mask,
                                s_meta.cost_mod_param,
                                None,
                                None,
                            )
                        )
                    elif (
                        ck == _CM_NEXT_TAG_ADD
                        or ck == _CM_NEXT_TAG_SET
                        or ck == _CM_NEXT_TAG_HALF
                    ):
                        cm_m = s_meta.cost_mod_mask
                        if s_i + 1 < n_p:
                            if suffix_masks is None:
                                suffix_masks = [0] * (n_p + 1)
                                acc_m = 0
                                for j in range(n_p - 1, -1, -1):
                                    pj = played[j]
                                    jm = pj.markers
                                    j_mask = (
                                        _ALL_TAGS_BITMASK
                                        if (
                                            jm
                                            and "_all_tags" in jm
                                            and jm["_all_tags"]
                                        )
                                        else meta_by_slug[
                                            pj.definition.slug
                                        ].tag_mask
                                    )
                                    acc_m |= j_mask
                                    suffix_masks[j] = acc_m
                            if suffix_masks[s_i + 1] & cm_m:
                                continue
                        mods.append(
                            (
                                _CM_TAG_ADD if ck == _CM_NEXT_TAG_ADD else ck,
                                cm_m,
                                s_meta.cost_mod_param,
                                None,
                                None,
                            )
                        )
                    else:
                        mods.append(
                            (
                                _CM_GENERIC,
                                0,
                                0,
                                s,
                                s_meta.beh.modify_energy_cost,
                            )
                        )
            return mods

        def cards_energy_costs(
            self, player_index: int, cards: Sequence[CardInstance]
        ) -> list[int]:
            player = self.players[player_index]
            pf = self._p_flags[player_index]
            if not (pf & _PF_ALL_TAGS):
                cond = self.daily_condition
                char = player.character
                meta_by_slug = _HYDROGEN_META_BY_SLUG
                is_canon = self._all_canonical_defs
                has_board_mods = bool(pf & _PF_HAS_COST_MOD)
                if cond is None and char is None and not has_board_mods:
                    out_fast: list[int] = []
                    for card in cards:
                        cm = card.markers
                        c_meta = (
                            meta_by_slug[card.definition.slug]
                            if ((not cm or "_copied_definition" not in cm) and is_canon)
                            else _get_card_meta(card)
                        )
                        cost = c_meta.cost
                        if c_meta.has_own_cost_mod:
                            cost = c_meta.beh.modify_own_energy_cost(
                                self, player, card, cost
                            )
                        out_fast.append(cost if cost >= 0 else 0)
                    return out_fast
                cond_hot = cond is not None and cond.hot_day
                if char is not None:
                    _, _, _, _, _, char_e_mask, char_e_delta, _ = _get_char_fast(char)
                    char_custom = bool(char.energy_tag) and not char_e_mask
                else:
                    char_e_mask = 0
                    char_e_delta = 0
                    char_custom = False
                if (
                    is_canon
                    and not char_custom
                    and not any(
                        c.markers and "_all_tags" in c.markers and c.markers["_all_tags"]
                        for c in cards
                    )
                    and not (
                        (pf & _PF_HAS_PLAYED_COST_MOD)
                        and any(pc is c for pc in player.played_today for c in cards)
                    )
                ):
                    b_mods = (
                        self._resolve_board_cost_mods(player, pf)
                        if has_board_mods
                        else None
                    )
                    out_fast_m: list[int] = []
                    for card in cards:
                        cm = card.markers
                        c_meta = (
                            meta_by_slug[card.definition.slug]
                            if (not cm or "_copied_definition" not in cm)
                            else _get_card_meta(card)
                        )
                        cost = c_meta.cost
                        t_mask = (
                            c_meta.tag_mask
                            if (not cm or "_copied_definition" not in cm)
                            else meta_by_slug[card.definition.slug].tag_mask
                        )
                        if cond_hot:
                            if t_mask & _MASK_OUTDOORS:
                                cost += 1
                            if t_mask & _MASK_INDOORS:
                                cost -= 1
                        if char_e_mask and (t_mask & char_e_mask):
                            cost += char_e_delta
                        if c_meta.has_own_cost_mod:
                            cost = c_meta.beh.modify_own_energy_cost(
                                self, player, card, cost
                            )
                        if b_mods:
                            for bm_k, bm_m, bm_p, bm_s, bm_fn in b_mods:
                                if bm_k == _CM_TAG_ADD:
                                    if t_mask & bm_m:
                                        cost += bm_p
                                elif bm_k == _CM_ALL_ADD:
                                    cost += bm_p
                                elif bm_k == _CM_ALL_MUL2:
                                    cost *= 2
                                elif bm_k == _CM_NEXT_TAG_SET:
                                    if t_mask & bm_m:
                                        cost = bm_p
                                elif bm_k == _CM_NEXT_TAG_HALF:
                                    if t_mask & bm_m:
                                        cost //= 2
                                else:
                                    cost = bm_fn(self, player, bm_s, card, cost)
                        out_fast_m.append(cost if cost >= 0 else 0)
                    return out_fast_m
                tom_mods = None
                if pf & _PF_HAS_TOM_COST_MOD:
                    tom_mods = [
                        (
                            s,
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior,
                        )
                        for s in player.tomorrow_cards
                        if type(
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior
                        ).modify_tomorrow_energy_cost
                        is not CardBehavior.modify_tomorrow_energy_cost
                    ]
                played_mods = None
                if pf & _PF_HAS_PLAYED_COST_MOD:
                    played_mods = [
                        (
                            s,
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior,
                        )
                        for s in player.played_today
                        if type(
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior
                        ).modify_energy_cost
                        is not CardBehavior.modify_energy_cost
                    ]
                out_cc: list[int] = []
                for card in cards:
                    cm = card.markers
                    c_meta = (
                        meta_by_slug[card.definition.slug]
                        if ((not cm or "_copied_definition" not in cm) and is_canon)
                        else _get_card_meta(card)
                    )
                    cost = c_meta.cost
                    c_tags = (
                        _ALL_TAGS_CACHED
                        if (cm and "_all_tags" in cm and cm["_all_tags"] and _ALL_TAGS_CACHED is not None)
                        else card.definition.tags
                    )
                    if cond is not None:
                        cost = cond.modify_energy_cost(c_tags, cost)
                    if char is not None:
                        cost = char.modify_energy_cost(c_tags, cost)
                    if c_meta.has_own_cost_mod:
                        cost = c_meta.beh.modify_own_energy_cost(
                            self, player, card, cost
                        )
                    if tom_mods:
                        for s, s_beh in tom_mods:
                            cost = s_beh.modify_tomorrow_energy_cost(
                                self, player, s, card, cost
                            )
                    if played_mods:
                        for s, s_beh in played_mods:
                            cost = s_beh.modify_energy_cost(
                                self, player, s, card, cost
                            )
                    out_cc.append(cost if cost >= 0 else 0)
                return out_cc
            if (
                self._all_canonical_defs
                and self.daily_condition is None
                and player.character is None
                and not (pf & _PF_HAS_COST_MOD)
            ):
                meta_by_slug = _HYDROGEN_META_BY_SLUG
                out_aa: list[int] = []
                for card in cards:
                    cm = card.markers
                    c_meta = (
                        meta_by_slug[card.definition.slug]
                        if (not cm or "_copied_definition" not in cm)
                        else _get_card_meta(card)
                    )
                    if c_meta.has_own_cost_mod:
                        out_aa.append(self.energy_cost(player_index, card))
                    else:
                        c_v = c_meta.cost
                        out_aa.append(c_v if c_v >= 0 else 0)
                return out_aa
            return [self.energy_cost(player_index, c) for c in cards]

        def hand_playable_with_costs(
            self, player_index: int
        ) -> list[tuple[int, int]]:
            player = self.players[player_index]
            hand = player.hand
            if player.asleep or not hand:
                return []
            p_energy = player.energy
            pf = self._p_flags[player_index]

            if not (pf & _PF_ALL_TAGS):
                cond = self.daily_condition
                char = player.character
                meta_by_slug = _HYDROGEN_META_BY_SLUG
                is_canon = self._all_canonical_defs
                has_board_mods = bool(pf & _PF_HAS_COST_MOD)
                out: list[tuple[int, int]] = []
                if is_canon and not self._has_nonplain_hand:
                    cond_hot = cond is not None and cond.hot_day
                    if char is not None:
                        _, _, _, _, _, char_e_mask, char_e_delta, _ = _get_char_fast(char)
                        char_custom = bool(char.energy_tag) and not char_e_mask
                    else:
                        char_e_mask = 0
                        char_e_delta = 0
                        char_custom = False
                    if not cond_hot and not char_e_mask and not char_custom and not has_board_mods:
                        for index, card in enumerate(hand):
                            c_meta = meta_by_slug[card.definition.slug]
                            if c_meta.has_can_play and not c_meta.beh.can_play(
                                self, player, card
                            ):
                                continue
                            cost = c_meta.cost
                            if c_meta.has_own_cost_mod:
                                cost = c_meta.beh.modify_own_energy_cost(
                                    self, player, card, cost
                                )
                            if cost < 0:
                                cost = 0
                            if cost <= p_energy:
                                out.append((index, cost))
                        return out
                    if not char_custom:
                        b_mods = (
                            self._resolve_board_cost_mods(player, pf)
                            if has_board_mods
                            else None
                        )
                        for index, card in enumerate(hand):
                            c_meta = meta_by_slug[card.definition.slug]
                            if c_meta.has_can_play and not c_meta.beh.can_play(
                                self, player, card
                            ):
                                continue
                            cost = c_meta.cost
                            t_mask = c_meta.tag_mask
                            if cond_hot:
                                if t_mask & _MASK_OUTDOORS:
                                    cost += 1
                                if t_mask & _MASK_INDOORS:
                                    cost -= 1
                            if char_e_mask and (t_mask & char_e_mask):
                                cost += char_e_delta
                            if c_meta.has_own_cost_mod:
                                cost = c_meta.beh.modify_own_energy_cost(
                                    self, player, card, cost
                                )
                            if b_mods:
                                for bm_k, bm_m, bm_p, bm_s, bm_fn in b_mods:
                                    if bm_k == _CM_TAG_ADD:
                                        if t_mask & bm_m:
                                            cost += bm_p
                                    elif bm_k == _CM_ALL_ADD:
                                        cost += bm_p
                                    elif bm_k == _CM_ALL_MUL2:
                                        cost *= 2
                                    elif bm_k == _CM_NEXT_TAG_SET:
                                        if t_mask & bm_m:
                                            cost = bm_p
                                    elif bm_k == _CM_NEXT_TAG_HALF:
                                        if t_mask & bm_m:
                                            cost //= 2
                                    else:
                                        cost = bm_fn(self, player, bm_s, card, cost)
                            if cost < 0:
                                cost = 0
                            if cost <= p_energy:
                                out.append((index, cost))
                        return out
                tom_mods = None
                if pf & _PF_HAS_TOM_COST_MOD:
                    tom_mods = [
                        (
                            s,
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior,
                        )
                        for s in player.tomorrow_cards
                        if type(
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior
                        ).modify_tomorrow_energy_cost
                        is not CardBehavior.modify_tomorrow_energy_cost
                    ]
                played_mods = None
                if pf & _PF_HAS_PLAYED_COST_MOD:
                    played_mods = [
                        (
                            s,
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior,
                        )
                        for s in player.played_today
                        if type(
                            s.definition.behavior
                            if (not s.markers or "_copied_definition" not in s.markers)
                            else s.effective_behavior
                        ).modify_energy_cost
                        is not CardBehavior.modify_energy_cost
                    ]
                for index, card in enumerate(hand):
                    m = card.markers
                    c_meta = (
                        meta_by_slug[card.definition.slug]
                        if ((not m or "_copied_definition" not in m) and is_canon)
                        else _get_card_meta(card)
                    )
                    if c_meta.has_can_play and not c_meta.beh.can_play(
                        self, player, card
                    ):
                        continue
                    cost = c_meta.cost
                    c_tags = (
                        _ALL_TAGS_CACHED
                        if (m and "_all_tags" in m and m["_all_tags"] and _ALL_TAGS_CACHED is not None)
                        else card.definition.tags
                    )
                    if cond is not None:
                        cost = cond.modify_energy_cost(c_tags, cost)
                    if char is not None:
                        cost = char.modify_energy_cost(c_tags, cost)
                    if c_meta.has_own_cost_mod:
                        cost = c_meta.beh.modify_own_energy_cost(
                            self, player, card, cost
                        )
                    if tom_mods:
                        for s, s_beh in tom_mods:
                            cost = s_beh.modify_tomorrow_energy_cost(
                                self, player, s, card, cost
                            )
                    if played_mods:
                        for s, s_beh in played_mods:
                            cost = s_beh.modify_energy_cost(
                                self, player, s, card, cost
                            )
                    if cost < 0:
                        cost = 0
                    if cost <= p_energy:
                        out.append((index, cost))
                return out

            if (
                self._all_canonical_defs
                and not self._has_nonplain_hand
                and self.daily_condition is None
                and player.character is None
            ):
                meta_by_slug = _HYDROGEN_META_BY_SLUG
                out_aa_h: list[tuple[int, int]] = []
                if not (pf & _PF_HAS_COST_MOD):
                    for index, card in enumerate(hand):
                        c_meta = meta_by_slug[card.definition.slug]
                        if c_meta.has_can_play and not c_meta.beh.can_play(
                            self, player, card
                        ):
                            continue
                        cost = (
                            self.energy_cost(player_index, card)
                            if c_meta.has_own_cost_mod
                            else c_meta.cost
                        )
                        if cost < 0:
                            cost = 0
                        if cost <= p_energy:
                            out_aa_h.append((index, cost))
                    return out_aa_h
                grants_all = any(
                    _get_card_meta(tc).tomorrow_all_tags
                    for tc in player.tomorrow_cards
                ) or any(
                    _get_card_meta(s).after_all_tags
                    for s in player.played_today
                )
                if pf & _PF_HAS_PLAYED_COST_MOD:
                    if self._played_items_shared:
                        self._unshare_played_items()
                    self._sync_played_today_tags(player)
                b_mods = self._resolve_board_cost_mods(player, pf)
                has_gen_bm = any(bm[0] == _CM_GENERIC for bm in b_mods)
                if not has_gen_bm:
                    for index, card in enumerate(hand):
                        c_meta = meta_by_slug[card.definition.slug]
                        if c_meta.has_can_play and not c_meta.beh.can_play(
                            self, player, card
                        ):
                            continue
                        if c_meta.has_own_cost_mod:
                            cost = self.energy_cost(player_index, card)
                            if cost <= p_energy:
                                out_aa_h.append((index, cost))
                            continue
                        cost = c_meta.cost
                        t_mask = _ALL_TAGS_BITMASK if grants_all else c_meta.tag_mask
                        for bm_k, bm_m, bm_p, _, _ in b_mods:
                            if bm_k == _CM_TAG_ADD:
                                if t_mask & bm_m:
                                    cost += bm_p
                            elif bm_k == _CM_ALL_ADD:
                                cost += bm_p
                            elif bm_k == _CM_ALL_MUL2:
                                cost *= 2
                            elif bm_k == _CM_NEXT_TAG_SET:
                                if t_mask & bm_m:
                                    cost = bm_p
                            elif bm_k == _CM_NEXT_TAG_HALF:
                                if t_mask & bm_m:
                                    cost //= 2
                        if cost < 0:
                            cost = 0
                        if cost <= p_energy:
                            out_aa_h.append((index, cost))
                    return out_aa_h

            return super().hand_playable_with_costs(player_index)

        def score_played_today_fast(
            self, player_index: int
        ) -> tuple[int, list[CardInstance]]:
            player = self.players[player_index]
            played = player.played_today
            n_played = len(played)
            if n_played == 0:
                return 0, []
            cond = self.daily_condition
            char = player.character
            pf = self._p_flags[player_index]
            if not (pf & _PF_ALL_TAGS):
                for idx_chk in range(n_played):
                    cm_chk = played[idx_chk].markers
                    if cm_chk and "_all_tags" in cm_chk:
                        if self._played_items_shared:
                            self._unshare_played_items()
                            played = player.played_today
                        played[idx_chk].markers.pop("_all_tags", None)
            meta_by_slug = _HYDROGEN_META_BY_SLUG
            is_canon = self._all_canonical_defs

            if is_canon and not (pf & _PF_HAS_CUSTOM_TOM_FUN_MOD):
                if cond is not None:
                    _, cond_f_mask, cond_f_delta, _, cond_custom = _get_cond_fast(cond)
                else:
                    cond_f_mask = 0
                    cond_f_delta = 0
                    cond_custom = False
                if char is not None:
                    _, char_f_mask, char_f_delta, char_zero_mask, _, _, _, char_any_custom = _get_char_fast(char)
                    char_custom = (
                        (
                            (bool(char.fun_tag) and not char_f_mask)
                            or (bool(char.zero_fun_tag) and not char_zero_mask)
                        )
                        if char_any_custom
                        else False
                    )
                else:
                    char_f_mask = 0
                    char_f_delta = 0
                    char_zero_mask = 0
                    char_custom = False
                if not cond_custom and not char_custom:
                    has_tom_f = bool(pf & _PF_HAS_TOM_FUN_MOD) and bool(
                        player.tomorrow_cards
                    )
                    tom_all_f = 0
                    tom_first_out_f = 0
                    tom_tag_f: list[tuple[int, int]] | None = None
                    if has_tom_f:
                        for tc_s in player.tomorrow_cards:
                            tcm = tc_s.markers
                            tc_meta = (
                                meta_by_slug[tc_s.definition.slug]
                                if (not tcm or "_copied_definition" not in tcm)
                                else _get_card_meta(tc_s)
                            )
                            tk = tc_meta.tom_mod_fun_kind
                            if tk == _TMF_TAG_ADD:
                                if tom_tag_f is None:
                                    tom_tag_f = [
                                        (
                                            tc_meta.tom_mod_fun_mask,
                                            tc_meta.tom_mod_fun_param,
                                        )
                                    ]
                                else:
                                    tom_tag_f.append(
                                        (
                                            tc_meta.tom_mod_fun_mask,
                                            tc_meta.tom_mod_fun_param,
                                        )
                                    )
                            elif tk == _TMF_FIRST_OUTDOORS:
                                tom_first_out_f += tc_meta.tom_mod_fun_param
                            elif tk == _TMF_ALL_ADD:
                                tom_all_f += tc_meta.tom_mod_fun_param
                    plain_cc = (
                        not cond_f_mask
                        and not char_f_mask
                        and not char_zero_mask
                        and not has_tom_f
                    )
                    # Fast Path 1: zero fun mods, zero _all_tags
                    if not (pf & (_PF_ALL_TAGS | _PF_HAS_FUN_MOD)):
                        if not (pf & (_PF_HAS_COPIED | _PF_HAS_TOM_ACTION)):
                            scored_u = 0
                            if plain_cc:
                                for target in played:
                                    if not target.is_tomorrow:
                                        t_meta = meta_by_slug[target.definition.slug]
                                        scored_u += (
                                            t_meta.beh.fun_value(self, player, target)
                                            if t_meta.has_fun_val
                                            else t_meta.base_fun
                                        )
                                return scored_u, []
                            for t_i in range(n_played):
                                target = played[t_i]
                                if not target.is_tomorrow:
                                    t_meta = meta_by_slug[target.definition.slug]
                                    val = (
                                        t_meta.beh.fun_value(self, player, target)
                                        if t_meta.has_fun_val
                                        else t_meta.base_fun
                                    )
                                    t_mask = t_meta.tag_mask
                                    if cond_f_mask and (t_mask & cond_f_mask):
                                        val += cond_f_delta
                                    if char_f_mask and (t_mask & char_f_mask):
                                        val += char_f_delta
                                    if has_tom_f:
                                        val += tom_all_f
                                        if (
                                            t_i == 0
                                            and tom_first_out_f
                                            and (t_mask & _MASK_OUTDOORS)
                                        ):
                                            val += tom_first_out_f
                                        if tom_tag_f is not None:
                                            for tm_m, tm_p in tom_tag_f:
                                                if t_mask & tm_m:
                                                    val += tm_p
                                    if char_zero_mask and (t_mask & char_zero_mask):
                                        val = 0
                                    scored_u += val
                            return scored_u, []
                        scored_p1 = 0
                        next_tom_p1: list[CardInstance] = []
                        for t_i in range(n_played):
                            target = played[t_i]
                            tm = target.markers
                            t_meta = (
                                meta_by_slug[target.definition.slug]
                                if (not tm or "_copied_definition" not in tm)
                                else _get_card_meta(target)
                            )
                            if t_meta.has_tomorrow_action:
                                next_tom_p1.append(target)
                            if not target.is_tomorrow:
                                val = (
                                    t_meta.beh.fun_value(self, player, target)
                                    if t_meta.has_fun_val
                                    else t_meta.base_fun
                                )
                                if not plain_cc:
                                    t_mask = (
                                        t_meta.tag_mask
                                        if not tm
                                        else meta_by_slug[target.definition.slug].tag_mask
                                    )
                                    if cond_f_mask and (t_mask & cond_f_mask):
                                        val += cond_f_delta
                                    if char_f_mask and (t_mask & char_f_mask):
                                        val += char_f_delta
                                    if has_tom_f:
                                        val += tom_all_f
                                        if (
                                            t_i == 0
                                            and tom_first_out_f
                                            and (t_mask & _MASK_OUTDOORS)
                                        ):
                                            val += tom_first_out_f
                                        if tom_tag_f is not None:
                                            for tm_m, tm_p in tom_tag_f:
                                                if t_mask & tm_m:
                                                    val += tm_p
                                    if char_zero_mask and (t_mask & char_zero_mask):
                                        val = 0
                                scored_p1 += val
                        return scored_p1, next_tom_p1

                    # Fast Path 2: purely additive today_mods via 8-bit tag masks
                    if not (pf & _PF_HAS_NONADD_FUN_MOD):
                        if n_played == 1 and not (pf & _PF_ALL_TAGS):
                            t0 = played[0]
                            tm0 = t0.markers
                            t_meta0 = (
                                meta_by_slug[t0.definition.slug]
                                if (not tm0 or "_copied_definition" not in tm0)
                                else _get_card_meta(t0)
                            )
                            nt0 = [t0] if t_meta0.has_tomorrow_action else []
                            if t0.is_tomorrow:
                                return 0, nt0
                            val0 = (
                                t_meta0.beh.fun_value(self, player, t0)
                                if t_meta0.has_fun_val
                                else t_meta0.base_fun
                            )
                            if not plain_cc:
                                t_mask0 = meta_by_slug[t0.definition.slug].tag_mask
                                if cond_f_mask and (t_mask0 & cond_f_mask):
                                    val0 += cond_f_delta
                                if char_f_mask and (t_mask0 & char_f_mask):
                                    val0 += char_f_delta
                                if has_tom_f:
                                    val0 += tom_all_f
                                    if tom_first_out_f and (t_mask0 & _MASK_OUTDOORS):
                                        val0 += tom_first_out_f
                                    if tom_tag_f is not None:
                                        for tm_m, tm_p in tom_tag_f:
                                            if t_mask0 & tm_m:
                                                val0 += tm_p
                                if char_zero_mask and (t_mask0 & char_zero_mask):
                                    val0 = 0
                            return val0, nt0

                        if not (pf & (_PF_ALL_TAGS | _PF_HAS_COPIED)):
                            metas_u = [meta_by_slug[c.definition.slug] for c in played]
                            next_tom_u = (
                                [
                                    played[i]
                                    for i in range(n_played)
                                    if metas_u[i].has_tomorrow_action
                                ]
                                if (pf & _PF_HAS_TOM_ACTION)
                                else []
                            )
                            bonuses_u = [0] * n_played
                            for s_i in range(n_played):
                                s_meta = metas_u[s_i]
                                if not s_meta.has_fun_mod:
                                    continue
                                kind = s_meta.mod_fun_kind
                                if kind == _MF_AFTER_TAG_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        if metas_u[j].tag_mask & m_mask:
                                            bonuses_u[j] += param
                                elif kind == _MF_AFTER_ALL_ADD:
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        bonuses_u[j] += param
                                elif kind == _MF_BEFORE_TAG_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i):
                                        if metas_u[j].tag_mask & m_mask:
                                            bonuses_u[j] += param
                                elif kind == _MF_BEFORE_ALL_ADD:
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i):
                                        bonuses_u[j] += param
                                elif kind == _MF_OTHER_TAGS_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(n_played):
                                        if j != s_i and (metas_u[j].tag_mask & m_mask):
                                            bonuses_u[j] += param
                                elif kind == _MF_NEXT_TAG_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        if metas_u[j].tag_mask & m_mask:
                                            bonuses_u[j] += param
                                            break
                                elif kind == _MF_NEXT_TAG_COST_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        if metas_u[j].tag_mask & m_mask:
                                            bonuses_u[j] += param * metas_u[j].cost
                                            break
                                elif kind == _MF_NOS_BOOK:
                                    tc_ref = played[s_i].markers.get("target_card")
                                    if tc_ref is not None:
                                        for j in range(s_i):
                                            if played[j] is tc_ref:
                                                bonuses_u[j] += 3
                                                break
                            scored_u2 = 0
                            if plain_cc:
                                for t_i in range(n_played):
                                    target = played[t_i]
                                    if not target.is_tomorrow:
                                        t_meta = metas_u[t_i]
                                        scored_u2 += (
                                            (
                                                t_meta.beh.fun_value(self, player, target)
                                                if t_meta.has_fun_val
                                                else t_meta.base_fun
                                            )
                                            + bonuses_u[t_i]
                                        )
                                return scored_u2, next_tom_u
                            for t_i in range(n_played):
                                target = played[t_i]
                                if not target.is_tomorrow:
                                    t_meta = metas_u[t_i]
                                    val = (
                                        t_meta.beh.fun_value(self, player, target)
                                        if t_meta.has_fun_val
                                        else t_meta.base_fun
                                    ) + bonuses_u[t_i]
                                    t_mask = t_meta.tag_mask
                                    if cond_f_mask and (t_mask & cond_f_mask):
                                        val += cond_f_delta
                                    if char_f_mask and (t_mask & char_f_mask):
                                        val += char_f_delta
                                    if has_tom_f:
                                        val += tom_all_f
                                        if (
                                            t_i == 0
                                            and tom_first_out_f
                                            and (t_mask & _MASK_OUTDOORS)
                                        ):
                                            val += tom_first_out_f
                                        if tom_tag_f is not None:
                                            for tm_m, tm_p in tom_tag_f:
                                                if t_mask & tm_m:
                                                    val += tm_p
                                    if char_zero_mask and (t_mask & char_zero_mask):
                                        val = 0
                                    scored_u2 += val
                            return scored_u2, next_tom_u

                        seen_aa = False
                        if (pf & _PF_ALL_TAGS) and player.tomorrow_cards:
                            for tc in player.tomorrow_cards:
                                if _get_card_meta(tc).tomorrow_all_tags:
                                    seen_aa = True
                                    break
                        check_aa = bool(pf & _PF_ALL_TAGS) or seen_aa
                        if check_aa and self._played_items_shared:
                            self._unshare_played_items()
                            played = player.played_today
                        metas_p2: list[_HydrogenDefnMeta] = []
                        masks_p2: list[int] = []
                        next_tom_p2: list[CardInstance] = []
                        for source in played:
                            sm = source.markers
                            if not sm:
                                s_meta = meta_by_slug[source.definition.slug]
                                if seen_aa:
                                    source.markers["_all_tags"] = True
                                    s_mask = _ALL_TAGS_BITMASK
                                else:
                                    s_mask = s_meta.tag_mask
                            else:
                                s_meta = (
                                    meta_by_slug[source.definition.slug]
                                    if "_copied_definition" not in sm
                                    else _get_card_meta(source)
                                )
                                if check_aa:
                                    if seen_aa:
                                        sm["_all_tags"] = True
                                        s_mask = _ALL_TAGS_BITMASK
                                    else:
                                        if "_all_tags" in sm:
                                            sm.pop("_all_tags", None)
                                        s_mask = meta_by_slug[source.definition.slug].tag_mask
                                else:
                                    s_mask = (
                                        _ALL_TAGS_BITMASK
                                        if ("_all_tags" in sm and sm["_all_tags"])
                                        else meta_by_slug[source.definition.slug].tag_mask
                                    )
                            if s_meta.after_all_tags:
                                seen_aa = True
                                check_aa = True
                            if s_meta.has_tomorrow_action:
                                next_tom_p2.append(source)
                            metas_p2.append(s_meta)
                            masks_p2.append(s_mask)

                        bonuses = [0] * n_played
                        if n_played > 1:
                            for s_i in range(n_played):
                                s_meta = metas_p2[s_i]
                                if not s_meta.has_fun_mod:
                                    continue
                                kind = s_meta.mod_fun_kind
                                if kind == _MF_AFTER_TAG_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        if masks_p2[j] & m_mask:
                                            bonuses[j] += param
                                elif kind == _MF_AFTER_ALL_ADD:
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        bonuses[j] += param
                                elif kind == _MF_BEFORE_TAG_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i):
                                        if masks_p2[j] & m_mask:
                                            bonuses[j] += param
                                elif kind == _MF_BEFORE_ALL_ADD:
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i):
                                        bonuses[j] += param
                                elif kind == _MF_OTHER_TAGS_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(n_played):
                                        if j != s_i and (masks_p2[j] & m_mask):
                                            bonuses[j] += param
                                elif kind == _MF_NEXT_TAG_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        if masks_p2[j] & m_mask:
                                            bonuses[j] += param
                                            break
                                elif kind == _MF_NEXT_TAG_COST_ADD:
                                    m_mask = s_meta.mod_fun_mask
                                    param = s_meta.mod_fun_param
                                    for j in range(s_i + 1, n_played):
                                        if masks_p2[j] & m_mask:
                                            bonuses[j] += param * played[j].definition.cost
                                            break
                                elif kind == _MF_NOS_BOOK:
                                    tc_ref = played[s_i].markers.get("target_card")
                                    if tc_ref is not None:
                                        for j in range(s_i):
                                            if played[j] is tc_ref:
                                                bonuses[j] += 3
                                                break

                        scored_p2 = 0
                        if plain_cc:
                            for t_i in range(n_played):
                                target = played[t_i]
                                if not target.is_tomorrow:
                                    t_meta = metas_p2[t_i]
                                    scored_p2 += (
                                        (
                                            t_meta.beh.fun_value(self, player, target)
                                            if t_meta.has_fun_val
                                            else t_meta.base_fun
                                        )
                                        + bonuses[t_i]
                                    )
                            return scored_p2, next_tom_p2
                        for t_i in range(n_played):
                            target = played[t_i]
                            if not target.is_tomorrow:
                                t_meta = metas_p2[t_i]
                                val = (
                                    t_meta.beh.fun_value(self, player, target)
                                    if t_meta.has_fun_val
                                    else t_meta.base_fun
                                ) + bonuses[t_i]
                                t_mask = masks_p2[t_i]
                                if cond_f_mask and (t_mask & cond_f_mask):
                                    val += cond_f_delta
                                if char_f_mask and (t_mask & char_f_mask):
                                    val += char_f_delta
                                if has_tom_f:
                                    val += tom_all_f
                                    if (
                                        t_i == 0
                                        and tom_first_out_f
                                        and (t_mask & _MASK_OUTDOORS)
                                    ):
                                        val += tom_first_out_f
                                    if tom_tag_f is not None:
                                        for tm_m, tm_p in tom_tag_f:
                                            if t_mask & tm_m:
                                                val += tm_p
                                if char_zero_mask and (t_mask & char_zero_mask):
                                    val = 0
                                scored_p2 += val
                        return scored_p2, next_tom_p2

                    # Fast Path 3: non-additive today_mods (Bring a Friend / Work Call / Water Trampoline / Schwank / Christmas Name Draw)
                    if not (pf & (_PF_ALL_TAGS | _PF_HAS_COPIED)):
                        metas_u3 = [meta_by_slug[c.definition.slug] for c in played]
                        next_tom_u3 = (
                            [
                                played[i]
                                for i in range(n_played)
                                if metas_u3[i].has_tomorrow_action
                            ]
                            if (pf & _PF_HAS_TOM_ACTION)
                            else []
                        )
                        vals_u3 = [0] * n_played
                        if plain_cc:
                            for t_i in range(n_played):
                                target = played[t_i]
                                if not target.is_tomorrow:
                                    t_meta = metas_u3[t_i]
                                    vals_u3[t_i] = (
                                        t_meta.beh.fun_value(self, player, target)
                                        if t_meta.has_fun_val
                                        else t_meta.base_fun
                                    )
                        else:
                            for t_i in range(n_played):
                                target = played[t_i]
                                if not target.is_tomorrow:
                                    t_meta = metas_u3[t_i]
                                    val = (
                                        t_meta.beh.fun_value(self, player, target)
                                        if t_meta.has_fun_val
                                        else t_meta.base_fun
                                    )
                                    t_mask = t_meta.tag_mask
                                    if cond_f_mask and (t_mask & cond_f_mask):
                                        val += cond_f_delta
                                    if char_f_mask and (t_mask & char_f_mask):
                                        val += char_f_delta
                                    if has_tom_f:
                                        val += tom_all_f
                                        if (
                                            t_i == 0
                                            and tom_first_out_f
                                            and (t_mask & _MASK_OUTDOORS)
                                        ):
                                            val += tom_first_out_f
                                        if tom_tag_f is not None:
                                            for tm_m, tm_p in tom_tag_f:
                                                if t_mask & tm_m:
                                                    val += tm_p
                                    vals_u3[t_i] = val
                        for s_i in range(n_played):
                            s_meta = metas_u3[s_i]
                            if not s_meta.has_fun_mod:
                                continue
                            kind = s_meta.mod_fun_kind
                            if kind == _MF_AFTER_TAG_ADD:
                                m_mask = s_meta.mod_fun_mask
                                param = s_meta.mod_fun_param
                                for j in range(s_i + 1, n_played):
                                    if metas_u3[j].tag_mask & m_mask:
                                        vals_u3[j] += param
                            elif kind == _MF_AFTER_ALL_ADD:
                                param = s_meta.mod_fun_param
                                for j in range(s_i + 1, n_played):
                                    vals_u3[j] += param
                            elif kind == _MF_AFTER_ALL_MUL2:
                                for j in range(s_i + 1, n_played):
                                    vals_u3[j] *= 2
                            elif kind == _MF_BEFORE_TAG_ADD:
                                m_mask = s_meta.mod_fun_mask
                                param = s_meta.mod_fun_param
                                for j in range(s_i):
                                    if metas_u3[j].tag_mask & m_mask:
                                        vals_u3[j] += param
                            elif kind == _MF_BEFORE_ALL_ADD:
                                param = s_meta.mod_fun_param
                                for j in range(s_i):
                                    vals_u3[j] += param
                            elif kind == _MF_OTHER_TAGS_ADD:
                                m_mask = s_meta.mod_fun_mask
                                param = s_meta.mod_fun_param
                                for j in range(n_played):
                                    if j != s_i and (metas_u3[j].tag_mask & m_mask):
                                        vals_u3[j] += param
                            elif kind == _MF_NEXT_CARD_MUL:
                                if s_i + 1 < n_played:
                                    vals_u3[s_i + 1] *= s_meta.mod_fun_param
                            elif kind == _MF_NEXT_TAG_ADD:
                                m_mask = s_meta.mod_fun_mask
                                param = s_meta.mod_fun_param
                                for j in range(s_i + 1, n_played):
                                    if metas_u3[j].tag_mask & m_mask:
                                        vals_u3[j] += param
                                        break
                            elif kind == _MF_NEXT_TAG_COST_ADD:
                                m_mask = s_meta.mod_fun_mask
                                param = s_meta.mod_fun_param
                                for j in range(s_i + 1, n_played):
                                    if metas_u3[j].tag_mask & m_mask:
                                        vals_u3[j] += param * metas_u3[j].cost
                                        break
                            elif kind == _MF_NEXT_TAG_MUL2:
                                m_mask = s_meta.mod_fun_mask
                                for j in range(s_i + 1, n_played):
                                    if metas_u3[j].tag_mask & m_mask:
                                        vals_u3[j] *= 2
                                        break
                            elif kind == _MF_NTH_TAG_MUL2:
                                m_mask = s_meta.mod_fun_mask
                                rem_ord = s_meta.mod_fun_ord
                                for j in range(s_i + 1, n_played):
                                    if metas_u3[j].tag_mask & m_mask:
                                        rem_ord -= 1
                                        if rem_ord == 0:
                                            vals_u3[j] *= 2
                                            break
                            elif kind == _MF_NOS_BOOK:
                                tc_ref = played[s_i].markers.get("target_card")
                                if tc_ref is not None:
                                    for j in range(s_i):
                                        if played[j] is tc_ref:
                                            vals_u3[j] += 3
                                            break
                            else:
                                s_beh = s_meta.beh
                                s_card = played[s_i]
                                for j in range(n_played):
                                    if not played[j].is_tomorrow:
                                        vals_u3[j] = s_beh.modify_fun(
                                            self, player, s_card, played[j], vals_u3[j]
                                        )
                        scored_u3 = 0
                        if not char_zero_mask:
                            for t_i in range(n_played):
                                if not played[t_i].is_tomorrow:
                                    scored_u3 += vals_u3[t_i]
                        else:
                            for t_i in range(n_played):
                                if not played[t_i].is_tomorrow and not (
                                    metas_u3[t_i].tag_mask & char_zero_mask
                                ):
                                    scored_u3 += vals_u3[t_i]
                        return scored_u3, next_tom_u3

                    # Fast Path 3b: non-additive today_mods WITH _all_tags or copied cards
                    seen_aa3 = False
                    if (pf & _PF_ALL_TAGS) and player.tomorrow_cards:
                        for tc in player.tomorrow_cards:
                            if _get_card_meta(tc).tomorrow_all_tags:
                                seen_aa3 = True
                                break
                    check_aa3 = bool(pf & _PF_ALL_TAGS) or seen_aa3
                    if check_aa3 and self._played_items_shared:
                        self._unshare_played_items()
                        played = player.played_today
                    metas_p3: list[_HydrogenDefnMeta] = []
                    masks_p3: list[int] = []
                    next_tom_p3: list[CardInstance] = []
                    for source in played:
                        sm = source.markers
                        if not sm:
                            s_meta = meta_by_slug[source.definition.slug]
                            if seen_aa3:
                                source.markers["_all_tags"] = True
                                s_mask = _ALL_TAGS_BITMASK
                            else:
                                s_mask = s_meta.tag_mask
                        else:
                            s_meta = (
                                meta_by_slug[source.definition.slug]
                                if "_copied_definition" not in sm
                                else _get_card_meta(source)
                            )
                            if check_aa3:
                                if seen_aa3:
                                    sm["_all_tags"] = True
                                    s_mask = _ALL_TAGS_BITMASK
                                else:
                                    if "_all_tags" in sm:
                                        sm.pop("_all_tags", None)
                                    s_mask = meta_by_slug[source.definition.slug].tag_mask
                            else:
                                s_mask = (
                                    _ALL_TAGS_BITMASK
                                    if ("_all_tags" in sm and sm["_all_tags"])
                                    else meta_by_slug[source.definition.slug].tag_mask
                                )
                        if s_meta.after_all_tags:
                            seen_aa3 = True
                            check_aa3 = True
                        if s_meta.has_tomorrow_action:
                            next_tom_p3.append(source)
                        metas_p3.append(s_meta)
                        masks_p3.append(s_mask)

                    vals_p3 = [0] * n_played
                    if plain_cc:
                        for t_i in range(n_played):
                            target = played[t_i]
                            if not target.is_tomorrow:
                                t_meta = metas_p3[t_i]
                                vals_p3[t_i] = (
                                    t_meta.beh.fun_value(self, player, target)
                                    if t_meta.has_fun_val
                                    else t_meta.base_fun
                                )
                    else:
                        for t_i in range(n_played):
                            target = played[t_i]
                            if not target.is_tomorrow:
                                t_meta = metas_p3[t_i]
                                val = (
                                    t_meta.beh.fun_value(self, player, target)
                                    if t_meta.has_fun_val
                                    else t_meta.base_fun
                                )
                                t_mask = masks_p3[t_i]
                                if cond_f_mask and (t_mask & cond_f_mask):
                                    val += cond_f_delta
                                if char_f_mask and (t_mask & char_f_mask):
                                    val += char_f_delta
                                if has_tom_f:
                                    val += tom_all_f
                                    if (
                                        t_i == 0
                                        and tom_first_out_f
                                        and (t_mask & _MASK_OUTDOORS)
                                    ):
                                        val += tom_first_out_f
                                    if tom_tag_f is not None:
                                        for tm_m, tm_p in tom_tag_f:
                                            if t_mask & tm_m:
                                                val += tm_p
                                vals_p3[t_i] = val
                    for s_i in range(n_played):
                        s_meta = metas_p3[s_i]
                        if not s_meta.has_fun_mod:
                            continue
                        kind = s_meta.mod_fun_kind
                        if kind == _MF_AFTER_TAG_ADD:
                            m_mask = s_meta.mod_fun_mask
                            param = s_meta.mod_fun_param
                            for j in range(s_i + 1, n_played):
                                if masks_p3[j] & m_mask:
                                    vals_p3[j] += param
                        elif kind == _MF_AFTER_ALL_ADD:
                            param = s_meta.mod_fun_param
                            for j in range(s_i + 1, n_played):
                                vals_p3[j] += param
                        elif kind == _MF_AFTER_ALL_MUL2:
                            for j in range(s_i + 1, n_played):
                                vals_p3[j] *= 2
                        elif kind == _MF_BEFORE_TAG_ADD:
                            m_mask = s_meta.mod_fun_mask
                            param = s_meta.mod_fun_param
                            for j in range(s_i):
                                if masks_p3[j] & m_mask:
                                    vals_p3[j] += param
                        elif kind == _MF_BEFORE_ALL_ADD:
                            param = s_meta.mod_fun_param
                            for j in range(s_i):
                                vals_p3[j] += param
                        elif kind == _MF_OTHER_TAGS_ADD:
                            m_mask = s_meta.mod_fun_mask
                            param = s_meta.mod_fun_param
                            for j in range(n_played):
                                if j != s_i and (masks_p3[j] & m_mask):
                                    vals_p3[j] += param
                        elif kind == _MF_NEXT_CARD_MUL:
                            if s_i + 1 < n_played:
                                vals_p3[s_i + 1] *= s_meta.mod_fun_param
                        elif kind == _MF_NEXT_TAG_ADD:
                            m_mask = s_meta.mod_fun_mask
                            param = s_meta.mod_fun_param
                            for j in range(s_i + 1, n_played):
                                if masks_p3[j] & m_mask:
                                    vals_p3[j] += param
                                    break
                        elif kind == _MF_NEXT_TAG_COST_ADD:
                            m_mask = s_meta.mod_fun_mask
                            param = s_meta.mod_fun_param
                            for j in range(s_i + 1, n_played):
                                if masks_p3[j] & m_mask:
                                    vals_p3[j] += param * played[j].definition.cost
                                    break
                        elif kind == _MF_NEXT_TAG_MUL2:
                            m_mask = s_meta.mod_fun_mask
                            for j in range(s_i + 1, n_played):
                                if masks_p3[j] & m_mask:
                                    vals_p3[j] *= 2
                                    break
                        elif kind == _MF_NTH_TAG_MUL2:
                            m_mask = s_meta.mod_fun_mask
                            rem_ord = s_meta.mod_fun_ord
                            for j in range(s_i + 1, n_played):
                                if masks_p3[j] & m_mask:
                                    rem_ord -= 1
                                    if rem_ord == 0:
                                        vals_p3[j] *= 2
                                        break
                        elif kind == _MF_NOS_BOOK:
                            tc_ref = played[s_i].markers.get("target_card")
                            if tc_ref is not None:
                                for j in range(s_i):
                                    if played[j] is tc_ref:
                                        vals_p3[j] += 3
                                        break
                        else:
                            s_beh = s_meta.beh
                            s_card = played[s_i]
                            for j in range(n_played):
                                if not played[j].is_tomorrow:
                                    vals_p3[j] = s_beh.modify_fun(
                                        self, player, s_card, played[j], vals_p3[j]
                                    )
                    scored_p3 = 0
                    if not char_zero_mask:
                        for t_i in range(n_played):
                            if not played[t_i].is_tomorrow:
                                scored_p3 += vals_p3[t_i]
                    else:
                        for t_i in range(n_played):
                            if not played[t_i].is_tomorrow and not (
                                masks_p3[t_i] & char_zero_mask
                            ):
                                scored_p3 += vals_p3[t_i]
                    return scored_p3, next_tom_p3

            all_tags_fs = _ALL_TAGS_CACHED
            seen_after_all = False
            tom_mods: list[tuple[CardInstance, CardBehavior]] | None = None
            if (pf & (_PF_ALL_TAGS | _PF_HAS_TOM_FUN_MOD)) and player.tomorrow_cards:
                for source in player.tomorrow_cards:
                    s_meta = _get_card_meta(source)
                    if s_meta.tomorrow_all_tags:
                        seen_after_all = True
                    if s_meta.has_tom_fun_mod:
                        if tom_mods is None:
                            tom_mods = []
                        tom_mods.append((source, s_meta.beh))

            metas: list[_HydrogenDefnMeta] = []
            tags_by_pos: list[frozenset[str]] = []
            has_any_fun_mod = False
            next_tomorrow_cards: list[CardInstance] = []
            check_all_tags = bool(pf & _PF_ALL_TAGS) or seen_after_all
            if check_all_tags and self._played_items_shared:
                self._unshare_played_items()
                played = player.played_today

            for source in played:
                sm = source.markers
                if not sm:
                    s_meta = (
                        meta_by_slug[source.definition.slug]
                        if is_canon
                        else _get_card_meta(source)
                    )
                    if seen_after_all:
                        source.markers["_all_tags"] = True
                        s_tags = (
                            all_tags_fs
                            if all_tags_fs is not None
                            else source.tags
                        )
                    else:
                        s_tags = s_meta.tags
                else:
                    s_meta = (
                        meta_by_slug[source.definition.slug]
                        if (is_canon and "_copied_definition" not in sm)
                        else _get_card_meta(source)
                    )
                    if check_all_tags:
                        if seen_after_all:
                            sm["_all_tags"] = True
                            s_tags = (
                                all_tags_fs
                                if all_tags_fs is not None
                                else source.tags
                            )
                        else:
                            if "_all_tags" in sm:
                                sm.pop("_all_tags", None)
                            s_tags = source.definition.tags
                    else:
                        s_tags = (
                            all_tags_fs
                            if ("_all_tags" in sm and sm["_all_tags"] and all_tags_fs is not None)
                            else source.definition.tags
                        )
                if s_meta.after_all_tags:
                    seen_after_all = True
                    check_all_tags = True
                if s_meta.has_fun_mod:
                    has_any_fun_mod = True
                if s_meta.has_tomorrow_action:
                    next_tomorrow_cards.append(source)
                metas.append(s_meta)
                tags_by_pos.append(s_tags)

            # Fast path when no today_mods and no tom_mods exist
            if not has_any_fun_mod and tom_mods is None:
                scored_plain = 0
                if cond is None and char is None:
                    for t_i in range(n_played):
                        target = played[t_i]
                        if not target.is_tomorrow:
                            t_meta = metas[t_i]
                            scored_plain += (
                                t_meta.beh.fun_value(self, player, target)
                                if t_meta.has_fun_val
                                else t_meta.base_fun
                            )
                    return scored_plain, next_tomorrow_cards
                for t_i in range(n_played):
                    target = played[t_i]
                    if not target.is_tomorrow:
                        t_meta = metas[t_i]
                        value = (
                            t_meta.beh.fun_value(self, player, target)
                            if t_meta.has_fun_val
                            else t_meta.base_fun
                        )
                        t_tags = tags_by_pos[t_i]
                        if cond is not None:
                            value = cond.modify_fun(t_tags, value)
                        if char is not None:
                            value = char.modify_fun(t_tags, value)
                            value = char.finalize_fun(t_tags, value)
                        scored_plain += value
                return scored_plain, next_tomorrow_cards

            # Resolve active today_mods in exact left-to-right source order
            resolved_mods: list[tuple[int, int, Any, int, Any]] = []
            if has_any_fun_mod:
                for s_i in range(n_played):
                    s_meta = metas[s_i]
                    if not s_meta.has_fun_mod:
                        continue
                    kind = s_meta.mod_fun_kind
                    if kind == _MF_AFTER_TAG_ADD:
                        if s_i + 1 < n_played:
                            resolved_mods.append(
                                (
                                    _RMF_AFTER_TAG_ADD,
                                    s_i,
                                    s_meta.mod_fun_tag,
                                    s_meta.mod_fun_param,
                                    None,
                                )
                            )
                    elif kind == _MF_AFTER_ALL_ADD:
                        if s_i + 1 < n_played:
                            resolved_mods.append(
                                (
                                    _RMF_AFTER_ALL_ADD,
                                    s_i,
                                    None,
                                    s_meta.mod_fun_param,
                                    None,
                                )
                            )
                    elif kind == _MF_AFTER_ALL_MUL2:
                        if s_i + 1 < n_played:
                            resolved_mods.append(
                                (_RMF_AFTER_ALL_MUL2, s_i, None, 2, None)
                            )
                    elif kind == _MF_BEFORE_TAG_ADD:
                        if s_i > 0:
                            resolved_mods.append(
                                (
                                    _RMF_BEFORE_TAG_ADD,
                                    s_i,
                                    s_meta.mod_fun_tag,
                                    s_meta.mod_fun_param,
                                    None,
                                )
                            )
                    elif kind == _MF_BEFORE_ALL_ADD:
                        if s_i > 0:
                            resolved_mods.append(
                                (
                                    _RMF_BEFORE_ALL_ADD,
                                    s_i,
                                    None,
                                    s_meta.mod_fun_param,
                                    None,
                                )
                            )
                    elif kind == _MF_OTHER_TAGS_ADD:
                        if n_played > 1:
                            resolved_mods.append(
                                (
                                    _RMF_OTHER_TAGS_ADD,
                                    s_i,
                                    s_meta.mod_fun_tags,
                                    s_meta.mod_fun_param,
                                    None,
                                )
                            )
                    elif kind == _MF_NEXT_CARD_MUL:
                        if s_i + 1 < n_played:
                            resolved_mods.append(
                                (
                                    _RMF_EXACT_POS_MUL,
                                    s_i + 1,
                                    None,
                                    s_meta.mod_fun_param,
                                    None,
                                )
                            )
                    elif kind == _MF_NEXT_TAG_ADD:
                        m_tag = s_meta.mod_fun_tag
                        for j in range(s_i + 1, n_played):
                            if m_tag in tags_by_pos[j]:
                                resolved_mods.append(
                                    (
                                        _RMF_EXACT_POS_ADD,
                                        j,
                                        None,
                                        s_meta.mod_fun_param,
                                        None,
                                    )
                                )
                                break
                    elif kind == _MF_NEXT_TAG_COST_ADD:
                        m_tag = s_meta.mod_fun_tag
                        for j in range(s_i + 1, n_played):
                            if m_tag in tags_by_pos[j]:
                                resolved_mods.append(
                                    (
                                        _RMF_EXACT_POS_ADD,
                                        j,
                                        None,
                                        s_meta.mod_fun_param
                                        * played[j].definition.cost,
                                        None,
                                    )
                                )
                                break
                    elif kind == _MF_NEXT_TAG_MUL2:
                        m_tag = s_meta.mod_fun_tag
                        for j in range(s_i + 1, n_played):
                            if m_tag in tags_by_pos[j]:
                                resolved_mods.append(
                                    (_RMF_EXACT_POS_MUL, j, None, 2, None)
                                )
                                break
                    elif kind == _MF_NTH_TAG_MUL2:
                        m_tag = s_meta.mod_fun_tag
                        rem_ord = s_meta.mod_fun_ord
                        for j in range(s_i + 1, n_played):
                            if m_tag in tags_by_pos[j]:
                                rem_ord -= 1
                                if rem_ord == 0:
                                    resolved_mods.append(
                                        (_RMF_EXACT_POS_MUL, j, None, 2, None)
                                    )
                                    break
                    elif kind == _MF_NOS_BOOK:
                        tc_ref = played[s_i].markers.get("target_card")
                        if tc_ref is not None:
                            for j in range(s_i):
                                if played[j] is tc_ref:
                                    resolved_mods.append(
                                        (_RMF_EXACT_POS_ADD, j, None, 3, None)
                                    )
                                    break
                    else:
                        resolved_mods.append(
                            (
                                _RMF_GENERIC,
                                s_i,
                                played[s_i],
                                0,
                                s_meta.beh,
                            )
                        )

            scored_today = 0
            for t_i in range(n_played):
                target = played[t_i]
                if not target.is_tomorrow:
                    t_meta = metas[t_i]
                    value = (
                        t_meta.beh.fun_value(self, player, target)
                        if t_meta.has_fun_val
                        else t_meta.base_fun
                    )
                    t_tags = tags_by_pos[t_i]
                    if cond is not None:
                        value = cond.modify_fun(t_tags, value)
                    if char is not None:
                        value = char.modify_fun(t_tags, value)
                    if tom_mods is not None:
                        for source, s_beh in tom_mods:
                            value = s_beh.modify_tomorrow_fun(
                                self, player, source, target, value
                            )
                    for r_kind, r_pos, r_tag, r_param, r_beh in resolved_mods:
                        if r_kind == _RMF_AFTER_TAG_ADD:
                            if t_i > r_pos and r_tag in t_tags:
                                value += r_param
                        elif r_kind == _RMF_EXACT_POS_ADD:
                            if t_i == r_pos:
                                value += r_param
                        elif r_kind == _RMF_EXACT_POS_MUL:
                            if t_i == r_pos:
                                value *= r_param
                        elif r_kind == _RMF_AFTER_ALL_MUL2:
                            if t_i > r_pos:
                                value *= 2
                        elif r_kind == _RMF_BEFORE_TAG_ADD:
                            if t_i < r_pos and r_tag in t_tags:
                                value += r_param
                        elif r_kind == _RMF_BEFORE_ALL_ADD:
                            if t_i < r_pos:
                                value += r_param
                        elif r_kind == _RMF_AFTER_ALL_ADD:
                            if t_i > r_pos:
                                value += r_param
                        elif r_kind == _RMF_OTHER_TAGS_ADD:
                            if t_i != r_pos and (r_tag & t_tags):
                                value += r_param
                        else:
                            value = r_beh.modify_fun(
                                self, player, r_tag, target, value
                            )
                    if char is not None:
                        value = char.finalize_fun(t_tags, value)
                    scored_today += value
            return scored_today, next_tomorrow_cards

    _HYDROGEN_SIM_GAME_CLS = _HydrogenSimGame
    return _HydrogenSimGame


def _hydrogen_copy_game(src: Game) -> Game:
    sim_cls = _HYDROGEN_SIM_GAME_CLS or _get_hydrogen_sim_game_cls()
    is_hsim = type(src) is sim_cls
    sim = object.__new__(sim_cls)
    sim.rng = src.rng if is_hsim else _clone_random(src.rng)
    sim.ais = _HYDROGEN_SIM_AIS
    sim._trunk_shared = False
    sim._discard_shared = False
    sim._cond_shared = False
    sim._suitcase_shared = False
    sim._played_items_shared = False
    sim._shared_played_len = 0
    sim._shared_opp_mask = 0
    sim._cow_acting_idx = -1
    sim._all_canonical_defs = getattr(src, "_all_canonical_defs", False)
    sim._has_opp_suitcase_ref = (
        src._has_opp_suitcase_ref  # type: ignore[attr-defined]
        if is_hsim
        else any(
            oc.markers
            and ("suitcase_target" in oc.markers or "target_card" in oc.markers)
            for opp in src.players
            for oc in opp.played_today
        )
    )

    cards_with_refs: list[CardInstance] = []
    need_map = False
    for p in src.players:
        for c in p.played_today:
            m = c.markers
            if m and ("target_card" in m or "suitcase_target" in m):
                need_map = True
                break
        if need_map:
            break
    sim._has_card_ref = need_map
    sim._has_marked_hand = (
        src._has_marked_hand  # type: ignore[attr-defined]
        if is_hsim
        else any(
            bool(c.markers) or c.is_tomorrow for p in src.players for c in p.hand
        )
    )
    sim._has_nonplain_hand = (
        src._has_nonplain_hand  # type: ignore[attr-defined]
        if is_hsim
        else any(
            bool(c.markers) and not _is_plain_hand_markers(c.markers)
            for p in src.players
            for c in p.hand
        )
    )

    if need_map:
        card_map: dict[int, CardInstance] = {}

        def clone_zone_mapped(zone: list[CardInstance]) -> list[CardInstance]:
            out: list[CardInstance] = []
            for c in zone:
                m = c.markers
                if m:
                    nm = dict(m)
                    nc = CardInstance(c.instance_id, c.definition, c.is_tomorrow, nm)
                    if "target_card" in nm or "suitcase_target" in nm:
                        cards_with_refs.append(nc)
                else:
                    nc = CardInstance(c.instance_id, c.definition, c.is_tomorrow, {})
                card_map[id(c)] = nc
                out.append(nc)
            return out

        sim.players = [
            PlayerState(
                name=p.name,
                energy=p.energy,
                fun=p.fun,
                character=p.character,
                hand=(
                    [
                        CardInstance(
                            c.instance_id,
                            c.definition,
                            c.is_tomorrow,
                            dict(c.markers),
                        )
                        if (c.markers or c.is_tomorrow)
                        else c
                        for c in p.hand
                    ]
                    if is_hsim
                    else [
                        CardInstance(
                            c.instance_id,
                            c.definition,
                            c.is_tomorrow,
                            dict(c.markers) if c.markers else {},
                        )
                        for c in p.hand
                    ]
                ),
                played_today=clone_zone_mapped(p.played_today),
                tomorrow_cards=clone_zone_mapped(p.tomorrow_cards),
                asleep=p.asleep,
                skipped_turns=p.skipped_turns,
                acquired_cards=_NOOP_COUNTER,
                picked_cards=_NOOP_COUNTER,
            )
            for p in src.players
        ]
        sim.suitcase = clone_zone_mapped(src.suitcase)
        for cloned in cards_with_refs:
            nm = cloned.markers
            tc = nm.get("target_card")
            if isinstance(tc, CardInstance):
                mapped = card_map.get(id(tc))
                if mapped is not None:
                    nm["target_card"] = mapped
            sc = nm.get("suitcase_target")
            if isinstance(sc, CardInstance):
                mapped = card_map.get(id(sc))
                if mapped is not None:
                    nm["suitcase_target"] = mapped
    else:
        has_mh = sim._has_marked_hand
        sim.players = [
            PlayerState(
                name=p.name,
                energy=p.energy,
                fun=p.fun,
                character=p.character,
                hand=(
                    (
                        p.hand.copy()
                        if not has_mh
                        else [
                            CardInstance(
                                c.instance_id,
                                c.definition,
                                c.is_tomorrow,
                                dict(c.markers),
                            )
                            if (c.markers or c.is_tomorrow)
                            else c
                            for c in p.hand
                        ]
                    )
                    if is_hsim
                    else [
                        CardInstance(
                            c.instance_id,
                            c.definition,
                            c.is_tomorrow,
                            dict(c.markers) if c.markers else {},
                        )
                        for c in p.hand
                    ]
                ),
                played_today=[
                    CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(c.markers) if c.markers else {},
                    )
                    for c in p.played_today
                ],
                tomorrow_cards=[
                    CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(c.markers) if c.markers else {},
                    )
                    for c in p.tomorrow_cards
                ],
                asleep=p.asleep,
                skipped_turns=p.skipped_turns,
                acquired_cards=_NOOP_COUNTER,
                picked_cards=_NOOP_COUNTER,
            )
            for p in src.players
        ]
        sim.suitcase = [
            CardInstance(
                c.instance_id,
                c.definition,
                c.is_tomorrow,
                dict(c.markers) if c.markers else {},
            )
            for c in src.suitcase
        ]

    if is_hsim:
        sim.trunk = src.trunk.copy()
        sim.discard = src.discard.copy()
        sim._p_flags = src._p_flags.copy()  # type: ignore[attr-defined]
    else:
        sim.trunk = [
            CardInstance(
                c.instance_id,
                c.definition,
                c.is_tomorrow,
                dict(c.markers) if c.markers else {},
            )
            for c in src.trunk
        ]
        sim.discard = [
            CardInstance(
                c.instance_id,
                c.definition,
                c.is_tomorrow,
                dict(c.markers) if c.markers else {},
            )
            for c in src.discard
        ]
        sim._recompute_all_flags()
    sim._known_trunk_state = getattr(src, "_known_trunk_state", None)
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


def _hydrogen_copy_game_for_play(src: Game, player_index: int) -> Game:
    """Fast copy-on-write clone for single-player `play_card` / `go_to_bed` search nodes."""
    sim_cls = _HYDROGEN_SIM_GAME_CLS or _get_hydrogen_sim_game_cls()
    if type(src) is not sim_cls or src._has_opp_suitcase_ref:  # type: ignore[attr-defined]
        return _hydrogen_copy_game(src)

    sim = object.__new__(sim_cls)
    sim.rng = src.rng
    sim.ais = _HYDROGEN_SIM_AIS
    sim._trunk_shared = True
    sim._discard_shared = True
    sim._cond_shared = True
    sim._shared_opp_mask = 0b1111 ^ (1 << player_index)
    sim._cow_acting_idx = player_index
    sim._all_canonical_defs = src._all_canonical_defs  # type: ignore[attr-defined]
    sim._has_opp_suitcase_ref = False
    has_ref = src._has_card_ref  # type: ignore[attr-defined]
    has_marked_hand = src._has_marked_hand  # type: ignore[attr-defined]
    sim._has_card_ref = has_ref
    sim._has_marked_hand = has_marked_hand
    sim._has_nonplain_hand = src._has_nonplain_hand  # type: ignore[attr-defined]
    p_flags = src._p_flags.copy()  # type: ignore[attr-defined]
    sim._p_flags = p_flags

    src_players = src.players
    p = src_players[player_index]

    if has_ref:
        sim._played_items_shared = False
        sim._shared_played_len = 0
        new_played = [
            CardInstance(
                c.instance_id,
                c.definition,
                c.is_tomorrow,
                dict(c.markers) if c.markers else {},
            )
            for c in p.played_today
        ]
        sim._suitcase_shared = False
        new_suitcase = [
            CardInstance(
                c.instance_id,
                c.definition,
                c.is_tomorrow,
                dict(c.markers) if c.markers else {},
            )
            for c in src.suitcase
        ]
        orig_played = p.played_today
        orig_suitcase = src.suitcase
        for nc in new_played:
            nm = nc.markers
            if not nm:
                continue
            tc = nm.get("target_card")
            if isinstance(tc, CardInstance):
                for oc, cc in zip(orig_played, new_played):
                    if oc is tc:
                        nm["target_card"] = cc
                        break
            sc = nm.get("suitcase_target")
            if isinstance(sc, CardInstance):
                for os_c, cs_c in zip(orig_suitcase, new_suitcase):
                    if os_c is sc:
                        nm["suitcase_target"] = cs_c
                        break
    else:
        sim._suitcase_shared = True
        new_suitcase = src.suitcase
        if p.played_today and not (
            p_flags[player_index]
            & (_PF_ALL_TAGS | _PF_HAS_CARD_PLAY | _PF_HAS_ACQUIRE | _PF_HAS_PICK)
        ):
            new_played = p.played_today.copy()
            sim._played_items_shared = True
            sim._shared_played_len = len(new_played)
        else:
            new_played = [
                CardInstance(
                    c.instance_id,
                    c.definition,
                    c.is_tomorrow,
                    dict(c.markers) if c.markers else {},
                )
                for c in p.played_today
            ]
            sim._played_items_shared = False
            sim._shared_played_len = 0

    new_p = PlayerState(
        name=p.name,
        energy=p.energy,
        fun=p.fun,
        character=p.character,
        hand=(
            p.hand.copy()
            if not has_marked_hand
            else [
                CardInstance(
                    c.instance_id,
                    c.definition,
                    c.is_tomorrow,
                    dict(c.markers),
                )
                if (c.markers or c.is_tomorrow)
                else c
                for c in p.hand
            ]
        ),
        played_today=new_played,
        tomorrow_cards=p.tomorrow_cards,
        asleep=p.asleep,
        skipped_turns=p.skipped_turns,
        acquired_cards=_NOOP_COUNTER,
        picked_cards=_NOOP_COUNTER,
    )

    players_list = list(src_players)
    players_list[player_index] = new_p
    sim.players = players_list
    sim.suitcase = new_suitcase
    sim.trunk = src.trunk
    sim.discard = src.discard
    sim._known_trunk_state = src._known_trunk_state  # type: ignore[attr-defined]
    sim.day = src.day
    sim.daily_conditions = src.daily_conditions
    sim.characters = src.characters
    sim.daily_condition = src.daily_condition
    sim._condition_deck = src._condition_deck
    sim.revealed_conditions = src.revealed_conditions
    sim._condition_knowledge = src._condition_knowledge
    sim.starting_player = src.starting_player
    sim._is_setup = src._is_setup
    sim.stats = src.stats
    sim._fun_at_start_of_scoring = src._fun_at_start_of_scoring
    return sim


def _card_allows_extra_plays(
    game: Game, player: PlayerState, card: CardInstance
) -> bool:
    m = card.markers
    if not m or "_copied_definition" not in m:
        d = card.definition
        meta = _HYDROGEN_META_BY_SLUG.get(d.slug)
        if meta is not None and meta.defn is d:
            return (
                bool(meta.beh.allows_extra_card_plays(game, player, card))
                if meta.has_allows_extra
                else False
            )
    return bool(card.effective_behavior.allows_extra_card_plays(game, player, card))


class HydrogenAI(CarbonAI):
    """Aggressive multi-big-day opponent-aware lookahead policy."""

    BEAM_WIDTH = 48
    DRAFT_BEAM_WIDTH = 18
    BRANCH_LIMIT = 13
    MAX_TURN_PLAYS = 14
    RESERVE_TOMORROW_WEIGHT = 0.74
    RESERVE_SLOT_WEIGHTS_COMBO = (1.0, 0.99, 0.79, 0.44)
    RESERVE_SLOT_WEIGHTS_PLAIN = (1.0, 0.90, 0.46, 0.21)
    RESERVE_SLOT_TAIL_WEIGHT = 0.06
    FIRST_TO_BED_BONUS = 0.65
    DRAFT_SAMPLES = 8
    PLAY_SAMPLES = 4
    _reserve_eval_cache = _HYDROGEN_RESERVE_EVAL_CACHE

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)
        _ensure_hydrogen_caches()
        global _HYDROGEN_SIM_GAME_CLS
        if _HYDROGEN_SIM_GAME_CLS is None:
            _HYDROGEN_SIM_GAME_CLS = _get_hydrogen_sim_game_cls()
        self.__dict__.pop("_reserve_eval_cache", None)

    def _prepare_observable_game(self, game: Game, player_index: int) -> Game:
        global _HYDROGEN_SIM_GAME_CLS
        if _HYDROGEN_SIM_GAME_CLS is None:
            _HYDROGEN_SIM_GAME_CLS = _get_hydrogen_sim_game_cls()
        sim = super()._prepare_observable_game(game, player_index)
        sim.__class__ = _HYDROGEN_SIM_GAME_CLS
        sim.ais = _HYDROGEN_SIM_AIS
        sim._suitcase_shared = False  # type: ignore[attr-defined]
        sim._has_nonplain_hand = any(  # type: ignore[attr-defined]
            bool(c.markers) and not _is_plain_hand_markers(c.markers)
            for p in sim.players
            for c in p.hand
        )
        sim._recompute_all_flags()  # type: ignore[attr-defined]
        return sim

    def _get_live_known_trunk_top(
        self, game: Game, player_index: int
    ) -> tuple[CardInstance, ...]:
        if (
            self._known_trunk_state is not None
            and self._known_trunk_state[0] == id(game)
            and self._known_trunk_state[1] == game.day
            and len(game.trunk) > self._known_trunk_state[2]
        ):
            self._known_trunk_state = None
            return ()
        return super()._get_live_known_trunk_top(game, player_index)

    def _simulate_play_step(
        self,
        state: Game,
        player_index: int,
        hand_idx: int,
        known_cost: int | None = None,
    ) -> Game:
        child = _hydrogen_copy_game_for_play(state, player_index)
        card_title = child.players[player_index].hand[hand_idx].definition.title
        is_hsim = type(child) is _HYDROGEN_SIM_GAME_CLS
        use_fast = known_cost is not None and is_hsim

        if card_title in _BLIND_TRUNK_STEP_TITLES and not _get_sim_known_trunk_top(child):
            if card_title == "Assorted Cutlery" and child.trunk:
                if getattr(child, "_trunk_shared", False):
                    child.trunk = child.trunk.copy()
                    child._trunk_shared = False  # type: ignore[attr-defined]
                best_t_idx = len(child.trunk) - 1
                best_diff = float("inf")
                wanted_tags = _wanted_tags_for_player_fast(
                    child.players[player_index]
                )
                meta_by_slug = _HYDROGEN_META_BY_SLUG
                is_canon = is_hsim and child._all_canonical_defs  # type: ignore[attr-defined]
                if is_canon:
                    seen_slugs: set[str] = set()
                    for t_i, tc in enumerate(child.trunk):
                        if not tc.markers:
                            s = tc.definition.slug
                            if s in seen_slugs:
                                continue
                            seen_slugs.add(s)
                            t_prof = meta_by_slug[s].prof
                        else:
                            t_prof = _get_card_meta(tc).prof
                        if (
                            t_prof.ends_day_immediately
                            or t_prof.penalizes_before
                            or t_prof.penalizes_after
                        ):
                            continue
                        diff = abs(
                            _hydrogen_free_play_value(
                                child, player_index, tc, wanted_tags
                            )
                            - 2.2
                        )
                        if diff < best_diff:
                            best_diff = diff
                            best_t_idx = t_i
                else:
                    seen_defn_ids: set[int] = set()
                    for t_i, tc in enumerate(child.trunk):
                        if not tc.markers:
                            d = tc.definition
                            did = id(d)
                            if did in seen_defn_ids:
                                continue
                            seen_defn_ids.add(did)
                            t_meta_opt = meta_by_slug.get(d.slug)
                            t_prof = (
                                t_meta_opt.prof
                                if (t_meta_opt is not None and t_meta_opt.defn is d)
                                else _beryllium_build_profile(tc)
                            )
                        else:
                            t_prof = _beryllium_build_profile(tc)
                        if (
                            t_prof.ends_day_immediately
                            or t_prof.penalizes_before
                            or t_prof.penalizes_after
                        ):
                            continue
                        diff = abs(
                            _hydrogen_free_play_value(
                                child, player_index, tc, wanted_tags
                            )
                            - 2.2
                        )
                        if diff < best_diff:
                            best_diff = diff
                            best_t_idx = t_i
                child.trunk[-1], child.trunk[best_t_idx] = (
                    child.trunk[best_t_idx],
                    child.trunk[-1],
                )
                if use_fast:
                    child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
                else:
                    child.play_card(player_index, hand_idx)
                return child
            if card_title == "Poker":
                if use_fast:
                    child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
                else:
                    child.play_card(player_index, hand_idx)
                p_card = child.players[player_index].played_today[-1]
                p_card.markers["poker_success"] = False
                p_card.markers["_blind_poker"] = True
                if is_hsim:
                    child._p_flags[player_index] |= _PF_HAS_LOOP2  # type: ignore[attr-defined]
                return child
            if card_title == "Quacks":
                if use_fast:
                    child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
                else:
                    child.play_card(player_index, hand_idx)
                q_card = child.players[player_index].played_today[-1]
                q_card.markers["_quacks_outdoors"] = False
                q_card.markers["energy_cubes"] = 1
                q_card.markers["_blind_quacks"] = True
                if is_hsim:
                    child._p_flags[player_index] |= (  # type: ignore[attr-defined]
                        _PF_HAS_LOOP2 | _PF_HAS_SIG_MARKER
                    )
                return child

        if use_fast:
            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
        else:
            child.play_card(player_index, hand_idx)

        if card_title == "Tell a Story":
            p_child = child.players[player_index]
            has_epic_duels = any(
                _effective_card_title(pc) == "Epic Duels"
                for pc in p_child.played_today
            )
            awake_opps = sum(
                1
                for idx, opp in enumerate(child.players)
                if idx != player_index and not opp.asleep
            )
            if not has_epic_duels and awake_opps >= 2:
                child._known_trunk_state = None

        return child

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if self._evaluation_depth >= 1 or len(options) <= 1:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        _ensure_hydrogen_caches()
        self._ensure_live_game(game)
        live_player = game.players[player_index]
        own_markers = _carbon_own_suitcase_markers(live_player)
        pr_markers, scout_markers = _opponent_suitcase_markers(
            game, player_index
        )
        remaining_draw_picks = self._remaining_draw_picks_today(
            game, player_index
        )
        if live_player.played_today and (
            len(live_player.played_today) >= 8
            or len(live_player.hand) >= 10
            or len(options) < 2
        ):
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        wanted_tags, live_ctx = _build_wanted_and_ctx(
            live_player, getattr(game, "_all_canonical_defs", False)
        )
        total_future_my_picks = remaining_draw_picks
        quick_scores: list[float] = []
        for option in options:
            qs = (
                _hydrogen_quick_card_desirability(
                    game,
                    player_index,
                    option,
                    wanted_tags,
                    remaining_draw_picks=total_future_my_picks,
                    _ctx=live_ctx,
                )
                + self._draft_synergy_adjustment(
                    game, player_index, option, wanted_tags
                )
                + _suitcase_denial_value(
                    game, player_index, option, pr_markers, scout_markers
                )
            )
            if own_markers and any(m in option.markers for m in own_markers):
                qs -= 3.0
            quick_scores.append(qs)
        max_qs = max(quick_scores)

        has_stochastic_option = any(
            opt.title in _STOCHASTIC_PLAY_TITLES for opt in options
        )
        if not live_player.played_today:
            if remaining_draw_picks >= 2:
                num_samples = self.DRAFT_SAMPLES
            elif remaining_draw_picks == 1:
                num_samples = min(self.DRAFT_SAMPLES, 4)
            elif game.day < DAY_COUNT or has_stochastic_option:
                num_samples = min(self.DRAFT_SAMPLES, 2)
            else:
                num_samples = 1
            draft_beam = self.DRAFT_BEAM_WIDTH
            draft_max_plays = 10
        else:
            num_samples = (
                2
                if (has_stochastic_option or len(live_player.played_today) < 5)
                else 1
            )
            draft_beam = 12 if len(live_player.played_today) < 6 else 6
            draft_max_plays = 9 if len(live_player.played_today) < 6 else 5

        samples: list[tuple[Game, list[tuple[CardInstance, ...]]]] = []
        for _ in range(num_samples):
            base_sim = self._prepare_observable_game(game, player_index)
            future_pools: list[tuple[CardInstance, ...]] = []
            if total_future_my_picks > 0 and len(base_sim.trunk) >= SUITCASE_SIZE * (
                total_future_my_picks + 1
            ):
                known_ids = {
                    kc.instance_id for kc in _get_sim_known_trunk_top(base_sim)
                }
                pool_candidates = [
                    tc
                    for tc in base_sim.trunk
                    if tc.instance_id not in known_ids
                ]
                cursor = 0
                for _ in range(total_future_my_picks):
                    if cursor + SUITCASE_SIZE <= len(pool_candidates):
                        future_pools.append(
                            tuple(
                                pool_candidates[
                                    cursor : cursor + SUITCASE_SIZE
                                ]
                            )
                        )
                        cursor += SUITCASE_SIZE
            samples.append((base_sim, future_pools))

        best_indices: list[int] = []
        best_value = float("-inf")

        for idx, option in enumerate(options):
            synergy_val = self._draft_synergy_adjustment(
                samples[0][0], player_index, option, wanted_tags
            ) + _suitcase_denial_value(
                game, player_index, option, pr_markers, scout_markers
            )
            if own_markers and any(m in option.markers for m in own_markers):
                synergy_val -= 3.0

            eval_samples = (
                samples[:1] if quick_scores[idx] < max_qs - 5.5 else samples
            )
            eff_beam = (
                max(5, draft_beam - 4)
                if quick_scores[idx] < max_qs - 5.5
                else draft_beam
            )

            plan_sum = 0.0
            for base_sim, future_pools in eval_samples:
                sim = _hydrogen_copy_game(base_sim)
                if (
                    idx < len(sim.suitcase)
                    and sim.suitcase[idx].instance_id == option.instance_id
                ):
                    picked = sim.suitcase.pop(idx)
                else:
                    picked = _clone_card(option)
                sim.give_card(player_index, picked)
                sim.players[player_index].picked_cards[picked.title] += 1
                sim.record_card_pick_or_draw(player_index, picked)
                if sim.trunk or sim.discard:
                    try:
                        sim.suitcase.insert(idx, sim._draw_from_trunk())
                    except Exception:
                        pass

                for pool in future_pools:
                    cloned_pool = tuple(_clone_card(c) for c in pool)
                    pick_i = _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_card(
                        sim, player_index, cloned_pool
                    )
                    extra_card = cloned_pool[pick_i]
                    for t_i, tc in enumerate(sim.trunk):
                        if tc.instance_id == extra_card.instance_id:
                            sim.trunk.pop(t_i)
                            break
                    sim.give_card(player_index, extra_card)
                    sim.players[player_index].picked_cards[
                        extra_card.title
                    ] += 1

                if (
                    not live_player.played_today
                    and sim.suitcase
                    and len(sim.trunk) >= SUITCASE_SIZE
                ):
                    known_top_cards = _get_sim_known_trunk_top(sim)
                    known_ids = (
                        {kc.instance_id for kc in known_top_cards}
                        if known_top_cards
                        else None
                    )
                    needed = len(sim.suitcase)
                    chosen_trunk_indices: list[int] = []
                    trunk_list = sim.trunk
                    sim_is_canon = getattr(sim, "_all_canonical_defs", False)
                    sim_ctx = _build_quick_zone_ctx(
                        sim.players[player_index], sim_is_canon
                    )
                    if sim_is_canon:
                        seen_ok_slug: dict[str, bool] = {}
                        meta_by_slug = _HYDROGEN_META_BY_SLUG
                        for i in range(len(trunk_list) - 1, -1, -1):
                            tc = trunk_list[i]
                            if known_ids is not None and tc.instance_id in known_ids:
                                continue
                            if not tc.markers:
                                s = tc.definition.slug
                                ok = seen_ok_slug.get(s)
                                if ok is None:
                                    ok = (
                                        0.2
                                        <= _hydrogen_quick_card_desirability(
                                            sim,
                                            player_index,
                                            tc,
                                            {},
                                            _ctx=sim_ctx,
                                            _in_hand=False,
                                            _meta=meta_by_slug[s],
                                        )
                                        <= 1.6
                                    )
                                    seen_ok_slug[s] = ok
                            else:
                                ok = (
                                    0.2
                                    <= _hydrogen_quick_card_desirability(
                                        sim,
                                        player_index,
                                        tc,
                                        {},
                                        _ctx=sim_ctx,
                                        _in_hand=False,
                                    )
                                    <= 1.6
                                )
                            if ok:
                                chosen_trunk_indices.append(i)
                                if len(chosen_trunk_indices) == needed:
                                    break
                    else:
                        seen_ok: dict[int, bool] = {}
                        for i in range(len(trunk_list) - 1, -1, -1):
                            tc = trunk_list[i]
                            if known_ids is not None and tc.instance_id in known_ids:
                                continue
                            if not tc.markers:
                                did = id(tc.definition)
                                ok = seen_ok.get(did)
                                if ok is None:
                                    ok = (
                                        0.2
                                        <= _hydrogen_quick_card_desirability(
                                            sim,
                                            player_index,
                                            tc,
                                            {},
                                            _ctx=sim_ctx,
                                            _in_hand=False,
                                        )
                                        <= 1.6
                                    )
                                    seen_ok[did] = ok
                            else:
                                ok = (
                                    0.2
                                    <= _hydrogen_quick_card_desirability(
                                        sim,
                                        player_index,
                                        tc,
                                        {},
                                        _ctx=sim_ctx,
                                        _in_hand=False,
                                    )
                                    <= 1.6
                                )
                            if ok:
                                chosen_trunk_indices.append(i)
                                if len(chosen_trunk_indices) == needed:
                                    break
                    for slot_i, t_idx in enumerate(chosen_trunk_indices):
                        sim.suitcase[slot_i], trunk_list[t_idx] = (
                            _clone_card(trunk_list[t_idx]),
                            sim.suitcase[slot_i],
                        )

                _, sample_val = self._search_best_day_plan(
                    sim,
                    player_index,
                    beam_width=eff_beam,
                    max_plays=draft_max_plays,
                )
                plan_sum += sample_val

            plan_val = plan_sum / len(eval_samples)
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
        bonus = super()._draft_synergy_adjustment(
            game, player_index, card, wanted_tags
        )
        player = game.players[player_index]
        days_left = DAY_COUNT - game.day
        title = card.title
        cost = game.energy_cost(player_index, card)
        zero_tag = (
            player.character.zero_fun_tag
            if player.character is not None
            else ""
        )
        is_zeroed = bool(zero_tag and zero_tag in card.tags)

        # Apply non-overlapping draft calibrations for under/over-performing cards
        if title == "Floating" and (days_left >= 1 or len(player.hand) >= 2):
            bonus += 0.50
        elif title == "Couch Tube" and not is_zeroed:
            has_heavy = any(
                c is not card and c.effective_cost >= 4
                for c in player.hand
            )
            bonus += 0.55 + (0.35 if has_heavy else 0.0)
        elif title == "Make Plans" and (days_left >= 1 or len(player.hand) <= 3):
            bonus += 0.45
        elif title == "Chuck a Frisbee" and days_left >= 1 and not is_zeroed:
            bonus += 0.55
        elif title in {"Wakeboard", "Risk", "Screamer Battle"} and not is_zeroed:
            if days_left >= 2 or cost <= player.energy + 3:
                bonus += 0.35
        elif title == "Keeper":
            if any(
                c is not card and "Outdoors" in c.tags
                for c in (*player.hand, *player.tomorrow_cards)
            ):
                bonus += 0.40
        elif title == "Forced Family Fun":
            if days_left >= 2 or any(
                c is not card and "Board Game" in c.tags
                for c in (*player.hand, *player.tomorrow_cards)
            ):
                bonus += 0.35
        elif title == "High-End Red":
            other_food = sum(
                1
                for c in (*player.hand, *player.played_today)
                if c is not card
                and "Food" in c.tags
                and not _get_card_meta(c).must_play_first
            )
            if other_food < 2:
                bonus -= 0.75
        elif title == "Johnny Appleseed":
            bonus -= 0.60
        elif title == "Stretch":
            has_later_ex = any(
                c is not card
                and "Exercise" in c.tags
                and c.effective_cost <= 5
                and not _get_card_meta(c).must_play_first
                for c in player.hand
            )
            if not has_later_ex:
                bonus -= 0.55
        elif title == "Cities and Knights":
            has_dump = any(
                c is not card
                and (
                    c.title in {"Wakesurf", "Drop a Ski"}
                    or (
                        c.effective_base_fun <= 0
                        and not _get_card_meta(c).goes_to_tomorrow
                    )
                )
                for c in player.hand
            )
            if not has_dump:
                bonus -= 0.65
        elif title == "Scrabble":
            if not any(
                _effective_card_title(c) == "Paddleboat"
                for c in (*player.hand, *player.played_today)
            ):
                bonus -= 0.65
        elif title == "Sunrise":
            bonus -= 0.60
        elif title == "Doxology":
            bonus -= 0.45
        elif title == "Bug Spray":
            out_cnt = sum(
                1
                for c in (*player.hand, *player.tomorrow_cards)
                if c is not card and "Outdoors" in c.tags
            )
            if out_cnt < 2:
                bonus -= 0.45
        elif title == "Evening Chat":
            bonus -= 0.50
        elif title == "Paint":
            bonus -= 0.45
        elif title == "Cliff Climbing":
            has_out = any(
                c is not card and "Outdoors" in c.tags
                for c in (*player.hand, *player.tomorrow_cards, *player.played_today)
            )
            if not has_out:
                bonus -= 0.65
        elif title == "Water Volleyball":
            if not any(
                c is not card and "Exercise" in c.tags
                for c in (*player.hand, *player.tomorrow_cards)
            ):
                bonus -= 0.40
        elif title == "Bracelet Making":
            if not any(
                c is not card and "Social" in c.tags
                for c in (*player.hand, *player.tomorrow_cards)
            ):
                bonus -= 0.40

        return bonus

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        self._ensure_live_game(game)
        cached = self._take_cached_play_choice(
            game, player_index, playable_hand_indices
        )
        if cached is not None and cached in playable_hand_indices:
            return cached
        if self._evaluation_depth >= 1:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_play(
                game, player_index, playable_hand_indices
            )
        best_action = self._select_root_play_action(
            game,
            player_index,
            playable_hand_indices,
            allow_bed_now=False,
        )
        if best_action is not None and best_action in playable_hand_indices:
            return best_action
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_play(
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
        self._ensure_live_game(game)
        if (
            len(game.trunk) + len(game.discard) <= 6
            or len(game.players[player_index].played_today) >= 20
        ):
            return None
        if self._evaluation_depth >= 1:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_extra_card_to_play(
                game, player_index, playable_hand_indices
            )
        best_action = self._select_root_play_action(
            game,
            player_index,
            playable_hand_indices,
            allow_bed_now=True,
            in_extra_play=True,
        )
        if best_action is not None and best_action in playable_hand_indices:
            return best_action
        return None

    def _select_root_play_action(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
        *,
        allow_bed_now: bool,
        in_extra_play: bool = False,
    ) -> int | None:
        played_count = len(game.players[player_index].played_today)
        use_multi = (
            self.PLAY_SAMPLES > 1
            and played_count < 8
            and self._has_active_stochastic_play(
                game, player_index, playable_hand_indices
            )
        )
        base_sim = self._prepare_observable_game(game, player_index)
        first_action_0, max_val_0, table_0 = self._search_best_day_plan_with_table(
            base_sim,
            player_index,
            beam_width=self.BEAM_WIDTH,
            allow_bed_now=allow_bed_now,
            allowed_first_indices=playable_hand_indices,
            in_extra_play=in_extra_play,
        )
        if not use_multi or len(table_0) <= 1:
            return first_action_0

        player_live = game.players[player_index]
        ranked_act = sorted(
            table_0.items(), key=lambda kv: kv[1][0], reverse=True
        )
        contenders: list[int | None] = []
        for act, (val0, _) in ranked_act:
            if len(contenders) < 4 and val0 >= max_val_0 - 3.2:
                contenders.append(act)
            elif (
                act is not None
                and act < len(player_live.hand)
                and player_live.hand[act].title in _STOCHASTIC_PLAY_TITLES
                and val0 >= max_val_0 - 4.5
                and len(contenders) < 5
            ):
                contenders.append(act)
        if len(contenders) <= 1:
            return first_action_0

        extra_samples = [
            self._prepare_observable_game(game, player_index)
            for _ in range(self.PLAY_SAMPLES - 1)
        ]
        top_is_stochastic = (
            first_action_0 is not None
            and first_action_0 < len(player_live.hand)
            and player_live.hand[first_action_0].title in _STOCHASTIC_PLAY_TITLES
        )
        if top_is_stochastic:
            extra_samples.append(
                self._prepare_observable_game(game, player_index)
            )

        best_actions: list[int | None] = []
        best_mean = float("-inf")
        n_total = 2.0 + len(extra_samples)

        for cand in contenders:
            total = 2.0 * table_0[cand][0]
            valid = True
            for sim in extra_samples:
                root_skipped = sim.players[player_index].skipped_turns
                if cand is None:
                    bed_sim = _hydrogen_copy_game_for_play(sim, player_index)
                    if not in_extra_play:
                        bed_sim.go_to_bed(player_index)
                    total += self._evaluate_terminal_day(
                        bed_sim,
                        player_index,
                        turns_from_root=1 if in_extra_play else 0,
                        root_skipped_turns=root_skipped,
                    )
                else:
                    try:
                        cand_card = sim.players[player_index].hand[cand]
                        child_extra = in_extra_play or _card_allows_extra_plays(
                            sim, sim.players[player_index], cand_card
                        )
                        child = self._simulate_play_step(
                            sim, player_index, cand
                        )
                        if child.players[player_index].asleep:
                            val = self._evaluate_terminal_day(
                                child,
                                player_index,
                                turns_from_root=1,
                                root_skipped_turns=root_skipped,
                            )
                        else:
                            _, val = self._search_best_day_plan(
                                child,
                                player_index,
                                beam_width=16,
                                max_plays=10,
                                allow_bed_now=True,
                                base_turns_offset=1,
                                root_skipped_override=root_skipped,
                                start_extra_active=child_extra,
                            )
                        total += val
                    except Exception:
                        valid = False
                        break
            if not valid:
                continue
            mean_val = total / n_total
            if mean_val > best_mean + 1e-9:
                best_mean = mean_val
                best_actions = [cand]
            elif abs(mean_val - best_mean) <= 1e-9:
                best_actions.append(cand)

        if not best_actions:
            return first_action_0
        if None in best_actions:
            return None
        return self.rng.choice(best_actions)

    def _state_signature(self, sim: Game, player_index: int) -> tuple:
        p = sim.players[player_index]
        hand = p.hand
        if len(hand) > 1:
            h_titles = [c.definition.title for c in hand]
            h_titles.sort()
            hand_sig: tuple = tuple(h_titles)
        elif hand:
            hand_sig = (hand[0].definition.title,)
        else:
            hand_sig = ()
        played = p.played_today
        if type(sim) is _HYDROGEN_SIM_GAME_CLS and not (
            sim._p_flags[player_index] & _PF_HAS_SIG_MARKER  # type: ignore[attr-defined]
        ):
            played_sig = tuple([c.definition.title for c in played]) if played else ()
        else:
            played_list = []
            for c in played:
                m = c.markers
                if not m:
                    played_list.append(c.definition.title)
                elif (
                    "_copied_definition" not in m
                    and not m.get("energy_cubes")
                    and m.get("target_card") is None
                    and m.get("target_player_index") is None
                ):
                    played_list.append(c.definition.title)
                else:
                    played_list.append(
                        (
                            c.definition.title,
                            _effective_card_title(c),
                            m.get("energy_cubes", 0),
                            getattr(m.get("target_card"), "instance_id", None),
                            m.get("target_player_index"),
                        )
                    )
            played_sig = tuple(played_list)
        tom = p.tomorrow_cards
        return (
            p.energy,
            p.fun,
            played_sig,
            hand_sig,
            tuple([c.definition.title for c in tom]) if tom else (),
        )

    def _continuation_potential(self, sim: Game, player_index: int) -> float:
        """PYTHONHASHSEED-deterministic continuation upside for active/pending scalers."""
        player = sim.players[player_index]
        hand = player.hand
        if player.asleep or not hand:
            return 0.0

        meta_by_slug = _HYDROGEN_META_BY_SLUG
        if not meta_by_slug:
            _ensure_hydrogen_caches()
        is_hsim = type(sim) is _HYDROGEN_SIM_GAME_CLS
        is_canon = is_hsim and sim._all_canonical_defs  # type: ignore[attr-defined]
        played_today = player.played_today
        energy_budget = player.energy

        if (
            is_canon
            and not sim._has_nonplain_hand  # type: ignore[attr-defined]
            and _ALL_CANONICAL_META_TAGS_OK
        ):
            pf = sim._p_flags[player_index]  # type: ignore[attr-defined]
            aa_mask = 0
            aa_arr: list[float] | None = None
            global_after_bonus = 0.0
            has_after_all_tags = False
            if pf & _PF_HAS_CONT_AFTER:
                for played in played_today:
                    pm = played.markers
                    meta = (
                        meta_by_slug[played.definition.slug]
                        if (not pm or "_copied_definition" not in pm)
                        else _get_card_meta(played)
                    )
                    if meta.after_all_tags:
                        has_after_all_tags = True
                    if (
                        meta.is_multi_scaler
                        and not meta.is_before_scaler
                        and meta.signed_match_benefit > 0.0
                    ):
                        if aa_arr is None:
                            aa_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        smb = meta.signed_match_benefit
                        for t in meta.sorted_target_tags:
                            b_idx = _TAG_BIT[t]
                            aa_arr[b_idx] += smb
                            aa_mask |= 1 << b_idx
                    elif meta.is_global_after_scaler:
                        global_after_bonus += meta.signed_match_benefit

            pending_before_bonus = 0.0
            for hc in hand:
                h_meta = meta_by_slug[hc.definition.slug]
                if h_meta.is_before_scaler and h_meta.cost <= energy_budget:
                    if h_meta.target_mask:
                        t_m = h_meta.target_mask
                        matches_played = 0
                        for p in played_today:
                            pm = p.markers
                            p_mask = (
                                _ALL_TAGS_BITMASK
                                if (pm and "_all_tags" in pm and pm["_all_tags"])
                                else meta_by_slug[p.definition.slug].tag_mask
                            )
                            if p_mask & t_m:
                                matches_played += 1
                        pending_before_bonus += (
                            matches_played * h_meta.signed_match_benefit * 0.90
                        )
                        if aa_arr is None:
                            aa_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        b_delta = h_meta.signed_match_benefit * 0.70
                        for t in h_meta.sorted_target_tags:
                            b_idx = _TAG_BIT[t]
                            aa_arr[b_idx] += b_delta
                            aa_mask |= 1 << b_idx
                    elif h_meta.title == "Evening on the Dock":
                        pending_before_bonus += len(played_today) * 0.90
                    elif h_meta.title == "Couch Tube":
                        cost_sum = 0
                        for p in played_today:
                            pm = p.markers
                            cost_sum += (
                                p.definition.cost
                                if (not pm or "_copied_definition" not in pm)
                                else p.effective_cost
                            )
                        pending_before_bonus += cost_sum * 0.85

            if not aa_mask and global_after_bonus == 0.0:
                return (
                    (energy_budget if energy_budget < 7 else 7) * 0.72
                    + pending_before_bonus
                )
            matching_playable = 0.0
            if has_after_all_tags:
                all_after_sum = 0.0
                if aa_arr is not None:
                    for b_idx in range(8):
                        if aa_mask & (1 << b_idx):
                            all_after_sum += aa_arr[b_idx]
                for hc in hand:
                    c_meta = meta_by_slug[hc.definition.slug]
                    if c_meta.cost <= energy_budget:
                        matching_playable += global_after_bonus
                        matching_playable += all_after_sum
            else:
                for hc in hand:
                    c_meta = meta_by_slug[hc.definition.slug]
                    if c_meta.cost <= energy_budget:
                        matching_playable += global_after_bonus
                        if aa_mask & c_meta.tag_mask:
                            for b_idx in c_meta.sorted_tag_bits:
                                if aa_mask & (1 << b_idx):
                                    matching_playable += aa_arr[b_idx]  # type: ignore[index]
            return (
                (energy_budget if energy_budget < 7 else 7) * 0.72
                + matching_playable * 0.95
                + pending_before_bonus
            )

        active_after_tags: dict[str, float] = {}
        global_after_bonus = 0.0
        has_after_all_tags = False
        if not is_hsim or (sim._p_flags[player_index] & _PF_HAS_CONT_AFTER):  # type: ignore[attr-defined]
            for played in played_today:
                pm = played.markers
                p_uncopied = not pm or "_copied_definition" not in pm
                if p_uncopied:
                    d = played.definition
                    meta = (
                        meta_by_slug[d.slug]
                        if is_canon
                        else meta_by_slug.get(d.slug)
                    )
                    if not is_canon and (meta is None or meta.defn is not d):
                        meta = None
                else:
                    meta = _get_card_meta(played)
                if meta is not None:
                    if meta.after_all_tags:
                        has_after_all_tags = True
                    if (
                        meta.is_multi_scaler
                        and not meta.is_before_scaler
                        and meta.signed_match_benefit > 0.0
                    ):
                        smb = meta.signed_match_benefit
                        for t in meta.sorted_target_tags:
                            active_after_tags[t] = (
                                active_after_tags[t] + smb
                                if t in active_after_tags
                                else smb
                            )
                    elif meta.is_global_after_scaler:
                        global_after_bonus += meta.signed_match_benefit
                else:
                    prof = _beryllium_build_profile(played)
                    p_beh = (
                        played.definition.behavior
                        if p_uncopied
                        else played.effective_behavior
                    )
                    if p_beh.after_all_tags:
                        has_after_all_tags = True
                    if (
                        prof.is_multi_scaler
                        and not prof.is_before_scaler
                        and prof.signed_match_benefit > 0
                    ):
                        p_targets = (
                            prof.target_tags
                            if len(prof.target_tags) <= 1
                            else sorted(prof.target_tags)
                        )
                        smb = prof.signed_match_benefit
                        for t in p_targets:
                            active_after_tags[t] = (
                                active_after_tags[t] + smb
                                if t in active_after_tags
                                else smb
                            )
                    elif prof.is_global_after_scaler:
                        global_after_bonus += prof.signed_match_benefit

        pending_before_bonus = 0.0

        for hc in hand:
            hm = hc.markers
            h_uncopied = not hm or "_copied_definition" not in hm
            if h_uncopied:
                hd = hc.definition
                h_meta_opt = meta_by_slug.get(hd.slug)
                if h_meta_opt is not None and h_meta_opt.defn is not hd:
                    h_meta_opt = None
            else:
                h_meta_opt = None
            if h_meta_opt is not None:
                if h_meta_opt.is_before_scaler and h_meta_opt.cost <= energy_budget:
                    if h_meta_opt.target_tags:
                        t_tags = h_meta_opt.target_tags
                        matches_played = 0
                        for p in played_today:
                            if (
                                p.definition.tags if not p.markers else p.tags
                            ) & t_tags:
                                matches_played += 1
                        pending_before_bonus += (
                            matches_played * h_meta_opt.signed_match_benefit * 0.90
                        )
                        b_delta = h_meta_opt.signed_match_benefit * 0.70
                        for t in h_meta_opt.sorted_target_tags:
                            active_after_tags[t] = (
                                active_after_tags[t] + b_delta
                                if t in active_after_tags
                                else b_delta
                            )
                    elif h_meta_opt.title == "Evening on the Dock":
                        pending_before_bonus += len(played_today) * 0.90
                    elif h_meta_opt.title == "Couch Tube":
                        cost_sum = 0
                        for p in played_today:
                            pm = p.markers
                            cost_sum += (
                                p.definition.cost
                                if (not pm or "_copied_definition" not in pm)
                                else p.effective_cost
                            )
                        pending_before_bonus += cost_sum * 0.85
            else:
                h_prof = _beryllium_build_profile(hc)
                h_cost = hc.definition.cost if h_uncopied else hc.effective_cost
                if h_prof.is_before_scaler and h_cost <= energy_budget:
                    if h_prof.target_tags:
                        t_tags = h_prof.target_tags
                        matches_played = 0
                        for p in played_today:
                            if (
                                p.definition.tags if not p.markers else p.tags
                            ) & t_tags:
                                matches_played += 1
                        pending_before_bonus += (
                            matches_played * h_prof.signed_match_benefit * 0.90
                        )
                        b_delta = h_prof.signed_match_benefit * 0.70
                        for t in (
                            t_tags if len(t_tags) <= 1 else sorted(t_tags)
                        ):
                            active_after_tags[t] = (
                                active_after_tags[t] + b_delta
                                if t in active_after_tags
                                else b_delta
                            )
                    elif hc.definition.title == "Evening on the Dock":
                        pending_before_bonus += len(played_today) * 0.90
                    elif hc.definition.title == "Couch Tube":
                        cost_sum = 0
                        for p in played_today:
                            pm = p.markers
                            cost_sum += (
                                p.definition.cost
                                if (not pm or "_copied_definition" not in pm)
                                else p.effective_cost
                            )
                        pending_before_bonus += cost_sum * 0.85

        matching_playable = 0.0
        if active_after_tags or global_after_bonus != 0.0:
            all_after_sum = (
                sum(active_after_tags[k] for k in sorted(active_after_tags))
                if has_after_all_tags
                else 0.0
            )
            for card in hand:
                cm = card.markers
                if _is_plain_hand_markers(cm):
                    cd = card.definition
                    c_meta_opt = meta_by_slug.get(cd.slug)
                    if c_meta_opt is not None and c_meta_opt.defn is cd:
                        if c_meta_opt.cost <= energy_budget:
                            matching_playable += global_after_bonus
                            if has_after_all_tags:
                                matching_playable += all_after_sum
                            else:
                                for t in c_meta_opt.sorted_tags:
                                    if t in active_after_tags:
                                        matching_playable += active_after_tags[t]
                        continue
                cost = (
                    card.definition.cost
                    if (not cm or "_copied_definition" not in cm)
                    else card.effective_cost
                )
                if cost <= energy_budget:
                    matching_playable += global_after_bonus
                    if has_after_all_tags:
                        matching_playable += all_after_sum
                    else:
                        c_tags = card.definition.tags if not cm else card.tags
                        for t in (
                            c_tags if len(c_tags) <= 1 else sorted(c_tags)
                        ):
                            if t in active_after_tags:
                                matching_playable += active_after_tags[t]
        return (
            min(player.energy, 7) * 0.72
            + matching_playable * 0.95
            + pending_before_bonus
        )

    def _ordered_candidate_indices_with_costs(
        self,
        sim: Game,
        player_index: int,
        limit: int,
    ) -> list[tuple[int, int]]:
        """Context- and opponent-aware left-to-right play ordering returning `(hand_idx, cost)`."""
        player = sim.players[player_index]
        if player.asleep or not player.hand:
            return []
        is_hsim = type(sim) is _HYDROGEN_SIM_GAME_CLS
        if is_hsim:
            playable_with_costs = sim.hand_playable_with_costs(player_index)  # type: ignore[attr-defined]
        else:
            p_energy = player.energy
            playable_with_costs = []
            for index, card in enumerate(player.hand):
                m = card.markers
                c_beh = (
                    card.definition.behavior
                    if (not m or "_copied_definition" not in m)
                    else card.effective_behavior
                )
                if (
                    type(c_beh).can_play is not CardBehavior.can_play
                    and not c_beh.can_play(sim, player, card)
                ):
                    continue
                cost = sim.energy_cost(player_index, card)
                if cost <= p_energy:
                    playable_with_costs.append((index, cost))
        if not playable_with_costs:
            return []
        if len(playable_with_costs) == 1 and limit >= 1:
            return playable_with_costs

        is_canon = is_hsim and sim._all_canonical_defs  # type: ignore[attr-defined]
        wanted_tags, ctx = _build_wanted_and_ctx(player, is_canon)
        meta_by_slug = _HYDROGEN_META_BY_SLUG
        plain_canon_hand = (
            is_canon and not sim._has_nonplain_hand  # type: ignore[attr-defined]
        )
        played_today = player.played_today
        last_card = played_today[-1] if played_today else None
        last_prof = (
            _get_card_meta(last_card).prof
            if last_card is not None
            else None
        )
        has_last_check = last_card is not None and last_prof is not None
        last_title = (
            _effective_card_title(last_card) if has_last_check else ""
        )
        last_tags = (
            (
                last_card.definition.tags
                if not last_card.markers
                else last_card.tags
            )
            if has_last_check
            else frozenset()
        )
        known_top = _get_sim_known_trunk_top(sim)

        players = sim.players
        awake_opps = (
            (0 if player_index == 0 or players[0].asleep else 1)
            + (0 if player_index == 1 or players[1].asleep else 1)
            + (0 if player_index == 2 or players[2].asleep else 1)
            + (0 if player_index == 3 or players[3].asleep else 1)
        )

        has_wildlife_first = (
            not played_today
            and any(
                _effective_card_title(tc) == "Wildlife Spotting"
                for tc in player.tomorrow_cards
            )
        )

        has_epic_duels_played = False
        has_early_food_played = False
        has_pb_played = False
        for pc in played_today:
            pm = pc.markers
            pc_title = (
                pc.definition.title
                if (not pm or "_copied_definition" not in pm)
                else _effective_card_title(pc)
            )
            if pc_title == "Epic Duels":
                has_epic_duels_played = True
            elif pc_title == "Paddleboat":
                has_pb_played = True
            if (
                not has_early_food_played
                and "Food" in (pc.definition.tags if not pm else pc.tags)
                and pc_title in _EARLY_COPYABLE_FOOD_TITLES
            ):
                has_early_food_played = True

        hand = player.hand
        has_trunk_consumer = False
        has_trunk_setter = False
        first_cutlery_card: CardInstance | None = None
        first_bcm_card: CardInstance | None = None
        for c in hand:
            ct = c.definition.title
            if ct in _TRUNK_CONSUMER_TITLES:
                has_trunk_consumer = True
                if ct == "Assorted Cutlery" and first_cutlery_card is None:
                    first_cutlery_card = c
                elif ct == "Bougie Coffee Machine" and first_bcm_card is None:
                    first_bcm_card = c
            if ct in _TRUNK_REORDER_TITLES:
                has_trunk_setter = True

        keyed: list[tuple[int, float, int, int]] = []
        for hand_idx, cost in playable_with_costs:
            card = hand[hand_idx]
            if plain_canon_hand:
                meta = meta_by_slug[card.definition.slug]
                c_tags = meta.tags
                eff_base_fun = meta.base_fun
            else:
                meta = _get_card_meta(card)
                c_tags = card.definition.tags if not card.markers else card.tags
                eff_base_fun = (
                    card.definition.base_fun
                    if not card.markers
                    else card.effective_base_fun
                )
            prof = meta.prof
            desirability = _hydrogen_quick_card_desirability(
                sim,
                player_index,
                card,
                wanted_tags,
                cost=cost,
                prof=prof,
                _ctx=ctx,
                _in_hand=True,
                _meta=meta,
            )
            bucket = prof.play_order_bucket

            if has_wildlife_first and "Outdoors" in c_tags:
                desirability += 2.0
                if bucket > 2:
                    bucket = 2

            if (
                prof.conditional_energy_gain > 0
                and prof.immediate_energy_gain == 0
            ):
                if _condition_met_in_played(player, card, prof):
                    bucket = 1
                    desirability += prof.conditional_energy_gain * 1.35
                else:
                    bucket = 4

            oq = meta.order_qcode
            if oq:
                if oq == _OQ_COLOURING:
                    if player.character is not None and player.character.zero_fun_tag:
                        bucket = 6
                        desirability -= 8.0
                elif oq == _OQ_NOS_BOOK:
                    if not played_today:
                        bucket = 6
                        desirability -= 6.0
                elif oq == _OQ_EPIC_DUELS:
                    if (
                        awake_opps > 0
                        and not has_epic_duels_played
                        and has_trunk_setter
                        and has_trunk_consumer
                    ):
                        bucket = 1
                        desirability += 3.8
                elif oq == _OQ_GROUP_DINNER:
                    bucket = 1
                elif oq == _OQ_SING_SONG:
                    if awake_opps > 0 and cost >= 2:
                        bucket = 5
                elif oq == _OQ_THE_CREW:
                    if awake_opps > 0:
                        bg_opps = 0
                        for idx in range(4):
                            if idx != player_index:
                                for oc in players[idx].played_today:
                                    if "Board Game" in oc.tags:
                                        bg_opps += 1
                                        break
                        if bg_opps < 2:
                            bucket = 5
                elif oq == _OQ_WEDDING_ANNIVERSARY:
                    rem_after_play = player.energy - cost
                    eligible_opp = [
                        oc
                        for idx, opp in enumerate(players)
                        if idx != player_index
                        for oc in opp.played_today
                        if oc.definition.cost <= rem_after_play
                        and oc.effective_behavior.can_play(sim, player, card)
                    ]
                    if eligible_opp:
                        best_cv = max(
                            _contextual_copy_value(
                                sim, player_index, oc, card, pay_cost=True
                            )
                            for oc in eligible_opp
                        )
                        if best_cv >= 3.5:
                            bucket = 0 if not played_today else 2
                            desirability += best_cv
                        elif awake_opps > 0:
                            bucket = 5
                    else:
                        bucket = 6
                        desirability -= 4.0
                elif oq == _OQ_MEDICAL_ADVICE:
                    if sim.day < DAY_COUNT and len(played_today) <= 1:
                        bucket = 2
                        desirability += 3.5
                    else:
                        bucket = 6
                        desirability -= 3.0
                elif oq == _OQ_TELL_A_STORY:
                    safe = has_epic_duels_played or awake_opps == 0
                    if safe and first_cutlery_card is not None:
                        c_cost = sim.energy_cost(player_index, first_cutlery_card)
                        if cost + c_cost <= player.energy and any(
                            other is not card
                            and other is not first_cutlery_card
                            and _hydrogen_free_play_value(
                                sim, player_index, other, wanted_tags
                            )
                            >= 4.5
                            for other in hand
                        ):
                            bucket = 2
                            desirability += 5.0
                    elif safe and first_bcm_card is not None:
                        b_cost = sim.energy_cost(player_index, first_bcm_card)
                        if cost + b_cost <= player.energy and any(
                            other is not card
                            and other is not first_bcm_card
                            and "Food" in other.tags
                            and _hydrogen_free_play_value(
                                sim, player_index, other, wanted_tags
                            )
                            >= 4.0
                            for other in hand
                        ):
                            bucket = 2
                            desirability += 4.5
                    else:
                        bucket = 5
                elif oq == _OQ_TRUNK_REORDER_4:
                    if has_trunk_consumer:
                        bucket = 2
                        desirability += 2.5
                elif oq == _OQ_ASSORTED_CUTLERY:
                    if known_top:
                        top_val = _hydrogen_free_play_value(
                            sim, player_index, known_top[0], wanted_tags
                        )
                        if top_val >= 3.5:
                            has_item_buff = any(
                                other.definition.title
                                in {"Boat Ride", "Fancy Craft", "Canoe"}
                                and cost + sim.energy_cost(player_index, other)
                                <= player.energy
                                for other in hand
                            )
                            bucket = 3 if has_item_buff else 2
                            desirability += top_val
                        elif top_val < 2.0 and cost >= 2:
                            bucket = 6
                            desirability -= 4.0
                elif oq == _OQ_QUACKS:
                    if known_top:
                        if "Outdoors" in known_top[0].tags:
                            bucket = 6
                            desirability -= 5.0
                        else:
                            safe_cnt = 0
                            for kc in known_top:
                                if "Outdoors" in kc.tags:
                                    break
                                safe_cnt += 1
                            bucket = 2
                            desirability += safe_cnt * 1.6
                elif oq == _OQ_POKER:
                    if known_top and known_top[0].tags:
                        bucket = 2
                        desirability += 3.5
                elif oq == _OQ_SHIRT_SCRABBLE:
                    has_pb_hand = any(
                        c.definition.title == "Paddleboat"
                        and cost + sim.energy_cost(player_index, c) <= player.energy
                        for c in hand
                    )
                    if len(hand) >= 2 and sim.suitcase:
                        bucket = 3 if has_pb_hand else 2
                        if has_pb_played:
                            desirability += 3.5
                    elif meta.title == "Last Year's Shirt":
                        bucket = 6
                        desirability -= 4.0
                elif oq == _OQ_SOLO:
                    opp_played_bg = any(
                        "Board Game" in oc.tags
                        for idx, opp in enumerate(players)
                        if idx != player_index
                        for oc in opp.played_today
                    )
                    if not opp_played_bg and awake_opps > 0:
                        has_bg_setup = any(
                            other.definition.title
                            in {
                                "Bend the Rules",
                                "Forced Family Fun",
                                "Trekking Through History",
                            }
                            and sim.energy_cost(player_index, other)
                            + max(0, cost - 1)
                            <= player.energy
                            for other in hand
                        )
                        bucket = 2 if has_bg_setup else 1
                        desirability += 2.2
                elif oq == _OQ_SAN_JUAN:
                    urgent_target = any(
                        not opp.asleep
                        and len(opp.played_today) == 2
                        and not any(
                            "Board Game" in oc.tags for oc in opp.played_today
                        )
                        for idx, opp in enumerate(players)
                        if idx != player_index
                    )
                    if urgent_target:
                        bucket = 1
                        desirability += 2.2
                elif oq == _OQ_CITIES_AND_KNIGHTS:
                    if len(hand) >= 2:
                        bucket = 5
                elif oq == _OQ_ADDRESS_THE_FOOD:
                    if any(
                        "Food" in sc.tags
                        and sc.effective_behavior.can_play(sim, player, sc)
                        for sc in sim.suitcase
                    ):
                        bucket = 2
                        desirability += 3.5
                    else:
                        bucket = 6
                        desirability -= 4.0
                elif oq == _OQ_SUITCASE_SWEEPER_2:
                    tag_needed = (
                        "Food" if meta.title == "Treat Store Run" else "Relax"
                    )
                    matches_in_sc = sum(
                        1 for sc in sim.suitcase if tag_needed in sc.tags
                    )
                    if matches_in_sc >= 2:
                        bucket = 2
                        desirability += matches_in_sc * 1.5
                    elif matches_in_sc == 0:
                        desirability -= 2.0
                elif oq == _OQ_PADDLEBOARD:
                    if len(hand) >= 3:
                        bucket = 5
                        desirability -= 2.0
                elif oq == _OQ_ISTANBUL:
                    if not _allows_energy_gain_today(sim, player, card):
                        bucket = 5
                        desirability -= 3.0
                    elif len(hand) >= 2:
                        other_cost = sum(
                            sim.energy_cost(player_index, other)
                            for other in hand
                            if other is not card
                            and other.effective_behavior.can_play(sim, player, other)
                        )
                        if (
                            other_cost > player.energy - cost
                            or any(
                                _effective_card_title(other)
                                in {"Kayak", "Decaf", "Wakesurf", "Drop a Ski"}
                                for other in hand
                                if other is not card
                            )
                            or any(
                                _effective_card_title(pc)
                                in {"Decaf", "Wakesurf", "Drop a Ski"}
                                for pc in played_today
                            )
                        ):
                            bucket = 2
                            desirability += 2.4
                elif oq == _OQ_SHADY_SPOT:
                    has_sunscreen_hand = any(
                        other.definition.title == "Sunscreen"
                        and sim.energy_cost(player_index, other) == 1
                        for other in hand
                    )
                    bucket = 3 if has_sunscreen_hand else 2

            if has_last_check:
                title = meta.title
                if last_title == "Work Call":
                    if (
                        eff_base_fun >= 3
                        or prof.conditional_fun_bonus >= 3
                    ):
                        bucket = 0
                        desirability += (
                            eff_base_fun + prof.conditional_fun_bonus
                        ) * 2.0
                elif (
                    last_title == "Beach Volleyball"
                    and "Indoors" in c_tags
                ):
                    bucket = 1
                    desirability += 2.6
                elif (
                    title == "Beach Conversation"
                    and "Outdoors" in last_tags
                ):
                    bucket = 1
                    desirability += 2.6
                elif title == "Swimming" and "Exercise" in last_tags:
                    bucket = 2
                    desirability += 3.0
                elif (
                    title == "Carcassonne"
                    and "Board Game" in last_tags
                    and any(
                        other is not card and "Board Game" in other.tags
                        for other in hand
                    )
                ):
                    bucket = 2
                    desirability += 4.0
                elif (
                    last_prof.is_single_buff  # type: ignore[union-attr]
                    and last_prof.target_tags  # type: ignore[union-attr]
                    and (c_tags & last_prof.target_tags)  # type: ignore[union-attr]
                    and not prof.is_multi_scaler
                    and not prof.is_single_buff
                    and (
                        title != "Nos Book"
                        or len(played_today) >= 1
                    )
                ):
                    bucket = 2
                    desirability += (
                        card.definition.cost * 0.85
                        + (
                            eff_base_fun + prof.conditional_fun_bonus
                        )
                        * 0.65
                    )

            keyed.append((bucket, -desirability, cost, hand_idx))

        keyed.sort()
        if has_early_food_played and len(keyed) > 1:
            for pos, item in enumerate(keyed):
                if hand[item[3]].definition.title == "Leftovers":
                    if pos > 0:
                        keyed.insert(0, keyed.pop(pos))
                    break
        if len(keyed) <= limit:
            return [(item[3], item[2]) for item in keyed]
        selected: list[tuple[int, int]] = []
        selected_indices: set[int] = set()
        seen_buckets: set[int] = set()
        for item in keyed:
            b = item[0]
            if b not in seen_buckets:
                seen_buckets.add(b)
                idx = item[3]
                selected.append((idx, item[2]))
                selected_indices.add(idx)
        for item in keyed:
            if len(selected) >= limit:
                break
            idx = item[3]
            if idx not in selected_indices:
                selected.append((idx, item[2]))
                selected_indices.add(idx)
        return selected

    def _search_best_day_plan(
        self,
        base_sim: Game,
        player_index: int,
        *,
        beam_width: int = 12,
        max_plays: int | None = None,
        allow_bed_now: bool = True,
        allowed_first_indices: Sequence[int] | None = None,
        base_turns_offset: int = 0,
        root_skipped_override: int | None = None,
        start_extra_active: bool = False,
    ) -> tuple[int | None, float]:
        best_action, best_val, _ = self._search_best_day_plan_with_table(
            base_sim,
            player_index,
            beam_width=beam_width,
            max_plays=max_plays,
            allow_bed_now=allow_bed_now,
            allowed_first_indices=allowed_first_indices,
            base_turns_offset=base_turns_offset,
            root_skipped_override=root_skipped_override,
            start_extra_active=start_extra_active,
        )
        return best_action, best_val

    def _search_best_day_plan_with_table(
        self,
        base_sim: Game,
        player_index: int,
        *,
        beam_width: int = 12,
        max_plays: int | None = None,
        allow_bed_now: bool = True,
        allowed_first_indices: Sequence[int] | None = None,
        in_extra_play: bool = False,
        base_turns_offset: int = 0,
        root_skipped_override: int | None = None,
        start_extra_active: bool = False,
    ) -> tuple[int | None, float, dict[int | None, tuple[float, Game]]]:
        """Multi-card beam search tracking turn counts accurately."""
        played_count = len(base_sim.players[player_index].played_today)
        is_live_play = max_plays is None and self._evaluation_depth == 0
        if max_plays is None:
            if played_count >= 12:
                max_plays = 6
                beam_width = 10
            elif played_count >= 7:
                max_plays = 10
                beam_width = 18
            else:
                max_plays = self.MAX_TURN_PLAYS
                beam_width = self.BEAM_WIDTH

        root_skipped = (
            base_sim.players[player_index].skipped_turns
            if root_skipped_override is None
            else root_skipped_override
        )
        b_ps = base_sim.players
        base_awake_opps = (
            (0 if player_index == 0 or b_ps[0].asleep else 1)
            + (0 if player_index == 1 or b_ps[1].asleep else 1)
            + (0 if player_index == 2 or b_ps[2].asleep else 1)
            + (0 if player_index == 3 or b_ps[3].asleep else 1)
        )
        ftb_ctx: tuple[
            PlayerState,
            PlayerState,
            PlayerState,
            PlayerState,
            int,
            dict[tuple[int, bool, int], float],
        ] = (b_ps[0], b_ps[1], b_ps[2], b_ps[3], base_awake_opps, {})
        best_by_action: dict[int | None, tuple[float, Game]] = {}
        turns_by_action: dict[int | None, int] = {}

        if allow_bed_now:
            bed_now = _hydrogen_copy_game_for_play(base_sim, player_index)
            if (
                not in_extra_play
                and not start_extra_active
                and not bed_now.players[player_index].asleep
            ):
                bed_now.go_to_bed(player_index)
                bed_turns = base_turns_offset + 1
            else:
                bed_turns = 1 if in_extra_play else base_turns_offset
            bed_val = self._evaluate_terminal_day(
                bed_now,
                player_index,
                turns_from_root=bed_turns,
                root_skipped_turns=root_skipped,
                _ftb_ctx=ftb_ctx,
            )
            best_value = bed_val
            best_first_actions: list[int | None] = [None]
            best_by_action[None] = (bed_val, bed_now)
            turns_by_action[None] = bed_turns
        else:
            best_value = float("-inf")
            best_first_actions = []

        # Frontier tuple: (first_action, sim_state, continuation_val, turns_taken, extra_active)
        frontier: list[tuple[int, Game, float, int, bool]] = []
        seen_signatures: set[tuple] = set()
        need_depth_1 = max_plays > 1

        init_limit = 16 if is_live_play else max(10, self.BRANCH_LIMIT + 1)
        initial_with_costs = self._ordered_candidate_indices_with_costs(
            base_sim, player_index, init_limit
        )
        initial_pairs: Sequence[tuple[int, int | None]]
        if allowed_first_indices is not None:
            allowed_set = set(allowed_first_indices)
            filtered_pairs: list[tuple[int, int | None]] = [
                (idx, cost)
                for idx, cost in initial_with_costs
                if idx in allowed_set
            ]
            if not filtered_pairs and allowed_first_indices:
                filtered_pairs = [
                    (idx, None) for idx in allowed_first_indices[:init_limit]
                ]
            initial_pairs = filtered_pairs
        else:
            initial_pairs = initial_with_costs

        base_player = base_sim.players[player_index]
        first_step_turns = (
            1
            if in_extra_play
            else (
                base_turns_offset
                if start_extra_active
                else base_turns_offset + 1
            )
        )
        for hand_idx, known_cost in initial_pairs:
            try:
                cand_card = base_player.hand[hand_idx]
                child_extra = (
                    in_extra_play
                    or start_extra_active
                    or _card_allows_extra_plays(
                        base_sim, base_player, cand_card
                    )
                )
                child = self._simulate_play_step(
                    base_sim, player_index, hand_idx, known_cost
                )
                stop_val = self._evaluate_terminal_day(
                    child,
                    player_index,
                    turns_from_root=first_step_turns,
                    root_skipped_turns=root_skipped,
                    _ftb_ctx=ftb_ctx,
                )
            except Exception:
                continue
            prev = best_by_action.get(hand_idx)
            if prev is None or stop_val > prev[0]:
                best_by_action[hand_idx] = (stop_val, child)
                turns_by_action[hand_idx] = first_step_turns
            if stop_val > best_value + 1e-9:
                best_value = stop_val
                best_first_actions = [hand_idx]
            elif (
                abs(stop_val - best_value) <= 1e-9
                and hand_idx not in best_first_actions
            ):
                best_first_actions.append(hand_idx)

            if need_depth_1 and not child.players[player_index].asleep:
                sig = self._state_signature(child, player_index)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    cont_val = stop_val + self._continuation_potential(
                        child, player_index
                    )
                    frontier.append(
                        (
                            hand_idx,
                            child,
                            cont_val,
                            first_step_turns,
                            child_extra,
                        )
                    )

        if frontier:
            frontier.sort(key=lambda item: item[2], reverse=True)
            frontier = frontier[:beam_width]

        for depth in range(1, max_plays):
            if not frontier:
                break
            need_next = depth + 1 < max_plays
            if is_live_play:
                active_beam = (
                    beam_width if depth < 7 else max(10, beam_width - 6)
                )
                branch_limit = 12 if depth < 6 else 9
            else:
                active_beam = (
                    beam_width if depth < 5 else max(5, beam_width - 3)
                )
                branch_limit = (
                    self.BRANCH_LIMIT
                    if depth < 5
                    else max(5, self.BRANCH_LIMIT - 2)
                )
            next_frontier: list[tuple[int, Game, float, int, bool]] = []
            for (
                first_action,
                state,
                _,
                turns_so_far,
                extra_active,
            ) in frontier:
                candidates = self._ordered_candidate_indices_with_costs(
                    state, player_index, branch_limit
                )
                s_player = state.players[player_index]
                step_turns = turns_so_far if extra_active else turns_so_far + 1
                for hand_idx, known_cost in candidates:
                    try:
                        cand_card = s_player.hand[hand_idx]
                        child_extra = extra_active or _card_allows_extra_plays(
                            state, s_player, cand_card
                        )
                        child = self._simulate_play_step(
                            state, player_index, hand_idx, known_cost
                        )
                        stop_val = self._evaluate_terminal_day(
                            child,
                            player_index,
                            turns_from_root=step_turns,
                            root_skipped_turns=root_skipped,
                            _ftb_ctx=ftb_ctx,
                        )
                    except Exception:
                        continue
                    prev = best_by_action.get(first_action)
                    if prev is None or stop_val > prev[0]:
                        best_by_action[first_action] = (stop_val, child)
                        turns_by_action[first_action] = step_turns
                    if stop_val > best_value + 1e-9:
                        best_value = stop_val
                        best_first_actions = [first_action]
                    elif (
                        abs(stop_val - best_value) <= 1e-9
                        and first_action not in best_first_actions
                    ):
                        best_first_actions.append(first_action)

                    if need_next and not child.players[player_index].asleep:
                        sig = self._state_signature(child, player_index)
                        if sig not in seen_signatures:
                            seen_signatures.add(sig)
                            cont_val = stop_val + self._continuation_potential(
                                child, player_index
                            )
                            next_frontier.append(
                                (
                                    first_action,
                                    child,
                                    cont_val,
                                    step_turns,
                                    child_extra,
                                )
                            )

            if need_next:
                next_frontier.sort(key=lambda item: item[2], reverse=True)
                frontier = next_frontier[:active_beam]

        if (
            is_live_play
            and base_sim.day < DAY_COUNT
            and len(best_by_action) >= 2
            and base_sim._is_setup
        ):
            best_first_actions, best_value = self._refine_with_next_day_rollout(
                base_sim,
                player_index,
                best_by_action,
                best_value,
                turns_by_action=turns_by_action,
                root_skipped_turns=root_skipped,
            )

        if not best_first_actions:
            return (
                (initial_pairs[0][0] if initial_pairs else None),
                best_value,
                best_by_action,
            )
        if None in best_first_actions:
            return None, best_value, best_by_action
        return (
            self.rng.choice(best_first_actions),
            best_value,
            best_by_action,
        )

    def _first_to_bed_value(
        self,
        sim: Game,
        player_index: int,
        *,
        turns_from_root: int = 0,
        root_skipped_turns: int = 0,
    ) -> float:
        """Estimate expected first-to-bed value using only observable opponent stamina."""
        if sim.day >= DAY_COUNT:
            return 0.0
        players = sim.players
        for idx in range(4):
            if idx != player_index and players[idx].asleep:
                return 0.0
        player = players[player_index]
        delta_skipped = player.skipped_turns - root_skipped_turns
        if delta_skipped < 0:
            delta_skipped = 0
        if turns_from_root <= 0:
            opp_rounds = 0
        elif player.asleep:
            opp_rounds = (turns_from_root - 1) + delta_skipped
            if opp_rounds < 0:
                opp_rounds = 0
        else:
            opp_rounds = turns_from_root + delta_skipped

        base_bonus = (
            0.85 if sim.day == DAY_COUNT - 1 else self.FIRST_TO_BED_BONUS
        )
        if opp_rounds <= 0:
            return base_bonus

        p_win = 1.0
        for idx in range(4):
            if idx == player_index:
                continue
            opp = players[idx]
            eff_turns = opp_rounds - opp.skipped_turns
            if eff_turns <= 0:
                continue
            opp_h = len(opp.hand)
            if opp_h <= 0:
                return 0.0
            opp_e = opp.energy
            for t in range(eff_turns):
                est_h = opp_h - t
                if est_h <= 0:
                    return 0.0
                est_e = opp_e - 2 * t
                if est_e <= 0:
                    p_stay = _FTB_ZERO_E_STAY[est_h if est_h < 4 else 4]
                elif est_e == 1:
                    p_stay = 0.48
                elif est_e == 2:
                    p_stay = 0.68
                elif est_e <= 4:
                    p_stay = 0.80
                else:
                    p_stay = 0.88
                p_win *= p_stay
                if p_win < 0.03:
                    return 0.0
        return base_bonus * p_win

    def _refine_with_next_day_rollout(
        self,
        base_sim: Game,
        player_index: int,
        best_by_action: dict[int | None, tuple[float, Game]],
        max_fast_val: float,
        *,
        turns_by_action: dict[int | None, int] | None = None,
        root_skipped_turns: int | None = None,
    ) -> tuple[list[int | None], float]:
        """Run Day d+1 play search with accurate first-to-bed and start_day accounting."""
        contenders = [
            (act, f_val, term_sim)
            for act, (f_val, term_sim) in best_by_action.items()
            if f_val >= max_fast_val - 2.8
        ]
        if len(contenders) <= 1:
            return [act for act, _, _ in contenders], max_fast_val

        configs = {
            (
                t_sim.players[player_index].energy,
                tuple(
                    opp.fun - t_sim.players[player_index].fun
                    for idx, opp in enumerate(t_sim.players)
                    if idx != player_index
                ),
                tuple(
                    sorted(
                        (c.title, _effective_card_title(c))
                        for c in t_sim.players[player_index].hand
                    )
                ),
                tuple(
                    (
                        c.title,
                        _effective_card_title(c),
                        c.markers.get("chosen_tag"),
                    )
                    for c in t_sim.players[player_index].played_today
                    if c.effective_behavior.has_tomorrow_action
                    or _effective_card_title(c)
                    in {"Chuck a Frisbee", "Scout Other Cottages"}
                ),
                tuple(
                    (
                        c.title,
                        tuple(
                            sorted(
                                k
                                for k in c.markers
                                if k.startswith(("_puerto_rico", "_scout"))
                            )
                        ),
                    )
                    for c in t_sim.suitcase
                ),
                tuple(c.instance_id for c in _get_sim_known_trunk_top(t_sim)),
            )
            for _, _, t_sim in contenders
        }
        if len(configs) <= 1:
            best_acts = [
                act
                for act, f_val, _ in contenders
                if abs(f_val - max_fast_val) <= 1e-9
            ]
            return best_acts, max_fast_val

        r_skipped = (
            base_sim.players[player_index].skipped_turns
            if root_skipped_turns is None
            else root_skipped_turns
        )
        awake_opps = sum(
            1
            for idx, p in enumerate(base_sim.players)
            if idx != player_index and not p.asleep
        )

        rollout_cache: dict[tuple, float] = {}
        best_blended = float("-inf")
        best_acts: list[int | None] = []

        contenders.sort(key=lambda item: item[1], reverse=True)
        for act, f_val, term_sim in contenders[:6]:
            contender_turns = (
                turns_by_action.get(act, 0 if act is None else 1)
                if turns_by_action is not None
                else (0 if act is None else 1)
            )
            contender_first_bed = self._first_to_bed_value(
                term_sim,
                player_index,
                turns_from_root=contender_turns,
                root_skipped_turns=r_skipped,
            )

            next_sim = _hydrogen_copy_game(term_sim)
            gd_cnt = sum(
                1
                for c in next_sim.players[player_index].played_today
                if _effective_card_title(c) == "Group Dinner"
            )
            scout_cnt = sum(
                1
                for c in next_sim.players[player_index].played_today
                if _effective_card_title(c) == "Scout Other Cottages"
            )
            for p_i, p in enumerate(next_sim.players):
                if not p.asleep:
                    next_sim.go_to_bed(p_i)
            try:
                next_sim.end_day()
            except Exception:
                blended = f_val
                if blended > best_blended + 1e-9:
                    best_blended = blended
                    best_acts = [act]
                elif abs(blended - best_blended) <= 1e-9:
                    best_acts.append(act)
                continue

            p_next = next_sim.players[player_index]
            post_end_day_fun = float(p_next.fun)
            end_day_fun = (
                post_end_day_fun
                - 1.35 * gd_cnt
                - (0.45 * min(awake_opps, 3) if scout_cnt and awake_opps else 0.0)
            )
            cache_key = (
                p_next.energy,
                next_sim._condition_deck[-1].title
                if (next_sim.daily_conditions and next_sim._condition_deck)
                else None,
                tuple(
                    (len(opp.hand), opp.skipped_turns, opp.fun - p_next.fun)
                    for idx, opp in enumerate(next_sim.players)
                    if idx != player_index
                ),
                tuple(
                    sorted(
                        (c.title, _effective_card_title(c)) for c in p_next.hand
                    )
                ),
                tuple(
                    (
                        c.title,
                        _effective_card_title(c),
                        c.markers.get("chosen_tag"),
                    )
                    for c in p_next.tomorrow_cards
                ),
                tuple(c.title for c in next_sim.suitcase),
                tuple(
                    c.instance_id for c in _get_sim_known_trunk_top(next_sim)
                ),
            )
            delta_tomorrow = rollout_cache.get(cache_key)
            if delta_tomorrow is None:
                try:
                    next_sim.day += 1
                    next_sim.start_day()
                    wanted_next = _wanted_tags_for_player_fast(p_next)
                    top_tag = max(
                        wanted_next.items(),
                        key=lambda kv: (kv[1], kv[0]),
                        default=("", 0.0),
                    )
                    for n_i, n_defn in enumerate(_NEUTRAL_DRAFT_DEFNS):
                        if (
                            n_i >= 1
                            and top_tag[1] >= 1.4
                            and top_tag[0] in _TAGGED_NEUTRAL_3_BY_TAG
                        ):
                            p_next.hand.append(
                                CardInstance(
                                    -900 - n_i,
                                    _TAGGED_NEUTRAL_3_BY_TAG[top_tag[0]],
                                )
                            )
                        else:
                            p_next.hand.append(
                                CardInstance(-900 - n_i, n_defn)
                            )
                    for opp_i, opp_p in enumerate(next_sim.players):
                        if opp_i != player_index:
                            for n_i, n_defn in enumerate(_NEUTRAL_DRAFT_DEFNS):
                                opp_p.hand.append(
                                    CardInstance(-910 - opp_i * 3 - n_i, n_defn)
                                )

                    prev_depth = self._evaluation_depth
                    self._evaluation_depth = 1
                    try:
                        _, day2_total = self._search_best_day_plan(
                            next_sim,
                            player_index,
                            beam_width=9,
                            max_plays=9,
                            allow_bed_now=True,
                        )
                    finally:
                        self._evaluation_depth = prev_depth
                    # Compare against post_end_day_fun (BEFORE start_day) so character daily Fun isn't cancelled
                    delta_tomorrow = (day2_total - post_end_day_fun) - 7.0
                    rollout_cache[cache_key] = delta_tomorrow
                except Exception:
                    delta_tomorrow = (
                        f_val - end_day_fun - contender_first_bed
                    )

            rollout_total = end_day_fun + contender_first_bed + delta_tomorrow
            blended = 0.55 * f_val + 0.45 * rollout_total
            best_by_action[act] = (blended, term_sim)
            if blended > best_blended + 1e-9:
                best_blended = blended
                best_acts = [act]
            elif abs(blended - best_blended) <= 1e-9:
                best_acts.append(act)

        return best_acts, best_blended

    def _evaluate_terminal_day(
        self,
        terminal_sim: Game,
        player_index: int,
        *,
        turns_from_root: int = 1,
        root_skipped_turns: int = 0,
        _ftb_ctx: tuple | None = None,
    ) -> float:
        """Opponent-aware terminal-day evaluation with turn-decayed FTB and Hydrogen reserve."""
        players = terminal_sim.players
        player = players[player_index]

        if _ftb_ctx is not None and (
            (player_index == 0 or players[0] is _ftb_ctx[0])
            and (player_index == 1 or players[1] is _ftb_ctx[1])
            and (player_index == 2 or players[2] is _ftb_ctx[2])
            and (player_index == 3 or players[3] is _ftb_ctx[3])
        ):
            awake_opps = _ftb_ctx[4]
            opponents_awake = awake_opps > 0
            if terminal_sim.day < DAY_COUNT and awake_opps == 3:
                f_key = (turns_from_root, player.asleep, player.skipped_turns)
                f_dict = _ftb_ctx[5]
                cached_ftb = f_dict.get(f_key)
                if cached_ftb is None:
                    cached_ftb = self._first_to_bed_value(
                        terminal_sim,
                        player_index,
                        turns_from_root=turns_from_root,
                        root_skipped_turns=root_skipped_turns,
                    )
                    f_dict[f_key] = cached_ftb
                first_to_bed_value = cached_ftb
            else:
                first_to_bed_value = 0.0
        else:
            awake_opps = 0
            for idx in range(4):
                if idx != player_index and not players[idx].asleep:
                    awake_opps += 1
            opponents_awake = awake_opps > 0
            first_to_bed_value = (
                self._first_to_bed_value(
                    terminal_sim,
                    player_index,
                    turns_from_root=turns_from_root,
                    root_skipped_turns=root_skipped_turns,
                )
                if (terminal_sim.day < DAY_COUNT and awake_opps == 3)
                else 0.0
            )

        played_today = player.played_today
        if type(terminal_sim) is _HYDROGEN_SIM_GAME_CLS:
            pf = terminal_sim._p_flags[player_index]  # type: ignore[attr-defined]
            needs_full_end_day = bool(pf & _PF_HAS_END_DAY)
            has_loop2 = bool(pf & _PF_HAS_LOOP2)
            has_loop3 = opponents_awake and bool(pf & _PF_HAS_AWAKE_OPP_TERM)
        else:
            needs_full_end_day = False
            has_loop2 = False
            has_loop3 = False
            for c in played_today:
                m = c.markers
                uncopied = not m or "_copied_definition" not in m
                c_beh = (
                    c.definition.behavior if uncopied else c.effective_behavior
                )
                b_type = type(c_beh)
                if (
                    b_type.on_score is not CardBehavior.on_score
                    or b_type.on_end_day is not CardBehavior.on_end_day
                    or b_type.on_after_end_day
                    is not CardBehavior.on_after_end_day
                ):
                    needs_full_end_day = True
                    has_loop2 = True
                    has_loop3 = opponents_awake
                    break
                eff_title = (
                    c.definition.title if uncopied else _effective_card_title(c)
                )
                if eff_title == "Group Dinner" or (
                    m and ("_blind_poker" in m or "_blind_quacks" in m)
                ):
                    has_loop2 = True
                if opponents_awake and eff_title in _TERMINAL_AWAKE_OPP_TITLES:
                    has_loop3 = True

        eval_sim = terminal_sim
        if needs_full_end_day:
            scoring = _hydrogen_copy_game(terminal_sim)
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
                        for c in played_today
                    )
                )
                next_tomorrow_cards = [
                    c
                    for c in played_today
                    if c.effective_behavior.has_tomorrow_action
                ]
        else:
            prev_scores = terminal_sim._fun_at_start_of_scoring
            terminal_sim._fun_at_start_of_scoring = (
                players[0].fun,
                players[1].fun,
                players[2].fun,
                players[3].fun,
            )
            if type(terminal_sim) is _HYDROGEN_SIM_GAME_CLS:
                scored_today, next_tomorrow_cards = (
                    terminal_sim.score_played_today_fast(player_index)  # type: ignore[attr-defined]
                )
            else:
                scored_today = 0
                next_tomorrow_cards = []
                for c in played_today:
                    scored_today += terminal_sim.card_fun(player_index, c)
                    c_beh = (
                        c.definition.behavior
                        if not c.markers
                        else c.effective_behavior
                    )
                    if getattr(c_beh, "has_tomorrow_action", False):
                        next_tomorrow_cards.append(c)
            terminal_sim._fun_at_start_of_scoring = prev_scores
            today_total_fun = float(player.fun + scored_today)

        if has_loop2:
            for c in played_today:
                m = c.markers
                eff_title = (
                    c.definition.title
                    if (not m or "_copied_definition" not in m)
                    else _effective_card_title(c)
                )
                if eff_title == "Group Dinner":
                    today_total_fun -= 1.35
                if m:
                    t = c.definition.title
                    if t == "Poker" and m.get("_blind_poker"):
                        today_total_fun += 1.25
                    elif t == "Quacks" and m.get("_blind_quacks"):
                        today_total_fun += 0.15

        if has_loop3:
            opp_has_sweeper: bool | None = None
            played_len = len(played_today)
            hand_len = len(player.hand)
            for c in played_today:
                m = c.markers
                eff_title = (
                    c.definition.title
                    if (not m or "_copied_definition" not in m)
                    else _effective_card_title(c)
                )
                if eff_title == "Fit to Print":
                    if played_len <= 4 and all(
                        played_len >= len(players[idx].played_today)
                        for idx in range(4)
                        if idx != player_index
                    ):
                        today_total_fun -= (
                            3.6 if played_len <= 3 else 1.9
                        )
                elif eff_title == "Play With the Kids":
                    if hand_len <= 2 and all(
                        hand_len > len(players[idx].hand)
                        for idx in range(4)
                        if idx != player_index
                    ):
                        today_total_fun -= 2.2
                elif eff_title == "Puerto Rico":
                    if opp_has_sweeper is None:
                        opp_has_sweeper = any(
                            any(
                                _effective_card_title(tc) == "Photo Shoot"
                                for tc in players[idx].tomorrow_cards
                            )
                            or any(
                                _effective_card_title(oc)
                                == "Euchre Awards Ceremony"
                                for oc in players[idx].played_today
                            )
                            for idx in range(4)
                            if idx != player_index and not players[idx].asleep
                        )
                    today_total_fun -= 1.25 + (0.6 if opp_has_sweeper else 0.0)
                elif eff_title == "Scout Other Cottages":
                    if opp_has_sweeper is None:
                        opp_has_sweeper = any(
                            any(
                                _effective_card_title(tc) == "Photo Shoot"
                                for tc in players[idx].tomorrow_cards
                            )
                            or any(
                                _effective_card_title(oc)
                                == "Euchre Awards Ceremony"
                                for oc in players[idx].played_today
                            )
                            for idx in range(4)
                            if idx != player_index and not players[idx].asleep
                        )
                    today_total_fun -= 0.40 * awake_opps + (
                        0.75 if opp_has_sweeper else 0.0
                    )
                elif eff_title == "San Juan":
                    t_idx = c.markers.get("target_player_index")
                    if (
                        isinstance(t_idx, int)
                        and t_idx != player_index
                        and not players[t_idx].asleep
                    ):
                        today_total_fun -= 1.4
                elif eff_title == "Cheesy Phone Game":
                    relax_opps = 0
                    for idx in range(4):
                        if idx != player_index:
                            for oc in players[idx].played_today:
                                if "Relax" in oc.tags:
                                    relax_opps += 1
                                    break
                    if relax_opps < 2:
                        today_total_fun += 0.35 * awake_opps

        if terminal_sim.day >= DAY_COUNT:
            return today_total_fun

        if eval_sim is terminal_sim:
            orig_fun = player.fun
            player.fun = int(round(today_total_fun))
            try:
                reserve_value = self._evaluate_tomorrow_and_reserve_fast(
                    eval_sim,
                    player_index,
                    next_tomorrow_cards,
                    days_left=DAY_COUNT - terminal_sim.day,
                )
            finally:
                player.fun = orig_fun
        else:
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
        """Reserve & tomorrow valuation preserving core combos while discouraging multi-day hoarding."""
        player = sim.players[player_index]
        hand = player.hand
        if not hand and not next_tomorrow_cards:
            return 0.0

        meta_by_slug = _HYDROGEN_META_BY_SLUG
        if not meta_by_slug:
            _ensure_hydrogen_caches()
        is_hsim = type(sim) is _HYDROGEN_SIM_GAME_CLS
        is_canonical = (
            sim._all_canonical_defs  # type: ignore[attr-defined]
            if is_hsim
            else getattr(sim, "_all_canonical_defs", False)
        )
        plain_canon_hand = (
            is_hsim
            and is_canonical
            and not sim._has_nonplain_hand  # type: ignore[attr-defined]
        )
        h_metas: list[_HydrogenDefnMeta] | None = None
        all_tags_canonical = True
        if plain_canon_hand:
            can_cache = True
            all_tags_canonical = _ALL_CANONICAL_META_TAGS_OK
            if hand:
                hand_slugs = tuple([hc.definition.slug for hc in hand])
                has_ski_shoulders = "ski-on-shoulders" in hand_slugs
            else:
                hand_slugs = ()
                has_ski_shoulders = False
        elif hand:
            meta_by_id = _HYDROGEN_META_BY_DEFN_ID
            h_metas_tmp: list[_HydrogenDefnMeta] = []
            slug_list: list[str] = []
            has_ski_shoulders = False
            can_cache = True
            for hc in hand:
                if not _is_plain_hand_markers(hc.markers):
                    can_cache = False
                    break
                meta = meta_by_id.get(id(hc.definition))
                if meta is None:
                    can_cache = False
                    break
                if not meta.tags_canonical or not meta.target_canonical:
                    all_tags_canonical = False
                h_metas_tmp.append(meta)
                s = meta.slug
                slug_list.append(s)
                if s == "ski-on-shoulders":
                    has_ski_shoulders = True
            if can_cache:
                hand_slugs = tuple(slug_list)
                h_metas = h_metas_tmp
            else:
                hand_slugs = ()
        else:
            hand_slugs = ()
            can_cache = True
            has_ski_shoulders = False

        tom_key: tuple = ()
        if can_cache and next_tomorrow_cards:
            if is_canonical and len(next_tomorrow_cards) == 1:
                tc0 = next_tomorrow_cards[0]
                tm0 = tc0.markers
                if not tm0:
                    tom_key = ((tc0.definition.slug, tc0.definition.title, None),)
                elif (
                    "_copied_definition" in tm0
                    and id(_effective_source_definition(tc0))
                    not in _HYDROGEN_META_BY_DEFN_ID
                ):
                    can_cache = False
                else:
                    tt0 = (
                        tc0.definition.title
                        if "_copied_definition" not in tm0
                        else _effective_card_title(tc0)
                    )
                    tom_key = (
                        (tc0.definition.slug, tt0, tm0.get("chosen_tag")),
                    )
            else:
                meta_by_id = _HYDROGEN_META_BY_DEFN_ID
                tom_key_list = []
                for tc in next_tomorrow_cards:
                    if not is_canonical and id(tc.definition) not in meta_by_id:
                        can_cache = False
                        break
                    tm = tc.markers
                    if (
                        tm
                        and "_copied_definition" in tm
                        and id(_effective_source_definition(tc))
                        not in meta_by_id
                    ):
                        can_cache = False
                        break
                    t_title = (
                        tc.definition.title
                        if (not tm or "_copied_definition" not in tm)
                        else _effective_card_title(tc)
                    )
                    tom_key_list.append(
                        (
                            tc.definition.slug,
                            t_title,
                            tm.get("chosen_tag") if tm else None,
                        )
                    )
                if can_cache:
                    tom_key = tuple(tom_key_list)

        cache = self._reserve_eval_cache
        cache_key: tuple | None = None
        ski_ahead_ge2: bool | None = None
        if can_cache:
            is_default_cfg = (
                self.RESERVE_TOMORROW_WEIGHT == 0.74
                and self.RESERVE_SLOT_WEIGHTS_COMBO
                is HydrogenAI.RESERVE_SLOT_WEIGHTS_COMBO
                and self.RESERVE_SLOT_WEIGHTS_PLAIN
                is HydrogenAI.RESERVE_SLOT_WEIGHTS_PLAIN
                and self.RESERVE_SLOT_TAIL_WEIGHT == 0.06
            )
            if is_default_cfg and not next_tomorrow_cards and not has_ski_shoulders:
                cache_key = (days_left, player.character, hand_slugs)
            else:
                if has_ski_shoulders:
                    p_fun = player.fun
                    opps_ahead_cnt = 0
                    for idx, opp in enumerate(sim.players):
                        if idx != player_index and opp.fun > p_fun:
                            opps_ahead_cnt += 1
                    ski_ahead_ge2 = opps_ahead_cnt >= 2
                cfg_key = (
                    0
                    if is_default_cfg
                    else (
                        self.RESERVE_TOMORROW_WEIGHT,
                        self.RESERVE_SLOT_WEIGHTS_COMBO,
                        self.RESERVE_SLOT_WEIGHTS_PLAIN,
                        self.RESERVE_SLOT_TAIL_WEIGHT,
                    )
                )
                cache_key = (
                    days_left,
                    cfg_key,
                    player.character,
                    ski_ahead_ge2,
                    tom_key,
                    hand_slugs,
                )
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val

        if hand and can_cache and h_metas is None:
            h_metas = [meta_by_slug[s] for s in hand_slugs]

        char = player.character
        rtw = self.RESERVE_TOMORROW_WEIGHT
        zero_tag = char.zero_fun_tag if char is not None else ""

        tom_fun = 0.0
        tom_energy_delta = 0
        tom_tag_bonuses: dict[str, float] | None = None
        tom_all_tags = False
        has_colouring = False
        has_photo_shoot = False
        has_dates_first = False
        tom_pos_energy_sum = 0

        if next_tomorrow_cards:
            for tc in next_tomorrow_cards:
                tm = tc.markers
                tc_uncopied = not tm or "_copied_definition" not in tm
                if tc_uncopied:
                    td = tc.definition
                    t_meta = (
                        meta_by_slug[td.slug]
                        if is_canonical
                        else meta_by_slug.get(td.slug)
                    )
                    if not is_canonical and (
                        t_meta is None or t_meta.defn is not td
                    ):
                        t_meta = None
                else:
                    t_meta = None
                if t_meta is not None:
                    t_title = t_meta.title
                    tom_fun += t_meta.tomorrow_start_fun
                    tse = t_meta.tomorrow_start_energy
                    tom_energy_delta += tse
                    if tse > 0:
                        tom_pos_energy_sum += tse
                    if t_title == "Photo Shoot":
                        tom_fun += 3.5
                    elif t_title == "Date's First Nos":
                        tom_fun += 2.2
                    elif t_title == "Wildlife Spotting" and zero_tag != "Outdoors":
                        has_out = any(
                            "Outdoors"
                            in (hc.definition.tags if not hc.markers else hc.tags)
                            for hc in hand
                        )
                        tom_fun += 1.10 if has_out else 0.70

                    smb = t_meta.signed_match_benefit
                    if smb != 0.0:
                        if tom_tag_bonuses is None:
                            tom_tag_bonuses = {}
                        for t in t_meta.sorted_target_tags:
                            tom_tag_bonuses[t] = (
                                tom_tag_bonuses.get(t, 0.0) + smb
                            )
                        if smb < 0.0:
                            tom_fun += smb * 1.15
                        else:
                            tom_fun += smb * 0.50
                    if tm and "chosen_tag" in tm:
                        chosen_tag = tm["chosen_tag"]
                        if isinstance(chosen_tag, str):
                            if tom_tag_bonuses is None:
                                tom_tag_bonuses = {}
                            tom_tag_bonuses[chosen_tag] = (
                                tom_tag_bonuses.get(chosen_tag, 0.0) + 1.2
                            )
                    if t_meta.tomorrow_all_tags:
                        tom_all_tags = True
                    if t_title == "Colouring":
                        has_colouring = True
                    elif t_title == "Photo Shoot":
                        has_photo_shoot = True
                    elif t_title == "Date's First Nos":
                        has_dates_first = True
                else:
                    t_prof = _beryllium_build_profile(tc)
                    t_title = (
                        tc.definition.title
                        if tc_uncopied
                        else _effective_card_title(tc)
                    )
                    tom_fun += t_prof.tomorrow_start_fun
                    tse = t_prof.tomorrow_start_energy
                    tom_energy_delta += tse
                    if tse > 0:
                        tom_pos_energy_sum += tse
                    if t_title == "Photo Shoot":
                        tom_fun += 3.5
                    elif t_title == "Date's First Nos":
                        tom_fun += 2.2
                    elif t_title == "Wildlife Spotting" and zero_tag != "Outdoors":
                        has_out = any(
                            "Outdoors"
                            in (hc.definition.tags if not hc.markers else hc.tags)
                            for hc in hand
                        )
                        tom_fun += 1.10 if has_out else 0.70

                    if t_prof.signed_match_benefit != 0.0:
                        if tom_tag_bonuses is None:
                            tom_tag_bonuses = {}
                        t_targets = (
                            t_prof.target_tags
                            if len(t_prof.target_tags) <= 1
                            else sorted(t_prof.target_tags)
                        )
                        for t in t_targets:
                            tom_tag_bonuses[t] = (
                                tom_tag_bonuses.get(t, 0.0)
                                + t_prof.signed_match_benefit
                            )
                        if t_prof.signed_match_benefit < 0.0:
                            tom_fun += t_prof.signed_match_benefit * 1.15
                        else:
                            tom_fun += t_prof.signed_match_benefit * 0.50
                    if tm and "chosen_tag" in tm:
                        chosen_tag = tm["chosen_tag"]
                        if isinstance(chosen_tag, str):
                            if tom_tag_bonuses is None:
                                tom_tag_bonuses = {}
                            tom_tag_bonuses[chosen_tag] = (
                                tom_tag_bonuses.get(chosen_tag, 0.0) + 1.2
                            )
                    tc_beh = (
                        tc.definition.behavior
                        if tc_uncopied
                        else tc.effective_behavior
                    )
                    if tc_beh.tomorrow_all_tags:
                        tom_all_tags = True
                    if t_title == "Colouring":
                        has_colouring = True
                    elif t_title == "Photo Shoot":
                        has_photo_shoot = True
                    elif t_title == "Date's First Nos":
                        has_dates_first = True

        tom_energy_value = (
            tom_energy_delta * 1.22
            if tom_energy_delta >= 0
            else tom_energy_delta * 1.15
        )

        if not hand:
            res_val = tom_fun + tom_energy_value
            if has_colouring:
                res_val += -18.0 if zero_tag else 2.2
            if has_photo_shoot:
                res_val += 1.8
            if has_dates_first and not zero_tag:
                res_val += -1.15
            if cache_key is not None:
                if len(cache) > 1048576:
                    cache.clear()
                cache[cache_key] = res_val
            return res_val

        if can_cache and all_tags_canonical:
            assert h_metas is not None
            hand_len = len(h_metas)
            h_m1 = 0
            h_m2 = 0
            relax_cnt_total = 0
            out_cnt_total = 0
            ex_cnt_total = 0
            cost5_cnt = 0
            cutlery_combo_cnt = 0
            cutlery_real_combo_cnt = 0
            cheap2_cnt = 0
            copy_food_cnt = 0
            acquirer_cnt = 0
            cutlery_cnt = 0
            story_cnt = 0
            pb_cnt = 0
            total_adv_sup = 0
            has_any_multi_or_global = False
            held_scaler_benefit_by_tag: dict[str, float] | None = None

            for hm in h_metas:
                cmask = hm.tag_mask
                h_m2 |= h_m1 & cmask
                h_m1 |= cmask
                if cmask & _MASK_RELAX:
                    relax_cnt_total += 1
                if cmask & _MASK_OUTDOORS:
                    out_cnt_total += 1
                if cmask & _MASK_EXERCISE:
                    ex_cnt_total += 1
                if hm.cost <= 2:
                    cheap2_cnt += 1
                if hm.cost >= 5:
                    cost5_cnt += 1
                if hm.is_cutlery_combo:
                    cutlery_combo_cnt += 1
                if hm.is_cutlery_real_combo:
                    cutlery_real_combo_cnt += 1
                if hm.is_copy_food:
                    copy_food_cnt += 1
                if hm.is_acquirer:
                    acquirer_cnt += 1
                mt = hm.title
                if mt == "Assorted Cutlery":
                    cutlery_cnt += 1
                elif mt == "Tell a Story":
                    story_cnt += 1
                elif mt == "Paddleboat":
                    pb_cnt += 1
                total_adv_sup += hm.reserve_adv_race_sup
                if hm.is_multi_scaler or hm.is_global_after_scaler:
                    has_any_multi_or_global = True
                    if hm.is_multi_scaler and hm.signed_match_benefit > 0.0:
                        if held_scaler_benefit_by_tag is None:
                            held_scaler_benefit_by_tag = {}
                        smb = hm.signed_match_benefit
                        for t in hm.sorted_target_tags:
                            held_scaler_benefit_by_tag[t] = (
                                held_scaler_benefit_by_tag.get(t, 0.0) + smb
                            )

            paired_partner_counts: dict[str, int] | None = None
            seen_must_play_first = False
            surpluses: list[float] = []
            if hand_len == 1:
                ordered_metas: Sequence[_HydrogenDefnMeta] = h_metas
            elif hand_len == 2:
                m0, m1 = h_metas[0], h_metas[1]
                ordered_metas = (
                    (m0, m1)
                    if m0.reserve_rank <= m1.reserve_rank
                    else (m1, m0)
                )
            elif hand_len == 3:
                m0, m1, m2 = h_metas[0], h_metas[1], h_metas[2]
                if m0.reserve_rank > m1.reserve_rank:
                    m0, m1 = m1, m0
                if m1.reserve_rank > m2.reserve_rank:
                    m1, m2 = m2, m1
                    if m0.reserve_rank > m1.reserve_rank:
                        m0, m1 = m1, m0
                ordered_metas = (m0, m1, m2)
            else:
                ordered_metas = sorted(h_metas, key=_RESERVE_RANK_GETTER)
            pos_tom_all = (
                sum(
                    tom_tag_bonuses[k]
                    for k in sorted(tom_tag_bonuses)
                    if tom_tag_bonuses[k] > 0.0
                )
                if (tom_all_tags and tom_tag_bonuses is not None)
                else 0.0
            )
            if char is not None:
                (
                    _,
                    char_f_mask,
                    char_f_delta,
                    zero_mask,
                    zero_bit,
                    char_e_mask,
                    char_e_delta,
                    char_custom,
                ) = _get_char_fast(char)
            else:
                char_f_mask = 0
                char_f_delta = 0
                zero_mask = 0
                zero_bit = -2
                char_e_mask = 0
                char_e_delta = 0
                char_custom = False

            for c_meta in ordered_metas:
                if c_meta.must_play_first:
                    if seen_must_play_first:
                        continue
                    seen_must_play_first = True
                    surpluses.append(0.45)
                    continue

                c_title = c_meta.title
                cmask = c_meta.tag_mask
                ordered_c_tags = c_meta.sorted_tags
                cost = c_meta.cost if c_meta.cost > 0 else 0
                if char is None:
                    is_zeroed = False
                    base_f = c_meta.reserve_base_f
                elif not char_custom:
                    is_zeroed = bool(cmask & zero_mask)
                    if char_e_mask and (cmask & char_e_mask):
                        cost += char_e_delta
                        if cost < 0:
                            cost = 0
                    if is_zeroed:
                        base_f = 0.0
                    elif char_f_mask and (cmask & char_f_mask):
                        base_f = float(c_meta.reserve_base_f_int + char_f_delta)
                    else:
                        base_f = c_meta.reserve_base_f
                else:
                    c_tags = c_meta.tags
                    is_zeroed = bool(zero_tag and zero_tag in c_tags)
                    cost = char.modify_energy_cost(c_tags, cost)
                    if cost < 0:
                        cost = 0
                    base_f = float(
                        char.finalize_fun(
                            c_tags,
                            char.modify_fun(c_tags, c_meta.reserve_base_f_int),
                        )
                    )

                if cost >= 8:
                    target_hc = next(
                        hc for hc in hand if hc.definition is c_meta.defn
                    )
                    support = (
                        tom_energy_delta if tom_energy_delta > 0 else 0
                    ) + _carbon_extra_energy_support(
                        sim,
                        player_index,
                        target_hc,
                        extra_fodder=3,
                        ignore_today_block=True,
                    )
                    if days_left == 1 and cost > 7 + support:
                        if not (cutlery_cnt > 0 and story_cnt > 0):
                            continue
                    over_budget = cost - (6 + support)
                    if over_budget < 0:
                        over_budget = 0
                else:
                    over_budget = 0

                rie = c_meta.reserve_imm_e_mode
                if rie == _RIE_STATIC:
                    imm_e = c_meta.reserve_static_imm_e
                elif rie == _RIE_COND_MASK:
                    imm_e = c_meta.conditional_energy_gain * (
                        0.75
                        if (c_meta.target_mask & ((h_m1 & ~cmask) | h_m2))
                        else 0.35
                    )
                elif rie == _RIE_WURSTCHEN:
                    wc = cheap2_cnt - (1 if c_meta.cost <= 2 else 0) + 2
                    imm_e = float(wc if wc < 5 else 5)
                elif rie == _RIE_ISTANBUL:
                    e_per = c_meta.istanbul_e_per
                    disc_tom = min(2, max(1, hand_len - 1))
                    imm_e = max(
                        0.0,
                        float(disc_tom) * max(0.0, e_per - 1.45),
                    )
                else:
                    imm_e = c_meta.conditional_energy_gain * (
                        0.75
                        if (cost5_cnt - (1 if c_meta.cost >= 5 else 0)) > 0
                        else 0.35
                    )

                card_draw_val = (
                    c_meta.reserve_draw_cheap
                    if cost <= 1
                    else c_meta.reserve_draw_exp
                )
                raw_val = (
                    base_f
                    + imm_e * 0.95
                    + card_draw_val
                    - over_budget * 1.50
                )
                is_synergy_piece = False

                if tom_all_tags:
                    if pos_tom_all > 0.0 and not is_zeroed:
                        raw_val += pos_tom_all * 1.25
                        is_synergy_piece = True
                elif tom_tag_bonuses is not None and not is_zeroed:
                    for t in ordered_c_tags:
                        tb = tom_tag_bonuses.get(t, 0.0)
                        if tb > 0.0:
                            raw_val += tb * 1.25
                            is_synergy_piece = True

                if (
                    held_scaler_benefit_by_tag is not None
                    and not is_zeroed
                    and not c_meta.is_multi_scaler
                    and days_left >= 1
                ):
                    best_partner_tag: str | None = None
                    best_partner_benefit = 0.0
                    for t in ordered_c_tags:
                        b_val = held_scaler_benefit_by_tag.get(t, 0.0)
                        if b_val > best_partner_benefit:
                            best_partner_benefit = b_val
                            best_partner_tag = t
                    if best_partner_tag is not None:
                        if paired_partner_counts is None:
                            paired_partner_counts = {best_partner_tag: 1}
                            raw_val += best_partner_benefit * 1.15
                            is_synergy_piece = True
                        else:
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

                if c_meta.reserve_has_syn_branch:
                    pure_zero_target = (
                        c_meta.target_single_bit == zero_bit
                        and c_meta.energy_delta >= 0
                    )
                    if c_meta.is_multi_scaler and not pure_zero_target:
                        is_synergy_piece = True
                        has_owned_match = bool(
                            c_meta.target_mask & ((h_m1 & ~cmask) | h_m2)
                        )
                        abundance = (
                            1.0
                            if has_owned_match
                            else c_meta.tag_abundance_target
                        )
                        wait_factor = (
                            1.55 if days_left >= 2 else 0.90
                        ) * abundance
                        raw_val += wait_factor * c_meta.signed_match_benefit
                    elif c_meta.is_global_after_scaler:
                        if zero_tag and c_title == "Colouring":
                            raw_val -= 8.0
                        else:
                            is_synergy_piece = True
                            wait_factor = 2.10 if days_left >= 2 else 1.35
                            raw_val += wait_factor * c_meta.signed_match_benefit
                    elif c_title == "Couch Tube" and days_left >= 1 and not is_zeroed:
                        is_synergy_piece = True
                        raw_val += 3.40 if days_left >= 2 else 2.40
                    elif (
                        c_title == "Cheesy Phone Game"
                        and days_left >= 1
                        and not is_zeroed
                    ):
                        raw_val += 1.50
                    elif c_title == "Assorted Cutlery" and days_left >= 1:
                        has_combo = (
                            cutlery_combo_cnt
                            - (1 if c_meta.is_cutlery_combo else 0)
                        ) > 0
                        if has_combo:
                            is_synergy_piece = True
                            raw_val += 4.50
                        elif days_left >= 2:
                            raw_val += 3.50
                    elif (
                        c_title == "Ski on Shoulders"
                        and days_left >= 1
                        and not is_zeroed
                    ):
                        if ski_ahead_ge2 is None:
                            ski_ahead_ge2 = (
                                sum(
                                    1
                                    for idx, opp in enumerate(sim.players)
                                    if idx != player_index and opp.fun > player.fun
                                )
                                >= 2
                            )
                        if ski_ahead_ge2:
                            is_synergy_piece = True
                            raw_val += 3.60
                    elif (
                        c_title == "Epic Prank"
                        and days_left >= 1
                        and not is_zeroed
                        and bool(_MASK_ITEM & ((h_m1 & ~cmask) | h_m2))
                    ):
                        is_synergy_piece = True
                        raw_val += 3.20
                    elif c_meta.is_tomorrow_scaler and not pure_zero_target:
                        if days_left >= 2:
                            raw_val += 0.50 * c_meta.signed_match_benefit
                    elif (
                        c_meta.is_single_buff
                        and days_left >= 1
                        and not pure_zero_target
                    ):
                        has_partner_now = bool(
                            c_meta.target_mask & ((h_m1 & ~cmask) | h_m2)
                        )
                        is_synergy_piece = True
                        raw_val += (
                            0.85 if has_partner_now else 0.45
                        ) * c_meta.signed_match_benefit
                    elif (
                        c_meta.is_count_finisher
                        and base_f <= 2
                        and days_left >= 1
                        and not is_zeroed
                    ):
                        is_synergy_piece = True
                        raw_val += 1.50 if days_left >= 2 else 0.75

                net_energy_cost = cost - imm_e
                if net_energy_cost < 0.0:
                    net_energy_cost = 0.0
                if is_synergy_piece:
                    energy_tax = (
                        (net_energy_cost if net_energy_cost < 3.0 else 3.0)
                        * 0.42
                        + (
                            (net_energy_cost - 3.0)
                            if net_energy_cost > 3.0
                            else 0.0
                        )
                        * 0.88
                    )
                else:
                    energy_tax = net_energy_cost * 0.92
                surplus = raw_val - energy_tax
                if surplus > 0.0:
                    surpluses.append(surplus)

            if len(surpluses) > 1:
                surpluses.sort(reverse=True)
            has_combo_or_energy = (
                has_any_multi_or_global or tom_energy_delta >= 2
            )
            slot_weights = (
                self.RESERVE_SLOT_WEIGHTS_COMBO
                if has_combo_or_energy
                else self.RESERVE_SLOT_WEIGHTS_PLAIN
            )
            tail_weight = self.RESERVE_SLOT_TAIL_WEIGHT
            held_value = 0.0
            for i, s in enumerate(surpluses):
                held_value += s * (slot_weights[i] if i < 4 else tail_weight)
            val = tom_fun + tom_energy_value + rtw * held_value

            extra = 0.0
            if has_colouring:
                if zero_tag:
                    extra -= 18.0
                else:
                    extra += 2.2
                    for hm in h_metas:
                        ce = hm.reserve_colouring_extra
                        if ce != 0.0:
                            extra += ce

            if has_photo_shoot:
                cheap_held = 0
                for hm in h_metas:
                    if hm.cost <= 2:
                        cheap_held += 1
                extra += 1.8 + 0.65 * (cheap_held if cheap_held < 3 else 3)
                if pb_cnt > 0:
                    extra += 3.5

            if has_dates_first and not zero_tag:
                cheap_or_energy = 0
                for hm in h_metas:
                    if hm.cost <= 1 or hm.immediate_energy_gain >= 2:
                        cheap_or_energy += 1
                extra += (cheap_or_energy - 1) * 1.15

            for hm in h_metas:
                if not hm.has_reserve_extra:
                    continue
                flat1 = hm.reserve_flat1
                if flat1 is not None:
                    extra += flat1 * rtw
                else:
                    d1 = hm.reserve_dyn1_code
                    if d1:
                        if d1 == _RD1_LEFTOVERS:
                            if (
                                copy_food_cnt - (1 if hm.is_copy_food else 0)
                            ) > 0:
                                extra += 2.2 * rtw
                        elif d1 == _RD1_PADDLEBOAT:
                            if (
                                has_photo_shoot
                                or (
                                    acquirer_cnt - (1 if hm.is_acquirer else 0)
                                )
                                > 0
                            ):
                                extra += 2.5 * rtw
                        elif d1 == _RD1_POTATO_PANCAKES:
                            rc = relax_cnt_total - (
                                1 if (hm.tag_mask & _MASK_RELAX) else 0
                            )
                            if rc >= 1:
                                extra += (rc if rc < 3 else 3) * 1.6 * rtw
                        elif d1 == _RD1_KEEPER:
                            oc = out_cnt_total - (
                                1 if (hm.tag_mask & _MASK_OUTDOORS) else 0
                            )
                            if oc >= 2:
                                extra += (oc if oc < 3 else 3) * 0.9 * rtw
                        elif d1 == _RD1_CAMP_CROSSROADS:
                            ec = ex_cnt_total - (
                                1 if (hm.tag_mask & _MASK_EXERCISE) else 0
                            )
                            if ec >= 2:
                                extra += (ec if ec < 3 else 3) * 0.9 * rtw
                        elif d1 == _RD1_PUDDING_CHOMEUR:
                            if hand_len >= 2:
                                extra += 1.4 * rtw

                flat2 = hm.reserve_flat2
                if flat2 is not None:
                    extra += flat2 * rtw
                else:
                    d2 = hm.reserve_dyn2_code
                    if d2:
                        if d2 == _RD2_ASSORTED_CUTLERY:
                            if (
                                cutlery_real_combo_cnt
                                - (1 if hm.is_cutlery_real_combo else 0)
                            ) == 0:
                                extra -= 2.0 * rtw
                        elif d2 == _RD2_SKI_ON_SHOULDERS:
                            if ski_ahead_ge2 is None:
                                ski_ahead_ge2 = (
                                    sum(
                                        1
                                        for idx in range(4)
                                        if idx != player_index
                                        and sim.players[idx].fun > player.fun
                                    )
                                    >= 2
                                )
                            if not ski_ahead_ge2:
                                extra -= 1.2 * rtw
                        elif d2 == _RD2_ADVENTURE_RACE:
                            sup = (
                                tom_pos_energy_sum
                                + total_adv_sup
                                - hm.reserve_adv_race_sup
                            )
                            if sup >= 3:
                                extra += 3.0 * rtw
                            elif sup >= 2 or days_left >= 3:
                                extra += 1.6 * rtw
                            elif sup == 0 and days_left <= 1:
                                extra -= 1.2 * rtw
                        elif d2 == _RD2_WURSTCHEN:
                            if (cheap2_cnt - (1 if hm.cost <= 2 else 0)) >= 2:
                                extra += 1.2 * rtw
                        elif d2 == _RD2_MEDICAL_ADVICE:
                            if days_left >= 2:
                                extra += 1.8 * rtw
                        elif d2 == _RD2_EMPTY_HAND_PAIR:
                            if hand_len >= 3:
                                extra -= 0.9 * rtw

            res_val = val + extra
            if cache_key is not None:
                if len(cache) > 1048576:
                    cache.clear()
                cache[cache_key] = res_val
            return res_val

        prof_by_id = _CARBON_PROFILE_BY_DEFN_ID
        held_multi_scalers: list[tuple[CardInstance, CardSynergyProfile]] = []
        for c in hand:
            cm = c.markers
            c_prof = (
                prof_by_id.get(id(c.definition))
                if (not cm or "_copied_definition" not in cm)
                else None
            )
            if c_prof is None:
                c_prof = _beryllium_build_profile(c)
            if c_prof.is_multi_scaler or c_prof.is_global_after_scaler:
                held_multi_scalers.append((c, c_prof))

        held_scaler_benefit_by_tag = {}
        for _, s_prof in held_multi_scalers:
            if s_prof.is_multi_scaler and s_prof.signed_match_benefit > 0:
                s_targets = (
                    s_prof.target_tags
                    if len(s_prof.target_tags) <= 1
                    else sorted(s_prof.target_tags)
                )
                for t in s_targets:
                    held_scaler_benefit_by_tag[t] = (
                        held_scaler_benefit_by_tag.get(t, 0.0)
                        + s_prof.signed_match_benefit
                    )

        paired_partner_counts = {}
        seen_must_play_first = False
        surpluses = []

        if len(hand) <= 1:
            ordered_hand: Sequence[CardInstance] = hand
        else:
            ordered_hand = sorted(
                hand,
                key=lambda c: (
                    (
                        c.definition.cost
                        if not c.markers
                        else c.effective_cost
                    ),
                    -(
                        c.definition.base_fun
                        if not c.markers
                        else c.effective_base_fun
                    ),
                    c.definition.slug,
                ),
            )

        pos_tom_all = (
            sum(
                tom_tag_bonuses[k]
                for k in sorted(tom_tag_bonuses)
                if tom_tag_bonuses[k] > 0.0
            )
            if tom_all_tags
            else 0.0
        )

        for c in ordered_hand:
            cm = c.markers
            c_uncopied = not cm or "_copied_definition" not in cm
            c_prof = (
                prof_by_id.get(id(c.definition)) if c_uncopied else None
            )
            if c_prof is None:
                c_prof = _beryllium_build_profile(c)
            c_defn = c.definition if c_uncopied else c.effective_definition
            c_title = c_defn.title
            c_tags = c.definition.tags if not cm else c.tags
            ordered_c_tags = (
                c_tags if len(c_tags) <= 1 else sorted(c_tags)
            )
            is_zeroed = bool(zero_tag and zero_tag in c_tags)
            cost = max(0, c_defn.cost)
            if char is not None:
                cost = max(0, char.modify_energy_cost(c_tags, cost))
            base_f = float(c_defn.base_fun + c_prof.immediate_fun_gain)
            if c_title == "Nos Book":
                base_f += 3.0
            if char is not None:
                base_f = float(
                    char.finalize_fun(
                        c_tags, char.modify_fun(c_tags, int(base_f))
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
                ) + _carbon_extra_energy_support(
                    sim,
                    player_index,
                    c,
                    extra_fodder=3,
                    ignore_today_block=True,
                )
                if days_left == 1 and cost > 7 + support:
                    if not (
                        any(
                            hc.definition.title == "Assorted Cutlery"
                            for hc in hand
                        )
                        and any(
                            hc.definition.title == "Tell a Story"
                            for hc in hand
                        )
                    ):
                        continue
                over_budget = max(0, cost - (6 + support))
            else:
                over_budget = 0

            if c_title == "Würstchen":
                cheap_cnt = sum(
                    1
                    for other in hand
                    if other is not c and other.definition.cost <= 2
                )
                imm_e = float(min(5, cheap_cnt + 2))
            elif c_title == "Istanbul":
                e_per = _istanbul_energy_per_discard(c)
                disc_tom = min(2, max(1, len(hand) - 1))
                imm_e = max(
                    0.0,
                    float(disc_tom) * max(0.0, e_per - 1.45),
                )
            elif (
                c_prof.conditional_energy_gain > 0
                and c_title
                not in {
                    "Potato Pancakes",
                    "Keeper",
                    "Camp Crossroads Dessert",
                }
            ):
                t_tags = c_prof.target_tags
                has_trig = any(
                    other is not c
                    and (
                        other.definition.cost >= 5
                        if c_title == "Steak"
                        else bool(
                            (
                                other.definition.tags
                                if not other.markers
                                else other.tags
                            )
                            & t_tags
                        )
                    )
                    for other in hand
                )
                imm_e = c_prof.conditional_energy_gain * (
                    0.75 if has_trig else 0.35
                )
            else:
                imm_e = (
                    0.0
                    if c_title == "Pudding Chômeur"
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

            if tom_all_tags:
                if pos_tom_all > 0.0 and not is_zeroed:
                    raw_val += pos_tom_all * 1.25
                    is_synergy_piece = True
            elif tom_tag_bonuses and not is_zeroed:
                for t in ordered_c_tags:
                    tb = tom_tag_bonuses.get(t, 0.0)
                    if tb > 0.0:
                        raw_val += tb * 1.25
                        is_synergy_piece = True

            best_partner_tag = None
            best_partner_benefit = 0.0
            if held_scaler_benefit_by_tag and not is_zeroed:
                for t in ordered_c_tags:
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

            pure_zero_target = bool(
                zero_tag
                and c_prof.target_tags == frozenset({zero_tag})
                and c_prof.energy_delta >= 0
            )
            if c_prof.is_multi_scaler and not pure_zero_target:
                is_synergy_piece = True
                t_tags = c_prof.target_tags
                has_owned_match = any(
                    other is not c
                    and (
                        (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        )
                        & t_tags
                    )
                    for other in hand
                )
                abundance = (
                    1.0 if has_owned_match else _tag_abundance(t_tags)
                )
                wait_factor = (
                    1.55 if days_left >= 2 else 0.90
                ) * abundance
                raw_val += wait_factor * c_prof.signed_match_benefit
            elif c_prof.is_global_after_scaler:
                if zero_tag and c_title == "Colouring":
                    raw_val -= 8.0
                else:
                    is_synergy_piece = True
                    wait_factor = 2.10 if days_left >= 2 else 1.35
                    raw_val += wait_factor * c_prof.signed_match_benefit
            elif c_title == "Couch Tube" and days_left >= 1 and not is_zeroed:
                is_synergy_piece = True
                raw_val += 3.40 if days_left >= 2 else 2.40
            elif c_title == "Cheesy Phone Game" and days_left >= 1 and not is_zeroed:
                raw_val += 1.50
            elif c_title == "Assorted Cutlery" and days_left >= 1:
                has_combo = any(
                    other is not c
                    and (
                        other.definition.title in _TRUNK_REORDER_TITLES
                        or other.definition.title
                        in {"Boat Ride", "Fancy Craft", "Canoe"}
                        or (
                            other.definition.cost
                            if not other.markers
                            else other.effective_cost
                        )
                        >= 6
                    )
                    for other in hand
                )
                if has_combo:
                    is_synergy_piece = True
                    raw_val += 4.50
                elif days_left >= 2:
                    raw_val += 3.50
            elif c_title == "Ski on Shoulders" and days_left >= 1 and not is_zeroed:
                if ski_ahead_ge2 is None:
                    ski_ahead_ge2 = (
                        sum(
                            1
                            for idx, opp in enumerate(sim.players)
                            if idx != player_index and opp.fun > player.fun
                        )
                        >= 2
                    )
                if ski_ahead_ge2:
                    is_synergy_piece = True
                    raw_val += 3.60
            elif (
                c_title == "Epic Prank"
                and days_left >= 1
                and not is_zeroed
                and any(
                    "Item"
                    in (
                        other.definition.tags
                        if not other.markers
                        else other.tags
                    )
                    for other in hand
                    if other is not c
                )
            ):
                is_synergy_piece = True
                raw_val += 3.20
            elif c_prof.is_tomorrow_scaler and not pure_zero_target:
                if days_left >= 2:
                    raw_val += 0.50 * c_prof.signed_match_benefit
            elif c_prof.is_single_buff and days_left >= 1 and not pure_zero_target:
                t_tags = c_prof.target_tags
                has_partner_now = any(
                    other is not c
                    and (
                        (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        )
                        & t_tags
                    )
                    for other in hand
                )
                is_synergy_piece = True
                raw_val += (
                    0.85 if has_partner_now else 0.45
                ) * c_prof.signed_match_benefit
            elif (
                c_prof.is_count_finisher
                and base_f <= 2
                and days_left >= 1
                and not is_zeroed
            ):
                is_synergy_piece = True
                raw_val += 1.50 if days_left >= 2 else 0.75

            net_energy_cost = max(0.0, cost - imm_e)
            if is_synergy_piece:
                energy_tax = (
                    min(net_energy_cost, 3.0) * 0.42
                    + max(0.0, net_energy_cost - 3.0) * 0.88
                )
            else:
                energy_tax = net_energy_cost * 0.92
            surplus = raw_val - energy_tax
            if surplus > 0.0:
                surpluses.append(surplus)

        if len(surpluses) > 1:
            surpluses.sort(reverse=True)
        has_combo_or_energy = (
            bool(held_multi_scalers) or tom_energy_delta >= 2
        )
        slot_weights = (
            self.RESERVE_SLOT_WEIGHTS_COMBO
            if has_combo_or_energy
            else self.RESERVE_SLOT_WEIGHTS_PLAIN
        )
        tail_weight = self.RESERVE_SLOT_TAIL_WEIGHT
        held_value = 0.0
        for i, s in enumerate(surpluses):
            held_value += s * (slot_weights[i] if i < 4 else tail_weight)
        val = tom_fun + tom_energy_value + rtw * held_value

        extra = 0.0
        if has_colouring:
            if zero_tag:
                extra -= 18.0
            else:
                extra += 2.2
                for hc in hand:
                    hm = hc.markers
                    h_prof = (
                        prof_by_id.get(id(hc.definition))
                        if (not hm or "_copied_definition" not in hm)
                        else None
                    )
                    if h_prof is None:
                        h_prof = _beryllium_build_profile(hc)
                    if h_prof.is_multi_scaler or h_prof.is_before_scaler:
                        extra += max(0.0, h_prof.signed_match_benefit * 2.2)
                    elif hc.definition.title == "Newfangled Tres Fute":
                        extra += 6.0
                    elif hc.definition.title in {
                        "Potato Pancakes",
                        "Keeper",
                        "Camp Crossroads Dessert",
                    }:
                        extra += 3.0

        if has_photo_shoot:
            cheap_held = 0
            for hc in hand:
                if (hc.definition.cost if not hc.markers else hc.effective_cost) <= 2:
                    cheap_held += 1
            extra += 1.8 + 0.65 * min(cheap_held, 3)
            if any(hc.definition.title == "Paddleboat" for hc in hand):
                extra += 3.5

        if has_dates_first and not zero_tag:
            cheap_or_energy = 0
            for hc in hand:
                hm = hc.markers
                h_uncopied = not hm or "_copied_definition" not in hm
                if (
                    hc.definition.cost if h_uncopied else hc.effective_cost
                ) <= 1:
                    cheap_or_energy += 1
                else:
                    hp = (
                        prof_by_id.get(id(hc.definition))
                        if h_uncopied
                        else None
                    )
                    if hp is None:
                        hp = _beryllium_build_profile(hc)
                    if hp.immediate_energy_gain >= 2:
                        cheap_or_energy += 1
            extra += (cheap_or_energy - 1) * 1.15

        for hc in hand:
            h_title = hc.definition.title
            flat1 = _RESERVE_FLAT_EXTRA.get(h_title)
            if flat1 is not None:
                extra += flat1 * rtw
            elif h_title in _RESERVE_DYNAMIC_EXTRA_1:
                if h_title == "Leftovers":
                    for other in hand:
                        if (
                            other is not hc
                            and other.definition.title
                            in _HYDROGEN_COPY_FOOD_TITLES
                        ):
                            extra += 2.2 * rtw
                            break
                elif h_title == "Paddleboat":
                    has_pb_syn = has_photo_shoot
                    if not has_pb_syn:
                        for other in hand:
                            if (
                                other is not hc
                                and other.definition.title in _ACQUIRER_TITLES
                            ):
                                has_pb_syn = True
                                break
                    if has_pb_syn:
                        extra += 2.5 * rtw
                elif h_title == "Potato Pancakes":
                    relax_cnt = sum(
                        1
                        for other in hand
                        if other is not hc
                        and "Relax"
                        in (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        )
                    )
                    if relax_cnt >= 1:
                        extra += min(relax_cnt, 3) * 1.6 * rtw
                elif h_title == "Keeper":
                    out_cnt = sum(
                        1
                        for other in hand
                        if other is not hc
                        and "Outdoors"
                        in (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        )
                    )
                    if out_cnt >= 2:
                        extra += min(out_cnt, 3) * 0.9 * rtw
                elif h_title == "Camp Crossroads Dessert":
                    ex_cnt = sum(
                        1
                        for other in hand
                        if other is not hc
                        and "Exercise"
                        in (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        )
                    )
                    if ex_cnt >= 2:
                        extra += min(ex_cnt, 3) * 0.9 * rtw
                elif h_title == "Pudding Chômeur" and len(hand) >= 2:
                    extra += 1.4 * rtw

            flat2 = _RESERVE_EMPIRICAL_FLAT.get(h_title)
            if flat2 is not None:
                extra += flat2 * rtw
            elif h_title in _RESERVE_DYNAMIC_EXTRA_2:
                if h_title == "Assorted Cutlery":
                    has_real_combo = any(
                        other is not hc
                        and other.definition.title
                        in _CUTLERY_REAL_COMBO_TITLES
                        for other in hand
                    )
                    if not has_real_combo:
                        extra -= 2.0 * rtw
                elif h_title == "Ski on Shoulders":
                    if ski_ahead_ge2 is None:
                        ski_ahead_ge2 = (
                            sum(
                                1
                                for idx in range(4)
                                if idx != player_index
                                and sim.players[idx].fun > player.fun
                            )
                            >= 2
                        )
                    if not ski_ahead_ge2:
                        extra -= 1.2 * rtw
                elif h_title == "Adventure Race":
                    sup = tom_pos_energy_sum
                    for other in hand:
                        if other is hc:
                            continue
                        ot = other.definition.title
                        om = other.markers
                        op = (
                            prof_by_id.get(id(other.definition))
                            if (not om or "_copied_definition" not in om)
                            else None
                        )
                        if op is None:
                            op = _beryllium_build_profile(other)
                        if ot == "Shady Spot":
                            sup += 4
                        elif ot in {"Butter Tarts", "Istanbul"}:
                            sup += 4
                        elif ot == "Würstchen":
                            sup += 1
                        elif op.immediate_energy_gain > other.definition.cost:
                            sup += op.immediate_energy_gain - other.definition.cost
                    if sup >= 3:
                        extra += 3.0 * rtw
                    elif sup >= 2 or days_left >= 3:
                        extra += 1.6 * rtw
                    elif sup == 0 and days_left <= 1:
                        extra -= 1.2 * rtw
                elif h_title == "Würstchen":
                    cheap_cnt = sum(
                        1
                        for other in hand
                        if other is not hc and other.definition.cost <= 2
                    )
                    if cheap_cnt >= 2:
                        extra += 1.2 * rtw
                elif h_title == "Medical Advice" and days_left >= 2:
                    extra += 1.8 * rtw
                elif h_title in {"Wakesurf", "Drop a Ski"} and len(hand) >= 3:
                    extra -= 0.9 * rtw

        res_val = val + extra
        if cache_key is not None:
            if len(cache) > 524288:
                cache.clear()
            cache[cache_key] = res_val
        return res_val

    # ------------------------------------------------------------------
    # Tactical Sub-Decisions Routed Through Hydrogen's Policy
    # ------------------------------------------------------------------

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> Sequence[int]:
        if not options:
            return ()
        player = game.players[player_index]
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played == "Bohnanza":
            return _select_bohnanza_discards_fast(game, player_index, options)
        if last_played == "Istanbul":
            ranked = _evaluate_istanbul_discards_fast(
                game, player_index, options
            )
            if (
                len(ranked) <= 1
                or self._evaluation_depth >= 1
                or len(player.played_today) >= 8
                or game.stats is _get_shared_sim_stats()
                or len(player.hand) != len(options)
                or any(
                    hc.instance_id != opt.instance_id
                    for hc, opt in zip(player.hand, options)
                )
            ):
                return ranked[0][1]

            to_verify: list[tuple[int, ...]] = [
                cand for _, cand in ranked[:7]
            ]
            if () not in to_verify:
                to_verify.append(())
            to_verify.sort(key=len)

            base_sim = self._prepare_observable_game(game, player_index)
            sim_p0 = base_sim.players[player_index]
            if not sim_p0.played_today or len(sim_p0.hand) != len(options):
                return ranked[0][1]
            raw_e_per = _istanbul_energy_per_discard(sim_p0.played_today[-1])

            best_choice = ranked[0][1]
            best_val = float("-inf")
            prev_depth = self._evaluation_depth
            self._evaluation_depth = 1
            try:
                for disc_tuple in to_verify:
                    sim = _hydrogen_copy_game(base_sim)
                    sim_player = sim.players[player_index]
                    sim_source = sim_player.played_today[-1]
                    try:
                        discarded = sim.discard_cards_from_hand(
                            player_index, disc_tuple
                        )
                        sim.gain_energy(
                            sim_player, raw_e_per * len(discarded), sim_source
                        )
                        for t_src in tuple(sim_player.tomorrow_cards):
                            t_src.effective_behavior.on_tomorrow_card_play(
                                sim, sim_player, t_src, sim_source
                            )
                        for p_src in tuple(sim_player.played_today):
                            p_src.effective_behavior.on_card_play(
                                sim, sim_player, p_src, sim_source
                            )
                        _, val = self._search_best_day_plan(
                            sim,
                            player_index,
                            beam_width=6,
                            max_plays=7,
                            allow_bed_now=True,
                            base_turns_offset=1,
                            root_skipped_override=base_sim.players[
                                player_index
                            ].skipped_turns,
                            start_extra_active=_card_allows_extra_plays(
                                sim, sim_player, sim_source
                            ),
                        )
                    except Exception:
                        continue
                    adj_val = val - 0.02 * len(disc_tuple)
                    if adj_val > best_val + 1e-9:
                        best_val = adj_val
                        best_choice = disc_tuple
            finally:
                self._evaluation_depth = prev_depth
            return best_choice
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_cards_to_discard(
            game, player_index, options
        )

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_discard(
            game, player_index, options
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
            or len(game.players[player_index].played_today) >= 7
            or game.stats is _get_shared_sim_stats()
        ):
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_copy(
                game, player_index, options
            )
        live_player = game.players[player_index]
        live_source_card = (
            live_player.played_today[-1] if live_player.played_today else None
        )
        base_sim = self._prepare_observable_game(game, player_index)
        player = base_sim.players[player_index]
        source_card = player.played_today[-1] if player.played_today else None
        if source_card is None or live_source_card is None:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_copy(
                game, player_index, options
            )

        pay_cost = (
            type(source_card.effective_behavior).__name__ != "LeftoversBehavior"
        )
        ranked_indices = sorted(
            range(len(options)),
            key=lambda i: _contextual_copy_value(
                game,
                player_index,
                options[i],
                live_source_card,
                pay_cost=pay_cost,
            )
            - (1.35 if _effective_card_title(options[i]) == "Group Dinner" else 0.0),
            reverse=True,
        )[:5]
        best_idx = ranked_indices[0]
        best_val = float("-inf")
        prev_depth = self._evaluation_depth
        self._evaluation_depth = 1
        try:
            for idx in ranked_indices:
                option = options[idx]
                sim = _hydrogen_copy_game(base_sim)
                sim_player = sim.players[player_index]
                sim_source = sim_player.played_today[-1]
                try:
                    sim.copy_card_effect(
                        player_index,
                        _clone_card(option),
                        sim_source,
                        pay_source_cost=pay_cost,
                    )
                    for t_src in tuple(sim_player.tomorrow_cards):
                        t_src.effective_behavior.on_tomorrow_card_play(
                            sim, sim_player, t_src, sim_source
                        )
                    for p_src in tuple(sim_player.played_today):
                        p_src.effective_behavior.on_card_play(
                            sim, sim_player, p_src, sim_source
                        )
                    _, val = self._search_best_day_plan(
                        sim,
                        player_index,
                        beam_width=5,
                        max_plays=5,
                        allow_bed_now=True,
                        base_turns_offset=1,
                        root_skipped_override=base_sim.players[
                            player_index
                        ].skipped_turns,
                        start_extra_active=_card_allows_extra_plays(
                            sim, sim_player, sim_source
                        ),
                    )
                except Exception:
                    val = _contextual_copy_value(
                        game,
                        player_index,
                        option,
                        live_source_card,
                        pay_cost=pay_cost,
                    )
                if val > best_val:
                    best_val = val
                    best_idx = idx
        finally:
            self._evaluation_depth = prev_depth
        return best_idx

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_target(
            game, player_index, options
        )

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_target(
            game, player_index, options
        )

    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_player(
            game, player_index, eligible_player_indices
        )

    def choose_tag(
        self,
        game: Game,
        player_index: int,
        options: Sequence[str],
    ) -> str:
        if game.stats is not _get_shared_sim_stats():
            player = game.players[player_index]
            last_played = (
                _effective_card_title(player.played_today[-1])
                if player.played_today
                else ""
            )
            if last_played == "Poker":
                known_top = self._get_live_known_trunk_top(game, player_index)
                if known_top and known_top[0].tags:
                    for t in options:
                        if t in known_top[0].tags:
                            return t
                unseen_counts, _ = _unseen_tag_counts_and_total(
                    game, player_index, known_top
                )
                return max(
                    options,
                    key=lambda t: (unseen_counts.get(t, 0), t == "Outdoors"),
                )
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_tag(
            game, player_index, options
        )

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
            or game.stats is _get_shared_sim_stats()
        ):
            return maximum
        base_sim = self._prepare_observable_game(game, player_index)
        player = base_sim.players[player_index]
        source_card = player.played_today[-1] if player.played_today else None
        if source_card is None:
            return maximum

        best_spend = maximum
        best_val = float("-inf")
        prev_depth = self._evaluation_depth
        self._evaluation_depth = 1
        try:
            for spend in range(maximum + 1):
                sim = _hydrogen_copy_game(base_sim)
                sim_player = sim.players[player_index]
                sim_source = sim_player.played_today[-1]
                sim_player.energy -= spend
                sim_source.markers["energy_cubes"] = spend
                if spend:
                    sim._p_flags[player_index] |= _PF_HAS_SIG_MARKER  # type: ignore[attr-defined]
                for t_src in tuple(sim_player.tomorrow_cards):
                    t_src.effective_behavior.on_tomorrow_card_play(
                        sim, sim_player, t_src, sim_source
                    )
                for p_src in tuple(sim_player.played_today):
                    p_src.effective_behavior.on_card_play(
                        sim, sim_player, p_src, sim_source
                    )
                _, val = self._search_best_day_plan(
                    sim,
                    player_index,
                    beam_width=3,
                    max_plays=3,
                    allow_bed_now=True,
                    base_turns_offset=1,
                    root_skipped_override=base_sim.players[
                        player_index
                    ].skipped_turns,
                    start_extra_active=_card_allows_extra_plays(
                        sim, sim_player, sim_source
                    ),
                )
                if val >= best_val:
                    best_val = val
                    best_spend = spend
        finally:
            self._evaluation_depth = prev_depth
        return best_spend

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        if "Quacks" in action and game.stats is not _get_shared_sim_stats():
            self._ensure_live_game(game)
            player = game.players[player_index]
            last_card = player.played_today[-1] if player.played_today else None
            if last_card is None:
                return False
            key = (id(game), game.day, len(player.played_today), id(last_card))
            if self._quacks_card_key != key:
                self._quacks_card_key = key
                self._quacks_draws = 0
            draws = self._quacks_draws
            self._quacks_draws += 1
            known_top = self._get_live_known_trunk_top(game, player_index)
            if known_top:
                return "Outdoors" not in known_top[0].tags
            unseen_counts, unseen_total = _unseen_tag_counts_and_total(
                game, player_index, known_top
            )
            p_outdoors = unseen_counts.get("Outdoors", 31) / max(1, unseen_total)
            ev_thresh = (2.0 / (3.0 + 2.0 * draws)) + (
                0.015 if draws <= 1 else 0.0
            )
            return p_outdoors < ev_thresh
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_optional_action(
            game, player_index, action
        )

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        self._ensure_live_game(game)
        order = _HYDROGEN_EVAL_AI_SINGLETON.order_cards_for_trunk(
            game, player_index, cards
        )
        if cards and game.stats is not _get_shared_sim_stats():
            rem_prev = self._get_live_known_trunk_top(game, player_index)
            ordered_cards = tuple(cards[i] for i in order) + rem_prev
            self._known_trunk_state = (
                id(game),
                game.day,
                len(game.trunk) + len(cards),
                ordered_cards,
                tuple(len(p.played_today) for p in game.players),
            )
        return order

    def choose_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_cards_for_trunk(
            game, player_index, cards
        )

    def order_daily_conditions(
        self,
        game: Game,
        player_index: int,
        conditions: Sequence[Any],
    ) -> Sequence[int]:
        return _HYDROGEN_EVAL_AI_SINGLETON.order_daily_conditions(
            game, player_index, conditions
        )

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> bool:
        player = game.players[player_index]
        if (
            player.energy < 1
            or not options
            or len(game.trunk) + len(game.discard) < 2
        ):
            return False
        wanted_tags, ctx = _build_wanted_and_ctx(
            player, getattr(game, "_all_canonical_defs", False)
        )
        if game.day < DAY_COUNT:
            best_des = max(
                (
                    _hydrogen_quick_card_desirability(
                        game,
                        player_index,
                        c,
                        wanted_tags,
                        _ctx=ctx,
                        _in_hand=False,
                    )
                    for c in options
                ),
                default=0.0,
            )
            take = player.energy >= 2 or best_des >= 2.5
            if take:
                self._record_extra_pick_if_live(game, player_index)
            return take

        if len(options) < 2:
            return False
        pr_markers, scout_markers = _opponent_suitcase_markers(
            game, player_index
        )
        scored = sorted(
            (
                (
                    _hydrogen_quick_card_desirability(
                        game,
                        player_index,
                        c,
                        wanted_tags,
                        _ctx=ctx,
                        _in_hand=False,
                    )
                    + _suitcase_denial_value(
                        game, player_index, c, pr_markers, scout_markers
                    ),
                    c,
                )
                for c in options
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        second_des, second_card = scored[1]
        second_cost = game.energy_cost(player_index, second_card)
        s_prof = _get_card_meta(second_card).prof
        net_e_pos = (
            s_prof.immediate_energy_gain > second_cost
            or (
                second_card.title == "Würstchen"
                and len(player.played_today)
                + sum(
                    1
                    for hc in player.hand
                    if game.energy_cost(player_index, hc) <= 1
                )
                > second_cost
            )
            or second_card.title
            in {"Istanbul", "Potato Pancakes", "Pudding Chômeur"}
        )
        playable_hand_cost = sum(
            game.energy_cost(player_index, hc)
            for hc in player.hand
            if hc.effective_behavior.can_play(game, player, hc)
        )
        take = (
            net_e_pos
            or (second_cost == 0 and second_des >= 2.0)
            or (
                player.energy >= playable_hand_cost + second_cost + 2
                and second_des >= 3.0
            )
        )
        if take:
            self._record_extra_pick_if_live(game, player_index)
        return take


Hydrogen = HydrogenAI
HydrogenBrain = HydrogenAI
HydrogenbrainAI = HydrogenAI
