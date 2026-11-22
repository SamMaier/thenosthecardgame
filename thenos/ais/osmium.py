"""OsmiumAI: On-policy DAgger-R4 Huber-IRLS policy with aligned drafting and deep Day-6 search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from operator import attrgetter
from typing import Sequence, TYPE_CHECKING

from thenos.ais.argonbrain import (
    DAY_COUNT,
    SUITCASE_SIZE,
    _clone_card,
    _clone_random,
    _get_shared_sim_stats,
    _get_sim_known_trunk_top,
    _unseen_tag_counts_and_total,
)
from thenos.ais.beryllium import (
    _STOCHASTIC_PLAY_TITLES,
    _contextual_copy_value,
    _effective_source_definition,
)
from thenos.ais.carbon import (
    _TERMINAL_AWAKE_OPP_TITLES,
    _carbon_own_suitcase_markers,
    _effective_card_title,
    _opponent_suitcase_markers,
    _suitcase_denial_value,
)
from thenos.ais.hydrogen import (
    _FTB_ZERO_E_STAY,
    _HYDROGEN_COPY_FOOD_TITLES,
    _HYDROGEN_EVAL_AI_SINGLETON,
    _HYDROGEN_META_BY_DEFN_ID,
    _HYDROGEN_META_BY_SLUG,
    _HYDROGEN_SIM_AIS,
    _PF_HAS_AWAKE_OPP_TERM,
    _PF_HAS_END_DAY,
    _PF_HAS_LOOP2,
    _PF_HAS_SIG_MARKER,
    _TRUNK_REORDER_TITLES,
    HydrogenAI,
    _HydrogenEvalAI,
    _build_quick_zone_ctx,
    _build_wanted_and_ctx,
    _card_allows_extra_plays,
    _ensure_hydrogen_caches,
    _evaluate_istanbul_discards_fast,
    _get_card_meta,
    _get_hydrogen_sim_game_cls,
    _hydrogen_copy_game,
    _hydrogen_copy_game_for_play,
    _hydrogen_quick_card_desirability,
    _istanbul_energy_per_discard,
    _select_bohnanza_discards_fast,
    _wanted_tags_for_player_fast,
)
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.models import PlayerState

if TYPE_CHECKING:
    from thenos.game import Game


_CLUSTER_TAG_TO_IDX: dict[str, int] = {
    "Board Game": 0,
    "Exercise": 1,
    "Food": 2,
    "Item": 3,
    "Outdoors": 4,
    "Relax": 5,
    "Social": 6,
}

# Fitted on-policy Huber-IRLS Ridge weights (DAgger-R4, soft-capped Fun utility).
_OSMIUM_BASE_WEIGHTS: tuple[float, ...] = (
    2.541002,
    10.060937,
    21.698141,
    35.873598,
    49.681343,
    1.106752,
    -0.308655,
    0.677362,
    0.000000,
    8.787086,
    1.196287,
    0.224964,
    0.534371,
    0.176963,
    -0.684015,
    0.490292,
    1.346276,
    -0.625493,
    -1.005359,
    1.046788,
    0.942057,
    -0.141541,
    0.042194,
    -2.863289,
    2.746686,
    -0.067539,
    -0.303192,
    -3.306632,
    6.765583,
    0.759789,
    -0.724101,
    -0.200200,
    0.524159,
    2.320231,
    2.532745,
    4.575963,
    -0.041580,
    0.988738,
    -0.093725,
    -0.211342,
    -0.342867,
    0.671528,
    0.000000,
    0.348217,
    -0.402139,
    0.227379,
    0.703678,
    -0.151605,
    -0.324852,
    0.351208,
    0.970183,
    0.439383,
    -0.139217,
    0.000000,
    0.000000,
    0.405083,
    -0.558768,
    0.293247,
    0.965426,
    0.405920,
    -0.586797,
    0.547032,
    0.214187,
    0.013300,
    1.279720,
    -0.004457,
    0.553639,
    0.529742,
    0.105549,
    0.905928,
    3.256662,
    0.777767,
    2.483128,
    -0.338754,
    0.717891,
    0.443562,
    2.044074,
    0.023697,
    -1.302493,
    -0.081781,
    0.116720,
)

_OSMIUM_SLUG_WEIGHTS: dict[str, float] = {
    "7-wonders": 0.305407,
    "address-the-food": -1.035186,
    "adventure-race": 0.465357,
    "after-dinner-entertainment": 0.734647,
    "afternoon-coffee": -0.265428,
    "agricola": -0.395413,
    "assorted-cutlery": -0.866406,
    "aunterview": 0.190504,
    "azul": 0.471858,
    "beach-conversation": -0.102730,
    "beach-volleyball": -0.024997,
    "beaver-burgers": 0.675907,
    "bend-the-rules": 1.348707,
    "boat-ride": 0.185435,
    "bohnanza": 2.321775,
    "booby-prize": 0.359240,
    "bougie-coffee-machine": -1.051905,
    "bracelet-making": -1.414079,
    "bring-a-friend": 2.352499,
    "bubly": -0.014524,
    "bug-spray": -0.773428,
    "butter-tarts": -1.116795,
    "calzones": 0.381994,
    "camp-crossroads-dessert": -0.742079,
    "campfire": -0.134794,
    "canoe": 0.561685,
    "carcassonne": -0.604247,
    "chalk-art": 0.555938,
    "charcuterie": 0.459509,
    "cheap-red": -1.674853,
    "cheap-white": -0.402200,
    "cheesy-phone-game": 0.889732,
    "christmas-name-draw": 0.202683,
    "chuck-a-frisbee": 1.161063,
    "cities-and-knights": 0.409093,
    "civilization-the-boardgame": 0.873473,
    "classic-book": 0.215115,
    "cliff-climbing": -0.828539,
    "colouring": 4.279967,
    "couch-tube": 2.521061,
    "dates-first-nos": 1.700527,
    "decaf": 0.389490,
    "dishwashing": -0.664983,
    "dock-fishing": 0.316217,
    "dominion": -0.090719,
    "doritos": 1.299810,
    "doxology": -0.992291,
    "drop-a-ski": -0.648155,
    "dumbed-down-settlers": -0.302026,
    "early-bedtime": 1.503427,
    "epic-duels": -0.004370,
    "epic-playlist": -1.618146,
    "epic-prank": -0.005243,
    "euchre": -1.040411,
    "euchre-awards-ceremony": -1.476773,
    "euchre-tournament": -2.315201,
    "evening-chat": 0.901286,
    "evening-on-the-dock": 0.734487,
    "fajitas": -0.559177,
    "family-baseball-game": 0.779877,
    "fancy-craft": 2.234590,
    "fancy-floatie": -0.236519,
    "fishing-boat": 0.216247,
    "fishing-evening": -0.565163,
    "fishing-morning": -0.386606,
    "fit-to-print": -0.768210,
    "floating": 0.403721,
    "forced-family-fun": -1.053715,
    "grocery-store-run": -1.115974,
    "group-dinner": -0.270222,
    "high-end-red": -0.169239,
    "high-end-white": -1.054279,
    "hold-the-baby": 1.224265,
    "ice-cream-sandwich": 0.192919,
    "ice-wine": 0.698685,
    "istanbul": 3.871684,
    "johnny-appleseed": -0.721993,
    "kayak": 0.365404,
    "keeper": 0.376220,
    "kneeboard": -0.576329,
    "last-years-shirt": -0.511820,
    "leftovers": -0.202424,
    "long-distance-visitors": -0.371222,
    "m-ms": -0.680769,
    "make-plans": 0.712748,
    "medical-advice": -0.199819,
    "morning-bike": 0.632107,
    "morning-coffee": -0.277148,
    "morning-run": 0.531876,
    "morning-walk": 0.180278,
    "movie": 0.563263,
    "nap": -1.386491,
    "navy-sealing": 0.994307,
    "new-nos-book-entry": 0.705850,
    "newfangled-tres-fute": 0.489753,
    "non-fiction-book": -0.577863,
    "nos-book": 0.877031,
    "nos-shirt": 1.210483,
    "outdoor-movie": 1.058520,
    "paddleboard": -0.391463,
    "paddleboat": 0.654633,
    "paint": -0.006023,
    "paint-rocks": 0.041167,
    "photo-shoot": 0.639717,
    "play-with-the-kids": -1.014027,
    "poker": -0.779202,
    "ponyback": -0.123043,
    "potato-pancakes": -0.736249,
    "prime-picnic-table": 0.850664,
    "pudding-chomeur": 0.810791,
    "puerto-rico": 0.487236,
    "quacks": -0.156225,
    "read-the-radar": -0.652858,
    "risk": -0.228392,
    "rouladen": 2.960619,
    "san-juan": -0.066760,
    "schwank": -1.036712,
    "scout-other-cottages": 0.032884,
    "scrabble": -0.288709,
    "screamer-battle": -0.616176,
    "shady-spot": -2.290423,
    "sing-song": 0.035976,
    "ski-boat": -0.225525,
    "ski-on-shoulders": -0.134483,
    "sleep-in": -0.651480,
    "solo": 0.235076,
    "spa-day": 0.273807,
    "spikeball": -0.231654,
    "splendor": -0.793402,
    "stay-up-late": 0.063644,
    "steak": -0.084136,
    "stretch": -0.019005,
    "sunrise": -0.348479,
    "sunscreen": 1.160225,
    "sweet-lawn-chair": -0.052277,
    "swimming": 0.208704,
    "tanning": 0.190058,
    "teach-kid-to-ski": -0.566749,
    "tell-a-story": -0.443683,
    "terra-mystica": -0.850489,
    "the-crew": -0.657453,
    "thriller-book": -1.421688,
    "throw-a-baseball": -0.628485,
    "treat-cereal": 0.209585,
    "treat-store-run": 0.185536,
    "trekking-through-history": -1.215113,
    "tres-fute": -0.868853,
    "wakeboard": -0.564222,
    "wakesurf": -1.182049,
    "water-trampoline": -1.228901,
    "water-volleyball": 0.087744,
    "waterski": -1.093430,
    "wedding-anniversary": 2.928889,
    "weird-chip-flavour": 0.059818,
    "wildlife-spotting": -0.742717,
    "wingspan": -0.951864,
    "work-call": 1.578914,
    "wrestle-the-kids": 0.800413,
    "wurstchen": 1.315645,
    "zero-gravity-chair": 0.826657,
    "zumba": 0.434499,
}

_OSMIUM_SLUG_D1_WEIGHTS: dict[str, float] = {
    "7-wonders": 0.064932,
    "address-the-food": 0.324543,
    "adventure-race": 0.825986,
    "after-dinner-entertainment": 0.756144,
    "afternoon-coffee": 0.181251,
    "agricola": -0.487581,
    "assorted-cutlery": -0.066577,
    "aunterview": -0.622299,
    "azul": -0.135485,
    "beach-conversation": -0.002381,
    "beach-volleyball": 0.094199,
    "beaver-burgers": -0.326356,
    "bend-the-rules": 0.187889,
    "boat-ride": 0.107194,
    "bohnanza": 0.298253,
    "booby-prize": -0.010162,
    "bougie-coffee-machine": -1.095412,
    "bracelet-making": 0.142024,
    "bring-a-friend": 0.159613,
    "bubly": -0.153021,
    "bug-spray": -0.065773,
    "butter-tarts": 0.022607,
    "calzones": 0.629760,
    "camp-crossroads-dessert": -0.177358,
    "campfire": -0.044999,
    "canoe": 0.248399,
    "carcassonne": -0.016450,
    "chalk-art": -0.309851,
    "charcuterie": 0.158834,
    "cheap-red": -0.166544,
    "cheap-white": -0.152297,
    "cheesy-phone-game": 0.257694,
    "christmas-name-draw": 0.667510,
    "chuck-a-frisbee": -0.652299,
    "cities-and-knights": 0.075735,
    "civilization-the-boardgame": -0.024953,
    "classic-book": 0.247484,
    "cliff-climbing": -0.045467,
    "colouring": -0.813336,
    "couch-tube": 0.358086,
    "dates-first-nos": -1.087655,
    "decaf": 0.474479,
    "dishwashing": -0.327700,
    "dock-fishing": 0.180803,
    "dominion": -0.156430,
    "doritos": 0.465300,
    "doxology": 0.081522,
    "drop-a-ski": 0.186812,
    "dumbed-down-settlers": 0.295273,
    "early-bedtime": -0.702800,
    "epic-duels": 0.155381,
    "epic-playlist": 0.152797,
    "epic-prank": -0.174874,
    "euchre": 0.001861,
    "euchre-awards-ceremony": -0.665218,
    "euchre-tournament": -0.363723,
    "evening-chat": 0.552937,
    "evening-on-the-dock": 0.580206,
    "fajitas": 0.170888,
    "family-baseball-game": -0.025568,
    "fancy-craft": -0.671844,
    "fancy-floatie": 0.211671,
    "fishing-boat": 0.317440,
    "fishing-evening": 0.026438,
    "fishing-morning": -0.350338,
    "fit-to-print": -0.511966,
    "floating": 0.017887,
    "forced-family-fun": 0.389975,
    "grocery-store-run": -0.465855,
    "group-dinner": 1.274692,
    "high-end-red": -0.423862,
    "high-end-white": -0.153815,
    "hold-the-baby": 0.198603,
    "ice-cream-sandwich": 0.787094,
    "ice-wine": -0.451407,
    "istanbul": 0.181347,
    "johnny-appleseed": -0.194783,
    "kayak": 0.023115,
    "keeper": 0.305646,
    "kneeboard": -0.275781,
    "last-years-shirt": 1.838252,
    "leftovers": 0.031120,
    "long-distance-visitors": -0.118030,
    "m-ms": -0.825724,
    "make-plans": -0.182007,
    "medical-advice": -0.165715,
    "morning-bike": -0.448993,
    "morning-coffee": -0.488909,
    "morning-run": -1.319455,
    "morning-walk": -0.844371,
    "movie": 0.367864,
    "nap": -0.247307,
    "navy-sealing": 1.093665,
    "new-nos-book-entry": 0.012758,
    "newfangled-tres-fute": 0.511586,
    "non-fiction-book": 0.003836,
    "nos-book": 0.822852,
    "nos-shirt": -0.030720,
    "outdoor-movie": 0.516912,
    "paddleboard": -0.055185,
    "paddleboat": -0.619066,
    "paint": -0.017621,
    "paint-rocks": -0.111405,
    "photo-shoot": -0.194591,
    "play-with-the-kids": -0.413593,
    "poker": -0.100519,
    "ponyback": -0.286219,
    "potato-pancakes": 0.268960,
    "prime-picnic-table": 0.419999,
    "pudding-chomeur": -0.060819,
    "puerto-rico": -0.059233,
    "quacks": 0.388054,
    "read-the-radar": -0.033005,
    "risk": -0.233915,
    "rouladen": 1.325888,
    "san-juan": 0.585364,
    "schwank": 0.208598,
    "scout-other-cottages": -0.747114,
    "scrabble": 0.054752,
    "screamer-battle": -0.005739,
    "shady-spot": -0.278368,
    "sing-song": 0.253669,
    "ski-boat": 0.196335,
    "ski-on-shoulders": -0.887746,
    "sleep-in": -0.978253,
    "solo": -0.529884,
    "spa-day": 0.018718,
    "spikeball": 0.417895,
    "splendor": 0.142196,
    "stay-up-late": -0.335456,
    "steak": 0.103285,
    "stretch": 0.077061,
    "sunrise": -0.145782,
    "sunscreen": 0.461499,
    "sweet-lawn-chair": -0.073439,
    "swimming": 0.106373,
    "tanning": 0.272445,
    "teach-kid-to-ski": 0.018196,
    "tell-a-story": -0.403809,
    "terra-mystica": -0.393198,
    "the-crew": -0.059281,
    "thriller-book": -0.345656,
    "throw-a-baseball": -0.205823,
    "treat-cereal": 0.432468,
    "treat-store-run": -0.361085,
    "trekking-through-history": -0.369084,
    "tres-fute": -0.512304,
    "wakeboard": 0.614707,
    "wakesurf": 0.448734,
    "water-trampoline": 0.080896,
    "water-volleyball": -0.344141,
    "waterski": -0.247631,
    "wedding-anniversary": -0.922552,
    "weird-chip-flavour": -0.454933,
    "wildlife-spotting": -0.200303,
    "wingspan": -0.054219,
    "work-call": 0.276484,
    "wrestle-the-kids": -0.574106,
    "wurstchen": 0.921519,
    "zero-gravity-chair": 0.479730,
    "zumba": -0.231079,
}

_OSMIUM_RESERVE_CACHE_MAX = 262144
_OSMIUM_LARGE_RESERVE_CACHE_MAX = 65536
_OSMIUM_RESERVE_EVAL_CACHE: dict[tuple | int, float] = {}
_OSMIUM_LARGE_RESERVE_CACHE: dict[tuple | int, float] = {}
_OSMIUM_SIM_GAME_CLS_CACHED: type | None = None


@dataclass(frozen=True, slots=True)
class _OsmiumCardSpec:
    defn: CardDefinition
    slug: str
    slug_rank: int
    title_id: int
    cost: int
    bf_int: int
    bf_float: float
    bf_pos: bool
    net_e_pos: float
    is_scaler: bool
    is_multi_scaler: bool
    smb: float
    scaler_cluster_idxs: tuple[int, ...]
    cluster_idxs: tuple[int, ...]
    packed_a: int
    packed_b: int
    bool_flags: int
    w_slug: float
    w_slug_d1: float
    tom_e: int
    tom_f: int
    tom_all_tags: bool
    tom_kind: int
    multiset_bit: int


_OSMIUM_SPEC_BY_SLUG: dict[str, _OsmiumCardSpec] = {}
_OSMIUM_RANK_BY_SLUG: dict[str, int] = {}
_OSMIUM_MSET_BIT_BY_SLUG: dict[str, int] = {}
_OSMIUM_TITLE_ID_BY_SLUG: dict[str, int] = {}
_OSMIUM_TITLE_MSET_BIT_BY_SLUG: dict[str, int] = {}
_SLUG_RANK_GETTER = attrgetter("slug_rank")


def _ensure_osmium_caches() -> None:
    global _OSMIUM_SIM_GAME_CLS_CACHED
    if _OSMIUM_SPEC_BY_SLUG and _OSMIUM_SIM_GAME_CLS_CACHED is not None:
        return
    _ensure_hydrogen_caches()
    _OSMIUM_SIM_GAME_CLS_CACHED = _get_hydrogen_sim_game_cls()
    if _OSMIUM_SPEC_BY_SLUG:
        return
    metas = sorted(_HYDROGEN_META_BY_SLUG.values(), key=lambda m: m.slug)
    unique_titles = sorted({m.title for m in metas})
    assert len(unique_titles) < 256 and len(metas) < 512
    title_to_id = {t: i + 1 for i, t in enumerate(unique_titles)}
    for m_item in metas:
        tid = title_to_id[m_item.title]
        _OSMIUM_TITLE_ID_BY_SLUG[m_item.slug] = tid
        _OSMIUM_TITLE_MSET_BIT_BY_SLUG[m_item.slug] = 1 << (5 * (tid - 1))
    for rank_i, meta in enumerate(metas, start=1):
        if not meta.tags_canonical or not meta.target_canonical:
            continue
        defn = meta.defn
        title = meta.title
        cost = defn.cost if defn.cost > 0 else 0
        bf_int = defn.base_fun + meta.immediate_fun_gain
        bf_float = float(bf_int)
        if meta.immediate_energy_gain > cost and title != "Sleep In":
            net_e = float(meta.immediate_energy_gain - cost)
        elif title == "Butter Tarts":
            net_e = 3.0
        elif title == "Potato Pancakes" or title == "Istanbul":
            net_e = 2.0
        elif title == "Shady Spot":
            net_e = 2.5
        else:
            net_e = 0.0
        is_scaler = (
            meta.is_multi_scaler or meta.is_global_after_scaler
        ) and meta.signed_match_benefit > 0.0
        scaler_c_idxs = tuple(
            _CLUSTER_TAG_TO_IDX[tg]
            for tg in meta.sorted_target_tags
            if tg in _CLUSTER_TAG_TO_IDX
        )
        c_idxs = tuple(
            _CLUSTER_TAG_TO_IDX[tg]
            for tg in defn.tags
            if tg in _CLUSTER_TAG_TO_IDX
        )
        pa = 0
        for ti in c_idxs:
            pa |= 1 << (6 * ti)
        pb = 0
        if cost == 0:
            pb |= 1 << 0
        if cost <= 1:
            pb |= 1 << 5
        elif cost <= 4:
            pb |= 1 << 10
        else:
            pb |= 1 << 15
            if cost >= 8:
                pb |= 1 << 20
        if meta.is_acquirer:
            pb |= 1 << 25
        if title in _HYDROGEN_COPY_FOOD_TITLES:
            pb |= 1 << 30
        if "Food" in defn.tags and cost >= 2 and title != "Leftovers":
            pb |= 1 << 35
        if title in _TRUNK_REORDER_TITLES:
            pb |= 1 << 40
        if title == "Wakesurf" or title == "Drop a Ski":
            pb |= 1 << 45
        if title == "Wrestle the Kids" or title == "Play With the Kids":
            pb |= 1 << 50
        if title == "Epic Duels":
            pb |= 1 << 55
        bf_flags = 0
        if title == "Colouring":
            bf_flags |= 1
        elif title == "Paddleboat":
            bf_flags |= 2
        elif title == "Leftovers":
            bf_flags |= 4
        elif title == "Assorted Cutlery":
            bf_flags |= 8
        elif title == "Adventure Race":
            bf_flags |= 16
        elif title == "Civilization the Boardgame":
            bf_flags |= 32
        if title == "Photo Shoot":
            t_kind = 1
        elif title == "Date's First Nos":
            t_kind = 2
        elif title == "Wildlife Spotting":
            t_kind = 3
        else:
            t_kind = 0
        _OSMIUM_SPEC_BY_SLUG[defn.slug] = _OsmiumCardSpec(
            defn=defn,
            slug=defn.slug,
            slug_rank=rank_i,
            title_id=title_to_id[title],
            cost=cost,
            bf_int=bf_int,
            bf_float=bf_float,
            bf_pos=bf_int > 0,
            net_e_pos=net_e,
            is_scaler=is_scaler,
            is_multi_scaler=meta.is_multi_scaler,
            smb=meta.signed_match_benefit,
            scaler_cluster_idxs=scaler_c_idxs,
            cluster_idxs=c_idxs,
            packed_a=pa,
            packed_b=pb,
            bool_flags=bf_flags,
            w_slug=_OSMIUM_SLUG_WEIGHTS.get(defn.slug, 0.0),
            w_slug_d1=_OSMIUM_SLUG_D1_WEIGHTS.get(defn.slug, 0.0),
            tom_e=meta.tomorrow_start_energy,
            tom_f=meta.tomorrow_start_fun,
            tom_all_tags=meta.tomorrow_all_tags,
            tom_kind=t_kind,
            multiset_bit=1 << (14 + 5 * (rank_i - 1)),
        )
        _OSMIUM_RANK_BY_SLUG[defn.slug] = rank_i
        _OSMIUM_MSET_BIT_BY_SLUG[defn.slug] = 1 << (14 + 5 * (rank_i - 1))


def _osmium_reserve_score_plain_specs(
    sorted_specs: Sequence[_OsmiumCardSpec],
    tom_specs: Sequence[_OsmiumCardSpec] | None,
    days_left: int,
) -> float:
    """Bit-exact fast path for `_osmium_reserve_score` on plain canonical cards with no character."""
    w = _OSMIUM_BASE_WEIGHTS
    val = 0.0
    tom_e = 0
    tom_photo = 0.0
    tom_dates = 0.0
    has_tom_bonuses = False
    tom_tag_arr: list[float] | None = None

    if tom_specs:
        tom_f = 0
        tom_all_tags = 0.0
        tom_wild = 0.0
        tom_pos_smb = 0.0
        tom_neg_smb = 0.0
        for ts in tom_specs:
            tom_e += ts.tom_e
            tom_f += ts.tom_f
            if ts.tom_all_tags:
                tom_all_tags = 1.0
            tk = ts.tom_kind
            if tk == 1:
                tom_photo += 1.0
            elif tk == 2:
                tom_dates += 1.0
            elif tk == 3:
                tom_wild += 1.0
            smb = ts.smb
            if smb > 0.0:
                tom_pos_smb += smb
                if ts.scaler_cluster_idxs:
                    if tom_tag_arr is None:
                        tom_tag_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        has_tom_bonuses = True
                    for ti in ts.scaler_cluster_idxs:
                        tom_tag_arr[ti] += smb
            elif smb < 0.0:
                tom_neg_smb += smb
                if ts.scaler_cluster_idxs:
                    if tom_tag_arr is None:
                        tom_tag_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        has_tom_bonuses = True
                    for ti in ts.scaler_cluster_idxs:
                        tom_tag_arr[ti] += smb
        if tom_e:
            val += w[5] * float(tom_e)
            if tom_e > 2:
                val += w[6] * float(tom_e - 2)
        if tom_f:
            val += w[7] * float(tom_f)
        if tom_all_tags:
            val += w[8] * tom_all_tags
        if tom_photo:
            val += w[9] * tom_photo
        if tom_dates:
            val += w[10] * tom_dates
        if tom_wild:
            val += w[11] * tom_wild
        if tom_pos_smb:
            val += w[12] * tom_pos_smb
        if tom_neg_smb:
            val += w[13] * tom_neg_smb

    eff_h_len = len(sorted_specs)
    acc_a = 0
    acc_b = 0
    b_flags = 0
    total_cost = 0
    net_e_pos = 0.0
    scaler_power = 0.0
    scaler_cnt = 0
    bf_pos_cnt = 0
    bf_pos_sum_int = 0
    hand_scaler_by_tag: list[float] | None = None
    tag_tom_match: list[float] | None = (
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if has_tom_bonuses else None
    )

    for sp in sorted_specs:
        acc_a += sp.packed_a
        acc_b += sp.packed_b
        b_flags |= sp.bool_flags
        total_cost += sp.cost
        net_e_pos += sp.net_e_pos
        if sp.bf_pos:
            bf_pos_cnt += 1
            bf_pos_sum_int += sp.bf_int
        if sp.is_scaler:
            scaler_power += sp.smb
            scaler_cnt += 1
            if sp.scaler_cluster_idxs:
                if hand_scaler_by_tag is None:
                    hand_scaler_by_tag = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                s_smb = sp.smb
                for ti in sp.scaler_cluster_idxs:
                    hand_scaler_by_tag[ti] += s_smb
        if has_tom_bonuses:
            for ti in sp.cluster_idxs:
                tb = tom_tag_arr[ti]  # type: ignore[index]
                if tb != 0.0:
                    tag_tom_match[ti] += tb  # type: ignore[index]

    if eff_h_len >= 1:
        val += w[16]
        if eff_h_len >= 2:
            val += w[17]
            if eff_h_len >= 3:
                val += w[18]
                if eff_h_len >= 4:
                    val += w[19]
                    if eff_h_len >= 5:
                        val += w[20]
                        if eff_h_len >= 6:
                            val += w[21]
                            if eff_h_len > 6:
                                c_ex = eff_h_len - 6
                                val += w[22] * float(c_ex if c_ex < 3 else 3)

    tag_hand_scaler_match: list[float] | None = None
    if hand_scaler_by_tag is not None:
        tag_hand_scaler_match = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for sp in sorted_specs:
            if not sp.is_multi_scaler:
                for ti in sp.cluster_idxs:
                    hs_v = hand_scaler_by_tag[ti]
                    if hs_v > 0.0:
                        tag_hand_scaler_match[ti] += hs_v

    avail_e_d6 = 7.0 + (float(tom_e) if tom_e > 0 else 0.0) + net_e_pos
    total_cost_f = float(total_cost)
    if days_left == 1:
        if eff_h_len > 3:
            val += w[23] * float(eff_h_len - 3)
        if eff_h_len > 5:
            val += w[24] * float(eff_h_len - 5)
        used_e = total_cost_f if total_cost_f < avail_e_d6 else avail_e_d6
        excess_cost = total_cost_f - avail_e_d6
        if used_e > 0.0:
            val += w[25] * used_e
        if excess_cost > 0.0:
            val += w[26] * excess_cost
        if tom_photo:
            val += w[27] * tom_photo
        if tom_dates:
            val += w[28] * tom_dates
        if tom_e:
            val += w[29] * float(tom_e)
    elif days_left == 2 and eff_h_len > 5:
        val += w[30] * float(eff_h_len - 5)

    if bf_pos_cnt:
        if bf_pos_cnt <= 4:
            val += w[31] * float(bf_pos_sum_int)
        else:
            bf_all = [sp.bf_float for sp in sorted_specs if sp.bf_pos]
            bf_all.sort(reverse=True)
            val += w[31] * (bf_all[0] + bf_all[1] + bf_all[2] + bf_all[3])

    zero_cost_cnt = acc_b & 0x1F
    cheap_cnt = (acc_b >> 5) & 0x1F
    mid_cnt = (acc_b >> 10) & 0x1F
    heavy_cnt = (acc_b >> 15) & 0x1F
    unplay_heavy_cnt = (acc_b >> 20) & 0x1F
    if zero_cost_cnt:
        val += w[32] * float(zero_cost_cnt)
    if cheap_cnt:
        val += w[33] * float(cheap_cnt)
    if mid_cnt:
        val += w[34] * float(mid_cnt)
    if heavy_cnt:
        val += w[35] * float(heavy_cnt)
    if unplay_heavy_cnt:
        val += w[36] * float(unplay_heavy_cnt)
    if net_e_pos:
        val += w[37] * net_e_pos
    if scaler_power:
        val += w[38] * scaler_power

    max_tag_cnt = 0
    if acc_a or tag_hand_scaler_match is not None or tag_tom_match is not None:
        for ti in range(7):
            cnt = (acc_a >> (6 * ti)) & 0x3F
            if cnt > max_tag_cnt:
                max_tag_cnt = cnt
            base_idx = 39 + 4 * ti
            if cnt >= 2:
                val += w[base_idx]
                if cnt >= 3:
                    val += w[base_idx + 1]
            if tag_hand_scaler_match is not None:
                hm = tag_hand_scaler_match[ti]
                if hm != 0.0:
                    val += w[base_idx + 2] * hm
            if tag_tom_match is not None:
                tm = tag_tom_match[ti]
                if tm != 0.0:
                    val += w[base_idx + 3] * tm
        if max_tag_cnt:
            val += w[67] * float(max_tag_cnt)

    combo_hi = acc_b >> 25
    if b_flags or combo_hi or tom_e > 0 or tom_dates > 0.0:
        acquirer_cnt = combo_hi & 0x1F
        copy_food_cnt = (combo_hi >> 5) & 0x1F
        high_food_cnt = (combo_hi >> 10) & 0x1F
        trunk_setter_cnt = (combo_hi >> 15) & 0x1F
        empty_hand_cnt = (combo_hi >> 20) & 0x1F
        wrestle_pwtk_cnt = (combo_hi >> 25) & 0x1F
        epic_duels_cnt = (combo_hi >> 30) & 0x1F

        if tom_e > 0 and heavy_cnt > 0:
            val += w[68] * float(tom_e * (heavy_cnt if heavy_cnt < 2 else 2))
        if tom_dates > 0.0 and (cheap_cnt > 0 or net_e_pos > 0.0):
            c_plus_e = float(cheap_cnt) + net_e_pos
            val += w[69] * float(tom_dates * (c_plus_e if c_plus_e < 5.0 else 5.0))
        if (b_flags & 1) and scaler_cnt > 0:
            val += w[70] * float(scaler_cnt if scaler_cnt < 3 else 3)
        if (b_flags & 2) and (acquirer_cnt + tom_photo) > 0:
            acq_p = float(acquirer_cnt) + tom_photo
            val += w[71] * float(acq_p if acq_p < 3.0 else 3.0)
        if (b_flags & 4) and copy_food_cnt > 0:
            val += w[72] * float(copy_food_cnt if copy_food_cnt < 2 else 2)
        if (b_flags & 8) and trunk_setter_cnt > 0:
            val += w[73] * float(trunk_setter_cnt if trunk_setter_cnt < 2 else 2)
        if b_flags & 48:
            e_sup = (float(tom_e) if tom_e > 0 else 0.0) + net_e_pos
            if e_sup > 0.0:
                e_cap = e_sup if e_sup < 6.0 else 6.0
                if b_flags & 16:
                    val += w[74] * e_cap
                if b_flags & 32:
                    val += w[75] * e_cap
        if empty_hand_cnt > 0 and eff_h_len < 3:
            val += w[76] * float(empty_hand_cnt * (3 - eff_h_len))
        if wrestle_pwtk_cnt > 0:
            val += w[77] * float(
                wrestle_pwtk_cnt * (eff_h_len if eff_h_len < 5 else 5)
            )
        if epic_duels_cnt > 0 and cheap_cnt > 0:
            val += w[78] * float(
                epic_duels_cnt * (cheap_cnt if cheap_cnt < 4 else 4)
            )
        if acquirer_cnt > 0:
            val += w[79] * float(
                acquirer_cnt * (days_left if days_left < 3 else 3)
            )
        if copy_food_cnt > 0 and high_food_cnt > 0:
            val += w[80] * float(
                copy_food_cnt * (high_food_cnt if high_food_cnt < 3 else 3)
            )

    if eff_h_len:
        is_d1 = days_left == 1
        i_s = 0
        while i_s < eff_h_len:
            sp_i = sorted_specs[i_s]
            rk = sp_i.slug_rank
            w_s = 1.0
            i_s += 1
            while i_s < eff_h_len and sorted_specs[i_s].slug_rank == rk:
                w_s += 0.5
                i_s += 1
            val += sp_i.w_slug * w_s
            if is_d1:
                val += sp_i.w_slug_d1 * w_s

    return val


def _suitcase_pick1_edge(sim: Game, player_index: int) -> float:
    """Measure best-versus-later quality gap across the visible Suitcase."""
    sc = sim.suitcase
    if not sc:
        return 0.0
    player = sim.players[player_index]
    is_hsim = type(sim) is _OSMIUM_SIM_GAME_CLS_CACHED
    wanted, ctx = _build_wanted_and_ctx(
        player, is_hsim and sim._all_canonical_defs  # type: ignore[attr-defined]
    )
    costs = (
        sim.cards_energy_costs(player_index, sc)  # type: ignore[attr-defined]
        if is_hsim
        else None
    )
    vals = sorted(
        (
            _hydrogen_quick_card_desirability(
                sim,
                player_index,
                sc[i],
                wanted,
                cost=costs[i] if costs is not None else None,
                _ctx=ctx,
                _in_hand=False,
            )
            for i in range(len(sc))
        ),
        reverse=True,
    )
    if len(vals) >= 4:
        later = 0.50 * vals[1] + 0.32 * vals[2] + 0.18 * vals[3]
    elif len(vals) >= 2:
        later = sum(vals[1:]) / (len(vals) - 1)
    else:
        later = vals[0]
    return max(0.0, vals[0] - later)


def _osmium_reserve_score(
    sim: Game,
    player_index: int,
    next_tomorrow_cards: Sequence[CardInstance],
    days_left: int,
) -> float:
    """Compute OsmiumAI's linear reserve score directly without allocating dense arrays."""
    _ensure_hydrogen_caches()
    w = _OSMIUM_BASE_WEIGHTS
    val = 0.0
    player = sim.players[player_index]
    hand = player.hand
    char = player.character
    zero_tag = char.zero_fun_tag if char is not None else ""

    tom_e = 0
    tom_f = 0
    tom_all_tags = 0.0
    tom_photo = 0.0
    tom_dates = 0.0
    tom_wild = 0.0
    tom_pos_smb = 0.0
    tom_neg_smb = 0.0
    tom_tag_bonuses: dict[str, float] = {}

    for tc in next_tomorrow_cards:
        t_meta = _get_card_meta(tc)
        t_title = _effective_card_title(tc)
        tom_e += t_meta.tomorrow_start_energy
        tom_f += t_meta.tomorrow_start_fun
        if t_meta.tomorrow_all_tags:
            tom_all_tags = 1.0
        if t_title == "Photo Shoot":
            tom_photo += 1.0
        elif t_title == "Date's First Nos":
            tom_dates += 1.0
        elif t_title == "Wildlife Spotting":
            tom_wild += 1.0
        smb = t_meta.signed_match_benefit
        if smb > 0.0:
            tom_pos_smb += smb
            for tg in t_meta.sorted_target_tags:
                tom_tag_bonuses[tg] = tom_tag_bonuses.get(tg, 0.0) + smb
        elif smb < 0.0:
            tom_neg_smb += smb
            for tg in t_meta.sorted_target_tags:
                tom_tag_bonuses[tg] = tom_tag_bonuses.get(tg, 0.0) + smb
        chosen_t = tc.markers.get("chosen_tag") if tc.markers else None
        if isinstance(chosen_t, str):
            tom_pos_smb += 1.2
            tom_tag_bonuses[chosen_t] = tom_tag_bonuses.get(chosen_t, 0.0) + 1.2

    all_cards_zeroed = bool(tom_all_tags and zero_tag)
    if tom_e:
        val += w[5] * float(tom_e)
        if tom_e > 2:
            val += w[6] * float(tom_e - 2)
    if tom_f:
        val += w[7] * float(tom_f)
    if tom_all_tags:
        val += w[8] * tom_all_tags
        if all_cards_zeroed:
            # Tomorrow all-tag effects make every played card trigger the zeroed tag.
            val -= 18.0
    if tom_photo:
        val += w[9] * tom_photo
    if tom_dates:
        val += w[10] * tom_dates
    if tom_wild:
        val += w[11] * tom_wild
    if tom_pos_smb and not all_cards_zeroed:
        val += w[12] * tom_pos_smb
    if tom_neg_smb:
        val += w[13] * tom_neg_smb

    eff_h_len = 0
    base_funs: list[float] = []
    total_cost = 0
    zero_cost_cnt = 0
    cheap_cnt = 0
    mid_cnt = 0
    heavy_cnt = 0
    unplay_heavy_cnt = 0
    net_e_pos = 0.0
    scaler_power = 0.0
    scaler_cnt = 0

    has_colouring = False
    has_paddleboat = False
    acquirer_cnt = 0
    has_leftovers = False
    copy_food_cnt = 0
    high_food_cnt = 0
    has_cutlery = False
    trunk_setter_cnt = 0
    has_adv_race = False
    has_civ = False
    empty_hand_cnt = 0
    wrestle_pwtk_cnt = 0
    epic_duels_cnt = 0

    tag_cnts = [0] * 7
    hand_scaler_by_tag = [0.0] * 7
    tag_hand_scaler_match = [0.0] * 7
    tag_tom_match = [0.0] * 7
    slug_seen: dict[str, float] = {}
    hand_meta_tags: list[tuple[object, frozenset[str]]] = []

    for hc in hand:
        meta = _get_card_meta(hc)
        ctags = hc.tags if hc.markers else meta.tags
        cost = max(0, hc.effective_cost)
        is_zeroed = all_cards_zeroed or bool(
            zero_tag and (zero_tag in ctags or meta.after_all_tags)
        )
        title = meta.title

        if is_zeroed:
            if meta.immediate_energy_gain > cost and title != "Sleep In":
                net_e_pos += float(meta.immediate_energy_gain - cost)
            elif title == "Butter Tarts":
                net_e_pos += 3.0
            elif title in {"Potato Pancakes", "Istanbul"}:
                net_e_pos += 2.0
            elif title == "Shady Spot":
                net_e_pos += 2.5
            continue

        eff_h_len += 1
        hand_meta_tags.append((meta, ctags))
        bf = float(hc.effective_base_fun + meta.immediate_fun_gain)
        total_cost += cost
        if bf > 0.0:
            base_funs.append(bf)
        if cost == 0:
            zero_cost_cnt += 1
        if cost <= 1:
            cheap_cnt += 1
        elif cost <= 4:
            mid_cnt += 1
        else:
            heavy_cnt += 1
            if cost >= 8:
                unplay_heavy_cnt += 1

        if meta.immediate_energy_gain > cost and title != "Sleep In":
            net_e_pos += float(meta.immediate_energy_gain - cost)
        elif title == "Butter Tarts":
            net_e_pos += 3.0
        elif title in {"Potato Pancakes", "Istanbul"}:
            net_e_pos += 2.0
        elif title == "Shady Spot":
            net_e_pos += 2.5

        if (
            meta.is_multi_scaler or meta.is_global_after_scaler
        ) and meta.signed_match_benefit > 0.0:
            scaler_power += meta.signed_match_benefit
            scaler_cnt += 1
            for tg in meta.sorted_target_tags:
                ti = _CLUSTER_TAG_TO_IDX.get(tg)
                if ti is not None:
                    hand_scaler_by_tag[ti] += meta.signed_match_benefit

        for tg in ctags:
            ti = _CLUSTER_TAG_TO_IDX.get(tg)
            if ti is not None:
                tag_cnts[ti] += 1
                tb = tom_tag_bonuses.get(tg, 0.0)
                if tb != 0.0:
                    tag_tom_match[ti] += tb

        if title == "Colouring":
            has_colouring = True
        elif title == "Paddleboat":
            has_paddleboat = True
        elif title == "Leftovers":
            has_leftovers = True
        elif title == "Assorted Cutlery":
            has_cutlery = True
        elif title == "Adventure Race":
            has_adv_race = True
        elif title == "Civilization the Boardgame":
            has_civ = True
        elif title == "Epic Duels":
            epic_duels_cnt += 1

        if meta.is_acquirer:
            acquirer_cnt += 1
        if title in _HYDROGEN_COPY_FOOD_TITLES:
            copy_food_cnt += 1
        if "Food" in ctags and cost >= 2 and title != "Leftovers":
            high_food_cnt += 1
        if title in _TRUNK_REORDER_TITLES:
            trunk_setter_cnt += 1
        if title in {"Wakesurf", "Drop a Ski"}:
            empty_hand_cnt += 1
        if title in {"Wrestle the Kids", "Play With the Kids"}:
            wrestle_pwtk_cnt += 1

        slug = hc.definition.slug
        prev_w = slug_seen.get(slug, 0.0)
        slug_seen[slug] = prev_w + (1.0 if prev_w == 0.0 else 0.5)

    for k in range(1, 7):
        if eff_h_len >= k:
            val += w[15 + k]
    if eff_h_len > 6:
        val += w[22] * float(min(eff_h_len - 6, 3))

    for meta, ctags in hand_meta_tags:
        if meta.is_multi_scaler:
            continue
        for tg in ctags:
            ti = _CLUSTER_TAG_TO_IDX.get(tg)
            if ti is not None and hand_scaler_by_tag[ti] > 0.0:
                tag_hand_scaler_match[ti] += hand_scaler_by_tag[ti]

    avail_e_d6 = 7.0 + max(0.0, float(tom_e)) + net_e_pos
    total_cost_f = float(total_cost)
    if days_left == 1:
        if eff_h_len > 3:
            val += w[23] * float(eff_h_len - 3)
        if eff_h_len > 5:
            val += w[24] * float(eff_h_len - 5)
        used_e = min(total_cost_f, avail_e_d6)
        excess_cost = max(0.0, total_cost_f - avail_e_d6)
        if used_e > 0.0:
            val += w[25] * used_e
        if excess_cost > 0.0:
            val += w[26] * excess_cost
        if tom_photo:
            val += w[27] * tom_photo
        if tom_dates:
            val += w[28] * tom_dates
        if tom_e:
            val += w[29] * float(tom_e)
    elif days_left == 2 and eff_h_len > 5:
        val += w[30] * float(eff_h_len - 5)

    if base_funs:
        base_funs.sort(reverse=True)
        val += w[31] * sum(base_funs[:4])
    if zero_cost_cnt:
        val += w[32] * float(zero_cost_cnt)
    if cheap_cnt:
        val += w[33] * float(cheap_cnt)
    if mid_cnt:
        val += w[34] * float(mid_cnt)
    if heavy_cnt:
        val += w[35] * float(heavy_cnt)
    if unplay_heavy_cnt:
        val += w[36] * float(unplay_heavy_cnt)
    if net_e_pos:
        val += w[37] * net_e_pos
    if scaler_power:
        val += w[38] * scaler_power

    max_tag_cnt = 0
    for ti in range(7):
        cnt = tag_cnts[ti]
        if cnt > max_tag_cnt:
            max_tag_cnt = cnt
        base_idx = 39 + 4 * ti
        if cnt >= 2:
            val += w[base_idx]
        if cnt >= 3:
            val += w[base_idx + 1]
        hm = tag_hand_scaler_match[ti]
        tm = tag_tom_match[ti]
        if hm != 0.0:
            val += w[base_idx + 2] * hm
        if tm != 0.0:
            val += w[base_idx + 3] * tm
    if max_tag_cnt:
        val += w[67] * float(max_tag_cnt)

    if tom_e > 0 and heavy_cnt > 0:
        val += w[68] * float(tom_e * min(heavy_cnt, 2))
    if tom_dates > 0 and (cheap_cnt > 0 or net_e_pos > 0.0):
        val += w[69] * float(tom_dates * min(float(cheap_cnt) + net_e_pos, 5.0))
    if has_colouring and scaler_cnt > 0:
        val += w[70] * float(min(scaler_cnt, 3))
    if has_paddleboat and (acquirer_cnt + tom_photo) > 0:
        val += w[71] * float(min(float(acquirer_cnt) + tom_photo, 3.0))
    if has_leftovers and copy_food_cnt > 0:
        val += w[72] * float(min(copy_food_cnt, 2))
    if has_cutlery and trunk_setter_cnt > 0:
        val += w[73] * float(min(trunk_setter_cnt, 2))
    e_sup = max(0.0, float(tom_e)) + net_e_pos
    if has_adv_race and e_sup > 0.0:
        val += w[74] * min(e_sup, 6.0)
    if has_civ and e_sup > 0.0:
        val += w[75] * min(e_sup, 6.0)
    if empty_hand_cnt > 0:
        val += w[76] * float(empty_hand_cnt * max(0, 3 - len(hand)))
    if wrestle_pwtk_cnt > 0:
        val += w[77] * float(wrestle_pwtk_cnt * min(len(hand), 5))
    if epic_duels_cnt > 0 and cheap_cnt > 0:
        val += w[78] * float(epic_duels_cnt * min(cheap_cnt, 4))
    if acquirer_cnt > 0:
        val += w[79] * float(acquirer_cnt * min(days_left, 3))
    if copy_food_cnt > 0 and high_food_cnt > 0:
        val += w[80] * float(copy_food_cnt * min(high_food_cnt, 3))

    is_d1 = days_left == 1
    slug_weights = _OSMIUM_SLUG_WEIGHTS
    slug_d1_weights = _OSMIUM_SLUG_D1_WEIGHTS
    for slug in sorted(slug_seen):
        w_slug = slug_seen[slug]
        val += slug_weights.get(slug, 0.0) * w_slug
        if is_d1:
            val += slug_d1_weights.get(slug, 0.0) * w_slug

    return val


def _colouring_tag_score(
    tags: frozenset[str],
    wanted_tags: dict[str, float],
    cond: object,
    char: object,
) -> tuple[float, int]:
    """Deterministic tie-break / ranking heuristic for a Colouring target's tag set."""
    score = sum(wanted_tags.get(t, 0.0) for t in sorted(tags))
    if cond is not None:
        if getattr(cond, "fun_tag", "") in tags:
            score += float(getattr(cond, "fun_delta", 0)) * 0.8
        if getattr(cond, "hot_day", False):
            if "Outdoors" in tags:
                score -= 0.8
            if "Indoors" in tags:
                score += 0.8
    if char is not None:
        fun_tag = getattr(char, "fun_tag", "")
        if fun_tag and fun_tag in tags:
            score += float(getattr(char, "fun_delta", 0)) * 0.8
        energy_tag = getattr(char, "energy_tag", "")
        if energy_tag and energy_tag in tags:
            score -= float(getattr(char, "energy_delta", 0)) * 0.8
    return (score, len(tags))


def _colouring_candidate_indices(
    options: Sequence[CardInstance],
    zero_tag: str | None,
) -> list[int]:
    """Select the highest-value representative index per distinct Colouring tag set."""
    non_zero_indices = [
        i
        for i, c in enumerate(options)
        if c.tags and not (zero_tag and zero_tag in c.tags)
    ]
    cand_pool = non_zero_indices or [
        i
        for i, c in enumerate(options)
        if not (zero_tag and zero_tag in c.tags)
    ]
    if not cand_pool:
        cand_pool = list(range(len(options)))
    seen_tag_sets: dict[frozenset[str], tuple[tuple[int, int, int], int]] = {}
    for i in cand_pool:
        card = options[i]
        tset = card.tags
        # Prefer higher-value targets so later Bohnanza discards won't strip the mark.
        rep_key = (card.effective_base_fun, card.definition.cost, -i)
        prev = seen_tag_sets.get(tset)
        if prev is None or rep_key > prev[0]:
            seen_tag_sets[tset] = (rep_key, i)
    return [item[1] for item in seen_tag_sets.values()]


class _OsmiumOverrideEvalAI(_HydrogenEvalAI):
    """Lightweight sub-decision override wrapper used inside search branching."""

    __slots__ = (
        "spend_override",
        "discard_tuple_override",
        "copy_idx_override",
        "optional_override",
        "target_idx_override",
    )

    def __init__(
        self,
        *,
        spend_override: int | None = None,
        discard_tuple_override: tuple[int, ...] | None = None,
        copy_idx_override: int | None = None,
        optional_override: bool | None = None,
        target_idx_override: int | None = None,
    ) -> None:
        super().__init__()
        self.spend_override = spend_override
        self.discard_tuple_override = discard_tuple_override
        self.copy_idx_override = copy_idx_override
        self.optional_override = optional_override
        self.target_idx_override = target_idx_override

    def choose_energy_to_spend(
        self, game: Game, player_index: int, card: CardInstance, maximum: int
    ) -> int:
        if self.spend_override is not None:
            return max(0, min(maximum, self.spend_override))
        return super().choose_energy_to_spend(game, player_index, card, maximum)

    def choose_cards_to_discard(
        self, game: Game, player_index: int, options: Sequence[CardInstance]
    ) -> Sequence[int]:
        if self.discard_tuple_override is not None and all(
            0 <= i < len(options) for i in self.discard_tuple_override
        ):
            return self.discard_tuple_override
        return super().choose_cards_to_discard(game, player_index, options)

    def choose_card_to_copy(
        self, game: Game, player_index: int, options: Sequence[CardInstance]
    ) -> int:
        if (
            self.copy_idx_override is not None
            and 0 <= self.copy_idx_override < len(options)
        ):
            return self.copy_idx_override
        return super().choose_card_to_copy(game, player_index, options)

    def choose_card_target(
        self, game: Game, player_index: int, options: Sequence[CardInstance]
    ) -> int:
        if (
            self.target_idx_override is not None
            and 0 <= self.target_idx_override < len(options)
        ):
            played = game.players[player_index].played_today
            if played and _effective_card_title(played[-1]) == "Colouring":
                return self.target_idx_override
        return super().choose_card_target(game, player_index, options)

    def choose_optional_action(
        self, game: Game, player_index: int, action: str
    ) -> bool:
        if self.optional_override is not None and (
            "Cities and Knights" in action or "Epic Prank" in action
        ):
            return self.optional_override
        return super().choose_optional_action(game, player_index, action)


class OsmiumAI(HydrogenAI):
    """On-policy DAgger-R4 Huber-IRLS AI with aligned drafting and deep Day-6 search."""

    BEAM_WIDTH = 44
    DAY35_BEAM_WIDTH = 52
    DAY6_BEAM_WIDTH = 96
    DAY6_BRANCH_LIMIT = 20
    DAY6_MAX_PLAYS = 16
    BRANCH_LIMIT = 14
    MAX_TURN_PLAYS = 14
    DRAFT_BEAM_WIDTH = 16
    DAY6_DRAFT_BEAM_WIDTH = 24
    DRAFT_SAMPLES = 3
    _reserve_eval_cache = _OSMIUM_RESERVE_EVAL_CACHE

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)
        _ensure_osmium_caches()
        if type(self) is OsmiumAI:
            self.__dict__.pop("_reserve_eval_cache", None)
        elif "_reserve_eval_cache" not in type(self).__dict__:
            # Keep subclasses isolated from the shared OsmiumAI reserve cache.
            self._reserve_eval_cache = {}

    def _state_signature(self, sim: Game, player_index: int) -> tuple:
        if (
            type(sim) is _OSMIUM_SIM_GAME_CLS_CACHED
            and sim._all_canonical_defs  # type: ignore[attr-defined]
        ):
            tid_by_slug = _OSMIUM_TITLE_ID_BY_SLUG
            mset_tid = _OSMIUM_TITLE_MSET_BIT_BY_SLUG
            p = sim.players[player_index]
            hand = p.hand
            nh = len(hand)
            if nh == 0:
                hand_sig: int | tuple = 0
            elif nh == 1:
                hand_sig = mset_tid[hand[0].definition.slug]
            elif nh == 2:
                hand_sig = (
                    mset_tid[hand[0].definition.slug]
                    + mset_tid[hand[1].definition.slug]
                )
            elif nh == 3:
                hand_sig = (
                    mset_tid[hand[0].definition.slug]
                    + mset_tid[hand[1].definition.slug]
                    + mset_tid[hand[2].definition.slug]
                )
            elif nh == 4:
                hand_sig = (
                    mset_tid[hand[0].definition.slug]
                    + mset_tid[hand[1].definition.slug]
                    + mset_tid[hand[2].definition.slug]
                    + mset_tid[hand[3].definition.slug]
                )
            elif nh <= 30:
                acc_h = 0
                for c_h in hand:
                    acc_h += mset_tid[c_h.definition.slug]
                hand_sig = acc_h
            else:
                h_ids = [tid_by_slug[c_h.definition.slug] for c_h in hand]
                h_ids.sort()
                hand_sig = tuple(h_ids)

            played = p.played_today
            np = len(played)
            if np == 0:
                played_sig: int | tuple = 0
            elif not (sim._p_flags[player_index] & _PF_HAS_SIG_MARKER):  # type: ignore[attr-defined]
                acc_p = 0
                for idx_p in range(np):
                    acc_p |= tid_by_slug[played[idx_p].definition.slug] << (
                        idx_p << 3
                    )
                played_sig = acc_p
            else:
                has_real_sig = False
                played_list: list = []
                for c_p in played:
                    m = c_p.markers
                    if not m or (
                        "_copied_definition" not in m
                        and not m.get("energy_cubes")
                        and not m.get("_ski_on_shoulders_energy_cube")
                        and not m.get("discarded_card")
                        and m.get("discarded_item") is None
                        and m.get("target_card") is None
                        and m.get("target_player_index") is None
                    ):
                        played_list.append(tid_by_slug[c_p.definition.slug])
                    else:
                        has_real_sig = True
                        played_list.append(
                            (
                                tid_by_slug[c_p.definition.slug],
                                _effective_card_title(c_p),
                                m.get("energy_cubes", 0),
                                getattr(m.get("target_card"), "instance_id", None),
                                m.get("target_player_index"),
                                bool(m.get("_ski_on_shoulders_energy_cube")),
                                bool(m.get("discarded_card")),
                                getattr(m.get("discarded_item"), "instance_id", None),
                            )
                        )
                if not has_real_sig:
                    acc_p = 0
                    for idx_p, tid_p in enumerate(played_list):
                        acc_p |= tid_p << (idx_p << 3)
                    played_sig = acc_p
                else:
                    played_sig = tuple(played_list)

            tom = p.tomorrow_cards
            nt = len(tom)
            if nt == 0:
                tom_sig: int | tuple = 0
            elif nt == 1:
                tom_sig = tid_by_slug[tom[0].definition.slug]
            else:
                tom_sig = tuple(tid_by_slug[c_t.definition.slug] for c_t in tom)
            return (p.energy, p.fun, played_sig, hand_sig, tom_sig)
        return super()._state_signature(sim, player_index)

    def _draft_synergy_adjustment(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        wanted_tags: dict[str, float],
    ) -> float:
        return 0.0

    def _ftb_edge_weight(self) -> float:
        return _OSMIUM_BASE_WEIGHTS[15]

    def _suitcase_pick1_bonus(self, sim: Game, player_index: int) -> float:
        return min(
            2.6,
            0.45
            + max(0.0, self._ftb_edge_weight())
            * _suitcase_pick1_edge(sim, player_index),
        )

    def _first_to_bed_value(
        self,
        sim: Game,
        player_index: int,
        *,
        turns_from_root: int = 0,
        root_skipped_turns: int = 0,
        _base_bonus_override: float | None = None,
    ) -> float:
        if sim.day >= DAY_COUNT:
            return 0.0
        players = sim.players
        for idx in range(4):
            if idx != player_index and players[idx].asleep:
                return 0.0
        player = players[player_index]
        delta_skipped = max(0, player.skipped_turns - root_skipped_turns)
        if turns_from_root <= 0:
            opp_rounds = 0
        elif player.asleep:
            opp_rounds = max(0, (turns_from_root - 1) + delta_skipped)
        else:
            opp_rounds = turns_from_root + delta_skipped

        base_bonus = (
            _base_bonus_override
            if _base_bonus_override is not None
            else self._suitcase_pick1_bonus(sim, player_index)
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

    def _evaluate_terminal_day(
        self,
        terminal_sim: Game,
        player_index: int,
        *,
        turns_from_root: int = 1,
        root_skipped_turns: int = 0,
        _ftb_ctx: tuple | None = None,
    ) -> float:
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
                f_dict = _ftb_ctx[5]
                if (
                    len(_ftb_ctx) >= 9
                    and 0 <= turns_from_root < 256
                    and 0 <= player.skipped_turns < 256
                    and 0 <= root_skipped_turns < 256
                ):
                    if terminal_sim.suitcase is _ftb_ctx[6]:
                        sc_ids = _ftb_ctx[7]
                        use_int_k = True
                    else:
                        sc_ids = tuple(
                            c.instance_id for c in terminal_sim.suitcase
                        )
                        use_int_k = sc_ids == _ftb_ctx[7]
                    if use_int_k:
                        fkey: int | tuple = (
                            (root_skipped_turns << 18)
                            | (turns_from_root << 10)
                            | (player.skipped_turns << 2)
                            | (2 if player.asleep else 0)
                            | 1
                        )
                    else:
                        fkey = (
                            turns_from_root,
                            player.asleep,
                            player.skipped_turns,
                            root_skipped_turns,
                            sc_ids,
                        )
                    ftb_val = f_dict.get(fkey)
                    if ftb_val is None:
                        p_hand_ids = tuple(c.instance_id for c in player.hand)
                        sc_bkey = ("b", p_hand_ids, _ftb_ctx[8], sc_ids)
                        base_b = f_dict.get(sc_bkey)
                        if base_b is None:
                            base_b = self._suitcase_pick1_bonus(
                                terminal_sim, player_index
                            )
                            f_dict[sc_bkey] = base_b
                        ftb_val = self._first_to_bed_value(
                            terminal_sim,
                            player_index,
                            turns_from_root=turns_from_root,
                            root_skipped_turns=root_skipped_turns,
                            _base_bonus_override=base_b,
                        )
                        f_dict[fkey] = ftb_val
                    first_to_bed_value = ftb_val
                else:
                    sc_ids = tuple(c.instance_id for c in terminal_sim.suitcase)
                    full_key = (
                        turns_from_root,
                        player.asleep,
                        player.skipped_turns,
                        root_skipped_turns,
                        sc_ids,
                    )
                    ftb_val = f_dict.get(full_key)
                    if ftb_val is None:
                        p_hand_ids = tuple(c.instance_id for c in player.hand)
                        p_tom_ids = (
                            _ftb_ctx[8]
                            if len(_ftb_ctx) >= 9
                            else tuple(
                                c.instance_id for c in player.tomorrow_cards
                            )
                        )
                        sc_bkey = ("b", p_hand_ids, p_tom_ids, sc_ids)
                        base_b = f_dict.get(sc_bkey)
                        if base_b is None:
                            base_b = self._suitcase_pick1_bonus(
                                terminal_sim, player_index
                            )
                            f_dict[sc_bkey] = base_b
                        ftb_val = self._first_to_bed_value(
                            terminal_sim,
                            player_index,
                            turns_from_root=turns_from_root,
                            root_skipped_turns=root_skipped_turns,
                            _base_bonus_override=base_b,
                        )
                        f_dict[full_key] = ftb_val
                    first_to_bed_value = ftb_val
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
        is_hsim = type(terminal_sim) is _OSMIUM_SIM_GAME_CLS_CACHED
        if is_hsim:
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
            if is_hsim:
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

        if (
            eval_sim is terminal_sim
            and type(self)._evaluate_tomorrow_and_reserve_fast
            is not OsmiumAI._evaluate_tomorrow_and_reserve_fast
        ):
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

    def _canonical_reserve_cache_key(
        self,
        sim: Game,
        player_index: int,
        next_tomorrow_cards: Sequence[CardInstance],
        days_left: int,
    ) -> tuple | None:
        player = sim.players[player_index]
        hand = player.hand
        _ensure_hydrogen_caches()
        meta_by_id = _HYDROGEN_META_BY_DEFN_ID
        hand_slugs_list: list[str] = []
        for hc in hand:
            if hc.markers:
                return None
            meta = meta_by_id.get(id(hc.definition))
            if meta is None or not meta.tags_canonical or not meta.target_canonical:
                return None
            hand_slugs_list.append(meta.slug)
        hand_slugs = tuple(sorted(hand_slugs_list))

        tom_key: tuple = ()
        if next_tomorrow_cards:
            tom_key_list: list[tuple[str, str, object]] = []
            for tc in next_tomorrow_cards:
                tm = tc.markers
                meta = meta_by_id.get(id(tc.definition))
                if meta is None or not meta.tags_canonical or not meta.target_canonical:
                    return None
                if tm and "_copied_definition" in tm:
                    if id(_effective_source_definition(tc)) not in meta_by_id:
                        return None
                    t_title = _effective_card_title(tc)
                else:
                    t_title = tc.definition.title
                tom_key_list.append(
                    (
                        tc.definition.slug,
                        t_title,
                        tm.get("chosen_tag") if tm else None,
                    )
                )
            tom_key = tuple(tom_key_list)
        return (days_left, player.character, hand_slugs, tom_key)

    @staticmethod
    def _character_reserve_adjustment(player: PlayerState, days_left: int) -> float:
        char = player.character
        if char is None or not player.hand:
            return 0.0
        zero_tag = char.zero_fun_tag
        fun_tag = char.fun_tag
        fun_delta = char.fun_delta
        energy_tag = char.energy_tag
        energy_delta = char.energy_delta
        fun_mult = 0.95 if days_left >= 2 else 0.70
        energy_mult = 0.75 if days_left >= 2 else 0.55
        adj = 0.0
        for hc in player.hand:
            hc_tags = hc.tags
            if zero_tag and zero_tag in hc_tags:
                continue
            if fun_tag and fun_delta != 0 and fun_tag in hc_tags:
                adj += fun_delta * fun_mult
            if energy_tag and energy_delta != 0 and energy_tag in hc_tags:
                adj += (-energy_delta) * energy_mult
        return adj

    def _evaluate_tomorrow_and_reserve_fast(
        self,
        sim: Game,
        player_index: int,
        next_tomorrow_cards: Sequence[CardInstance],
        *,
        days_left: int,
    ) -> float:
        player = sim.players[player_index]
        hand = player.hand
        if not hand and not next_tomorrow_cards:
            return 0.0

        spec_by_slug = _OSMIUM_SPEC_BY_SLUG
        if not spec_by_slug:
            _ensure_osmium_caches()

        if player.character is None and len(hand) <= 30 and 0 <= days_left <= 15:
            is_canon_sim = (
                type(sim) is _OSMIUM_SIM_GAME_CLS_CACHED
                and sim._all_canonical_defs  # type: ignore[attr-defined]
            )
            nh = len(hand)
            plain_ok = True
            h_specs_built: Sequence[_OsmiumCardSpec] | None = None
            hand_mset = 0
            small_packed = 0
            if nh == 0:
                h_specs_built = ()
            elif is_canon_sim and not sim._has_marked_hand:  # type: ignore[attr-defined]
                if nh <= 3:
                    rank_by_slug = _OSMIUM_RANK_BY_SLUG
                    if nh == 1:
                        small_packed = rank_by_slug[hand[0].definition.slug] << 13
                    elif nh == 2:
                        r0 = rank_by_slug[hand[0].definition.slug]
                        r1 = rank_by_slug[hand[1].definition.slug]
                        if r0 > r1:
                            r0, r1 = r1, r0
                        small_packed = (r0 << 13) | (r1 << 22)
                    else:
                        r0 = rank_by_slug[hand[0].definition.slug]
                        r1 = rank_by_slug[hand[1].definition.slug]
                        r2 = rank_by_slug[hand[2].definition.slug]
                        if r0 > r1:
                            r0, r1 = r1, r0
                        if r1 > r2:
                            r1, r2 = r2, r1
                            if r0 > r1:
                                r0, r1 = r1, r0
                        small_packed = (r0 << 13) | (r1 << 22) | (r2 << 31)
                else:
                    mset_by_slug = _OSMIUM_MSET_BIT_BY_SLUG
                    if nh == 4:
                        hand_mset = (
                            mset_by_slug[hand[0].definition.slug]
                            + mset_by_slug[hand[1].definition.slug]
                            + mset_by_slug[hand[2].definition.slug]
                            + mset_by_slug[hand[3].definition.slug]
                        )
                    else:
                        for c_h in hand:
                            hand_mset += mset_by_slug[c_h.definition.slug]
            else:
                tmp_specs: list[_OsmiumCardSpec] = []
                for c_h in hand:
                    if c_h.markers:
                        plain_ok = False
                        break
                    d_h = c_h.definition
                    sp_h = spec_by_slug.get(d_h.slug)
                    if sp_h is None or sp_h.defn is not d_h:
                        plain_ok = False
                        break
                    hand_mset += sp_h.multiset_bit
                    tmp_specs.append(sp_h)
                if plain_ok and nh <= 3:
                    if nh == 1:
                        small_packed = tmp_specs[0].slug_rank << 13
                    elif nh == 2:
                        r0, r1 = tmp_specs[0].slug_rank, tmp_specs[1].slug_rank
                        if r0 > r1:
                            r0, r1 = r1, r0
                        small_packed = (r0 << 13) | (r1 << 22)
                    elif nh == 3:
                        r0, r1, r2 = (
                            tmp_specs[0].slug_rank,
                            tmp_specs[1].slug_rank,
                            tmp_specs[2].slug_rank,
                        )
                        if r0 > r1:
                            r0, r1 = r1, r0
                        if r1 > r2:
                            r1, r2 = r2, r1
                            if r0 > r1:
                                r0, r1 = r1, r0
                        small_packed = (r0 << 13) | (r1 << 22) | (r2 << 31)
                h_specs_built = tmp_specs

            tom_specs: Sequence[_OsmiumCardSpec] | None = None
            nt = len(next_tomorrow_cards)
            if plain_ok and nt > 0:
                if nt == 1:
                    tc0 = next_tomorrow_cards[0]
                    tm0 = tc0.markers
                    if tm0 and ("_copied_definition" in tm0 or "chosen_tag" in tm0):
                        plain_ok = False
                    else:
                        d_t0 = tc0.definition
                        sp_t0 = (
                            spec_by_slug[d_t0.slug]
                            if is_canon_sim
                            else spec_by_slug.get(d_t0.slug)
                        )
                        if sp_t0 is None or (
                            not is_canon_sim and sp_t0.defn is not d_t0
                        ):
                            plain_ok = False
                        else:
                            tom_specs = (sp_t0,)
                else:
                    tmp_t: list[_OsmiumCardSpec] = []
                    for tc_i in next_tomorrow_cards:
                        tmi = tc_i.markers
                        if tmi and ("_copied_definition" in tmi or "chosen_tag" in tmi):
                            plain_ok = False
                            break
                        d_ti = tc_i.definition
                        sp_ti = (
                            spec_by_slug[d_ti.slug]
                            if is_canon_sim
                            else spec_by_slug.get(d_ti.slug)
                        )
                        if sp_ti is None or (
                            not is_canon_sim and sp_ti.defn is not d_ti
                        ):
                            plain_ok = False
                            break
                        tmp_t.append(sp_ti)
                    if plain_ok:
                        tom_specs = tmp_t

            if plain_ok:
                fast_key: int | tuple
                if nh <= 3 and nt <= 1:
                    tom_r = tom_specs[0].slug_rank if nt == 1 else 0  # type: ignore[index]
                    fast_key = days_left | (tom_r << 4) | small_packed
                elif nt == 0:
                    fast_key = days_left | hand_mset
                elif nt == 1:
                    fast_key = (
                        days_left
                        | 16
                        | (tom_specs[0].slug_rank << 5)  # type: ignore[index]
                        | hand_mset
                    )
                else:
                    if hand_mset == 0 and nh > 0:
                        mset_by_slug = _OSMIUM_MSET_BIT_BY_SLUG
                        for c_h in hand:
                            hand_mset += mset_by_slug[c_h.definition.slug]
                    fast_key = (
                        days_left,
                        hand_mset,
                        tuple(sp_t.slug_rank for sp_t in tom_specs),  # type: ignore[union-attr]
                    )

                main_cache = self._reserve_eval_cache
                if main_cache is _OSMIUM_RESERVE_EVAL_CACHE and (
                    nh >= 4 or nt >= 2
                ):
                    target_cache = _OSMIUM_LARGE_RESERVE_CACHE
                    max_sz = _OSMIUM_LARGE_RESERVE_CACHE_MAX
                else:
                    target_cache = main_cache
                    max_sz = _OSMIUM_RESERVE_CACHE_MAX

                cached_fast = target_cache.get(fast_key)
                if cached_fast is not None:
                    return cached_fast

                if h_specs_built is None:
                    h_specs: Sequence[_OsmiumCardSpec] = [
                        spec_by_slug[c_h.definition.slug] for c_h in hand
                    ]
                else:
                    h_specs = h_specs_built

                if nh <= 1:
                    sorted_specs: Sequence[_OsmiumCardSpec] = h_specs
                elif nh == 2:
                    s0, s1 = h_specs[0], h_specs[1]
                    sorted_specs = (
                        (s0, s1) if s0.slug_rank <= s1.slug_rank else (s1, s0)
                    )
                elif nh == 3:
                    s0, s1, s2 = h_specs[0], h_specs[1], h_specs[2]
                    if s0.slug_rank > s1.slug_rank:
                        s0, s1 = s1, s0
                    if s1.slug_rank > s2.slug_rank:
                        s1, s2 = s2, s1
                        if s0.slug_rank > s1.slug_rank:
                            s0, s1 = s1, s0
                    sorted_specs = (s0, s1, s2)
                else:
                    sorted_specs = sorted(h_specs, key=_SLUG_RANK_GETTER)

                val_fast = _osmium_reserve_score_plain_specs(
                    sorted_specs, tom_specs, days_left
                )
                if len(target_cache) >= max_sz:
                    target_cache.clear()
                target_cache[fast_key] = val_fast
                return val_fast

        cache_key = self._canonical_reserve_cache_key(
            sim, player_index, next_tomorrow_cards, days_left
        )
        if cache_key is not None:
            cached = self._reserve_eval_cache.get(cache_key)
            if cached is not None:
                return cached

        val = _osmium_reserve_score(sim, player_index, next_tomorrow_cards, days_left)
        zero_tag = player.character.zero_fun_tag if player.character is not None else ""
        if not (
            zero_tag
            and next_tomorrow_cards
            and any(tc.effective_behavior.tomorrow_all_tags for tc in next_tomorrow_cards)
        ):
            val += self._character_reserve_adjustment(player, days_left)
        if cache_key is not None:
            cache = self._reserve_eval_cache
            if len(cache) >= _OSMIUM_RESERVE_CACHE_MAX:
                cache.clear()
            cache[cache_key] = val
        return val

    def _simulate_play_branches(
        self,
        state: Game,
        player_index: int,
        hand_idx: int,
        known_cost: int | None = None,
    ) -> list[Game]:
        """Expand critical sub-decisions inside search without mutating parent state."""
        player = state.players[player_index]
        card = player.hand[hand_idx]
        title = card.definition.title
        is_hsim = type(state) is _OSMIUM_SIM_GAME_CLS_CACHED
        branches: list[Game] = []

        if title == "Kayak":
            cost = (
                state.energy_cost(player_index, card)
                if known_cost is None
                else known_cost
            )
            max_spend = max(0, player.energy - cost)
            if max_spend >= 2:
                spends = [0, max_spend]
                if max_spend >= 4:
                    spends.append(max_spend // 2)
                for sp in spends:
                    child = _hydrogen_copy_game_for_play(state, player_index)
                    child.ais = [_OsmiumOverrideEvalAI(spend_override=sp)] * 4
                    try:
                        if known_cost is not None and is_hsim:
                            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
                        else:
                            child.play_card(player_index, hand_idx)
                        branches.append(child)
                    except Exception:
                        pass
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                if branches:
                    return branches

        elif title == "Istanbul" and len(player.hand) >= 2:
            tmp = _hydrogen_copy_game_for_play(state, player_index)
            tmp_p = tmp.players[player_index]
            c_cost = (
                tmp.energy_cost(player_index, tmp_p.hand[hand_idx])
                if known_cost is None
                else known_cost
            )
            tmp_p.energy -= c_cost
            popped = tmp_p.hand.pop(hand_idx)
            tmp_p.played_today.append(popped)
            ranked = _evaluate_istanbul_discards_fast(tmp, player_index, tuple(tmp_p.hand))
            seen_cands: list[tuple[int, ...]] = []
            for _, cand_t in ranked:
                if cand_t not in seen_cands:
                    seen_cands.append(cand_t)
                if len(seen_cands) >= 2:
                    break
            if () not in seen_cands and len(seen_cands) < 3:
                seen_cands.append(())
            if len(seen_cands) >= 2:
                for disc_t in seen_cands[:2]:
                    child = _hydrogen_copy_game_for_play(state, player_index)
                    child.ais = [
                        _OsmiumOverrideEvalAI(discard_tuple_override=disc_t)
                    ] * 4
                    try:
                        if known_cost is not None and is_hsim:
                            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
                        else:
                            child.play_card(player_index, hand_idx)
                        branches.append(child)
                    except Exception:
                        pass
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                if branches:
                    return branches

        elif title in {"Cities and Knights", "Epic Prank"} and len(player.hand) >= 2:
            has_elig = (
                len(player.hand) >= 2
                if title == "Cities and Knights"
                else any(
                    i != hand_idx and "Item" in c.tags
                    for i, c in enumerate(player.hand)
                )
            )
            if has_elig:
                for opt_choice in (True, False):
                    child = _hydrogen_copy_game_for_play(state, player_index)
                    child.ais = [
                        _OsmiumOverrideEvalAI(optional_override=opt_choice)
                    ] * 4
                    try:
                        if known_cost is not None and is_hsim:
                            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
                        else:
                            child.play_card(player_index, hand_idx)
                        branches.append(child)
                    except Exception:
                        pass
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                if branches:
                    return branches

        elif title in {"Wedding Anniversary", "Leftovers"}:
            cost = (
                state.energy_cost(player_index, card)
                if known_cost is None
                else known_cost
            )
            rem_e = player.energy - cost
            if title == "Leftovers":
                copy_chain = card.markers.get("_copy_chain", ())
                elig = [
                    c
                    for c in player.played_today
                    if "Food" in c.tags
                    and c.instance_id not in copy_chain
                    and c.definition.behavior.can_play(state, player, card)
                ]
                pay_cost = False
            else:
                elig = [
                    c
                    for o_i, opp in enumerate(state.players)
                    if o_i != player_index
                    for c in opp.played_today
                    if c.definition.cost <= rem_e
                    and c.definition.behavior.can_play(state, player, card)
                ]
                pay_cost = True
            if len(elig) >= 2:
                scored_c = sorted(
                    range(len(elig)),
                    key=lambda i: _contextual_copy_value(
                        state, player_index, elig[i], card, pay_cost=pay_cost
                    ),
                    reverse=True,
                )
                for c_idx in scored_c[:2]:
                    child = _hydrogen_copy_game_for_play(state, player_index)
                    child.ais = [_OsmiumOverrideEvalAI(copy_idx_override=c_idx)] * 4
                    try:
                        if known_cost is not None and is_hsim:
                            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
                        else:
                            child.play_card(player_index, hand_idx)
                        branches.append(child)
                    except Exception:
                        pass
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                if branches:
                    return branches

        elif title == "Colouring" and len(player.played_today) >= 2:
            prior = player.played_today
            zero_tag = (
                player.character.zero_fun_tag
                if player.character is not None
                else None
            )
            distinct_indices = _colouring_candidate_indices(prior, zero_tag)
            if len(distinct_indices) >= 2:
                wanted_tags = _wanted_tags_for_player_fast(player)
                cond = state.daily_condition
                char = player.character

                def _colouring_tag_rank(opt_i: int) -> tuple[float, int, int]:
                    score, n_tags = _colouring_tag_score(
                        prior[opt_i].tags, wanted_tags, cond, char
                    )
                    return (score, n_tags, -opt_i)

                ranked_t = sorted(
                    distinct_indices,
                    key=_colouring_tag_rank,
                    reverse=True,
                )[:3]
                for t_idx in ranked_t:
                    child = _hydrogen_copy_game_for_play(state, player_index)
                    child.ais = [_OsmiumOverrideEvalAI(target_idx_override=t_idx)] * 4
                    try:
                        if known_cost is not None and is_hsim:
                            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
                        else:
                            child.play_card(player_index, hand_idx)
                        branches.append(child)
                    except Exception:
                        pass
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                if branches:
                    return branches

        return [self._simulate_play_step(state, player_index, hand_idx, known_cost)]

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
        """Adaptive day-scaled beam search with in-search sub-decision branching."""
        played_count = len(base_sim.players[player_index].played_today)
        is_live_play = max_plays is None and self._evaluation_depth == 0
        is_day6 = base_sim.day >= DAY_COUNT
        if max_plays is None:
            if is_day6 and played_count < 8:
                max_plays = self.DAY6_MAX_PLAYS
                beam_width = self.DAY6_BEAM_WIDTH
            elif played_count >= 12:
                max_plays = 6
                beam_width = 10
            elif played_count >= 7:
                max_plays = 10
                beam_width = 20
            elif base_sim.day >= 3:
                max_plays = self.MAX_TURN_PLAYS
                beam_width = self.DAY35_BEAM_WIDTH
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
        ftb_ctx: tuple = (
            b_ps[0],
            b_ps[1],
            b_ps[2],
            b_ps[3],
            base_awake_opps,
            {},
            base_sim.suitcase,
            tuple(c.instance_id for c in base_sim.suitcase),
            tuple(c.instance_id for c in b_ps[player_index].tomorrow_cards),
        )
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

        frontier: list[tuple[int, Game, float, int, bool]] = []
        seen_signatures: set[tuple] = set()
        need_depth_1 = max_plays > 1

        init_limit = (
            self.DAY6_BRANCH_LIMIT
            if (is_live_play and is_day6)
            else (14 if is_live_play else max(10, self.BRANCH_LIMIT + 1))
        )
        initial_with_costs = self._ordered_candidate_indices_with_costs(
            base_sim, player_index, init_limit
        )
        if allowed_first_indices is not None:
            allowed_set = set(allowed_first_indices)
            filtered_pairs = [
                (idx, cost) for idx, cost in initial_with_costs if idx in allowed_set
            ]
            if not filtered_pairs and allowed_first_indices:
                filtered_pairs = [
                    (idx, None) for idx in allowed_first_indices[:init_limit]
                ]
            initial_pairs: Sequence[tuple[int, int | None]] = filtered_pairs
        else:
            initial_pairs = initial_with_costs

        base_player = base_sim.players[player_index]
        first_step_turns = (
            1
            if in_extra_play
            else (base_turns_offset if start_extra_active else base_turns_offset + 1)
        )
        for hand_idx, known_cost in initial_pairs:
            cand_card = base_player.hand[hand_idx]
            child_extra = (
                in_extra_play
                or start_extra_active
                or _card_allows_extra_plays(base_sim, base_player, cand_card)
            )
            try:
                children = (
                    self._simulate_play_branches(
                        base_sim, player_index, hand_idx, known_cost
                    )
                    if is_live_play
                    else [
                        self._simulate_play_step(
                            base_sim, player_index, hand_idx, known_cost
                        )
                    ]
                )
            except Exception:
                continue
            for child in children:
                try:
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
                            (hand_idx, child, cont_val, first_step_turns, child_extra)
                        )

        if frontier:
            frontier.sort(key=lambda item: item[2], reverse=True)
            frontier = frontier[:beam_width]

        max_branch_depth = 3 if is_day6 else 2
        for depth in range(1, max_plays):
            if not frontier:
                break
            need_next = depth + 1 < max_plays
            if is_live_play:
                active_beam = beam_width if depth < 7 else max(12, beam_width - 10)
                branch_limit = (
                    (self.DAY6_BRANCH_LIMIT if is_day6 else 12) if depth < 6 else 8
                )
            else:
                active_beam = beam_width if depth < 5 else max(6, beam_width - 3)
                branch_limit = (
                    self.BRANCH_LIMIT
                    if depth < 5
                    else max(6, self.BRANCH_LIMIT - 2)
                )
            expand_branches = is_live_play and depth <= max_branch_depth
            next_frontier: list[tuple[int, Game, float, int, bool]] = []
            for first_action, state, _, turns_so_far, extra_active in frontier:
                candidates = self._ordered_candidate_indices_with_costs(
                    state, player_index, branch_limit
                )
                s_player = state.players[player_index]
                step_turns = turns_so_far if extra_active else turns_so_far + 1
                for hand_idx, known_cost in candidates:
                    cand_card = s_player.hand[hand_idx]
                    child_extra = extra_active or _card_allows_extra_plays(
                        state, s_player, cand_card
                    )
                    try:
                        children = (
                            self._simulate_play_branches(
                                state, player_index, hand_idx, known_cost
                            )
                            if expand_branches
                            else [
                                self._simulate_play_step(
                                    state, player_index, hand_idx, known_cost
                                )
                            ]
                        )
                    except Exception:
                        continue
                    for child in children:
                        try:
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
        return self.rng.choice(best_first_actions), best_value, best_by_action

    def _fast_v_pick_index(
        self,
        sim: Game,
        player_index: int,
        pool: Sequence[CardInstance],
        days_left: int,
    ) -> int:
        """Select future Suitcase pick using V_theta marginal + quick desirability."""
        if days_left <= 0 or len(pool) <= 1:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_card(
                sim, player_index, pool
            )
        p = sim.players[player_index]
        base_v = self._evaluate_tomorrow_and_reserve_fast(
            sim, player_index, p.tomorrow_cards, days_left=days_left
        )
        wanted_tags, sim_ctx = _build_wanted_and_ctx(
            p, getattr(sim, "_all_canonical_defs", False)
        )
        best_indices = [0]
        best_score = float("-inf")
        orig_marked = getattr(sim, "_has_marked_hand", None)
        for i, card in enumerate(pool):
            p.hand.append(card)
            if orig_marked is False and (card.markers or card.is_tomorrow):
                sim._has_marked_hand = True  # type: ignore[attr-defined]
            try:
                v_with = self._evaluate_tomorrow_and_reserve_fast(
                    sim, player_index, p.tomorrow_cards, days_left=days_left
                )
            finally:
                p.hand.pop()
                if orig_marked is False:
                    sim._has_marked_hand = False  # type: ignore[attr-defined]
            q_h = _hydrogen_quick_card_desirability(
                sim,
                player_index,
                card,
                wanted_tags,
                _ctx=sim_ctx,
                _in_hand=False,
            )
            score = 0.65 * (v_with - base_v) + 0.35 * q_h
            if score > best_score + 1e-9:
                best_score = score
                best_indices = [i]
            elif abs(score - best_score) <= 1e-9:
                best_indices.append(i)
        return self.rng.choice(best_indices)

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        """Unpruned stochastic pool drafting with V_theta-aligned future picks and wider Day-6 beam."""
        if self._evaluation_depth >= 1 or len(options) <= 1:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        _ensure_osmium_caches()
        self._ensure_live_game(game)
        live_player = game.players[player_index]
        own_markers = _carbon_own_suitcase_markers(live_player)
        pr_markers, scout_markers = _opponent_suitcase_markers(game, player_index)
        remaining_draw_picks = self._remaining_draw_picks_today(game, player_index)
        if live_player.played_today and (
            len(live_player.played_today) >= 8
            or len(live_player.hand) >= 10
            or len(options) < 2
        ):
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        has_custom_draft_syn = (
            type(self)._draft_synergy_adjustment
            is not OsmiumAI._draft_synergy_adjustment
        )
        wanted_tags = (
            _build_wanted_and_ctx(
                live_player, getattr(game, "_all_canonical_defs", False)
            )[0]
            if has_custom_draft_syn
            else {}
        )
        total_future_my_picks = remaining_draw_picks
        days_left = DAY_COUNT - game.day

        has_stochastic_option = any(
            opt.title in _STOCHASTIC_PLAY_TITLES for opt in options
        )
        if not live_player.played_today:
            if remaining_draw_picks >= 1:
                num_samples = self.DRAFT_SAMPLES
            elif game.day < DAY_COUNT or has_stochastic_option:
                num_samples = min(self.DRAFT_SAMPLES, 2)
            else:
                num_samples = 1
            if game.day == DAY_COUNT:
                draft_beam = self.DAY6_DRAFT_BEAM_WIDTH
                draft_max_plays = 14
            else:
                draft_beam = self.DRAFT_BEAM_WIDTH
                draft_max_plays = 12
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
                    tc for tc in base_sim.trunk if tc.instance_id not in known_ids
                ]
                cursor = 0
                for _ in range(total_future_my_picks):
                    if cursor + SUITCASE_SIZE <= len(pool_candidates):
                        future_pools.append(
                            tuple(pool_candidates[cursor : cursor + SUITCASE_SIZE])
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

            plan_sum = 0.0
            for base_sim, future_pools in samples:
                sim = _hydrogen_copy_game(base_sim)
                sim.rng = _clone_random(base_sim.rng)
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
                    pick_i = self._fast_v_pick_index(
                        sim, player_index, cloned_pool, days_left
                    )
                    extra_card = cloned_pool[pick_i]
                    for t_i, tc in enumerate(sim.trunk):
                        if tc.instance_id == extra_card.instance_id:
                            sim.trunk.pop(t_i)
                            break
                    sim.give_card(player_index, extra_card)
                    sim.players[player_index].picked_cards[extra_card.title] += 1

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
                    beam_width=draft_beam,
                    max_plays=draft_max_plays,
                )
                plan_sum += sample_val

            total_val = (plan_sum / len(samples)) + synergy_val
            if total_val > best_value + 1e-9:
                best_value = total_val
                best_indices = [idx]
            elif abs(total_val - best_value) <= 1e-9:
                best_indices.append(idx)

        return self.rng.choice(best_indices)

    def choose_card_to_discard(
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
        if (
            len(options) <= 1
            or self._evaluation_depth >= 1
            or game.stats is _get_shared_sim_stats()
            or last_played == "Tell a Story"
            or len(player.played_today) >= 9
        ):
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_discard(
                game, player_index, options
            )

        self._ensure_live_game(game)
        base_sim = self._prepare_observable_game(game, player_index)
        sim_p0 = base_sim.players[player_index]
        sim_hand_ids = {hc.instance_id for hc in sim_p0.hand}
        if not all(opt.instance_id in sim_hand_ids for opt in options):
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_discard(
                game, player_index, options
            )

        best_val = float("-inf")
        best_indices: list[int] = []
        prev_depth = self._evaluation_depth
        self._evaluation_depth = 1
        try:
            for opt_i, opt in enumerate(options):
                sim = _hydrogen_copy_game(base_sim)
                sim.rng = _clone_random(base_sim.rng)
                sim_player = sim.players[player_index]
                hand_idx = next(
                    (
                        i
                        for i, hc in enumerate(sim_player.hand)
                        if hc.instance_id == opt.instance_id
                    ),
                    None,
                )
                if hand_idx is None:
                    continue
                try:
                    target_card = sim_player.hand[hand_idx]
                    sim.discard_from_hand(player_index, hand_idx)
                    if sim_player.played_today:
                        sim_source = sim_player.played_today[-1]
                        if last_played == "Cities and Knights":
                            sim_source.markers["discarded_card"] = True
                            if hasattr(sim, "_p_flags"):
                                sim._p_flags[player_index] |= _PF_HAS_SIG_MARKER  # type: ignore[attr-defined]
                        elif last_played == "Epic Prank":
                            sim_source.markers["discarded_item"] = target_card
                            if hasattr(sim, "_p_flags"):
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
                            beam_width=8,
                            max_plays=6,
                            allow_bed_now=True,
                            base_turns_offset=1,
                            root_skipped_override=sim_p0.skipped_turns,
                            start_extra_active=_card_allows_extra_plays(
                                sim, sim_player, sim_source
                            ),
                        )
                    else:
                        _, val = self._search_best_day_plan(
                            sim,
                            player_index,
                            beam_width=8,
                            max_plays=6,
                            allow_bed_now=True,
                        )
                except Exception:
                    continue
                if val > best_val + 1e-9:
                    best_val = val
                    best_indices = [opt_i]
                elif abs(val - best_val) <= 1e-9:
                    best_indices.append(opt_i)
        finally:
            self._evaluation_depth = prev_depth

        if not best_indices:
            return _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_discard(
                game, player_index, options
            )
        return self.rng.choice(best_indices)

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
            ranked = _evaluate_istanbul_discards_fast(game, player_index, options)
            if (
                len(ranked) <= 1
                or self._evaluation_depth >= 1
                or len(player.played_today) >= 10
                or game.stats is _get_shared_sim_stats()
                or len(player.hand) != len(options)
                or any(
                    hc.instance_id != opt.instance_id
                    for hc, opt in zip(player.hand, options)
                )
            ):
                return ranked[0][1]

            to_verify: list[tuple[int, ...]] = [cand for _, cand in ranked[:7]]
            if () not in to_verify:
                to_verify.append(())
            to_verify.sort(key=len)

            base_sim = self._prepare_observable_game(game, player_index)
            sim_p0 = base_sim.players[player_index]
            if not sim_p0.played_today or len(sim_p0.hand) != len(options):
                return ranked[0][1]
            raw_e_per = _istanbul_energy_per_discard(sim_p0.played_today[-1])

            best_choices: list[tuple[int, ...]] = [ranked[0][1]]
            best_val = float("-inf")
            prev_depth = self._evaluation_depth
            self._evaluation_depth = 1
            try:
                for disc_tuple in to_verify:
                    sim = _hydrogen_copy_game(base_sim)
                    sim.rng = _clone_random(base_sim.rng)
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
                            beam_width=12,
                            max_plays=9,
                            allow_bed_now=True,
                            base_turns_offset=1,
                            root_skipped_override=sim_p0.skipped_turns,
                            start_extra_active=_card_allows_extra_plays(
                                sim, sim_player, sim_source
                            ),
                        )
                    except Exception:
                        continue
                    adj_val = val - 0.02 * len(disc_tuple)
                    if adj_val > best_val + 1e-9:
                        best_val = adj_val
                        best_choices = [disc_tuple]
                    elif abs(adj_val - best_val) <= 1e-9:
                        best_choices.append(disc_tuple)
            finally:
                self._evaluation_depth = prev_depth
            return self.rng.choice(best_choices)
        return _HYDROGEN_EVAL_AI_SINGLETON.choose_cards_to_discard(
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
            or len(game.players[player_index].played_today) >= 9
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

        pay_cost = type(source_card.effective_behavior).__name__ != "LeftoversBehavior"
        ranked_indices = sorted(
            range(len(options)),
            key=lambda i: _contextual_copy_value(
                game, player_index, options[i], live_source_card, pay_cost=pay_cost
            )
            - (1.35 if options[i].definition.title == "Group Dinner" else 0.0),
            reverse=True,
        )[:5]
        best_indices = [ranked_indices[0]]
        best_val = float("-inf")
        prev_depth = self._evaluation_depth
        self._evaluation_depth = 1
        try:
            for idx in ranked_indices:
                option = options[idx]
                sim = _hydrogen_copy_game(base_sim)
                sim.rng = _clone_random(base_sim.rng)
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
                        beam_width=10,
                        max_plays=7,
                        allow_bed_now=True,
                        base_turns_offset=1,
                        root_skipped_override=player.skipped_turns,
                        start_extra_active=_card_allows_extra_plays(
                            sim, sim_player, sim_source
                        ),
                    )
                except Exception:
                    val = _contextual_copy_value(
                        game, player_index, option, live_source_card, pay_cost=pay_cost
                    )
                if val > best_val + 1e-9:
                    best_val = val
                    best_indices = [idx]
                elif abs(val - best_val) <= 1e-9:
                    best_indices.append(idx)
        finally:
            self._evaluation_depth = prev_depth
        return self.rng.choice(best_indices)

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
            or len(game.players[player_index].played_today) >= 9
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
                sim.rng = _clone_random(base_sim.rng)
                sim_player = sim.players[player_index]
                sim_source = sim_player.played_today[-1]
                sim_player.energy -= spend
                sim_source.markers["energy_cubes"] = spend
                if spend and hasattr(sim, "_p_flags"):
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
                    beam_width=10,
                    max_plays=7,
                    allow_bed_now=True,
                    base_turns_offset=1,
                    root_skipped_override=player.skipped_turns,
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

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if (
            self._evaluation_depth >= 1
            or len(options) <= 1
            or game.stats is _get_shared_sim_stats()
            or len(game.players[player_index].played_today) >= 10
        ):
            return super().choose_card_target(game, player_index, options)

        player = game.players[player_index]
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )

        if last_played == "Colouring":
            zero_tag = (
                player.character.zero_fun_tag
                if player.character is not None
                else None
            )
            distinct_indices = _colouring_candidate_indices(options, zero_tag)
            if len(distinct_indices) <= 1:
                return distinct_indices[0]

            wanted_tags = _wanted_tags_for_player_fast(player)
            cond = game.daily_condition
            char = player.character

            def _tag_tie_score(opt_i: int) -> tuple[float, int]:
                return _colouring_tag_score(
                    options[opt_i].tags, wanted_tags, cond, char
                )

            self._ensure_live_game(game)
            base_sim = self._prepare_observable_game(game, player_index)
            sim_p0 = base_sim.players[player_index]
            if not sim_p0.hand or not sim_p0.played_today:
                return max(distinct_indices, key=lambda i: (_tag_tie_score(i), -i))

            prior_by_id = {
                c.instance_id: i for i, c in enumerate(sim_p0.played_today[:-1])
            }
            if not all(options[i].instance_id in prior_by_id for i in distinct_indices):
                return max(distinct_indices, key=lambda i: (_tag_tie_score(i), -i))

            best_val = float("-inf")
            best_tie: tuple[float, int] = (float("-inf"), -1)
            best_indices: list[int] = []
            prev_depth = self._evaluation_depth
            self._evaluation_depth = 1
            try:
                for opt_i in distinct_indices:
                    sim = _hydrogen_copy_game(base_sim)
                    sim.rng = _clone_random(base_sim.rng)
                    sim_player = sim.players[player_index]
                    sim_source = sim_player.played_today[-1]
                    t_pos = prior_by_id[options[opt_i].instance_id]
                    sim_target = sim_player.played_today[t_pos]
                    try:
                        sim_target.markers["energy_cube"] = True
                        sim_source.markers["target_card"] = sim_target
                        sim._has_card_ref = True  # type: ignore[attr-defined]
                        if hasattr(sim, "_p_flags"):
                            sim._p_flags[player_index] |= _PF_HAS_SIG_MARKER  # type: ignore[attr-defined]
                        sim._sync_played_today_tags(sim_player)
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
                            beam_width=10,
                            max_plays=7,
                            allow_bed_now=True,
                            base_turns_offset=1,
                            root_skipped_override=sim_p0.skipped_turns,
                            start_extra_active=_card_allows_extra_plays(
                                sim, sim_player, sim_source
                            ),
                        )
                    except Exception:
                        continue
                    tie_key = _tag_tie_score(opt_i)
                    if val > best_val + 1e-9:
                        best_val = val
                        best_tie = tie_key
                        best_indices = [opt_i]
                    elif abs(val - best_val) <= 1e-9:
                        if tie_key > best_tie:
                            best_tie = tie_key
                            best_indices = [opt_i]
                        elif tie_key == best_tie:
                            best_indices.append(opt_i)
            finally:
                self._evaluation_depth = prev_depth

            if not best_indices:
                return max(distinct_indices, key=lambda i: (_tag_tie_score(i), -i))
            return self.rng.choice(best_indices)

        if last_played == "Address the Food":
            self._ensure_live_game(game)
            own_markers = _carbon_own_suitcase_markers(player)
            pr_markers, scout_markers = _opponent_suitcase_markers(
                game, player_index
            )
            base_sim = self._prepare_observable_game(game, player_index)
            sim_p0 = base_sim.players[player_index]
            if not sim_p0.played_today:
                return super().choose_card_target(game, player_index, options)
            sc_by_id = {c.instance_id for c in base_sim.suitcase}
            if not all(opt.instance_id in sc_by_id for opt in options):
                return super().choose_card_target(game, player_index, options)

            best_val = float("-inf")
            best_indices = []
            prev_depth = self._evaluation_depth
            self._evaluation_depth = 1
            try:
                for opt_i, opt in enumerate(options):
                    sim = _hydrogen_copy_game(base_sim)
                    sim.rng = _clone_random(base_sim.rng)
                    sim_player = sim.players[player_index]
                    sim_source = sim_player.played_today[-1]
                    sim_target = next(
                        (
                            c
                            for c in sim.suitcase
                            if c.instance_id == opt.instance_id
                        ),
                        None,
                    )
                    if sim_target is None:
                        continue
                    try:
                        if sim.trunk or sim.discard:
                            picked = sim.pick_suitcase_cards(
                                player_index, (sim_target,)
                            )[0]
                        else:
                            sim.suitcase.remove(sim_target)
                            if sim_target.markers:
                                sim_target.markers.clear()
                            sim.give_card(player_index, sim_target)
                            sim.record_card_pick_or_draw(player_index, sim_target)
                            picked = sim_target
                        sim.play_card_for_effect(
                            player_index, picked, pay_energy=False
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
                            beam_width=10,
                            max_plays=7,
                            allow_bed_now=True,
                            base_turns_offset=1,
                            root_skipped_override=sim_p0.skipped_turns,
                            start_extra_active=_card_allows_extra_plays(
                                sim, sim_player, sim_source
                            ),
                        )
                    except Exception:
                        continue
                    adj_val = val + _suitcase_denial_value(
                        game, player_index, opt, pr_markers, scout_markers
                    )
                    if own_markers and any(m in opt.markers for m in own_markers):
                        adj_val -= 3.0
                    if adj_val > best_val + 1e-9:
                        best_val = adj_val
                        best_indices = [opt_i]
                    elif abs(adj_val - best_val) <= 1e-9:
                        best_indices.append(opt_i)
            finally:
                self._evaluation_depth = prev_depth

            if not best_indices:
                return super().choose_card_target(game, player_index, options)
            return self.rng.choice(best_indices)

        return super().choose_card_target(game, player_index, options)

    def _quacks_multiplier(self, game: Game, player_index: int) -> float:
        return 1.0

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
            p_outdoors = unseen_counts.get("Outdoors", 50) / max(1, unseen_total)
            mult = self._quacks_multiplier(game, player_index)
            return p_outdoors < (mult / (1.0 + draws))

        if (
            ("Cities and Knights" in action or "Epic Prank" in action)
            and self._evaluation_depth == 0
            and game.stats is not _get_shared_sim_stats()
        ):
            self._ensure_live_game(game)
            live_p = game.players[player_index]
            is_prank = "Epic Prank" in action
            live_opts = (
                tuple(c for c in live_p.hand if "Item" in c.tags)
                if is_prank
                else tuple(live_p.hand)
            )
            if live_opts and len(live_p.played_today) < 9:
                base_sim = self._prepare_observable_game(game, player_index)
                sim_p0 = base_sim.players[player_index]
                sim_src0 = sim_p0.played_today[-1] if sim_p0.played_today else None
                if sim_src0 is not None:
                    prev_depth = self._evaluation_depth
                    self._evaluation_depth = 1
                    try:
                        sim_no = _hydrogen_copy_game(base_sim)
                        sim_no.rng = _clone_random(base_sim.rng)
                        p_no = sim_no.players[player_index]
                        src_no = p_no.played_today[-1]
                        for t_src in tuple(p_no.tomorrow_cards):
                            t_src.effective_behavior.on_tomorrow_card_play(
                                sim_no, p_no, t_src, src_no
                            )
                        for p_src in tuple(p_no.played_today):
                            p_src.effective_behavior.on_card_play(
                                sim_no, p_no, p_src, src_no
                            )
                        _, val_no = self._search_best_day_plan(
                            sim_no,
                            player_index,
                            beam_width=10,
                            max_plays=6,
                            allow_bed_now=True,
                            base_turns_offset=1,
                            root_skipped_override=sim_p0.skipped_turns,
                            start_extra_active=_card_allows_extra_plays(
                                sim_no, p_no, src_no
                            ),
                        )

                        sim_yes = _hydrogen_copy_game(base_sim)
                        sim_yes.rng = _clone_random(base_sim.rng)
                        p_yes = sim_yes.players[player_index]
                        src_yes = p_yes.played_today[-1]
                        opts_yes = (
                            tuple(c for c in p_yes.hand if "Item" in c.tags)
                            if is_prank
                            else tuple(p_yes.hand)
                        )
                        val_yes = float("-inf")
                        if opts_yes:
                            d_idx = _HYDROGEN_EVAL_AI_SINGLETON.choose_card_to_discard(
                                sim_yes, player_index, opts_yes
                            )
                            discarded_c = opts_yes[d_idx]
                            hand_idx = next(
                                (
                                    i
                                    for i, hc in enumerate(p_yes.hand)
                                    if hc is discarded_c
                                ),
                                None,
                            )
                            if hand_idx is not None:
                                sim_yes.discard_from_hand(player_index, hand_idx)
                                if is_prank:
                                    src_yes.markers["discarded_item"] = discarded_c
                                else:
                                    src_yes.markers["discarded_card"] = True
                                if hasattr(sim_yes, "_p_flags"):
                                    sim_yes._p_flags[player_index] |= _PF_HAS_SIG_MARKER  # type: ignore[attr-defined]
                                for t_src in tuple(p_yes.tomorrow_cards):
                                    t_src.effective_behavior.on_tomorrow_card_play(
                                        sim_yes, p_yes, t_src, src_yes
                                    )
                                for p_src in tuple(p_yes.played_today):
                                    p_src.effective_behavior.on_card_play(
                                        sim_yes, p_yes, p_src, src_yes
                                    )
                                _, val_yes = self._search_best_day_plan(
                                    sim_yes,
                                    player_index,
                                    beam_width=10,
                                    max_plays=6,
                                    allow_bed_now=True,
                                    base_turns_offset=1,
                                    root_skipped_override=sim_p0.skipped_turns,
                                    start_extra_active=_card_allows_extra_plays(
                                        sim_yes, p_yes, src_yes
                                    ),
                                )
                        return val_yes >= val_no
                    except Exception:
                        pass
                    finally:
                        self._evaluation_depth = prev_depth

        return super().choose_optional_action(game, player_index, action)


Osmium = OsmiumAI
OsmiumBrain = OsmiumAI
OsmiumbrainAI = OsmiumAI
