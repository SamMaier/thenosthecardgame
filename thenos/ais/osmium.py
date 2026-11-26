"""OsmiumAI: On-policy DAgger-R4 Huber-IRLS policy with aligned drafting and deep Day-6 search."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from operator import attrgetter
from typing import Sequence, TYPE_CHECKING

from thenos.ais.argonbrain import (
    DAY_COUNT,
    SUITCASE_SIZE,
    _ACTIVE_PROBE_KEYS,
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
    _projected_opponent_day_fun,
    _suitcase_denial_value,
)
import thenos.ais.hydrogen as _hydrogen_mod
from thenos.ais.hydrogen import (
    _BLIND_TRUNK_STEP_TITLES,
    _CM_GENERIC,
    _CM_NEXT_TAG_ADD,
    _CM_NEXT_TAG_HALF,
    _CM_NEXT_TAG_SET,
    _COMM_GAIN_E,
    _COMM_PLAIN,
    _CQ_BOUGIE_COFFEE_MACHINE,
    _CQ_CHEESY_PHONE_GAME,
    _CQ_GROUP_DINNER,
    _CQ_NONE,
    _HQ_NONE,
    _HYDROGEN_COPY_FOOD_TITLES,
    _HYDROGEN_EVAL_AI_SINGLETON,
    _HYDROGEN_META_BY_DEFN_ID,
    _HYDROGEN_META_BY_SLUG,
    _HYDROGEN_SIM_AIS,
    _MASK_BOARD_GAME,
    _MASK_EXERCISE,
    _MASK_FOOD,
    _MASK_OUTDOORS,
    _MASK_RELAX,
    _MASK_SOCIAL,
    _MF_ADE,
    _MF_GENERIC,
    _MF_NEXT_TAG_ADD,
    _MF_NEXT_TAG_COST_ADD,
    _MF_NEXT_TAG_MUL2,
    _MF_NTH_TAG_MUL2,
    _OCM_CLASSIC_BOOK,
    _OCM_SING_SONG,
    _PF_ALL_TAGS,
    _PF_GAVE_ENERGY,
    _PF_HAS_ACQUIRE,
    _PF_HAS_ALLOWS_E,
    _PF_HAS_AWAKE_OPP_TERM,
    _PF_HAS_BUBLY,
    _PF_HAS_CARD_PLAY,
    _PF_HAS_CK_END_DAY,
    _PF_HAS_COMPLEX_END_DAY,
    _PF_HAS_COPIED,
    _PF_HAS_CONT_AFTER,
    _PF_HAS_CUSTOM_TOM_FUN_MOD,
    _PF_HAS_END_DAY,
    _PF_HAS_EUCHRE_AWARDS,
    _PF_HAS_FIRST_CARD_TOM,
    _PF_HAS_FUN_MOD,
    _PF_HAS_FUN_VAL,
    _PF_HAS_GENERIC_CARD_PLAY,
    _PF_HAS_GENERIC_TOM_PLAY,
    _PF_HAS_LOOP2,
    _PF_HAS_NEW_AWAKE_OPP,
    _PF_HAS_NEXT_TAG_MOD,
    _PF_HAS_NONADD_FUN_MOD,
    _PF_HAS_PHOTO_SHOOT,
    _PF_HAS_PICK,
    _PF_HAS_PLAYED_COST_MOD,
    _PF_HAS_PLAYED_IS_TOM,
    _PF_HAS_SCOUT_END_DAY,
    _PF_HAS_SIG_MARKER,
    _PF_HAS_TOM_ACTION,
    _PF_HAS_TOM_COST_MOD,
    _PF_HAS_TOM_FUN_MOD,
    _PF_HAS_TOM_PLAY,
    _PF_MAY_HAVE_TAG_MARKERS,
    _QGLOB_COUCH_TUBE,
    _QGLOB_DOCK,
    _QGLOB_NONE,
    _QSYN_NEG_SMB,
    _QSYN_NONE,
    _SIG_MARKER_KEYS,
    _TRUNK_CONSUMER_TITLES,
    _TRUNK_REORDER_TITLES,
    HydrogenAI,
    _HydrogenDefnMeta,
    _HydrogenEvalAI,
    _build_quick_zone_ctx,
    _build_w_arr_only,
    _build_wanted_and_ctx,
    _card_allows_extra_plays,
    _ensure_hydrogen_caches,
    _eval_q_const_with_arr,
    _evaluate_istanbul_discards_fast,
    _get_card_meta,
    _get_hydrogen_sim_game_cls,
    _hydrogen_copy_game,
    _hydrogen_copy_game_for_play,
    _hydrogen_free_play_value,
    _hydrogen_hand_keep_value,
    _hydrogen_quick_card_desirability,
    _inline_own_cost_mod,
    _istanbul_energy_per_discard,
    _select_bohnanza_discards_fast,
    _tags_to_mask,
    _wanted_tags_for_player_fast,
)
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.basic import (
    EpicPrankBehavior,
    SoloBehavior,
    cities_and_knights_condition_met,
)
from thenos.cards.copy_effects import NosShirtBehavior
from thenos.cards.exercise import ChuckAFrisbeeBehavior
from thenos.cards.social import (
    DatesFirstNosBehavior,
    ScoutTheOtherCottagesBehavior,
)
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

_SAFE_COMM_PREFIX_EXCLUDE: int = (
    _PF_HAS_COPIED
    | _PF_HAS_GENERIC_CARD_PLAY
    | _PF_HAS_CUSTOM_TOM_FUN_MOD
)
_CANON_SIG_EXCLUDE_PF: int = (
    _PF_ALL_TAGS
    | _PF_MAY_HAVE_TAG_MARKERS
    | _PF_HAS_COPIED
    | _PF_HAS_SIG_MARKER
    | _PF_HAS_PICK
    | _PF_HAS_NONADD_FUN_MOD
    | _PF_HAS_ALLOWS_E
    | _PF_HAS_CUSTOM_TOM_FUN_MOD
    | _PF_HAS_GENERIC_CARD_PLAY
    | _PF_HAS_PLAYED_IS_TOM
    | _PF_HAS_NEW_AWAKE_OPP
    | _PF_HAS_BUBLY
    # Solo and Bubly are order-dependent under C signature; Epic Prank sets _PF_HAS_SIG_MARKER.
)
_BLOCKED_MASK_TRIGGERS: int = (
    _PF_HAS_EUCHRE_AWARDS | _PF_HAS_TOM_PLAY | _PF_HAS_NEXT_TAG_MOD
)
_EUCHRE_TAG_MASK: int = _MASK_FOOD | _MASK_BOARD_GAME
_PHOTO_TAG_MASK: int = _MASK_SOCIAL | _MASK_RELAX
_INPLACE_PF_EXCLUDE: int = (
    _PF_HAS_GENERIC_TOM_PLAY
    | _PF_ALL_TAGS
    | _PF_HAS_GENERIC_CARD_PLAY
    | _PF_HAS_COPIED
)
_OPP_TODAY_DYNAMIC_PF: int = (
    _PF_HAS_AWAKE_OPP_TERM | _PF_HAS_COMPLEX_END_DAY | _PF_HAS_SIG_MARKER
)
_OSMIUM_SIG_MARKER_KEYS: frozenset[str] = _SIG_MARKER_KEYS | frozenset(
    {"_blind_cutlery_ev", "_trunk_risk_adj"}
)
_SIMPLE_SCORE_EXCL_PF: int = (
    _PF_HAS_END_DAY
    | _PF_HAS_FUN_MOD
    | _PF_HAS_TOM_FUN_MOD
    | _PF_HAS_FUN_VAL
    | _PF_HAS_COPIED
    | _PF_ALL_TAGS
    | _PF_MAY_HAVE_TAG_MARKERS
    | _PF_HAS_PLAYED_IS_TOM
    | _PF_HAS_TOM_ACTION
    | _PF_HAS_BUBLY
    | _PF_HAS_CUSTOM_TOM_FUN_MOD
)
_HAND_TO_TRUNK_TITLES: frozenset[str] = frozenset(
    {"Tell a Story", "Last Year's Shirt"}
)
_TRUNK_PEEK_TITLES: frozenset[str] = frozenset(
    {"Read the Radar", "Fishing Boat", "Ski Boat", "Grocery Store Run"}
)
_BCM_PEEK_TITLES: frozenset[str] = frozenset(
    {"Read the Radar", "Fishing Boat", "Ski Boat"}
)
_CUTLERY_BOMB_EXTRA_TITLES: frozenset[str] = frozenset(
    {"Date's First Nos", "Navy SEALing", "Morning Run"}
)
_OSMIUM_TRUNK_SIM_TITLES: frozenset[str] = _TRUNK_REORDER_TITLES | frozenset(
    {
        "Assorted Cutlery",
        "Bougie Coffee Machine",
        "Quacks",
        "Poker",
        "Leftovers",
        "Wedding Anniversary",
    }
)
_TRUNK_DRAW_TAKE_COUNTS: dict[str, int] = {
    "Booby Prize": 1,
    "Last Year's Shirt": 2,
    "Euchre Tournament": 4,
    "Make Plans": 1,
    "Long Distance Visitors": 1,
    "Medical Advice": 2,
    "Scrabble": 1,
}
_LIVE_TRUNK_FIXED_MAX_POPS: dict[str, int] = {
    "Euchre Tournament": 4,
    "Bougie Coffee Machine": 3,
    "Medical Advice": 3,
    "Long Distance Visitors": 2,
    "Make Plans": 2,
    "Last Year's Shirt": 2,
    "Grocery Store Run": 5,
    "Read the Radar": 5,
    "Booby Prize": 1,
    "Assorted Cutlery": 1,
    "Poker": 1,
    "Dominion": 1,
    "Throw a Baseball": 1,
    "Address the Food": 1,
}
_LIVE_TRUNK_UNBOUNDED_POPS: frozenset[str] = frozenset(
    {"Treat Store Run", "Fancy Floatie", "Scrabble", "Fishing Boat", "Ski Boat"}
)
_BCM_FOOD_SYNERGY_RANK: dict[str, int] = {
    "Rouladen": 0,
    "High-End Red": 0,
    "High-End White": 0,
    "Cheap Red": 0,
    "Fajitas": 1,
    "Morning Coffee": 1,
    "Afternoon Coffee": 1,
    "Calzones": 1,
    "Charcuterie": 1,
    "Potato Pancakes": 1,
    "Keeper": 1,
    "Camp Crossroads Dessert": 1,
    "Pudding Chômeur": 1,
    "Weird Chip Flavour": 1,
    "M&Ms": 1,
    "Doritos": 3,
    "Ice Cream Sandwich": 3,
    "Bubly": 3,
    "Butter Tarts": 4,
    "Ice Wine": 5,
    "Leftovers": 6,
    "Beaver Burgers": 7,
}
_INPLACE_OP_BY_SLUG: dict[str, int] = {
    "evening-chat": 1,
    "early-bedtime": 1,
    "wingspan": 2,
    "the-crew": 3,
    "paddleboard": 4,
    "morning-coffee": 5,
    "sleep-in": 6,
    "butter-tarts": 7,
    "wurstchen": 8,
    "camp-crossroads-dessert": 9,
    "keeper": 10,
    "potato-pancakes": 11,
    "ice-cream-sandwich": 12,
    "doritos": 13,
    "steak": 14,
    "beach-conversation": 15,
    "cheesy-phone-game": 16,
    "read-the-radar": 17,
    "christmas-name-draw": 18,
    "wedding-anniversary": 18,
    "address-the-food": 19,
    "treat-store-run": 19,
    "fancy-floatie": 20,
    "bohnanza": 21,
    "ski-on-shoulders": 22,
    "leftovers": 23,
    "pudding-chomeur": 24,
    "tell-a-story": 25,
}


def _resolve_inplace_on_play(
    op: int,
    state: Game,
    player_index: int,
    s_player: PlayerState,
    s_hand: Sequence[CardInstance],
    s_played: Sequence[CardInstance],
    s_pf: int,
    s_no_allows_e: bool,
    orig_e: int,
    known_cost: int,
    cand_card: CardInstance,
    cand_meta: _HydrogenDefnMeta,
    meta_by_slug: dict[str, _HydrogenDefnMeta],
    b_ps: Sequence[PlayerState],
    root_crew_trig: bool,
) -> tuple[int, int, bool, bool, dict[str, object] | None] | None:
    """Resolve pure-stat `on_play` delta for in-place evaluation, or return None."""
    if op == 1:
        if cand_meta.has_can_play and not cand_meta.beh.can_play(
            state, s_player, cand_card
        ):
            return None
        return (0, 0, False, True, None)
    if op == 2:
        if s_pf & _PF_MAY_HAVE_TAG_MARKERS:
            return None
        return (0, 0, True, False, None)
    if op == 3:
        s_ps = state.players
        if (
            (player_index == 0 or s_ps[0] is b_ps[0])
            and (player_index == 1 or s_ps[1] is b_ps[1])
            and (player_index == 2 or s_ps[2] is b_ps[2])
            and (player_index == 3 or s_ps[3] is b_ps[3])
        ):
            crew_trig = root_crew_trig
        else:
            bg_opps = 0
            for o_i, opp in enumerate(s_ps):
                if o_i != player_index:
                    for oc in opp.played_today:
                        if "Board Game" in oc.tags:
                            bg_opps += 1
                            break
            crew_trig = bg_opps * 2 >= (len(s_ps) - 1)
        return (
            0,
            0,
            False,
            False,
            {"energy_cube": True, "_the_crew_energy_cube": True}
            if crew_trig
            else None,
        )
    if op == 4:
        rem_h = len(s_hand) - 1
        return (
            0,
            _PF_HAS_SIG_MARKER if rem_h > 0 else 0,
            False,
            False,
            {"energy_cubes": rem_h},
        )
    if op == 16:
        return (0, 0, False, False, None)
    if op == 17:
        if (
            len(state.trunk) >= 5
            and not state._has_opp_suitcase_ref  # type: ignore[attr-defined]
            and not (s_pf & (_PF_HAS_CARD_PLAY | _PF_HAS_TOM_PLAY))
            and not _has_sim_trunk_consumer(state, player_index)
        ):
            return (0, 0, False, False, None)
        return None
    if op == 18:
        s_ps = state.players
        if (
            (player_index == 0 or not s_ps[0].played_today)
            and (player_index == 1 or not s_ps[1].played_today)
            and (player_index == 2 or not s_ps[2].played_today)
            and (player_index == 3 or not s_ps[3].played_today)
        ):
            return (0, 0, False, False, None)
        return None
    if op == 19:
        for sc_c in state.suitcase:
            if "Food" in (sc_c.definition.tags if not sc_c.markers else sc_c.tags):
                return None
        return (0, 0, False, False, None)
    if op == 20:
        for sc_c in state.suitcase:
            if "Relax" in (sc_c.definition.tags if not sc_c.markers else sc_c.tags):
                return None
        return (0, 0, False, False, None)
    if op == 21:
        return (0, 0, False, False, None) if not s_played else None
    if op == 22:
        s_ps = state.players
        my_f = s_player.fun
        higher = 0
        for o_i, opp in enumerate(s_ps):
            if o_i != player_index and my_f < opp.fun:
                higher += 1
        if higher * 2 >= len(s_ps) - 1:
            return (
                0,
                _PF_HAS_SIG_MARKER,
                False,
                False,
                {"energy_cube": True, "_ski_on_shoulders_energy_cube": True},
            )
        return (0, 0, False, False, None)
    if op == 23:
        for pc in s_played:
            if "Food" in (pc.definition.tags if not pc.markers else pc.tags):
                return None
        return (0, 0, False, False, None)
    if op == 25:
        return (0, 0, False, False, None) if len(s_hand) == 1 else None
    if not s_no_allows_e:
        return None
    if op == 24:
        return (3, 0, False, False, None) if len(s_hand) == 1 else None
    if op == 5:
        return (3, 0, False, False, None)
    if op == 6:
        if cand_meta.has_can_play and not cand_meta.beh.can_play(
            state, s_player, cand_card
        ):
            return None
        return (2, 0, False, False, None)
    if op == 7:
        rem_e = orig_e - known_cost
        return (rem_e if rem_e > 0 else 0, 0, False, False, None)
    if op == 8:
        return (len(s_played), 0, False, False, None)
    if op == 14:
        if s_pf & _PF_HAS_COPIED:
            return None
        for pc in s_played:
            if pc.definition.cost >= 5:
                return (3, 0, False, False, None)
        return (0, 0, False, False, None)
    if s_pf & (_PF_MAY_HAVE_TAG_MARKERS | _PF_HAS_COPIED):
        return None
    if op == 9:
        cnt = 0
        for pc in s_played:
            if meta_by_slug[pc.definition.slug].tag_mask & _MASK_EXERCISE:
                cnt += 1
        return (cnt, 0, False, False, None)
    if op == 10:
        cnt = 0
        for pc in s_played:
            if meta_by_slug[pc.definition.slug].tag_mask & _MASK_OUTDOORS:
                cnt += 1
        return (cnt, 0, False, False, None)
    if op == 11:
        cnt = 0
        for pc in s_played:
            if meta_by_slug[pc.definition.slug].tag_mask & _MASK_RELAX:
                cnt += 1
        return (cnt * 2, 0, False, False, None)
    if op == 12:
        for pc in s_played:
            if meta_by_slug[pc.definition.slug].tag_mask & _MASK_FOOD:
                return (3, 0, False, False, None)
        return (0, 0, False, False, None)
    if op == 13:
        for pc in s_played:
            if meta_by_slug[pc.definition.slug].tag_mask & _MASK_FOOD:
                return (2, 0, False, False, None)
        return (0, 0, False, False, None)
    if op == 15:
        gain = (
            2
            if (
                s_played
                and (
                    meta_by_slug[s_played[-1].definition.slug].tag_mask
                    & _MASK_OUTDOORS
                )
            )
            else 0
        )
        return (gain, 0, False, False, None)
    return None


def _materialize_plain_child(
    parent_st: Game,
    player_index: int,
    hand_idx: int,
    known_cost: int,
    gain_e: int,
    c_pf_new: int,
    gain_f: int = 0,
    set_asleep: bool = False,
    extra_markers: dict[str, object] | None = None,
) -> Game:
    """Materialize a deferred in-place plain or pure-stat play step."""
    child = _hydrogen_copy_game_for_play(parent_st, player_index)
    c_pl = child.players[player_index]
    c_pl.energy = c_pl.energy - known_cost + gain_e
    if gain_f:
        c_pl.fun += gain_f
    if set_asleep:
        c_pl.asleep = True
    raw_card = c_pl.hand.pop(hand_idx)
    if child._played_items_shared and (  # type: ignore[attr-defined]
        c_pf_new
        & (_PF_ALL_TAGS | _PF_HAS_CARD_PLAY | _PF_HAS_ACQUIRE | _PF_HAS_PICK)
    ):
        child._unshare_played_items()  # type: ignore[attr-defined]
    if extra_markers is not None:
        played_markers = dict(extra_markers)
        if gain_e:
            played_markers["_gave_energy"] = True
    elif gain_e:
        played_markers = {"_gave_energy": True}
    else:
        played_markers = {}
    played_card = CardInstance(
        raw_card.instance_id,
        raw_card.definition,
        False,
        played_markers,
    )
    c_pl.played_today.append(played_card)
    child._p_flags[player_index] = c_pf_new  # type: ignore[attr-defined]
    return child


def _compute_blocked_tag_mask(
    player: PlayerState,
    pf: int,
    meta_by_slug: dict[str, _HydrogenDefnMeta],
) -> int:
    """Return bitmask of card tags that remain order-sensitive on this board."""
    if not (pf & _BLOCKED_MASK_TRIGGERS):
        return 0
    if pf & (_PF_ALL_TAGS | _PF_MAY_HAVE_TAG_MARKERS | _PF_HAS_COPIED):
        return 0xFF
    blocked = 0
    if pf & _PF_HAS_EUCHRE_AWARDS:
        blocked |= _MASK_FOOD | _MASK_BOARD_GAME
    if pf & _PF_HAS_GENERIC_TOM_PLAY:
        return 0xFF
    if pf & _PF_HAS_PHOTO_SHOOT:
        blocked |= _MASK_SOCIAL | _MASK_RELAX
    if pf & _PF_HAS_NEXT_TAG_MOD:
        played = player.played_today
        suffix_union = 0
        suffix_ex_cnt = 0
        for i in range(len(played) - 1, -1, -1):
            c = played[i]
            cs = c.definition.slug
            m = meta_by_slug[cs]
            if m.played_board_flags & _PF_HAS_NEXT_TAG_MOD:
                mf_k = m.mod_fun_kind
                if (
                    mf_k == _MF_NEXT_TAG_ADD
                    or mf_k == _MF_NEXT_TAG_COST_ADD
                    or mf_k == _MF_NEXT_TAG_MUL2
                ):
                    if not (suffix_union & m.mod_fun_mask):
                        blocked |= m.mod_fun_mask
                elif mf_k == _MF_NTH_TAG_MUL2:
                    if m.mod_fun_mask == _MASK_EXERCISE and m.mod_fun_ord == 2:
                        if suffix_ex_cnt < 2:
                            blocked |= _MASK_EXERCISE
                    else:
                        return 0xFF
                elif mf_k == _MF_ADE:
                    if (
                        not (suffix_union & _MASK_EXERCISE)
                        and any(
                            meta_by_slug[played[j].definition.slug].tag_mask
                            & _MASK_FOOD
                            for j in range(i)
                        )
                    ):
                        blocked |= _MASK_EXERCISE
                elif mf_k == _MF_GENERIC:
                    return 0xFF
                cm_k = m.cost_mod_kind
                if (
                    cm_k == _CM_NEXT_TAG_ADD
                    or cm_k == _CM_NEXT_TAG_SET
                    or cm_k == _CM_NEXT_TAG_HALF
                ):
                    if not (suffix_union & m.cost_mod_mask):
                        blocked |= m.cost_mod_mask
                elif cm_k == _CM_GENERIC:
                    return 0xFF
            tm = m.tag_mask
            suffix_union |= tm
            if tm & _MASK_EXERCISE:
                suffix_ex_cnt += 1
    return blocked


def _inline_opp_today_fun(
    terminal_sim: Game,
    players: Sequence[PlayerState],
    p_flags_arr: Sequence[int],
    baseline: Sequence[int],
    opp_fun_local: list[int | None],
    o_i: int,
    ctx_dict: dict | None,
) -> int:
    """Return opponent today Fun, caching only when opponent score is static."""
    opp_val = opp_fun_local[o_i]
    if opp_val is not None:
        return opp_val
    if ctx_dict is not None and not (p_flags_arr[o_i] & _OPP_TODAY_DYNAMIC_PF):
        ot_key = ("ot", o_i)
        opp_val = ctx_dict.get(ot_key)
        if opp_val is None:
            o_sc, _ = terminal_sim.score_played_today_fast(o_i)  # type: ignore[attr-defined]
            opp_val = (players[o_i].fun + o_sc) - baseline[o_i]
            ctx_dict[ot_key] = opp_val
    else:
        o_sc, _ = terminal_sim.score_played_today_fast(o_i)  # type: ignore[attr-defined]
        opp_val = (players[o_i].fun + o_sc) - baseline[o_i]
    opp_fun_local[o_i] = opp_val
    return opp_val

_SC_PLAIN_CQ_CODES: frozenset[int] = frozenset(
    {
        _CQ_NONE,
        _CQ_GROUP_DINNER,
        _CQ_BOUGIE_COFFEE_MACHINE,
        _CQ_CHEESY_PHONE_GAME,
    }
)

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
    "7-wonders": 0.050000,
    "address-the-food": -1.035186,
    "adventure-race": 0.465357,
    "after-dinner-entertainment": 0.200000,
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
    "cities-and-knights": 1.150000,
    "civilization-the-boardgame": 0.873473,
    "classic-book": 0.215115,
    "cliff-climbing": -0.828539,
    "colouring": 0.150000,
    "couch-tube": 2.521061,
    "dates-first-nos": 0.600000,
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
    "epic-prank": 0.550000,
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
    "fit-to-print": -0.350000,
    "floating": 0.403721,
    "forced-family-fun": -1.053715,
    "grocery-store-run": -1.115974,
    "group-dinner": -0.270222,
    "high-end-red": -0.169239,
    "high-end-white": -1.054279,
    "hold-the-baby": 1.224265,
    "ice-cream-sandwich": 0.192919,
    "ice-wine": 0.698685,
    "istanbul": 3.150000,
    "johnny-appleseed": -0.721993,
    "kayak": 0.365404,
    "keeper": 0.376220,
    "kneeboard": -0.576329,
    "last-years-shirt": 0.250000,
    "leftovers": -0.750000,
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
    "nos-shirt": 1.300000,
    "outdoor-movie": 1.058520,
    "paddleboard": 1.950000,
    "paddleboat": 0.654633,
    "paint": -0.006023,
    "paint-rocks": 0.041167,
    "photo-shoot": 0.639717,
    "play-with-the-kids": -0.300000,
    "poker": -0.779202,
    "ponyback": -0.123043,
    "potato-pancakes": -0.736249,
    "prime-picnic-table": 0.850664,
    "pudding-chomeur": 0.810791,
    "puerto-rico": 0.487236,
    "quacks": -0.156225,
    "read-the-radar": -0.652858,
    "risk": -0.228392,
    "rouladen": -0.080000,
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
    "solo": 0.650000,
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
    "7-wonders": 0.000000,
    "address-the-food": 0.324543,
    "adventure-race": 0.825986,
    "after-dinner-entertainment": 0.150000,
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
    "cities-and-knights": -0.450000,
    "civilization-the-boardgame": -0.024953,
    "classic-book": 0.247484,
    "cliff-climbing": -0.045467,
    "colouring": -0.200000,
    "couch-tube": 0.358086,
    "dates-first-nos": -1.500000,
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
    "epic-prank": -1.200000,
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
    "fit-to-print": -0.250000,
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
    "last-years-shirt": -0.050000,
    "leftovers": -0.250000,
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
    "nos-shirt": -0.350000,
    "outdoor-movie": 0.516912,
    "paddleboard": 0.650000,
    "paddleboat": -0.619066,
    "paint": -0.017621,
    "paint-rocks": -0.111405,
    "photo-shoot": -0.194591,
    "play-with-the-kids": -0.200000,
    "poker": -0.100519,
    "ponyback": -0.286219,
    "potato-pancakes": 0.268960,
    "prime-picnic-table": 0.419999,
    "pudding-chomeur": -0.060819,
    "puerto-rico": -0.059233,
    "quacks": 0.388054,
    "read-the-radar": -0.033005,
    "risk": -0.233915,
    "rouladen": 0.700000,
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
    "solo": 0.350000,
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
_OSMIUM_CONT_SPECIAL_MSET_MASK = 0
_OSMIUM_CONT_AFTER_MSET_MASK = 0
_OSMIUM_NEUTRAL3_MSET_MASK = 0
_CONT_CACHE_EXCL_PF = _PF_ALL_TAGS | _PF_MAY_HAVE_TAG_MARKERS | _PF_HAS_COPIED
_OSMIUM_SIM_GAME_CLS_CACHED: type | None = None


_RELAX_PAYOFF_TITLES: frozenset[str] = frozenset(
    {
        "Sweet Lawn Chair",
        "Zero Gravity Chair",
        "Spa Day",
        "Outdoor Movie",
        "Potato Pancakes",
        "Classic Book",
        "Water Trampoline",
        "Fancy Floatie",
        "Photo Shoot",
    }
)


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
    bf_first_ok: bool
    ade_ex_fun: int
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
_NEUTRAL_Q_OK_D0: dict[str, bool] = {}
_NEUTRAL_Q_OK_D1: dict[str, bool] = {}
_SLUG_RANK_GETTER = attrgetter("slug_rank")


def _init_cont_special_masks() -> None:
    """Populate title-multiset bitmasks for continuation caching."""
    global _OSMIUM_CONT_SPECIAL_MSET_MASK
    global _OSMIUM_CONT_AFTER_MSET_MASK, _OSMIUM_NEUTRAL3_MSET_MASK
    _OSMIUM_NEUTRAL3_MSET_MASK = 31 * _OSMIUM_TITLE_MSET_BIT_BY_SLUG.get(
        "_beryllium_neutral_3", 0
    )
    _OSMIUM_CONT_SPECIAL_MSET_MASK = (
        sum(
            31 * _OSMIUM_TITLE_MSET_BIT_BY_SLUG[s]
            for s in (
                "after-dinner-entertainment",
                "dates-first-nos",
                "cities-and-knights",
                "epic-prank",
                "nos-shirt",
            )
        )
        | _OSMIUM_NEUTRAL3_MSET_MASK
    )
    cont_after_m = 0
    for m_ca in _HYDROGEN_META_BY_SLUG.values():
        if m_ca.played_board_flags & _PF_HAS_CONT_AFTER:
            cont_after_m |= 31 * _OSMIUM_TITLE_MSET_BIT_BY_SLUG.get(m_ca.slug, 0)
    _OSMIUM_CONT_AFTER_MSET_MASK = cont_after_m


def _ensure_osmium_caches() -> None:
    global _OSMIUM_SIM_GAME_CLS_CACHED
    if (
        _OSMIUM_SPEC_BY_SLUG
        and _NEUTRAL_Q_OK_D0
        and _OSMIUM_SIM_GAME_CLS_CACHED is not None
        and _OSMIUM_CONT_SPECIAL_MSET_MASK
        and _OSMIUM_CONT_AFTER_MSET_MASK
    ):
        return
    _ensure_hydrogen_caches()
    _OSMIUM_SIM_GAME_CLS_CACHED = _get_hydrogen_sim_game_cls()
    if (
        (not _OSMIUM_CONT_SPECIAL_MSET_MASK or not _OSMIUM_CONT_AFTER_MSET_MASK)
        and _OSMIUM_TITLE_MSET_BIT_BY_SLUG
    ):
        _init_cont_special_masks()
    if _OSMIUM_SPEC_BY_SLUG and _NEUTRAL_Q_OK_D0:
        return
    metas = sorted(_HYDROGEN_META_BY_SLUG.values(), key=lambda m: m.slug)
    zero_w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for m_q in metas:
        if m_q.q_const_eligible and not m_q.has_own_cost_mod:
            q0 = _eval_q_const_with_arr(m_q, zero_w, 0, True)
            q1 = _eval_q_const_with_arr(m_q, zero_w, 0, False)
            _NEUTRAL_Q_OK_D0[m_q.slug] = 0.2 <= q0 <= 1.6
            _NEUTRAL_Q_OK_D1[m_q.slug] = 0.2 <= q1 <= 1.6
    if _OSMIUM_SPEC_BY_SLUG:
        return
    unique_titles = sorted({m.title for m in metas})
    assert len(unique_titles) < 256 and len(metas) < 512
    title_to_id = {t: i + 1 for i, t in enumerate(unique_titles)}
    for m_item in metas:
        tid = title_to_id[m_item.title]
        _OSMIUM_TITLE_ID_BY_SLUG[m_item.slug] = tid
        _OSMIUM_TITLE_MSET_BIT_BY_SLUG[m_item.slug] = 1 << (5 * (tid - 1))
    _init_cont_special_masks()
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
            net_e = 1.5
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
        if title == "Wrestle the Kids":
            pb |= 1 << 50
        if title == "Epic Duels":
            pb |= 1 << 55
        if title in _RELAX_PAYOFF_TITLES:
            pb |= 1 << 60
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
        elif title == "After Dinner Entertainment":
            bf_flags |= 64
        if title in _HAND_TO_TRUNK_TITLES:
            bf_flags |= 128
        if title in _TRUNK_PEEK_TITLES:
            bf_flags |= 256
        if (cost >= 7 and bf_int >= 8) or title in _CUTLERY_BOMB_EXTRA_TITLES:
            bf_flags |= 512
        if title == "Bougie Coffee Machine":
            bf_flags |= 1024
        elif title == "Quacks":
            bf_flags |= 2048
        elif title == "Poker":
            bf_flags |= 4096
        if title in _BCM_PEEK_TITLES:
            bf_flags |= 8192
        if title == "Photo Shoot":
            t_kind = 1
        elif title == "Date's First Nos":
            t_kind = 2
        elif title == "Wildlife Spotting":
            t_kind = 3
        elif title == "Epic Prank":
            t_kind = 4
        else:
            t_kind = 0
        ade_ex_f = (
            bf_int
            if (
                "Exercise" in defn.tags
                and not meta.must_play_first
                and bf_int >= 4
            )
            else 0
        )
        _OSMIUM_SPEC_BY_SLUG[defn.slug] = _OsmiumCardSpec(
            defn=defn,
            slug=defn.slug,
            slug_rank=rank_i,
            title_id=title_to_id[title],
            cost=cost,
            bf_int=bf_int,
            bf_float=bf_float,
            bf_pos=bf_int > 0,
            bf_first_ok=(bf_int > 0 and not meta.must_play_late),
            ade_ex_fun=ade_ex_f,
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
    tom_epic_prank = 0.0
    tom_has_relax_bonus = False
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
            elif tk == 4:
                tom_epic_prank += 1.0
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
        tom_has_relax_bonus = tom_tag_arr is not None and tom_tag_arr[5] > 0.0

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
    max_first_cost = 7 + (tom_e if tom_e > 0 else 0)
    best_first_fun = 0.0
    max_ade_ex_fun = 0
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
        if (
            sp.bf_first_ok
            and sp.cost <= max_first_cost
            and sp.bf_float > best_first_fun
        ):
            best_first_fun = sp.bf_float
        if sp.ade_ex_fun > max_ade_ex_fun:
            max_ade_ex_fun = sp.ade_ex_fun
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
        if (b_flags & 1) and (
            ((combo_hi >> 35) & 0x1F) > 0
            or tom_photo > 0.0
            or tom_has_relax_bonus
        ):
            relax_syn = (
                float((combo_hi >> 35) & 0x1F)
                + tom_photo
                + (1.0 if (tom_has_relax_bonus and tom_photo == 0.0) else 0.0)
            )
            val += 1.9 * (relax_syn if relax_syn < 2.0 else 2.0)
        if (b_flags & 2) and (acquirer_cnt + tom_photo) > 0:
            acq_p = float(acquirer_cnt) + tom_photo
            val += w[71] * float(acq_p if acq_p < 3.0 else 3.0)
        if (b_flags & 4) and copy_food_cnt > 0:
            val += w[72] * float(copy_food_cnt if copy_food_cnt < 2 else 2)
        if (b_flags & 8) and trunk_setter_cnt > 0:
            if (b_flags & 128) and (b_flags & 512):
                val += 3.0
            elif b_flags & 256:
                val += 1.8
            elif (b_flags & 128) and heavy_cnt > 0:
                val += 1.6
            else:
                val += 0.8
        if b_flags & 1024:
            if (b_flags & 128) and high_food_cnt > 0:
                val += 1.6
            elif b_flags & 8192:
                val += 1.2
        if (b_flags & 2048) and (b_flags & 256):
            val += 1.5
        if (b_flags & 4096) and trunk_setter_cnt > 0:
            val += 1.1
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

    if tom_epic_prank > 0.0:
        capped_bf = best_first_fun if best_first_fun < 18.0 else 18.0
        bf_bonus = 0.75 * (capped_bf - 3.0) if capped_bf > 3.0 else 0.0
        val += tom_epic_prank * (3.6 + bf_bonus)
    if (b_flags & 64) and ((acc_a >> 12) & 0x3F) and max_ade_ex_fun >= 4:
        ex_gain = float(max_ade_ex_fun - 2)
        val += 0.35 * (ex_gain if ex_gain < 8.0 else 8.0)

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


def _projected_end_hand_size(sim: Game, player_index: int) -> int:
    """Estimate a player's hand size after end-of-day hand-returning effects."""
    player = sim.players[player_index]
    hand_size = len(player.hand)
    has_scout = False
    nos_shirt_cnt = 0
    ck_cnt = 0
    for card in player.played_today:
        markers = card.markers
        title = (
            card.definition.title
            if (not markers or "_copied_definition" not in markers)
            else _effective_card_title(card)
        )
        if title == "Chuck a Frisbee":
            hand_size += 1
        elif title == "Nos Shirt":
            nos_shirt_cnt += 1
        elif title == "Cities and Knights":
            ck_cnt += 1
        elif title == "Scout Other Cottages":
            has_scout = True

    if nos_shirt_cnt > 0:
        start_f = (
            sim._fun_at_start_of_day[player_index]
            if sim._fun_at_start_of_day is not None
            else player.fun
        )
        proj_day_fun = (player.fun - start_f) + sum(
            sim.card_fun(player_index, pc) for pc in player.played_today
        )
        if proj_day_fun > 7:
            hand_size += nos_shirt_cnt

    if ck_cnt > 0 and sim.day < DAY_COUNT:
        if cities_and_knights_condition_met(
            pc.tags for pc in player.played_today
        ):
            avail_pool = len(sim.suitcase) + len(sim.trunk) + len(sim.discard)
            hand_size += min(ck_cnt * 2, avail_pool)

    if not has_scout or not sim.suitcase:
        return hand_size

    # Mirror Game.end_day() resolution order so contested or duplicate
    # Scout Other Cottages marks never double-count the same Suitcase card.
    claimed_sc: set[int] = set()
    for p_idx in sim.player_order():
        for card in sim.players[p_idx].played_today:
            markers = card.markers
            title = (
                card.definition.title
                if (not markers or "_copied_definition" not in markers)
                else _effective_card_title(card)
            )
            if title != "Scout Other Cottages":
                continue
            owner_marker = f"_scout_energy_cube_{card.instance_id}"
            for sc in sim.suitcase:
                sc_id = sc.instance_id
                if sc_id not in claimed_sc and sc.markers.get(owner_marker):
                    claimed_sc.add(sc_id)
                    if p_idx == player_index:
                        hand_size += 1
        if p_idx == player_index:
            break
    return hand_size


def _eval_plain_sc_spec(
    sp: tuple[float, bool, bool, int, tuple[int, ...]],
    w_arr: Sequence[float],
    w_mask: int,
) -> float:
    base_v, is_cheap, must_first, t_mask, t_bits = sp
    val_q = base_v
    if w_mask & t_mask:
        tm_q = 0.0
        if is_cheap:
            for b_q in t_bits:
                wq = w_arr[b_q]
                if wq != 0.0:
                    tm_q += wq * 0.78
                    if wq > 0.0:
                        tm_q += 0.38 * (wq if wq < 2.5 else 2.5)
        else:
            for b_q in t_bits:
                wq = w_arr[b_q]
                if wq != 0.0:
                    tm_q += wq * 0.78
        if must_first and tm_q > 0.0:
            tm_q *= 0.30
        val_q += tm_q
    return val_q


def _suitcase_pick1_edge(
    sim: Game,
    player_index: int,
    *,
    _sc_costs: Sequence[int | None] | None = None,
    _sc_metas: Sequence | None = None,
    _sc_plain: Sequence | None = None,
) -> float:
    """Measure best-versus-later quality gap across the visible Suitcase."""
    sc = sim.suitcase
    if not sc:
        return 0.0
    player = sim.players[player_index]
    is_hsim = type(sim) is _OSMIUM_SIM_GAME_CLS_CACHED
    is_canon = is_hsim and sim._all_canonical_defs  # type: ignore[attr-defined]
    is_d6_edge = sim.day >= DAY_COUNT
    tom_unmarked = True
    for tc in player.tomorrow_cards:
        if tc.markers:
            tom_unmarked = False
            break
    if (
        _sc_plain is not None
        and None not in _sc_plain
        and is_canon
        and player.character is None
        and sim.daily_condition is None
        and not sim._has_nonplain_hand  # type: ignore[attr-defined]
        and not (
            sim._p_flags[player_index]  # type: ignore[attr-defined]
            & (_PF_HAS_COPIED | _PF_ALL_TAGS | _PF_MAY_HAVE_TAG_MARKERS)
        )
        and len(player.hand) < 250
        and len(player.played_today) < 250
        and tom_unmarked
    ):
        w_arr, w_mask = _build_w_arr_only(player)
        vals = [
            (
                _eval_q_const_with_arr(sp[1], w_arr, w_mask, is_d6_edge)
                if len(sp) == 2
                else _eval_plain_sc_spec(sp, w_arr, w_mask)
            )
            for sp in _sc_plain
        ]
    else:
        sc_plain_tags = False
        if _sc_metas is not None and is_canon and player.character is None:
            sc_plain_tags = True
            for c_sc in sc:
                cm_sc = c_sc.markers
                if cm_sc and (
                    ("_all_tags" in cm_sc and cm_sc["_all_tags"])
                    or ("_extra_tags" in cm_sc and cm_sc["_extra_tags"])
                ):
                    sc_plain_tags = False
                    break
        build_dict = not sc_plain_tags
        wanted, ctx = _build_wanted_and_ctx(
            player, is_canon, _build_dict=build_dict
        )
        if (
            _sc_plain is not None
            and is_canon
            and player.character is None
            and sim.daily_condition is None
            and ctx.all_canonical
            and ctx.wanted_ref is wanted
            and ctx.w_arr is not None
            and not sim._has_nonplain_hand  # type: ignore[attr-defined]
            and not (
                sim._p_flags[player_index]  # type: ignore[attr-defined]
                & (_PF_HAS_COPIED | _PF_ALL_TAGS | _PF_MAY_HAVE_TAG_MARKERS)
            )
            and tom_unmarked
        ):
            w_arr = ctx.w_arr
            w_mask = ctx.w_mask
            vals = [
                (
                    (
                        _eval_q_const_with_arr(sp[1], w_arr, w_mask, is_d6_edge)
                        if len(sp) == 2
                        else _eval_plain_sc_spec(sp, w_arr, w_mask)
                    )
                    if sp is not None
                    else _hydrogen_quick_card_desirability(
                        sim,
                        player_index,
                        sc[i],
                        wanted,
                        cost=(
                            _sc_costs[i]
                            if (_sc_costs is not None and _sc_costs[i] is not None)
                            else sim.energy_cost(player_index, sc[i])
                        ),
                        _ctx=ctx,
                        _in_hand=False,
                        _meta=_sc_metas[i] if _sc_metas is not None else None,
                    )
                )
                for i, sp in enumerate(_sc_plain)
            ]
        else:
            costs = (
                _sc_costs
                if _sc_costs is not None
                else (
                    sim.cards_energy_costs(player_index, sc)  # type: ignore[attr-defined]
                    if is_hsim
                    else None
                )
            )
            vals = [
                _hydrogen_quick_card_desirability(
                    sim,
                    player_index,
                    sc[i],
                    wanted,
                    cost=costs[i] if costs is not None else None,
                    _ctx=ctx,
                    _in_hand=False,
                    _meta=_sc_metas[i] if _sc_metas is not None else None,
                )
                for i in range(len(sc))
            ]
    vals.sort(reverse=True)
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
    tom_epic_prank = 0.0
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
        elif t_title == "Epic Prank":
            tom_epic_prank += 1.0
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
    tom_has_relax_bonus = (
        tom_tag_bonuses.get("Relax", 0.0) > 0.0 and not all_cards_zeroed
    )
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
    has_ade = False
    has_paddleboat = False
    acquirer_cnt = 0
    has_leftovers = False
    copy_food_cnt = 0
    high_food_cnt = 0
    has_cutlery = False
    trunk_setter_cnt = 0
    has_hand_to_trunk = False
    has_trunk_peek = False
    has_cutlery_bomb = False
    has_bcm = False
    has_quacks = False
    has_poker = False
    has_bcm_peek = False
    has_adv_race = False
    has_civ = False
    empty_hand_cnt = 0
    wrestle_pwtk_cnt = 0
    epic_duels_cnt = 0
    relax_payoff_cnt = 0
    max_first_cost = 7 + (tom_e if tom_e > 0 else 0)
    best_first_fun = 0.0
    max_ade_ex_fun = 0

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
                net_e_pos += 1.5
            elif title in {"Potato Pancakes", "Istanbul"}:
                net_e_pos += 2.0
            elif title == "Shady Spot":
                net_e_pos += 2.5
            continue

        eff_h_len += 1
        hand_meta_tags.append((meta, ctags))
        bf_int = hc.effective_base_fun + meta.immediate_fun_gain
        bf = float(bf_int)
        total_cost += cost
        if bf > 0.0:
            base_funs.append(bf)
        if (
            bf_int > 0
            and not meta.must_play_late
            and cost <= max_first_cost
            and bf > best_first_fun
        ):
            best_first_fun = bf
        if (
            "Exercise" in ctags
            and not meta.must_play_first
            and bf_int >= 4
            and bf_int > max_ade_ex_fun
        ):
            max_ade_ex_fun = bf_int
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
            net_e_pos += 1.5
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
        elif title == "After Dinner Entertainment":
            has_ade = True
        elif title == "Paddleboat":
            has_paddleboat = True
        elif title == "Leftovers":
            has_leftovers = True
        elif title == "Assorted Cutlery":
            has_cutlery = True
        elif title == "Bougie Coffee Machine":
            has_bcm = True
        elif title == "Quacks":
            has_quacks = True
        elif title == "Poker":
            has_poker = True
        elif title == "Adventure Race":
            has_adv_race = True
        elif title == "Civilization the Boardgame":
            has_civ = True
        elif title == "Epic Duels":
            epic_duels_cnt += 1
        if title in _RELAX_PAYOFF_TITLES:
            relax_payoff_cnt += 1

        if meta.is_acquirer:
            acquirer_cnt += 1
        if title in _HYDROGEN_COPY_FOOD_TITLES:
            copy_food_cnt += 1
        if "Food" in ctags and cost >= 2 and title != "Leftovers":
            high_food_cnt += 1
        if title in _TRUNK_REORDER_TITLES:
            trunk_setter_cnt += 1
        if title in _HAND_TO_TRUNK_TITLES:
            has_hand_to_trunk = True
        if title in _TRUNK_PEEK_TITLES:
            has_trunk_peek = True
        if title in _BCM_PEEK_TITLES:
            has_bcm_peek = True
        if (cost >= 7 and bf_int >= 8) or title in _CUTLERY_BOMB_EXTRA_TITLES:
            has_cutlery_bomb = True
        if title in {"Wakesurf", "Drop a Ski"}:
            empty_hand_cnt += 1
        if title == "Wrestle the Kids":
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
    if has_colouring and (
        relax_payoff_cnt > 0 or tom_photo > 0.0 or tom_has_relax_bonus
    ):
        relax_syn = (
            float(relax_payoff_cnt)
            + tom_photo
            + (1.0 if (tom_has_relax_bonus and tom_photo == 0.0) else 0.0)
        )
        val += 1.9 * (relax_syn if relax_syn < 2.0 else 2.0)
    if has_paddleboat and (acquirer_cnt + tom_photo) > 0:
        val += w[71] * float(min(float(acquirer_cnt) + tom_photo, 3.0))
    if has_leftovers and copy_food_cnt > 0:
        val += w[72] * float(min(copy_food_cnt, 2))
    if has_cutlery and trunk_setter_cnt > 0:
        if has_hand_to_trunk and has_cutlery_bomb:
            val += 3.0
        elif has_trunk_peek:
            val += 1.8
        elif has_hand_to_trunk and heavy_cnt > 0:
            val += 1.6
        else:
            val += 0.8
    if has_bcm:
        if has_hand_to_trunk and high_food_cnt > 0:
            val += 1.6
        elif has_bcm_peek:
            val += 1.2
    if has_quacks and has_trunk_peek:
        val += 1.5
    if has_poker and trunk_setter_cnt > 0:
        val += 1.1
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

    if tom_epic_prank > 0.0 and not all_cards_zeroed:
        capped_bf = best_first_fun if best_first_fun < 18.0 else 18.0
        bf_bonus = 0.75 * (capped_bf - 3.0) if capped_bf > 3.0 else 0.0
        val += tom_epic_prank * (3.6 + bf_bonus)
    if has_ade and tag_cnts[2] > 0 and max_ade_ex_fun >= 4:
        ex_gain = float(max_ade_ex_fun - 2)
        val += 0.35 * (ex_gain if ex_gain < 8.0 else 8.0)

    is_d1 = days_left == 1
    slug_weights = _OSMIUM_SLUG_WEIGHTS
    slug_d1_weights = _OSMIUM_SLUG_D1_WEIGHTS
    for slug in sorted(slug_seen):
        w_slug = slug_seen[slug]
        val += slug_weights.get(slug, 0.0) * w_slug
        if is_d1:
            val += slug_d1_weights.get(slug, 0.0) * w_slug

    return val


def _trunk_top_keep_prob(
    game: Game,
    player_index: int,
    *,
    has_epic_duels: bool = False,
    turns_ahead: int = 0,
) -> float:
    """Estimate P(top of Trunk remains undisturbed until our next play)."""
    if has_epic_duels:
        return 1.0
    is_hsim = type(game) is _OSMIUM_SIM_GAME_CLS_CACHED
    p_keep = 1.0
    for idx, opp in enumerate(game.players):
        if (
            idx == player_index
            or opp.asleep
            or not opp.hand
            or opp.skipped_turns > turns_ahead
        ):
            continue
        eff_e = opp.energy - 2 * turns_ahead
        eff_h = len(opp.hand) - turns_ahead
        if eff_h <= 0 or eff_e < 0:
            q = 0.02
        else:
            if is_hsim:
                has_trig = bool(
                    game._p_flags[idx]  # type: ignore[attr-defined]
                    & (_PF_HAS_PHOTO_SHOOT | _PF_HAS_EUCHRE_AWARDS)
                )
            else:
                has_trig = any(
                    _effective_card_title(c) == "Photo Shoot"
                    for c in opp.tomorrow_cards
                ) or any(
                    _effective_card_title(c) == "Euchre Awards Ceremony"
                    for c in opp.played_today
                )
            if has_trig:
                q = 0.75 if eff_e >= 2 else (0.45 if eff_e == 1 else 0.15)
            else:
                q = 0.13 if eff_e >= 2 else (0.10 if eff_e == 1 else 0.06)
        p_keep *= 1.0 - q
    return p_keep


def _fast_reachable_energy(
    game: Game,
    player_index: int,
    *,
    exclude_card: CardInstance | None = None,
) -> int:
    """Estimate reachable Energy today from current Energy plus easy hand gainers."""
    player = game.players[player_index]
    base_e = player.energy
    for pc in player.played_today:
        if _effective_card_title(pc) == "Sleep In":
            return base_e
    has_food = any("Food" in pc.tags for pc in player.played_today)
    has_food_hand = False
    for hc in player.hand:
        if hc is not exclude_card and "Food" in hc.tags:
            has_food_hand = True
            break
    extra = 0
    hand_len = len(player.hand) - (1 if exclude_card is not None else 0)
    played_nonempty = bool(player.played_today)
    for hc in player.hand:
        if hc is exclude_card:
            continue
        hm = _get_card_meta(hc)
        t = hm.title
        if t == "Sleep In":
            if not played_nonempty:
                extra += 2
            continue
        if hm.immediate_energy_gain > 0:
            if t == "Morning Coffee" and played_nonempty:
                continue
            if t == "Afternoon Coffee" and len(player.played_today) + hand_len < 3:
                continue
            hc_cost = game.energy_cost(player_index, hc)
            if hc_cost <= base_e + extra and hm.immediate_energy_gain > hc_cost:
                extra += hm.immediate_energy_gain - hc_cost
        elif t in {"Doritos", "Ice Cream Sandwich"} and (has_food or has_food_hand):
            hc_cost = game.energy_cost(player_index, hc)
            if hc_cost <= base_e + extra:
                gain = (2 if t == "Doritos" else 3) - hc_cost
                if gain > 0:
                    extra += gain
        elif t == "Istanbul" and hand_len >= 3:
            hc_cost = game.energy_cost(player_index, hc)
            if hc_cost <= base_e + extra:
                extra += 2
    return base_e + (extra if extra < 6 else 6)


def _osmium_free_play_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float] | None = None,
    *,
    rem_energy: int | None = None,
) -> float:
    """Accurate value of playing `card` from Trunk/effect for 0 Energy."""
    val = _hydrogen_free_play_value(game, player_index, card, wanted_tags)
    player = game.players[player_index]
    days_left = DAY_COUNT - game.day
    rem_e = player.energy if rem_energy is None else max(0, rem_energy)
    title = _effective_card_title(card)
    c_tags = card.tags
    zero_tag = player.character.zero_fun_tag if player.character is not None else ""
    is_zeroed = bool(
        zero_tag
        and (
            zero_tag in c_tags
            or getattr(card.effective_behavior, "after_all_tags", False)
        )
    )

    if title == "Evening Chat":
        if days_left == 0 and rem_e <= 1 and not is_zeroed:
            val = 5.0
        else:
            val = (
                (0.0 if is_zeroed else 5.0)
                - (1.2 if days_left >= 1 else 0.0)
                - 1.5 * max(0, rem_e - 1)
            )
    elif title == "Early Bedtime":
        val = (-1.5 * rem_e) if days_left == 0 else (3.6 - 1.5 * rem_e)
    elif title == "Butter Tarts":
        allows_e = all(
            pc.effective_behavior.allows_energy_gain(game, player, pc, card)
            for pc in player.played_today
        )
        val = min(rem_e, 8) * 1.15 if allows_e else 0.0
    elif title == "Sleep In":
        val = 2.2 if rem_e <= 1 else -1.5
    elif title == "Work Call":
        if rem_e == 0:
            val = -4.0
    elif title == "Paddleboard":
        if not is_zeroed:
            val -= float(len(player.hand))
    elif title == "Evening on the Dock":
        if not is_zeroed:
            val += float(len(player.played_today))
    elif title == "Nos Book":
        if not is_zeroed:
            val += 1.0
    elif title == "Bougie Coffee Machine":
        if not is_zeroed:
            val = max(val, 5.8 if days_left >= 1 else 3.8)

    if not is_zeroed:
        has_baf = False
        has_htb = False
        has_active_ade = False
        has_active_wt = False
        has_food_played = False
        for pc in player.played_today:
            pt = _effective_card_title(pc)
            if has_active_ade and "Exercise" in pc.tags:
                has_active_ade = False
            if has_active_wt and "Relax" in pc.tags:
                has_active_wt = False
            if "Food" in pc.tags:
                has_food_played = True
            if pt == "Bring a Friend":
                has_baf = True
            elif pt == "Hold the Baby":
                has_htb = True
            elif pt == "After Dinner Entertainment" and has_food_played:
                has_active_ade = True
            elif pt == "Water Trampoline":
                has_active_wt = True
        eff_f = float(card.effective_base_fun)
        if title == "Paddleboard":
            eff_f -= float(len(player.hand))
        elif title == "Nos Book" or title == "Evening on the Dock":
            eff_f += float(len(player.played_today))
        if eff_f > 0.0:
            if has_baf:
                val += eff_f
            if has_active_ade and "Exercise" in c_tags:
                val += eff_f
            if has_active_wt and "Relax" in c_tags:
                val += eff_f
        elif has_baf and title == "Work Call":
            val -= 4.0
        if has_htb:
            val += 3.0
        if any(
            _effective_card_title(tc) == "Date's First Nos"
            for tc in player.tomorrow_cards
        ):
            val += 2.0
    return val


def _blind_cutlery_expected_value(
    game: Game, player_index: int, ac_card: CardInstance
) -> float:
    """Fast deterministic EV of the unknown free play from blind Assorted Cutlery."""
    player = game.players[player_index]
    days_left = DAY_COUNT - game.day
    val = 2.65 if days_left >= 1 else 2.05
    rem_e = max(0, player.energy)
    played = player.played_today
    n_before = max(0, len(played) - 1)
    # Early-play risk (Evening Chat / Early Bedtime / Sleep In / Paint / Beaver Burgers)
    val -= 0.07 * float(rem_e if rem_e < 7 else 7)
    # Prior-card penalty risk (Medical Advice / Scout Other Cottages / Family Baseball Game)
    val -= 0.03 * float(n_before if n_before < 6 else 6)

    has_baf = False
    has_htb = False
    has_active_ade = False
    has_active_wt = False
    has_food_played = False
    has_couch_tube = False
    has_euchre_awards = False
    scaler_bonus = 0.0
    for idx_p in range(len(played)):
        pc = played[idx_p]
        if pc is ac_card:
            continue
        pt = _effective_card_title(pc)
        p_tags = pc.tags
        if has_active_ade and "Exercise" in p_tags:
            has_active_ade = False
        if has_active_wt and "Relax" in p_tags:
            has_active_wt = False
        if "Food" in p_tags:
            has_food_played = True
        if pt == "Bring a Friend":
            has_baf = True
        elif pt == "Hold the Baby":
            has_htb = True
        elif pt == "After Dinner Entertainment" and has_food_played:
            has_active_ade = True
        elif pt == "Water Trampoline":
            has_active_wt = True
        elif pt == "Couch Tube":
            has_couch_tube = True
        elif pt == "Euchre Awards Ceremony":
            has_euchre_awards = True
        elif pt in {
            "High-End Red",
            "Sweet Lawn Chair",
            "Stretch",
            "Doxology",
            "Trekking Through History",
            "Movie",
        }:
            scaler_bonus += 0.35
        elif pt in {
            "Prime Picnic Table",
            "Bug Spray",
            "Spa Day",
            "Epic Playlist",
            "Euchre",
            "High-End White",
        }:
            scaler_bonus += 0.20

    has_photo_shoot = False
    for tc in player.tomorrow_cards:
        tt = _effective_card_title(tc)
        if tt == "Date's First Nos":
            val += 2.0
        elif tt in {"Aunterview", "Teach Kid to Ski"}:
            scaler_bonus += 0.35
        elif tt == "Fancy Floatie":
            scaler_bonus += 0.20
        elif tt == "Photo Shoot":
            has_photo_shoot = True

    if has_couch_tube and n_before == 1:
        val -= 4.0
    if has_baf:
        val += 1.85
    if has_htb:
        val += 3.0
    if has_active_ade:
        val += 0.75
    if has_active_wt:
        val += 0.40
    if has_photo_shoot or has_euchre_awards:
        val += 0.35
    val += scaler_bonus
    if (
        player.character is not None
        and player.character.zero_fun_tag
        and val > 0.0
    ):
        val *= 0.79
    return val


def _record_sim_known_trunk_order(
    game: Game,
    ordered_cards: tuple[CardInstance, ...],
    pushed_count: int | None = None,
) -> None:
    """Update sim `_known_trunk_state` after `order_cards_for_trunk`, invalidating on reshuffle."""
    prev_state = getattr(game, "_known_trunk_state", None)
    prev_dlen = getattr(game, "_known_trunk_discard_len", -1)
    rem_prev: tuple[CardInstance, ...] = ()
    if prev_state is not None and (prev_dlen < 0 or len(game.discard) >= prev_dlen):
        prev_len, prev_cards = prev_state
        popped = prev_len - len(game.trunk)
        if 0 <= popped < len(prev_cards):
            rem_prev = prev_cards[popped:]
    n_pushed = len(ordered_cards) if pushed_count is None else pushed_count
    game._known_trunk_state = (  # type: ignore[attr-defined]
        len(game.trunk) + n_pushed,
        (ordered_cards + rem_prev)[:6],
    )
    game._known_trunk_discard_len = len(game.discard)  # type: ignore[attr-defined]


class _OsmiumEvalAI(_HydrogenEvalAI):
    """Osmium tactical evaluator for trunk ordering, trunk setup discards, and Quacks."""

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        if not cards:
            return ()
        n_cards = len(cards)
        if n_cards == 1:
            order: Sequence[int] = [0]
            if game.stats is _get_shared_sim_stats():
                _record_sim_known_trunk_order(game, (cards[0],), pushed_count=1)
            return order
        first_c = cards[0]
        first_slug = first_c.definition.slug
        if not first_c.markers:
            all_same = True
            for idx_s in range(1, n_cards):
                sc_s = cards[idx_s]
                if sc_s.definition.slug != first_slug or sc_s.markers:
                    all_same = False
                    break
            if all_same:
                if game.stats is _get_shared_sim_stats():
                    _record_sim_known_trunk_order(
                        game, tuple(cards[:6]), pushed_count=n_cards
                    )
                return list(range(n_cards))

        player = game.players[player_index]
        is_hsim = type(game) is _OSMIUM_SIM_GAME_CLS_CACHED
        is_canon = is_hsim and game._all_canonical_defs  # type: ignore[attr-defined]
        wanted_tags, ctx = _build_wanted_and_ctx(player, is_canon)
        costs = game.cards_energy_costs(player_index, cards) if is_hsim else None  # type: ignore[attr-defined]
        meta_by_slug = _HYDROGEN_META_BY_SLUG
        metas = (
            [
                meta_by_slug.get(c.definition.slug) if not c.markers else None
                for c in cards
            ]
            if is_canon
            else None
        )
        quick_vals = [
            _hydrogen_quick_card_desirability(
                game,
                player_index,
                cards[i],
                wanted_tags,
                cost=costs[i] if costs is not None else None,
                _ctx=ctx,
                _in_hand=False,
                _meta=metas[i] if metas is not None else None,
            )
            for i in range(n_cards)
        ]

        reach_e = _fast_reachable_energy(game, player_index)
        cutlery_cnt = 0
        cutlery_cost = 99
        has_bcm = False
        bcm_cost = 99
        has_quacks = False
        has_poker = False
        food_sweep = False
        relax_sweep = False
        take_k = 0
        has_bcm_played = False
        for pc in player.played_today:
            if _effective_card_title(pc) == "Bougie Coffee Machine":
                has_bcm_played = True
                break
        sc_has_food = False
        sc_has_relax = False
        for sc in game.suitcase:
            sc_tags = sc.tags
            if not sc_has_food and "Food" in sc_tags:
                sc_has_food = True
            if not sc_has_relax and "Relax" in sc_tags:
                sc_has_relax = True
            if sc_has_food and sc_has_relax:
                break
        opp_cutlery_cost: int | None = None
        opp_bcm_cost: int | None = None
        opp_has_quacks = False
        opp_has_poker = False
        has_wa = False
        for hc in player.hand:
            if hc.definition.title == "Wedding Anniversary":
                has_wa = True
                break
        if has_wa:
            for o_i, opp in enumerate(game.players):
                if o_i == player_index:
                    continue
                for oc in opp.played_today:
                    ot = _effective_card_title(oc)
                    if ot == "Assorted Cutlery":
                        if opp_cutlery_cost is None or oc.definition.cost < opp_cutlery_cost:
                            opp_cutlery_cost = oc.definition.cost
                    elif ot == "Bougie Coffee Machine":
                        if opp_bcm_cost is None or oc.definition.cost < opp_bcm_cost:
                            opp_bcm_cost = oc.definition.cost
                    elif ot == "Quacks":
                        opp_has_quacks = True
                    elif ot == "Poker":
                        opp_has_poker = True

        for hc in player.hand:
            ht = hc.definition.title
            if ht == "Assorted Cutlery":
                hc_cost = game.energy_cost(player_index, hc)
                if hc_cost <= reach_e:
                    cutlery_cnt += 1
                    if hc_cost < cutlery_cost:
                        cutlery_cost = hc_cost
            elif ht == "Bougie Coffee Machine":
                hc_cost = game.energy_cost(player_index, hc)
                if hc_cost <= reach_e:
                    has_bcm = True
                    if hc_cost < bcm_cost:
                        bcm_cost = hc_cost
            elif ht == "Leftovers" and has_bcm_played:
                hc_cost = game.energy_cost(player_index, hc)
                if hc_cost <= reach_e and hc.effective_behavior.can_play(
                    game, player, hc
                ):
                    has_bcm = True
                    if hc_cost < bcm_cost:
                        bcm_cost = hc_cost
            elif ht == "Wedding Anniversary" and has_wa:
                wa_base = game.energy_cost(player_index, hc)
                if opp_cutlery_cost is not None and wa_base + opp_cutlery_cost <= reach_e:
                    cutlery_cnt += 1
                    if wa_base + opp_cutlery_cost < cutlery_cost:
                        cutlery_cost = wa_base + opp_cutlery_cost
                if opp_bcm_cost is not None and wa_base + opp_bcm_cost <= reach_e:
                    has_bcm = True
                    if wa_base + opp_bcm_cost < bcm_cost:
                        bcm_cost = wa_base + opp_bcm_cost
                if opp_has_quacks and wa_base + 1 <= reach_e:
                    has_quacks = True
                if opp_has_poker and wa_base + 2 <= reach_e:
                    has_poker = True
            elif ht == "Quacks":
                if game.energy_cost(player_index, hc) <= reach_e:
                    has_quacks = True
            elif ht == "Poker":
                if game.energy_cost(player_index, hc) <= reach_e:
                    has_poker = True
            elif ht == "Treat Store Run" and sc_has_food:
                if game.energy_cost(player_index, hc) <= reach_e:
                    food_sweep = True
            elif ht == "Fancy Floatie" and sc_has_relax:
                if game.energy_cost(player_index, hc) <= reach_e:
                    relax_sweep = True
            else:
                tk = _TRUNK_DRAW_TAKE_COUNTS.get(ht, 0)
                if tk > 0 and (ht != "Scrabble" or len(player.hand) >= 2):
                    if game.energy_cost(player_index, hc) <= reach_e and tk > take_k:
                        take_k = tk

        def _order_bcm_subset(sub_indices: list[int]) -> list[int]:
            if len(sub_indices) <= 1:
                return sub_indices
            rem_bcm_e = max(0, reach_e - bcm_cost)
            bcm_scores = {
                i: (
                    _osmium_free_play_value(
                        game,
                        player_index,
                        cards[i],
                        wanted_tags,
                        rem_energy=rem_bcm_e,
                    )
                    + 1.5
                    if "Food" in cards[i].tags
                    else quick_vals[i]
                )
                for i in sub_indices
            }
            ranked = sorted(
                sub_indices, key=lambda i: (-bcm_scores[i], i)
            )
            top3 = ranked[:3]
            rest = ranked[3:]
            top3.sort(
                key=lambda i: (
                    0 if "Food" in cards[i].tags else 1,
                    _BCM_FOOD_SYNERGY_RANK.get(_effective_card_title(cards[i]), 2),
                    -bcm_scores[i],
                    i,
                )
            )
            if has_quacks:
                rest.sort(key=lambda i: ("Outdoors" in cards[i].tags, quick_vals[i], i))
            else:
                rest.sort(key=lambda i: (quick_vals[i], i))
            return [*top3, *rest]

        indices = list(range(n_cards))
        order_list: list[int] | None = None
        if cutlery_cnt > 0:
            rem_ac_e = max(0, reach_e - cutlery_cost)
            fp_vals = [
                _osmium_free_play_value(
                    game, player_index, cards[i], wanted_tags, rem_energy=rem_ac_e
                )
                for i in indices
            ]
            if max(fp_vals) >= 1.5:
                top_candidates = [i for i in indices if fp_vals[i] >= 1.5]
                top_candidates.sort(key=lambda i: (-fp_vals[i], i))
                chosen_top = top_candidates[:cutlery_cnt]
                chosen_set = set(chosen_top)
                rest = [i for i in indices if i not in chosen_set]
                if has_bcm and rest:
                    rest = _order_bcm_subset(rest)
                elif has_quacks and rest:
                    rest.sort(
                        key=lambda i: ("Outdoors" in cards[i].tags, quick_vals[i], i)
                    )
                elif take_k > 0 and rest:
                    rest_desc = sorted(rest, key=lambda i: (-quick_vals[i], i))
                    taken = rest_desc[:take_k]
                    taken_set = set(taken)
                    tail = sorted(
                        (i for i in rest if i not in taken_set),
                        key=lambda i: (quick_vals[i], i),
                    )
                    rest = [*taken, *tail]
                elif has_poker and rest:
                    tagged = [i for i in rest if cards[i].tags]
                    if tagged:
                        p_idx = min(tagged, key=lambda i: (quick_vals[i], i))
                        tail = sorted(
                            (i for i in rest if i != p_idx),
                            key=lambda i: (quick_vals[i], i),
                        )
                        rest = [p_idx, *tail]
                    else:
                        rest.sort(key=lambda i: (quick_vals[i], i))
                else:
                    rest.sort(key=lambda i: (quick_vals[i], i))
                order_list = [*chosen_top, *rest]

        if order_list is None:
            if has_bcm:
                order_list = _order_bcm_subset(indices)
            elif has_quacks and any("Outdoors" not in cards[i].tags for i in indices):
                order_list = sorted(
                    indices,
                    key=lambda i: ("Outdoors" in cards[i].tags, quick_vals[i], i),
                )
            elif food_sweep or relax_sweep:
                want_match = [
                    i
                    for i in indices
                    if (food_sweep and "Food" in cards[i].tags)
                    or (relax_sweep and "Relax" in cards[i].tags)
                ]
                if want_match:
                    want_match.sort(key=lambda i: (-quick_vals[i], i))
                    want_set = set(want_match)
                    tail = sorted(
                        (i for i in indices if i not in want_set),
                        key=lambda i: (quick_vals[i], i),
                    )
                    order_list = [*want_match, *tail]
                elif take_k > 0:
                    desc = sorted(indices, key=lambda i: (-quick_vals[i], i))
                    taken = desc[:take_k]
                    taken_set = set(taken)
                    tail = sorted(
                        (i for i in indices if i not in taken_set),
                        key=lambda i: (quick_vals[i], i),
                    )
                    order_list = [*taken, *tail]
                else:
                    order_list = sorted(indices, key=lambda i: (quick_vals[i], i))
            elif take_k > 0:
                desc = sorted(indices, key=lambda i: (-quick_vals[i], i))
                taken = desc[:take_k]
                taken_set = set(taken)
                tail = sorted(
                    (i for i in indices if i not in taken_set),
                    key=lambda i: (quick_vals[i], i),
                )
                order_list = [*taken, *tail]
            elif has_poker:
                tagged = [i for i in indices if cards[i].tags]
                if tagged:
                    p_idx = min(tagged, key=lambda i: (quick_vals[i], i))
                    tail = sorted(
                        (i for i in indices if i != p_idx),
                        key=lambda i: (quick_vals[i], i),
                    )
                    order_list = [p_idx, *tail]
                else:
                    order_list = sorted(indices, key=lambda i: (quick_vals[i], i))
            else:
                # No trunk consumer: bury strong cards away from opponents.
                order_list = sorted(indices, key=lambda i: (quick_vals[i], i))

        if game.stats is _get_shared_sim_stats():
            _record_sim_known_trunk_order(
                game,
                tuple(cards[i] for i in order_list[:6]),
                pushed_count=n_cards,
            )
        return order_list

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if not options:
            raise ValueError("No discard option was supplied")
        if len(options) == 1:
            return 0
        player = game.players[player_index]
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if (
            len(options) >= 2
            and last_played in _HAND_TO_TRUNK_TITLES
            and _has_sim_trunk_consumer(game, player_index)
        ):
            has_epic_duels = any(
                _effective_card_title(pc) == "Epic Duels"
                for pc in player.played_today
            )
            p_keep = _trunk_top_keep_prob(
                game, player_index, has_epic_duels=has_epic_duels
            )
            reach_e = _fast_reachable_energy(game, player_index)
            is_hsim = type(game) is _OSMIUM_SIM_GAME_CLS_CACHED
            is_canon = is_hsim and game._all_canonical_defs  # type: ignore[attr-defined]
            wanted_tags, ctx = _build_wanted_and_ctx(player, is_canon)

            has_wa = any(
                c.definition.title == "Wedding Anniversary" for c in options
            )
            opp_cutlery_cost: int | None = None
            opp_bcm_cost: int | None = None
            if has_wa:
                for o_i, opp in enumerate(game.players):
                    if o_i == player_index:
                        continue
                    for oc in opp.played_today:
                        ot = _effective_card_title(oc)
                        if ot == "Assorted Cutlery":
                            if opp_cutlery_cost is None or oc.definition.cost < opp_cutlery_cost:
                                opp_cutlery_cost = oc.definition.cost
                        elif ot == "Bougie Coffee Machine":
                            if opp_bcm_cost is None or oc.definition.cost < opp_bcm_cost:
                                opp_bcm_cost = oc.definition.cost

            # 1. Assorted Cutlery (or Wedding Anniversary copying opponent's)
            cutlery_cards = [
                c for c in options if c.definition.title == "Assorted Cutlery"
            ]
            wa_cutlery = [
                c
                for c in options
                if c.definition.title == "Wedding Anniversary"
                and opp_cutlery_cost is not None
            ]
            ac_holder = (
                cutlery_cards[0]
                if cutlery_cards
                else (wa_cutlery[0] if wa_cutlery else None)
            )
            if ac_holder is not None and p_keep >= 0.30:
                ac_cost = game.energy_cost(player_index, ac_holder) + (
                    0
                    if ac_holder.definition.title == "Assorted Cutlery"
                    else (opp_cutlery_cost or 0)
                )
                sole_ac = (
                    ac_holder
                    if (
                        len(cutlery_cards) + len(wa_cutlery) == 1
                    )
                    else None
                )
                if ac_cost <= reach_e:
                    best_i: int | None = None
                    best_net = float("-inf")
                    for i, cand in enumerate(options):
                        if cand is sole_ac:
                            continue
                        cand_reach = _fast_reachable_energy(
                            game, player_index, exclude_card=cand
                        )
                        if ac_cost > cand_reach:
                            continue
                        rem_e = max(0, cand_reach - ac_cost)
                        fp_val = _osmium_free_play_value(
                            game,
                            player_index,
                            cand,
                            wanted_tags,
                            rem_energy=rem_e,
                        )
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
                            opp_cost = max(0.0, fp_val - 0.95 * cand_cost)
                        elif game.day >= DAY_COUNT:
                            opp_cost = 0.0
                        else:
                            opp_cost = max(0.0, (cand_des - 0.4) * 0.55)
                        gift_pen = (
                            0.20 * max(0.0, cand_des) if cand_cost <= 3 else 0.0
                        )
                        net = (
                            p_keep * (fp_val - ac_cost * 0.90 - 1.6)
                            - opp_cost
                            - (1.0 - p_keep) * gift_pen
                        )
                        if net > best_net:
                            best_net = net
                            best_i = i
                    if best_i is not None and best_net >= 1.5:
                        return best_i

            # 2. Bougie Coffee Machine / Leftovers (after BCM) / WA copying BCM
            has_bcm_played = any(
                _effective_card_title(pc) == "Bougie Coffee Machine"
                for pc in player.played_today
            )
            bcm_cards = [
                c
                for c in options
                if c.definition.title == "Bougie Coffee Machine"
                or (
                    has_bcm_played
                    and c.definition.title == "Leftovers"
                    and c.effective_behavior.can_play(game, player, c)
                )
            ]
            wa_bcm = [
                c
                for c in options
                if c.definition.title == "Wedding Anniversary"
                and opp_bcm_cost is not None
            ]
            bcm_holder = (
                bcm_cards[0]
                if bcm_cards
                else (wa_bcm[0] if wa_bcm else None)
            )
            if bcm_holder is not None and p_keep >= 0.30:
                bcm_cost = game.energy_cost(player_index, bcm_holder) + (
                    0
                    if bcm_holder.definition.title != "Wedding Anniversary"
                    else (opp_bcm_cost or 0)
                )
                sole_bcm = (
                    bcm_holder if (len(bcm_cards) + len(wa_bcm) == 1) else None
                )
                if bcm_cost <= reach_e:
                    best_f_i: int | None = None
                    best_f_net = float("-inf")
                    for i, cand in enumerate(options):
                        if cand is sole_bcm or "Food" not in cand.tags:
                            continue
                        cand_reach = _fast_reachable_energy(
                            game, player_index, exclude_card=cand
                        )
                        if bcm_cost > cand_reach:
                            continue
                        rem_e = max(0, cand_reach - bcm_cost)
                        fp_val = _osmium_free_play_value(
                            game,
                            player_index,
                            cand,
                            wanted_tags,
                            rem_energy=rem_e,
                        )
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
                            opp_cost = max(0.0, fp_val - 0.95 * cand_cost)
                        elif game.day >= DAY_COUNT:
                            opp_cost = 0.0
                        else:
                            opp_cost = max(0.0, (cand_des - 0.4) * 0.55)
                        gift_pen = (
                            0.20 * max(0.0, cand_des) if cand_cost <= 3 else 0.0
                        )
                        net = (
                            p_keep * fp_val
                            - opp_cost
                            - (1.0 - p_keep) * gift_pen
                        )
                        if net > best_f_net:
                            best_f_net = net
                            best_f_i = i
                    if best_f_i is not None and best_f_net >= 1.5:
                        return best_f_i

            # 3. Quacks: prefer placing a weak non-Outdoors card on top rather than Outdoors
            quacks_cards = [c for c in options if c.definition.title == "Quacks"]
            if quacks_cards and p_keep >= 0.55:
                q_card = quacks_cards[0]
                q_cost = game.energy_cost(player_index, q_card)
                if q_cost <= reach_e:
                    sole_q = q_card if len(quacks_cards) == 1 else None
                    best_q_i: int | None = None
                    best_q_score = float("inf")
                    for i, cand in enumerate(options):
                        if cand is sole_q:
                            continue
                        cand_des = _hydrogen_hand_keep_value(
                            game, player_index, cand, wanted_tags, _ctx=ctx
                        )
                        q_score = cand_des - (
                            2.2 * p_keep if "Outdoors" not in cand.tags else 0.0
                        )
                        if q_score < best_q_score:
                            best_q_score = q_score
                            best_q_i = i
                    if best_q_i is not None:
                        return best_q_i

        return super().choose_card_to_discard(game, player_index, options)

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        if (
            "Quacks" in action
            and game.stats is _get_shared_sim_stats()
            and not _ACTIVE_PROBE_KEYS
        ):
            player = game.players[player_index]
            last_card = player.played_today[-1] if player.played_today else None
            if last_card is None:
                return False
            draws = int(last_card.markers.get("_quacks_draws", 0))
            known_top = _get_sim_known_trunk_top(game)
            if known_top:
                top_c = known_top[0]
                if "Outdoors" in top_c.tags:
                    return False
                reach_e = _fast_reachable_energy(game, player_index)
                has_ac = any(
                    hc.definition.title == "Assorted Cutlery"
                    and game.energy_cost(player_index, hc) <= reach_e
                    for hc in player.hand
                )
                if (
                    has_ac
                    and _osmium_free_play_value(game, player_index, top_c) >= 5.5
                ):
                    return False
                last_card.markers["_quacks_draws"] = draws + 1
                return True
            if draws >= 1:
                last_card.markers["_quacks_draws"] = draws + 1
                return False
        return super().choose_optional_action(game, player_index, action)


class _OsmiumNoConsumerTrunkEvalAI(_HydrogenEvalAI):
    """Fast sim evaluator for trunk reorder cards when no trunk consumer is held."""

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        return list(range(len(cards)))


_OSMIUM_NOOP_TRUNK_SINGLETON = _OsmiumNoConsumerTrunkEvalAI()
_OSMIUM_NOOP_TRUNK_SIM_AIS: list[object] = [
    _OSMIUM_NOOP_TRUNK_SINGLETON,
    _OSMIUM_NOOP_TRUNK_SINGLETON,
    _OSMIUM_NOOP_TRUNK_SINGLETON,
    _OSMIUM_NOOP_TRUNK_SINGLETON,
]


_OSMIUM_EVAL_AI_SINGLETON = _OsmiumEvalAI()
_OSMIUM_SIM_AIS: list[object] = [
    _OSMIUM_EVAL_AI_SINGLETON,
    _OSMIUM_EVAL_AI_SINGLETON,
    _OSMIUM_EVAL_AI_SINGLETON,
    _OSMIUM_EVAL_AI_SINGLETON,
]


class _OsmiumOverrideEvalAI(_OsmiumEvalAI):
    """Lightweight sub-decision override wrapper used inside search branching."""

    __slots__ = (
        "spend_override",
        "discard_tuple_override",
        "copy_idx_override",
        "discard_idx_override",
    )

    def __init__(
        self,
        *,
        spend_override: int | None = None,
        discard_tuple_override: tuple[int, ...] | None = None,
        copy_idx_override: int | None = None,
        discard_idx_override: int | None = None,
    ) -> None:
        super().__init__()
        self.spend_override = spend_override
        self.discard_tuple_override = discard_tuple_override
        self.copy_idx_override = copy_idx_override
        self.discard_idx_override = discard_idx_override

    def choose_card_to_discard(
        self, game: Game, player_index: int, options: Sequence[CardInstance]
    ) -> int:
        if (
            self.discard_idx_override is not None
            and 0 <= self.discard_idx_override < len(options)
        ):
            return self.discard_idx_override
        return super().choose_card_to_discard(game, player_index, options)

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


def _current_day_fun_est(sim: Game, p_idx: int) -> float:
    p = sim.players[p_idx]
    start_f = (
        sim._fun_at_start_of_day[p_idx]
        if sim._fun_at_start_of_day is not None
        else p.fun
    )
    val = float(p.fun - start_f)
    for pc in p.played_today:
        val += float(sim.card_fun(p_idx, pc))
    return val


def _poisson_beat_bg(opp: PlayerState, opp_bg: int, target_bg: int) -> float:
    if opp.asleep:
        return 1.0 if target_bg > opp_bg else 0.0
    opp_h = len(opp.hand)
    if opp_h <= 0:
        return 1.0 if target_bg > opp_bg else 0.0
    k_max = target_bg - opp_bg - 1
    if k_max < 0:
        return 0.0
    rem_plays = min(float(opp_h), 0.3 + opp.energy / 2.3)
    bg_rate = 0.28 if opp_bg > 0 else 0.17
    if opp.character is not None and opp.character.fun_tag == "Board Game":
        bg_rate = 0.35
    else:
        for tc in (*opp.played_today, *opp.tomorrow_cards):
            if _effective_card_title(tc) in {
                "Carcassonne",
                "Trekking Through History",
                "Forced Family Fun",
                "Bend the Rules",
            }:
                bg_rate = 0.35
                break
    lam = max(0.0, rem_plays * bg_rate)
    term = math.exp(-lam)
    cdf = term
    for k in range(1, k_max + 1):
        term *= lam / k
        cdf += term
    return min(1.0, cdf)


def _awake_opp_new_cards_adjustment(
    terminal_sim: Game,
    player_index: int,
    next_tomorrow_cards: Sequence[CardInstance],
    eval_sim: Game | None = None,
    *,
    solo_rescored_ids: set[int] | None = None,
    _my_day_fun: float | None = None,
    _opp_cache: dict | None = None,
    _known_has_target: bool = False,
) -> float:
    """Discount 7 Wonders, Solo, and Epic Prank when opponents can still act today."""
    players = terminal_sim.players
    player = players[player_index]
    played_today = player.played_today
    if not _known_has_target:
        has_target = False
        for c in played_today:
            m = c.markers
            eff_t = (
                c.definition.title
                if (not m or "_copied_definition" not in m)
                else _effective_card_title(c)
            )
            if eff_t == "7 Wonders" or eff_t == "Solo" or eff_t == "Epic Prank":
                has_target = True
                break
        if not has_target:
            return 0.0

    adj = 0.0
    n_players = len(players)
    my_day_fun: float | None = _my_day_fun
    max_opp_day_fun: float | None = None
    pre_after_fun: int | None = None
    opp_proj_cache: dict[int, float] = {}
    is_hsim = type(terminal_sim) is _OSMIUM_SIM_GAME_CLS_CACHED
    opp_cache_mask = (
        _PF_HAS_AWAKE_OPP_TERM | _PF_HAS_END_DAY | _PF_HAS_COMPLEX_END_DAY
    )

    def _opp_day_fun(idx: int) -> float:
        if (
            _opp_cache is not None
            and is_hsim
            and not (terminal_sim._p_flags[idx] & opp_cache_mask)  # type: ignore[attr-defined]
        ):
            ck = ("odf", idx)
            val = _opp_cache.get(ck)
            if val is None:
                val = _current_day_fun_est(terminal_sim, idx)
                _opp_cache[ck] = val
            return val
        return _current_day_fun_est(terminal_sim, idx)

    def _opp_proj(idx: int) -> float:
        val = opp_proj_cache.get(idx)
        if val is not None:
            return val
        if (
            _opp_cache is not None
            and is_hsim
            and not (terminal_sim._p_flags[idx] & opp_cache_mask)  # type: ignore[attr-defined]
        ):
            ck = ("opr", idx)
            val = _opp_cache.get(ck)
            if val is None:
                val = _projected_opponent_day_fun(terminal_sim, idx)
                _opp_cache[ck] = val
            opp_proj_cache[idx] = val
            return val
        val = _projected_opponent_day_fun(terminal_sim, idx)
        opp_proj_cache[idx] = val
        return val

    for c_idx, c in enumerate(played_today):
        m = c.markers
        eff_t = (
            c.definition.title
            if (not m or "_copied_definition" not in m)
            else _effective_card_title(c)
        )
        if eff_t == "7 Wonders":
            own_bg = sum(1 for pc in played_today if "Board Game" in pc.tags)
            left_idx = (player_index - 1) % n_players
            right_idx = (player_index + 1) % n_players
            left_opp = players[left_idx]
            right_opp = players[right_idx]
            if _opp_cache is not None:
                left_key = ("obg", left_idx)
                left_bg = _opp_cache.get(left_key)
                if left_bg is None:
                    left_bg = sum(
                        1 for oc in left_opp.played_today if "Board Game" in oc.tags
                    )
                    _opp_cache[left_key] = left_bg
                right_key = ("obg", right_idx)
                right_bg = _opp_cache.get(right_key)
                if right_bg is None:
                    right_bg = sum(
                        1 for oc in right_opp.played_today if "Board Game" in oc.tags
                    )
                    _opp_cache[right_key] = right_bg
            else:
                left_bg = sum(
                    1 for oc in left_opp.played_today if "Board Game" in oc.tags
                )
                right_bg = sum(
                    1 for oc in right_opp.played_today if "Board Game" in oc.tags
                )
            if (
                own_bg > left_bg
                and own_bg > right_bg
                and (not left_opp.asleep or not right_opp.asleep)
            ):
                p_left = _poisson_beat_bg(left_opp, left_bg, own_bg)
                p_right = _poisson_beat_bg(right_opp, right_bg, own_bg)
                mult = 1
                if c_idx == 0 and any(
                    _effective_card_title(tc) == "Epic Prank"
                    for tc in player.tomorrow_cards
                ):
                    mult *= 2
                if (
                    c_idx > 0
                    and _effective_card_title(played_today[c_idx - 1]) == "Work Call"
                ):
                    mult *= 2
                for p_i in range(c_idx):
                    if _effective_card_title(played_today[p_i]) == "Bring a Friend":
                        mult *= 2
                adj -= 4.0 * float(mult) * (1.0 - p_left * p_right)
        elif eff_t == "Solo":
            if solo_rescored_ids is not None:
                if id(c) not in solo_rescored_ids:
                    continue
                if my_day_fun is None:
                    my_day_fun = _current_day_fun_est(
                        terminal_sim, player_index
                    )
                can_score_solo = True
            elif eval_sim is not None and eval_sim is not terminal_sim:
                if pre_after_fun is None:
                    pre_after_fun = player.fun + sum(
                        terminal_sim.card_fun(player_index, pc)
                        for pc in played_today
                    )
                if eval_sim.players[player_index].fun <= pre_after_fun:
                    continue
                if my_day_fun is None:
                    my_day_fun = _current_day_fun_est(
                        terminal_sim, player_index
                    )
                can_score_solo = True
            else:
                if my_day_fun is None:
                    my_day_fun = _current_day_fun_est(
                        terminal_sim, player_index
                    )
                if max_opp_day_fun is None:
                    max_opp_day_fun = max(
                        (
                            _opp_day_fun(idx)
                            for idx in range(n_players)
                            if idx != player_index
                        ),
                        default=0.0,
                    )
                can_score_solo = my_day_fun >= max_opp_day_fun
            if can_score_solo:
                solo_fun = max(0, terminal_sim.card_fun(player_index, c))
                if solo_fun > 0:
                    p_win_day = 1.0
                    for idx in range(n_players):
                        if idx == player_index or players[idx].asleep:
                            continue
                        opp_proj = _opp_proj(idx)
                        scale = 2.0 + 0.35 * min(
                            float(players[idx].energy), 7.0
                        )
                        z = max(
                            -12.0,
                            min(12.0, (my_day_fun - opp_proj + 0.5) / scale),
                        )
                        p_win_day *= 1.0 / (1.0 + math.exp(-z))
                    adj -= float(solo_fun) * (1.0 - p_win_day)
        elif eff_t == "Epic Prank":
            if terminal_sim.day < DAY_COUNT and any(
                tc.instance_id == c.instance_id for tc in next_tomorrow_cards
            ):
                t_idx = m.get("target_player_index") if m else None
                if (
                    isinstance(t_idx, int)
                    and 0 <= t_idx < n_players
                    and not players[t_idx].asleep
                ):
                    if my_day_fun is None:
                        my_day_fun = _current_day_fun_est(
                            terminal_sim, player_index
                        )
                    opp_proj = _opp_proj(t_idx)
                    scale = 2.0 + 0.35 * min(
                        float(players[t_idx].energy), 7.0
                    )
                    z = max(
                        -12.0,
                        min(12.0, (my_day_fun - opp_proj + 0.5) / scale),
                    )
                    p_keep = 1.0 / (1.0 + math.exp(-z))
                    tom_e_ep = sum(
                        max(0, _get_card_meta(tc).tomorrow_start_energy)
                        for tc in next_tomorrow_cards
                    )
                    max_first_c = 7 + tom_e_ep
                    zero_tag = (
                        player.character.zero_fun_tag
                        if player.character is not None
                        else ""
                    )
                    best_first_f = 0.0
                    for hc in player.hand:
                        hm = hc.markers
                        hc_tags = hc.definition.tags if not hm else hc.tags
                        if zero_tag and zero_tag in hc_tags:
                            continue
                        hc_meta = _get_card_meta(hc)
                        if (
                            not hc_meta.must_play_late
                            and max(0, hc.effective_cost) <= max_first_c
                        ):
                            bf = float(
                                hc.effective_base_fun + hc_meta.immediate_fun_gain
                            )
                            if bf > best_first_f:
                                best_first_f = bf
                    capped_bf = best_first_f if best_first_f < 18.0 else 18.0
                    ep_tom_val = 3.6 + (
                        0.75 * (capped_bf - 3.0) if capped_bf > 3.0 else 0.0
                    )
                    adj -= (1.0 - p_keep) * ep_tom_val

    return adj


def _has_epic_duels_played(played: Sequence[CardInstance]) -> bool:
    for pc in played:
        m = pc.markers
        if not m or "_copied_definition" not in m:
            if pc.definition.title == "Epic Duels":
                return True
        elif _effective_card_title(pc) == "Epic Duels":
            return True
    return False


def _has_sim_trunk_consumer(sim: Game, player_index: int) -> bool:
    player = sim.players[player_index]
    has_leftovers = False
    has_wa = False
    for hc in player.hand:
        ht = hc.definition.title
        if ht in _TRUNK_CONSUMER_TITLES:
            return True
        if ht == "Leftovers":
            has_leftovers = True
        elif ht == "Wedding Anniversary":
            has_wa = True
    if has_leftovers:
        for pc in player.played_today:
            pm = pc.markers
            if not pm or "_copied_definition" not in pm:
                if pc.definition.title == "Bougie Coffee Machine":
                    return True
            elif _effective_card_title(pc) == "Bougie Coffee Machine":
                return True
    if has_wa:
        players = sim.players
        for o_i in range(4):
            if o_i == player_index:
                continue
            for oc in players[o_i].played_today:
                om = oc.markers
                if not om or "_copied_definition" not in om:
                    if oc.definition.title in _TRUNK_CONSUMER_TITLES:
                        return True
                elif _effective_card_title(oc) in _TRUNK_CONSUMER_TITLES:
                    return True
    return False


def _preserve_trunk_loop_flags(
    child: Game, player_index: int, state_pf: int
) -> None:
    cur_pf = child._p_flags[player_index]  # type: ignore[attr-defined]
    req_pf = state_pf & (_PF_HAS_LOOP2 | _PF_HAS_SIG_MARKER)
    if (cur_pf & req_pf) != req_pf:
        add_pf = 0
        for pc_m in child.players[player_index].played_today:
            m_pm = pc_m.markers
            if not m_pm:
                continue
            if "_trunk_risk_adj" in m_pm or "_blind_quacks" in m_pm:
                add_pf |= _PF_HAS_LOOP2 | _PF_HAS_SIG_MARKER
            elif "_blind_cutlery_ev" in m_pm or "_blind_poker" in m_pm:
                add_pf |= _PF_HAS_LOOP2
        if add_pf:
            child._p_flags[player_index] |= add_pf  # type: ignore[attr-defined]


def _update_trunk_sim_state_after_play(
    state: Game,
    child: Game,
    player_index: int,
    raw_title: str,
    consumer_id: int,
    pre_played_len: int,
    paid_cost: int,
    pre_known_top: tuple[CardInstance, ...],
    pre_discard_len: int,
    prev_keep_p: float,
    root_plen: int,
    state_pf: int,
    is_hsim: bool,
) -> None:
    """Propagate trunk knowledge, survival probability, and risk adjustments."""
    p_child = child.players[player_index]
    child._root_played_len = root_plen  # type: ignore[attr-defined]
    consumer_card = (
        p_child.played_today[pre_played_len]
        if pre_played_len < len(p_child.played_today)
        and p_child.played_today[pre_played_len].instance_id == consumer_id
        else next(
            (
                pc
                for pc in reversed(p_child.played_today)
                if pc.instance_id == consumer_id
            ),
            None,
        )
    )
    eff_title = (
        _effective_card_title(consumer_card)
        if consumer_card is not None
        else raw_title
    )

    if (
        (
            len(child.discard) < pre_discard_len
            or (
                pre_discard_len > 0
                and child.discard is not state.discard
                and len(state.discard) >= pre_discard_len
                and (
                    not child.discard
                    or child.discard[0].instance_id
                    != state.discard[0].instance_id
                )
            )
        )
        and eff_title not in _TRUNK_REORDER_TITLES
    ):
        child._known_trunk_state = None  # type: ignore[attr-defined]
    child._known_trunk_discard_len = len(child.discard)  # type: ignore[attr-defined]

    if eff_title in _TRUNK_REORDER_TITLES:
        if (
            getattr(child, "_known_trunk_state", None) is not None
            and _has_sim_trunk_consumer(child, player_index)
        ):
            turns_elapsed = max(
                0, len(state.players[player_index].played_today) - root_plen
            )
            has_ed = _has_epic_duels_played(p_child.played_today)
            child._trunk_keep_prob = _trunk_top_keep_prob(  # type: ignore[attr-defined]
                child,
                player_index,
                has_epic_duels=has_ed,
                turns_ahead=turns_elapsed,
            )
        else:
            child._known_trunk_state = None  # type: ignore[attr-defined]
            child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
    elif pre_known_top:
        if eff_title in {"Assorted Cutlery", "Bougie Coffee Machine", "Quacks", "Poker"}:
            risk_adj = 0.0
            if prev_keep_p < 0.999:
                if eff_title == "Assorted Cutlery":
                    rem_e_ac = max(
                        0, state.players[player_index].energy - paid_cost
                    )
                    fp = _osmium_free_play_value(
                        state,
                        player_index,
                        pre_known_top[0],
                        rem_energy=rem_e_ac,
                    )
                    blind_ev = (
                        _blind_cutlery_expected_value(
                            child, player_index, consumer_card
                        )
                        if consumer_card is not None
                        else 2.2
                    )
                    risk_adj -= (1.0 - prev_keep_p) * max(0.0, fp - blind_ev)
                elif eff_title == "Bougie Coffee Machine":
                    rem_e_bcm = max(
                        0, state.players[player_index].energy - paid_cost
                    )
                    food_surplus = sum(
                        max(
                            0.0,
                            _osmium_free_play_value(
                                state,
                                player_index,
                                kc,
                                rem_energy=rem_e_bcm,
                            )
                            - 2.0,
                        )
                        for kc in pre_known_top[:3]
                        if "Food" in kc.tags
                    )
                    risk_adj -= (1.0 - prev_keep_p) * food_surplus
                elif eff_title == "Quacks" and consumer_card is not None:
                    if not consumer_card.markers.get("_quacks_outdoors"):
                        draws_q = int(consumer_card.markers.get("energy_cubes", 0))
                        risk_adj -= (1.0 - prev_keep_p) * max(
                            0.0, 2.0 * draws_q - 2.1
                        )
                elif eff_title == "Poker" and (
                    consumer_card is None
                    or consumer_card.markers.get("poker_success")
                ):
                    risk_adj -= (1.0 - prev_keep_p) * 2.75

            if (
                eff_title == "Quacks"
                and consumer_card is not None
                and not consumer_card.markers.get("_quacks_outdoors")
                and not _get_sim_known_trunk_top(child)
            ):
                draws_done = int(consumer_card.markers.get("energy_cubes", 0))
                if draws_done in (1, 2):
                    unseen_counts, unseen_total = _unseen_tag_counts_and_total(
                        child, player_index, ()
                    )
                    p_out = unseen_counts.get("Outdoors", 50) / max(1, unseen_total)
                    risk_adj += prev_keep_p * max(
                        0.0,
                        (1.0 - p_out) * 2.0 * (draws_done + 1) - 2.0 * draws_done,
                    )

            if risk_adj != 0.0 and consumer_card is not None:
                consumer_card.markers["_trunk_risk_adj"] = risk_adj
                if is_hsim:
                    child._p_flags[player_index] |= (  # type: ignore[attr-defined]
                        _PF_HAS_LOOP2 | _PF_HAS_SIG_MARKER
                    )

        if (
            not _get_sim_known_trunk_top(child)
            or not _has_sim_trunk_consumer(child, player_index)
        ):
            child._known_trunk_state = None  # type: ignore[attr-defined]
            child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
        elif not _has_epic_duels_played(p_child.played_today):
            turns_elapsed = max(
                0, len(state.players[player_index].played_today) - root_plen
            )
            child._trunk_keep_prob = prev_keep_p * _trunk_top_keep_prob(  # type: ignore[attr-defined]
                child,
                player_index,
                has_epic_duels=False,
                turns_ahead=turns_elapsed,
            )
        else:
            child._trunk_keep_prob = prev_keep_p  # type: ignore[attr-defined]
    else:
        child._known_trunk_state = None  # type: ignore[attr-defined]
        child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]

    if is_hsim and (state_pf & _PF_HAS_LOOP2):
        _preserve_trunk_loop_flags(child, player_index, state_pf)


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
        self._draft_opp_pending_draws: tuple[int, int, int, int] | None = None
        self._last_eval_day_fun_sim: Game | None = None
        self._last_eval_day_fun: float | None = None
        self._known_trunk_discard_len: int = -1
        self._known_trunk_discard_ids: tuple[int, ...] = ()
        self._known_trunk_played_ids: tuple[frozenset[int], ...] = ()
        self._known_trunk_suit_ids: frozenset[int] = frozenset()
        if type(self) is OsmiumAI:
            self.__dict__.pop("_reserve_eval_cache", None)
        elif "_reserve_eval_cache" not in type(self).__dict__:
            # Keep subclasses isolated from the shared OsmiumAI reserve cache.
            self._reserve_eval_cache = {}

    def _clear_live_known_trunk(self) -> None:
        self._known_trunk_state = None
        self._known_trunk_discard_len = -1
        self._known_trunk_discard_ids = ()
        self._known_trunk_played_ids = ()
        self._known_trunk_suit_ids = frozenset()

    def _ensure_live_game(self, game: Game) -> None:
        if game.stats is _get_shared_sim_stats():
            return
        if self._live_game_ref is None or self._live_game_ref() is not game:
            self._clear_live_known_trunk()
        super()._ensure_live_game(game)

    def _get_live_known_trunk_top(
        self, game: Game, player_index: int
    ) -> tuple[CardInstance, ...]:
        self._ensure_live_game(game)
        if self._known_trunk_state is None:
            return ()
        game_id, day, expected_len, ordered_cards, opp_counts = (
            self._known_trunk_state
        )
        if game_id != id(game) or day != game.day:
            self._clear_live_known_trunk()
            return ()
        draws_since = expected_len - len(game.trunk)
        if draws_since < 0 or draws_since >= len(ordered_cards):
            self._clear_live_known_trunk()
            return ()
        prev_d = self._known_trunk_discard_ids
        d_len = len(prev_d)
        cur_discard = game.discard
        if (
            (
                self._known_trunk_discard_len >= 0
                and len(cur_discard) < self._known_trunk_discard_len
            )
            or len(cur_discard) < d_len
            or any(cur_discard[i].instance_id != cid for i, cid in enumerate(prev_d))
        ):
            self._clear_live_known_trunk()
            return ()
        if len(self._known_trunk_played_ids) == len(game.players):
            max_pops = 0
            any_opp_bohnanza = False
            unbounded_pop = False
            for idx, p in enumerate(game.players):
                prev_set = self._known_trunk_played_ids[idx]
                n_trigs = sum(
                    1
                    for tc in p.tomorrow_cards
                    if _effective_card_title(tc) == "Photo Shoot"
                ) + sum(
                    1
                    for pc in p.played_today
                    if _effective_card_title(pc) == "Euchre Awards Ceremony"
                )
                new_cnt = 0
                for pc in p.played_today:
                    if pc.instance_id in prev_set:
                        continue
                    new_cnt += 1
                    eff_t = _effective_card_title(pc)
                    raw_t = pc.definition.title
                    if idx != player_index:
                        if eff_t in _TRUNK_REORDER_TITLES or raw_t in {
                            "Wedding Anniversary",
                            "Leftovers",
                        }:
                            self._clear_live_known_trunk()
                            return ()
                        if eff_t == "Bohnanza":
                            any_opp_bohnanza = True
                    if eff_t == "Quacks":
                        max_pops += int(pc.markers.get("energy_cubes", 0))
                    elif eff_t == "Bohnanza":
                        max_pops += 4 * int(pc.markers.get("energy_cubes", 0))
                        unbounded_pop = True
                    elif eff_t in _LIVE_TRUNK_UNBOUNDED_POPS:
                        unbounded_pop = True
                    else:
                        max_pops += _LIVE_TRUNK_FIXED_MAX_POPS.get(eff_t, 0)
                max_pops += new_cnt * n_trigs
            if any_opp_bohnanza:
                all_prev_played = set().union(*self._known_trunk_played_ids)
                for dc in cur_discard[d_len:]:
                    dt = dc.definition.title
                    if dc.instance_id not in all_prev_played and (
                        dt in _TRUNK_REORDER_TITLES
                        or dt in {"Wedding Anniversary", "Leftovers"}
                    ):
                        self._clear_live_known_trunk()
                        return ()
            if self._known_trunk_suit_ids:
                new_suit = sum(
                    1
                    for sc in game.suitcase
                    if sc.instance_id not in self._known_trunk_suit_ids
                )
                if new_suit > max_pops:
                    max_pops = new_suit
            if max_pops >= expected_len or (
                unbounded_pop and d_len <= 1 and expected_len <= 40
            ):
                self._clear_live_known_trunk()
                return ()
        else:
            for idx, p in enumerate(game.players):
                if idx == player_index or idx >= len(opp_counts):
                    continue
                for played in p.played_today[opp_counts[idx] :]:
                    if (
                        _effective_card_title(played) in _TRUNK_REORDER_TITLES
                        or played.definition.title
                        in {"Wedding Anniversary", "Leftovers"}
                    ):
                        self._clear_live_known_trunk()
                        return ()
        rem = ordered_cards[draws_since:]
        if len(game.trunk) < len(rem):
            self._clear_live_known_trunk()
            return ()
        visible_ids = {
            c.instance_id
            for c in (
                *game.suitcase,
                *cur_discard,
                *game.players[player_index].hand,
                *(
                    pc
                    for p in game.players
                    for pc in (*p.played_today, *p.tomorrow_cards)
                ),
            )
        }
        if any(kc.instance_id in visible_ids for kc in rem):
            self._clear_live_known_trunk()
            return ()
        return rem

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        self._ensure_live_game(game)
        order = _OSMIUM_EVAL_AI_SINGLETON.order_cards_for_trunk(
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
            self._known_trunk_discard_len = len(game.discard)
            self._known_trunk_discard_ids = tuple(c.instance_id for c in game.discard)
            self._known_trunk_played_ids = tuple(
                frozenset(c.instance_id for c in p.played_today)
                for p in game.players
            )
            self._known_trunk_suit_ids = frozenset(c.instance_id for c in game.suitcase)
        return order

    def _prepare_observable_game(self, game: Game, player_index: int) -> Game:
        sim = super()._prepare_observable_game(game, player_index)
        sim._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
        sim._known_trunk_discard_len = len(sim.discard)  # type: ignore[attr-defined]
        return sim

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
                    if (
                        not m
                        or _OSMIUM_SIG_MARKER_KEYS.isdisjoint(m)
                        or (
                            "_blind_cutlery_ev" not in m
                            and "_trunk_risk_adj" not in m
                            and "_copied_definition" not in m
                            and not m.get("energy_cubes")
                            and not m.get("_ski_on_shoulders_energy_cube")
                            and not m.get("discarded_card")
                            and m.get("discarded_item") is None
                            and m.get("target_card") is None
                            and m.get("target_player_index") is None
                        )
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
                                m.get("_blind_cutlery_ev"),
                                m.get("_trunk_risk_adj"),
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
            if sim._known_trunk_state is not None:  # type: ignore[attr-defined]
                kt = _get_sim_known_trunk_top(sim)
                if kt:
                    return (
                        p.energy,
                        p.fun,
                        played_sig,
                        hand_sig,
                        tom_sig,
                        tuple(kc.instance_id for kc in kt),
                        round(getattr(sim, "_trunk_keep_prob", 1.0), 3),
                    )
            return (p.energy, p.fun, played_sig, hand_sig, tom_sig)
        base_sig = super()._state_signature(sim, player_index)
        played = sim.players[player_index].played_today
        extra_m = tuple(
            (i, m.get("_blind_cutlery_ev"), m.get("_trunk_risk_adj"))
            for i, c_p in enumerate(played)
            if (m := c_p.markers)
            and ("_blind_cutlery_ev" in m or "_trunk_risk_adj" in m)
        )
        kt = (
            _get_sim_known_trunk_top(sim)
            if getattr(sim, "_known_trunk_state", None) is not None
            else ()
        )
        if not extra_m and not kt:
            return base_sig
        return (
            base_sig,
            extra_m,
            tuple(kc.instance_id for kc in kt) if kt else (),
            round(getattr(sim, "_trunk_keep_prob", 1.0), 3) if kt else 1.0,
        )

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

    def _suitcase_pick1_bonus(
        self,
        sim: Game,
        player_index: int,
        *,
        _sc_costs: Sequence[int | None] | None = None,
        _sc_metas: Sequence | None = None,
        _sc_plain: Sequence | None = None,
    ) -> float:
        return min(
            2.6,
            0.45
            + max(0.0, self._ftb_edge_weight())
            * _suitcase_pick1_edge(
                sim,
                player_index,
                _sc_costs=_sc_costs,
                _sc_metas=_sc_metas,
                _sc_plain=_sc_plain,
            ),
        )

    def _next_day_starter_prob(
        self,
        sim: Game,
        player_index: int,
        my_end_hand: int,
    ) -> float:
        """Estimate P(player_index starts tomorrow) under fewest-cards-in-hand."""
        players = sim.players
        starter = sim.starting_player
        my_pos = (player_index - starter) % 4
        pending_draws = self._draft_opp_pending_draws
        p_win = 1.0
        for idx in range(4):
            if idx == player_index:
                continue
            opp = players[idx]
            opp_pos = (idx - starter) % 4
            # Ties go to the first player clockwise starting from sim.starting_player.
            need_ge = my_end_hand if my_pos < opp_pos else my_end_hand + 1
            if need_ge <= 0:
                continue

            extra = pending_draws[idx] if pending_draws is not None else 0
            opp_eod_h = _projected_end_hand_size(sim, idx) + extra

            if opp.asleep:
                if opp_eod_h < need_ge:
                    return 0.0
                continue

            opp_h = len(opp.hand) + extra
            if opp_h <= 0:
                if opp_eod_h < need_ge:
                    return 0.0
                continue

            opp_e = opp.energy
            if opp_eod_h < need_ge:
                if opp_e <= 0 or need_ge - opp_eod_h > 2:
                    return 0.0
                p_beat = 0.08 if need_ge - opp_eod_h == 1 else 0.02
            else:
                if opp_e <= 0:
                    est_plays = min(float(opp_h), 0.25)
                else:
                    est_plays = min(float(opp_h), 0.35 + opp_e / 2.6)
                mu = opp_eod_h - est_plays
                diff = mu - (need_ge - 0.5)
                scale = 0.25 + 0.15 * min(float(opp_h), max(0.0, opp_e * 0.5))
                z = diff / scale
                if z >= 12.0:
                    p_beat = 1.0
                elif z <= -12.0:
                    p_beat = 0.0
                else:
                    p_beat = 1.0 / (1.0 + math.exp(-z))
            p_win *= p_beat
            if p_win < 0.02:
                return 0.0
        return p_win

    def _first_to_bed_value(
        self,
        sim: Game,
        player_index: int,
        *,
        turns_from_root: int = 0,
        root_skipped_turns: int = 0,
        _base_bonus_override: float | None = None,
        _my_end_hand_override: int | None = None,
    ) -> float:
        """Expected value of being tomorrow's starting player (fewest cards in hand)."""
        if sim.day >= DAY_COUNT:
            return 0.0
        my_end_hand = (
            _my_end_hand_override
            if _my_end_hand_override is not None
            else _projected_end_hand_size(sim, player_index)
        )
        p_win = self._next_day_starter_prob(sim, player_index, my_end_hand)
        if p_win <= 0.0:
            return 0.0
        base_bonus = (
            _base_bonus_override
            if _base_bonus_override is not None
            else self._suitcase_pick1_bonus(sim, player_index)
        )
        return base_bonus * p_win

    def _evaluate_terminal_day(
        self,
        terminal_sim: Game,
        player_index: int,
        *,
        turns_from_root: int = 1,
        root_skipped_turns: int = 0,
        _ftb_ctx: tuple | None = None,
        _h_mset: int = -1,
        _p_mset: int = -1,
        _h_res_mset: int = -1,
        _ctx_ok: bool = False,
        _sc_ok: bool = False,
    ) -> float:
        players = terminal_sim.players
        player = players[player_index]
        ctx_matches = _ctx_ok or (
            _ftb_ctx is not None
            and (
                (player_index == 0 or players[0] is _ftb_ctx[0])
                and (player_index == 1 or players[1] is _ftb_ctx[1])
                and (player_index == 2 or players[2] is _ftb_ctx[2])
                and (player_index == 3 or players[3] is _ftb_ctx[3])
            )
        )
        if ctx_matches:
            awake_opps = _ftb_ctx[4]
            opponents_awake = awake_opps > 0
        else:
            awake_opps = 0
            for idx in range(4):
                if idx != player_index and not players[idx].asleep:
                    awake_opps += 1
            opponents_awake = awake_opps > 0

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
                    m
                    and (
                        "_blind_poker" in m
                        or "_blind_quacks" in m
                        or "_blind_cutlery_ev" in m
                        or "_trunk_risk_adj" in m
                    )
                ):
                    has_loop2 = True
                if opponents_awake and eff_title in _TERMINAL_AWAKE_OPP_TITLES:
                    has_loop3 = True

        eval_sim = terminal_sim
        returned_to_hand: list[CardInstance] | None = None
        solo_rescored_ids: set[int] | None = None
        inline_end_day = False
        can_inline_end_day = (
            needs_full_end_day
            and is_hsim
            and not (pf & _PF_HAS_COMPLEX_END_DAY)
        )
        if (
            can_inline_end_day
            and (pf & _PF_HAS_CK_END_DAY)
            and terminal_sim.day < DAY_COUNT
            and len(played_today) >= 3
        ):
            if (
                not terminal_sim._all_canonical_defs  # type: ignore[attr-defined]
                or not _hydrogen_mod._ALL_CANONICAL_META_TAGS_OK
                or (pf & (_PF_ALL_TAGS | _PF_MAY_HAVE_TAG_MARKERS))
            ):
                can_inline_end_day = False
            else:
                meta_by_slug_ck = _HYDROGEN_META_BY_SLUG
                packed_ck = 0
                ck_plain = True
                for c_ck in played_today:
                    cm_ck = c_ck.markers
                    if cm_ck and ("_all_tags" in cm_ck or "_extra_tags" in cm_ck):
                        ck_plain = False
                        break
                    packed_ck += meta_by_slug_ck[c_ck.definition.slug].tag_packed_cnt
                if not ck_plain:
                    can_inline_end_day = False
                else:
                    has3 = 0
                    ge2 = 0
                    while packed_ck:
                        cnt_b = packed_ck & 0xFF
                        if cnt_b >= 2:
                            ge2 += 1
                            if cnt_b >= 3:
                                has3 = 1
                        packed_ck >>= 8
                    if has3 and ge2 >= 2:
                        can_inline_end_day = False
        p_flags_arr: Sequence[int] = (
            terminal_sim._p_flags if can_inline_end_day else ()  # type: ignore[attr-defined]
        )
        if (
            can_inline_end_day
            and (pf & _PF_HAS_SCOUT_END_DAY)
            and terminal_sim.day < DAY_COUNT
        ):
            opp_end_mask = (
                _PF_HAS_COMPLEX_END_DAY | _PF_HAS_SCOUT_END_DAY | _PF_HAS_CK_END_DAY
            )
            for o_i in range(4):
                if o_i != player_index and (p_flags_arr[o_i] & opp_end_mask):
                    can_inline_end_day = False
                    break
        if can_inline_end_day and (pf & _PF_HAS_NEW_AWAKE_OPP):
            for o_i in range(4):
                if o_i != player_index and (
                    p_flags_arr[o_i] & _PF_HAS_COMPLEX_END_DAY
                ):
                    can_inline_end_day = False
                    break
        my_day_fun_pre: float | None = None
        if can_inline_end_day:
            inline_end_day = True
            start_of_scoring = (
                players[0].fun,
                players[1].fun,
                players[2].fun,
                players[3].fun,
            )
            prev_scores = terminal_sim._fun_at_start_of_scoring
            terminal_sim._fun_at_start_of_scoring = start_of_scoring
            try:
                scored_today, next_tomorrow_cards = (
                    terminal_sim.score_played_today_fast(player_index)  # type: ignore[attr-defined]
                )
                baseline = (
                    terminal_sim._fun_at_start_of_day
                    if terminal_sim._fun_at_start_of_day is not None
                    else start_of_scoring
                )
                my_fun_today = (player.fun + scored_today) - baseline[player_index]
                opp_fun_local: list[int | None] | None = None
                opp_ctx_dict = _ftb_ctx[5] if ctx_matches else None
                dropped_tom_ids: set[int] | None = None
                scout_acq_ids: set[int] | None = None
                solo_bonus_total = 0
                is_canon_eval = terminal_sim._all_canonical_defs  # type: ignore[attr-defined]
                meta_by_slug = _HYDROGEN_META_BY_SLUG
                for c in played_today:
                    cm = c.markers
                    uncopied = not cm or "_copied_definition" not in cm
                    if uncopied and is_canon_eval:
                        c_beh = meta_by_slug[c.definition.slug].beh
                    else:
                        c_beh = (
                            c.definition.behavior if uncopied else c.effective_behavior
                        )
                    b_type = type(c_beh)
                    if b_type is ChuckAFrisbeeBehavior:
                        if returned_to_hand is None:
                            returned_to_hand = [CardInstance(c.instance_id, c.definition)]
                        else:
                            returned_to_hand.append(CardInstance(c.instance_id, c.definition))
                    elif b_type is NosShirtBehavior:
                        if my_fun_today > 7:
                            if returned_to_hand is None:
                                returned_to_hand = [CardInstance(c.instance_id, c.definition)]
                            else:
                                returned_to_hand.append(CardInstance(c.instance_id, c.definition))
                    elif b_type is ScoutTheOtherCottagesBehavior:
                        if terminal_sim.day < DAY_COUNT:
                            owner_marker = f"_scout_energy_cube_{c.instance_id}"
                            for sc_c in terminal_sim.suitcase:
                                if sc_c.markers and sc_c.markers.get(owner_marker):
                                    sc_obj_id = id(sc_c)
                                    if scout_acq_ids is None:
                                        scout_acq_ids = {sc_obj_id}
                                    elif sc_obj_id in scout_acq_ids:
                                        continue
                                    else:
                                        scout_acq_ids.add(sc_obj_id)
                                    if returned_to_hand is None:
                                        returned_to_hand = [
                                            CardInstance(sc_c.instance_id, sc_c.definition)
                                        ]
                                    else:
                                        returned_to_hand.append(
                                            CardInstance(sc_c.instance_id, sc_c.definition)
                                        )
                    elif b_type is DatesFirstNosBehavior:
                        if my_fun_today < 2:
                            if dropped_tom_ids is None:
                                dropped_tom_ids = {id(c)}
                            else:
                                dropped_tom_ids.add(id(c))
                    elif b_type is EpicPrankBehavior:
                        t_idx = cm.get("target_player_index") if cm else None
                        if isinstance(t_idx, int) and 0 <= t_idx < 4:
                            if opp_fun_local is None:
                                opp_fun_local = [None, None, None, None]
                            opp_val = _inline_opp_today_fun(
                                terminal_sim,
                                players,
                                p_flags_arr,
                                baseline,
                                opp_fun_local,
                                t_idx,
                                opp_ctx_dict,
                            )
                            if my_fun_today < opp_val:
                                if dropped_tom_ids is None:
                                    dropped_tom_ids = {id(c)}
                                else:
                                    dropped_tom_ids.add(id(c))
                    elif b_type is SoloBehavior:
                        if opp_fun_local is None:
                            opp_fun_local = [None, None, None, None]
                        solo_beats = True
                        for o_i in range(4):
                            if o_i == player_index:
                                continue
                            opp_val = _inline_opp_today_fun(
                                terminal_sim,
                                players,
                                p_flags_arr,
                                baseline,
                                opp_fun_local,
                                o_i,
                                opp_ctx_dict,
                            )
                            if my_fun_today < opp_val:
                                solo_beats = False
                                break
                        if solo_beats:
                            solo_bonus_total += terminal_sim.card_fun(player_index, c)
                            if solo_rescored_ids is None:
                                solo_rescored_ids = {id(c)}
                            else:
                                solo_rescored_ids.add(id(c))
                if dropped_tom_ids is not None and next_tomorrow_cards:
                    next_tomorrow_cards = [
                        tc for tc in next_tomorrow_cards if id(tc) not in dropped_tom_ids
                    ]
                my_day_fun_pre = float(my_fun_today)
                today_total_fun = float(player.fun + scored_today + solo_bonus_total)
            finally:
                terminal_sim._fun_at_start_of_scoring = prev_scores
        elif needs_full_end_day:
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
            if is_hsim and terminal_sim._all_canonical_defs:  # type: ignore[attr-defined]
                if (
                    terminal_sim.daily_condition is None
                    and player.character is None
                    and not (pf & _SIMPLE_SCORE_EXCL_PF)
                ):
                    meta_by_slug_s = _HYDROGEN_META_BY_SLUG
                    scored_today = 0
                    for c_s in played_today:
                        scored_today += meta_by_slug_s[c_s.definition.slug].base_fun
                    next_tomorrow_cards = ()
                else:
                    scored_today, next_tomorrow_cards = (
                        terminal_sim.score_played_today_fast(player_index)  # type: ignore[attr-defined]
                    )
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
            my_day_fun_pre = (
                float(
                    (
                        player.fun
                        - (
                            terminal_sim._fun_at_start_of_day[player_index]
                            if terminal_sim._fun_at_start_of_day is not None
                            else player.fun
                        )
                    )
                    + scored_today
                )
                if (
                    (has_loop3 and (not is_hsim or (pf & _PF_HAS_NEW_AWAKE_OPP)))
                    or (
                        is_hsim
                        and terminal_sim.day < DAY_COUNT
                        and (pf & _PF_HAS_CONT_AFTER)
                    )
                )
                else None
            )
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
                    bce = m.get("_blind_cutlery_ev")
                    if bce is not None:
                        today_total_fun += bce
                    ra = m.get("_trunk_risk_adj")
                    if ra:
                        today_total_fun += ra

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
                            4.5 if played_len <= 3 else 2.4
                        )
                elif eff_title == "Play With the Kids":
                    if played_len <= 4 and all(
                        played_len >= len(players[idx].played_today)
                        for idx in range(4)
                        if idx != player_index
                    ):
                        today_total_fun -= (
                            3.6 if played_len <= 3 else 1.9
                        )
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
            if not is_hsim or (pf & _PF_HAS_NEW_AWAKE_OPP):
                today_total_fun += _awake_opp_new_cards_adjustment(
                    terminal_sim,
                    player_index,
                    next_tomorrow_cards,
                    eval_sim,
                    solo_rescored_ids=solo_rescored_ids,
                    _my_day_fun=my_day_fun_pre,
                    _opp_cache=_ftb_ctx[5] if ctx_matches else None,
                    _known_has_target=is_hsim,
                )

        if terminal_sim.day >= DAY_COUNT:
            return today_total_fun
        self._last_eval_day_fun_sim = terminal_sim
        self._last_eval_day_fun = my_day_fun_pre

        my_proj_h = (
            len(eval_sim.players[player_index].hand)
            if eval_sim is not terminal_sim
            else (
                len(player.hand) + (len(returned_to_hand) if returned_to_hand else 0)
                if inline_end_day
                else (
                    len(player.hand)
                    if not needs_full_end_day
                    else _projected_end_hand_size(terminal_sim, player_index)
                )
            )
        )
        if ctx_matches:
            ftb_len = len(_ftb_ctx)
            f_dict = _ftb_ctx[5]
            has_sc_ctx = _sc_ok or (
                ftb_len >= 9 and terminal_sim.suitcase is _ftb_ctx[6]
            )
            has_fast_ctx = has_sc_ctx and ftb_len >= 15
            if has_fast_ctx:
                p_cache = _ftb_ctx[12]
                p_win = p_cache.get(my_proj_h)
                if p_win is None:
                    p_win = self._next_day_starter_prob(
                        terminal_sim, player_index, my_proj_h
                    )
                    p_cache[my_proj_h] = p_win
                sc_ids = _ftb_ctx[7]
            else:
                sc_ids = (
                    _ftb_ctx[7]
                    if has_sc_ctx
                    else tuple(c.instance_id for c in terminal_sim.suitcase)
                )
                pkey = ("p", my_proj_h, sc_ids)
                p_win = f_dict.get(pkey)
                if p_win is None:
                    p_win = self._next_day_starter_prob(
                        terminal_sim, player_index, my_proj_h
                    )
                    f_dict[pkey] = p_win
            if p_win <= 0.0:
                first_to_bed_value = 0.0
            else:
                use_osmium_pick1 = (
                    type(self)._suitcase_pick1_bonus
                    is OsmiumAI._suitcase_pick1_bonus
                )
                if (
                    has_sc_ctx
                    and ftb_len >= 11
                    and use_osmium_pick1
                    and _ftb_ctx[9] is not None
                    and is_hsim
                    and terminal_sim._all_canonical_defs  # type: ignore[attr-defined]
                    and not terminal_sim._has_nonplain_hand  # type: ignore[attr-defined]
                    and not (
                        pf
                        & (
                            _PF_HAS_COPIED
                            | _PF_ALL_TAGS
                            | _PF_HAS_PLAYED_COST_MOD
                            | _PF_HAS_SIG_MARKER
                        )
                    )
                    and len(player.hand) <= 30
                    and len(played_today) <= 30
                ):
                    if _h_mset >= 0 and _p_mset >= 0:
                        h_mset = _h_mset
                        p_mset = _p_mset
                    else:
                        mset_tid = _OSMIUM_TITLE_MSET_BIT_BY_SLUG
                        h_mset = 0
                        for hc in player.hand:
                            h_mset += mset_tid[hc.definition.slug]
                        p_mset = 0
                        for pc in played_today:
                            p_mset += mset_tid[pc.definition.slug]
                    if has_fast_ctx and player.tomorrow_cards is _ftb_ctx[14]:
                        bm_cache = _ftb_ctx[13]
                        bm_key = (h_mset, p_mset)
                        base_b = bm_cache.get(bm_key)
                        b_store_cache = bm_cache
                        b_store_key: object = bm_key
                    else:
                        p_tom_ids = (
                            _ftb_ctx[8]
                            if (ftb_len >= 9 and player.tomorrow_cards is (_ftb_ctx[14] if ftb_len >= 15 else None))
                            else tuple(c.instance_id for c in player.tomorrow_cards)
                        )
                        sc_bkey = ("bm", h_mset, p_mset, p_tom_ids, sc_ids)
                        base_b = f_dict.get(sc_bkey)
                        b_store_cache = f_dict
                        b_store_key = sc_bkey
                else:
                    p_tom_ids = (
                        _ftb_ctx[8]
                        if (ftb_len >= 9 and (ftb_len < 15 or player.tomorrow_cards is _ftb_ctx[14]))
                        else tuple(c.instance_id for c in player.tomorrow_cards)
                    )
                    p_hand_ids = (
                        ()
                        if not player.hand
                        else tuple(c.instance_id for c in player.hand)
                    )
                    sc_bkey = ("b", p_hand_ids, p_tom_ids, sc_ids)
                    base_b = f_dict.get(sc_bkey)
                    b_store_cache = f_dict
                    b_store_key = sc_bkey
                if base_b is None:
                    if use_osmium_pick1:
                        no_sc_mod = not (pf & (_PF_HAS_PLAYED_COST_MOD | _PF_ALL_TAGS))
                        sc_pre_costs = (
                            _ftb_ctx[9]
                            if (has_sc_ctx and ftb_len >= 11 and no_sc_mod)
                            else None
                        )
                        sc_pre_metas = (
                            _ftb_ctx[10]
                            if (has_sc_ctx and ftb_len >= 11)
                            else None
                        )
                        sc_pre_plain = (
                            _ftb_ctx[11]
                            if (
                                has_sc_ctx
                                and no_sc_mod
                                and ftb_len >= 12
                            )
                            else None
                        )
                        base_b = self._suitcase_pick1_bonus(
                            terminal_sim,
                            player_index,
                            _sc_costs=sc_pre_costs,
                            _sc_metas=sc_pre_metas,
                            _sc_plain=sc_pre_plain,
                        )
                    else:
                        base_b = self._suitcase_pick1_bonus(
                            terminal_sim, player_index
                        )
                    b_store_cache[b_store_key] = base_b
                first_to_bed_value = base_b * p_win
        else:
            first_to_bed_value = self._first_to_bed_value(
                terminal_sim,
                player_index,
                _my_end_hand_override=my_proj_h,
            )

        orig_h_len = len(player.hand)
        if returned_to_hand:
            player.hand.extend(returned_to_hand)
        try:
            reserve_value: float | None = None
            days_left_eval = DAY_COUNT - terminal_sim.day
            nt_eval = len(next_tomorrow_cards)
            if (
                eval_sim is terminal_sim
                and nt_eval <= 2
                and not returned_to_hand
                and player.character is None
                and is_hsim
                and 0 <= days_left_eval <= 15
                and terminal_sim._all_canonical_defs  # type: ignore[attr-defined]
                and _hydrogen_mod._ALL_CANONICAL_META_TAGS_OK
                and not terminal_sim._has_marked_hand  # type: ignore[attr-defined]
                and type(self) is OsmiumAI
                and self._reserve_eval_cache is _OSMIUM_RESERVE_EVAL_CACHE
            ):
                tom_r_eval = 0
                tom_key2: tuple[int, int] | None = None
                tom_inline_ok = True
                if nt_eval == 1:
                    tc0_eval = next_tomorrow_cards[0]
                    tm0_eval = tc0_eval.markers
                    if tm0_eval and (
                        "_copied_definition" in tm0_eval
                        or "chosen_tag" in tm0_eval
                    ):
                        tom_inline_ok = False
                    else:
                        tom_r_eval = _OSMIUM_RANK_BY_SLUG[
                            tc0_eval.definition.slug
                        ]
                elif nt_eval == 2:
                    tc0_eval = next_tomorrow_cards[0]
                    tc1_eval = next_tomorrow_cards[1]
                    tm0_eval = tc0_eval.markers
                    tm1_eval = tc1_eval.markers
                    if (
                        tm0_eval
                        and (
                            "_copied_definition" in tm0_eval
                            or "chosen_tag" in tm0_eval
                        )
                    ) or (
                        tm1_eval
                        and (
                            "_copied_definition" in tm1_eval
                            or "chosen_tag" in tm1_eval
                        )
                    ):
                        tom_inline_ok = False
                    else:
                        rank_by_slug = _OSMIUM_RANK_BY_SLUG
                        tom_key2 = (
                            rank_by_slug[tc0_eval.definition.slug],
                            rank_by_slug[tc1_eval.definition.slug],
                        )
                if tom_inline_ok:
                    p_hand = player.hand
                    nh = len(p_hand)
                    if tom_key2 is not None:
                        if nh <= 30:
                            if nh == 0:
                                hand_mset = 0
                            elif _h_res_mset >= 0:
                                hand_mset = _h_res_mset
                            else:
                                mset_by_slug = _OSMIUM_MSET_BIT_BY_SLUG
                                hand_mset = 0
                                for c_h in p_hand:
                                    hand_mset += mset_by_slug[
                                        c_h.definition.slug
                                    ]
                            reserve_value = _OSMIUM_LARGE_RESERVE_CACHE.get(
                                (days_left_eval, hand_mset, tom_key2)
                            )
                    elif nh == 0:
                        if nt_eval == 0:
                            reserve_value = 0.0
                        else:
                            reserve_value = _OSMIUM_RESERVE_EVAL_CACHE.get(
                                days_left_eval | (tom_r_eval << 4)
                            )
                    elif nh <= 3:
                        rank_by_slug = _OSMIUM_RANK_BY_SLUG
                        if nh == 1:
                            small_packed = (
                                rank_by_slug[p_hand[0].definition.slug] << 13
                            )
                        elif nh == 2:
                            r0 = rank_by_slug[p_hand[0].definition.slug]
                            r1 = rank_by_slug[p_hand[1].definition.slug]
                            if r0 > r1:
                                r0, r1 = r1, r0
                            small_packed = (r0 << 13) | (r1 << 22)
                        else:
                            r0 = rank_by_slug[p_hand[0].definition.slug]
                            r1 = rank_by_slug[p_hand[1].definition.slug]
                            r2 = rank_by_slug[p_hand[2].definition.slug]
                            if r0 > r1:
                                r0, r1 = r1, r0
                            if r1 > r2:
                                r1, r2 = r2, r1
                                if r0 > r1:
                                    r0, r1 = r1, r0
                            small_packed = (r0 << 13) | (r1 << 22) | (r2 << 31)
                        reserve_value = _OSMIUM_RESERVE_EVAL_CACHE.get(
                            days_left_eval | (tom_r_eval << 4) | small_packed
                        )
                    elif nh <= 30:
                        if _h_res_mset >= 0:
                            hand_mset = _h_res_mset
                        else:
                            mset_by_slug = _OSMIUM_MSET_BIT_BY_SLUG
                            hand_mset = 0
                            for c_h in p_hand:
                                hand_mset += mset_by_slug[c_h.definition.slug]
                        if nt_eval == 0:
                            reserve_value = _OSMIUM_LARGE_RESERVE_CACHE.get(
                                days_left_eval | hand_mset
                            )
                        else:
                            reserve_value = _OSMIUM_LARGE_RESERVE_CACHE.get(
                                days_left_eval
                                | 16
                                | (tom_r_eval << 5)
                                | hand_mset
                            )

            if reserve_value is None:
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
                            days_left=days_left_eval,
                        )
                    finally:
                        player.fun = orig_fun
                else:
                    reserve_value = self._evaluate_tomorrow_and_reserve_fast(
                        eval_sim,
                        player_index,
                        next_tomorrow_cards,
                        days_left=days_left_eval,
                    )
        finally:
            if returned_to_hand:
                del player.hand[orig_h_len:]
        return today_total_fun + first_to_bed_value + reserve_value

    def _rollout_awake_opp_adjustment(
        self,
        term_sim: Game,
        next_sim: Game,
        player_index: int,
        awake_opps: int,
    ) -> float:
        adj = 0.0
        players = term_sim.players
        played_today = players[player_index].played_today
        played_len = len(played_today)
        if played_len <= 4:
            ftp_cnt = 0
            pwtk_cnt = 0
            for c in played_today:
                eff_t = _effective_card_title(c)
                if eff_t == "Fit to Print":
                    ftp_cnt += 1
                elif eff_t == "Play With the Kids":
                    pwtk_cnt += 1
            if (ftp_cnt or pwtk_cnt) and all(
                played_len >= len(players[idx].played_today)
                for idx in range(len(players))
                if idx != player_index
            ):
                if ftp_cnt:
                    adj -= float(ftp_cnt) * (4.5 if played_len <= 3 else 2.4)
                if pwtk_cnt:
                    adj -= float(pwtk_cnt) * (3.6 if played_len <= 3 else 1.9)
        adj += _awake_opp_new_cards_adjustment(
            term_sim,
            player_index,
            next_sim.players[player_index].tomorrow_cards,
            next_sim,
        )
        return adj

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

    def _simulate_play_step(
        self,
        state: Game,
        player_index: int,
        hand_idx: int,
        known_cost: int | None = None,
    ) -> Game:
        child = _hydrogen_copy_game_for_play(state, player_index)
        is_hsim = type(child) is _OSMIUM_SIM_GAME_CLS_CACHED
        p_ch = child.players[player_index]
        hand_card = p_ch.hand[hand_idx]
        card_title = hand_card.definition.title
        if (
            is_hsim
            and child._known_trunk_state is None  # type: ignore[attr-defined]
            and card_title not in _OSMIUM_TRUNK_SIM_TITLES
            and not state._has_opp_suitcase_ref  # type: ignore[attr-defined]
        ):
            state_pf = child._p_flags[player_index]  # type: ignore[attr-defined]
            if known_cost is not None:
                child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
            else:
                child.play_card(player_index, hand_idx)
            if child._known_trunk_state is not None:  # type: ignore[attr-defined]
                child._known_trunk_state = None  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
            if state_pf & _PF_HAS_LOOP2:
                _preserve_trunk_loop_flags(child, player_index, state_pf)
            return child
        raw_kts = (
            child._known_trunk_state  # type: ignore[attr-defined]
            if is_hsim
            else getattr(child, "_known_trunk_state", None)
        )
        state_pf = child._p_flags[player_index] if is_hsim else 0  # type: ignore[attr-defined]
        use_fast = known_cost is not None and is_hsim
        pre_known_top = _get_sim_known_trunk_top(child) if raw_kts is not None else ()

        if not pre_known_top and (
            card_title not in _OSMIUM_TRUNK_SIM_TITLES
            or (
                card_title not in _BLIND_TRUNK_STEP_TITLES
                and (
                    card_title == "Bougie Coffee Machine"
                    or not _has_sim_trunk_consumer(child, player_index)
                )
            )
        ):
            if raw_kts is not None:
                child._known_trunk_state = None  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
            if not is_hsim or state._has_opp_suitcase_ref:  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                child._root_played_len = int(  # type: ignore[attr-defined]
                    getattr(
                        state,
                        "_root_played_len",
                        len(state.players[player_index].played_today),
                    )
                )
                child._known_trunk_discard_len = len(child.discard)  # type: ignore[attr-defined]
            if card_title in _TRUNK_REORDER_TITLES:
                child.ais = _OSMIUM_NOOP_TRUNK_SIM_AIS
                try:
                    if use_fast:
                        child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
                    else:
                        child.play_card(player_index, hand_idx)
                finally:
                    child.ais = _HYDROGEN_SIM_AIS
                if (
                    is_hsim
                    and card_title == "Read the Radar"
                    and not hand_card.markers
                    and len(state.trunk) >= 5
                    and not state._has_opp_suitcase_ref  # type: ignore[attr-defined]
                    and not (state_pf & (_PF_HAS_CARD_PLAY | _PF_HAS_TOM_PLAY))
                    and len(child.trunk) == len(state.trunk)
                    and len(child.discard) == len(state.discard)
                ):
                    child.trunk = state.trunk
                    child._trunk_shared = True  # type: ignore[attr-defined]
            elif use_fast:
                child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
            else:
                child.play_card(player_index, hand_idx)
            if is_hsim:
                if child._known_trunk_state is not None:  # type: ignore[attr-defined]
                    child._known_trunk_state = None  # type: ignore[attr-defined]
                    child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                if state_pf & _PF_HAS_LOOP2:
                    _preserve_trunk_loop_flags(child, player_index, state_pf)
            elif getattr(child, "_known_trunk_state", None) is not None:
                child._known_trunk_state = None  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
            return child

        prev_keep_p = float(getattr(state, "_trunk_keep_prob", 1.0))
        root_plen = int(
            getattr(
                state,
                "_root_played_len",
                len(state.players[player_index].played_today),
            )
        )
        child._trunk_keep_prob = prev_keep_p  # type: ignore[attr-defined]
        child._root_played_len = root_plen  # type: ignore[attr-defined]
        consumer_id = hand_card.instance_id
        pre_played_len = len(p_ch.played_today)
        pre_discard_len = len(child.discard)
        child._known_trunk_discard_len = pre_discard_len  # type: ignore[attr-defined]

        if not pre_known_top:
            if card_title == "Assorted Cutlery":
                if known_cost is None:
                    if p_ch.asleep:
                        raise ValueError(
                            f"{p_ch.name} has already gone to bed"
                        )
                    rm_raw = hand_card.markers
                    c_beh = (
                        hand_card.definition.behavior
                        if not rm_raw
                        else hand_card.effective_behavior
                    )
                    if (
                        type(c_beh).can_play is not CardBehavior.can_play
                        and not c_beh.can_play(child, p_ch, hand_card)
                    ):
                        raise ValueError(
                            f"{hand_card.title} cannot legally be played"
                        )
                    cost = child.energy_cost(player_index, hand_card)
                    if cost > p_ch.energy:
                        raise ValueError(
                            f"Not enough Energy to play {hand_card.title}"
                        )
                else:
                    cost = known_cost
                popped_c = child._draw_from_trunk()
                if child._fun_at_start_of_day is None:
                    child._capture_day_start_fun()
                p_ch.energy -= cost
                raw_card = p_ch.hand.pop(hand_idx)
                rm = raw_card.markers
                ac_card = CardInstance(
                    raw_card.instance_id,
                    raw_card.definition,
                    raw_card.is_tomorrow,
                    dict(rm) if rm else {},
                )
                if is_hsim:
                    c_meta = (
                        _HYDROGEN_META_BY_SLUG[raw_card.definition.slug]
                        if (
                            (not rm or "_copied_definition" not in rm)
                            and child._all_canonical_defs  # type: ignore[attr-defined]
                        )
                        else _get_card_meta(raw_card)
                    )
                    pf = (
                        child._p_flags[player_index]  # type: ignore[attr-defined]
                        | c_meta.played_board_flags
                        | (_PF_HAS_PLAYED_IS_TOM if ac_card.is_tomorrow else 0)
                        | _PF_HAS_LOOP2
                        | _PF_HAS_SIG_MARKER
                    )
                    child._p_flags[player_index] = pf  # type: ignore[attr-defined]
                    if child._played_items_shared and (  # type: ignore[attr-defined]
                        pf
                        & (
                            _PF_ALL_TAGS
                            | _PF_HAS_CARD_PLAY
                            | _PF_HAS_ACQUIRE
                            | _PF_HAS_PICK
                        )
                    ):
                        child._unshare_played_items()  # type: ignore[attr-defined]
                p_ch.played_today.append(ac_card)
                if is_hsim:
                    if child._p_flags[player_index] & _PF_ALL_TAGS:  # type: ignore[attr-defined]
                        child._sync_played_today_tags(p_ch)  # type: ignore[attr-defined]
                else:
                    child._apply_tomorrow_tag_effects(p_ch, ac_card)
                child.discard_card(popped_c)
                if raw_kts is not None:
                    child._known_trunk_state = None  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                need_tom_play = (
                    not is_hsim
                    or bool(
                        (child._p_flags[player_index] & _PF_HAS_TOM_PLAY)  # type: ignore[attr-defined]
                        and p_ch.tomorrow_cards
                    )
                )
                need_card_play = not is_hsim or bool(
                    child._p_flags[player_index] & _PF_HAS_CARD_PLAY  # type: ignore[attr-defined]
                )
                if need_tom_play or need_card_play:
                    child.ais = _OSMIUM_SIM_AIS
                    try:
                        if need_tom_play:
                            for source in tuple(p_ch.tomorrow_cards):
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
                                        child, p_ch, source, ac_card
                                    )
                        if need_card_play:
                            for source in tuple(p_ch.played_today):
                                sm = source.markers
                                s_beh = (
                                    source.definition.behavior
                                    if (not sm or "_copied_definition" not in sm)
                                    else source.effective_behavior
                                )
                                if (
                                    type(s_beh).on_card_play
                                    is not CardBehavior.on_card_play
                                ):
                                    s_beh.on_card_play(
                                        child, p_ch, source, ac_card
                                    )
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                ac_card.markers["_blind_cutlery_ev"] = (
                    _blind_cutlery_expected_value(child, player_index, ac_card)
                )
                if is_hsim:
                    child._p_flags[player_index] |= _PF_HAS_LOOP2  # type: ignore[attr-defined]
                return child

            if card_title == "Poker":
                if use_fast:
                    child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
                else:
                    child.play_card(player_index, hand_idx)
                p_card = child.players[player_index].played_today[-1]
                p_card.markers["poker_success"] = False
                p_card.markers["_blind_poker"] = True
                if raw_kts is not None:
                    child._known_trunk_state = None  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                if is_hsim:
                    child._p_flags[player_index] |= _PF_HAS_LOOP2  # type: ignore[attr-defined]
                    if state_pf & _PF_HAS_SIG_MARKER:
                        _preserve_trunk_loop_flags(child, player_index, state_pf)
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
                if raw_kts is not None:
                    child._known_trunk_state = None  # type: ignore[attr-defined]
                child._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                if is_hsim:
                    child._p_flags[player_index] |= (  # type: ignore[attr-defined]
                        _PF_HAS_LOOP2 | _PF_HAS_SIG_MARKER
                    )
                return child

        if card_title in _OSMIUM_TRUNK_SIM_TITLES:
            child.ais = _OSMIUM_SIM_AIS
            try:
                if use_fast:
                    child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
                else:
                    child.play_card(player_index, hand_idx)
            finally:
                child.ais = _HYDROGEN_SIM_AIS
        else:
            if use_fast:
                child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
            else:
                child.play_card(player_index, hand_idx)

        paid_cost = known_cost if known_cost is not None else -1
        if paid_cost < 0 and pre_known_top and card_title in {
            "Assorted Cutlery",
            "Bougie Coffee Machine",
        }:
            paid_cost = state.energy_cost(
                player_index, state.players[player_index].hand[hand_idx]
            )
        _update_trunk_sim_state_after_play(
            state,
            child,
            player_index,
            card_title,
            consumer_id,
            pre_played_len,
            max(0, paid_cost),
            pre_known_top,
            pre_discard_len,
            prev_keep_p,
            root_plen,
            state_pf,
            is_hsim,
        )
        return child

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
        consumer_id = card.instance_id
        pre_played_len = len(player.played_today)
        pre_discard_len = len(state.discard)
        is_hsim = type(state) is _OSMIUM_SIM_GAME_CLS_CACHED
        raw_kts = (
            state._known_trunk_state  # type: ignore[attr-defined]
            if is_hsim
            else getattr(state, "_known_trunk_state", None)
        )
        pre_known_top = _get_sim_known_trunk_top(state) if raw_kts is not None else ()
        prev_keep_p = float(getattr(state, "_trunk_keep_prob", 1.0))
        root_plen = int(getattr(state, "_root_played_len", pre_played_len))
        state_pf = state._p_flags[player_index] if is_hsim else 0  # type: ignore[attr-defined]
        need_branch_post = (
            bool(pre_known_top)
            or (title in _OSMIUM_TRUNK_SIM_TITLES)
            or bool(state_pf & _PF_HAS_LOOP2)
        )
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
                        if need_branch_post:
                            _update_trunk_sim_state_after_play(
                                state,
                                child,
                                player_index,
                                title,
                                consumer_id,
                                pre_played_len,
                                cost,
                                pre_known_top,
                                pre_discard_len,
                                prev_keep_p,
                                root_plen,
                                state_pf,
                                is_hsim,
                            )
                        else:
                            child._root_played_len = root_plen  # type: ignore[attr-defined]
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
                        if need_branch_post:
                            _update_trunk_sim_state_after_play(
                                state,
                                child,
                                player_index,
                                title,
                                consumer_id,
                                pre_played_len,
                                c_cost,
                                pre_known_top,
                                pre_discard_len,
                                prev_keep_p,
                                root_plen,
                                state_pf,
                                is_hsim,
                            )
                        else:
                            child._root_played_len = root_plen  # type: ignore[attr-defined]
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
                        _update_trunk_sim_state_after_play(
                            state,
                            child,
                            player_index,
                            title,
                            consumer_id,
                            pre_played_len,
                            cost,
                            pre_known_top,
                            pre_discard_len,
                            prev_keep_p,
                            root_plen,
                            state_pf,
                            is_hsim,
                        )
                        branches.append(child)
                    except Exception:
                        pass
                    finally:
                        child.ais = _HYDROGEN_SIM_AIS
                if branches:
                    return branches

        elif title == "Tell a Story" and len(player.hand) >= 3:
            if not _has_sim_trunk_consumer(state, player_index):
                return [self._simulate_play_step(state, player_index, hand_idx, known_cost)]
            b1 = self._simulate_play_step(state, player_index, hand_idx, known_cost)
            rem_hand = [c for i, c in enumerate(player.hand) if i != hand_idx]
            best_dump_i = min(
                range(len(rem_hand)),
                key=lambda i: (
                    _hydrogen_hand_keep_value(state, player_index, rem_hand[i]),
                    i,
                ),
            )
            b1_kts = getattr(b1, "_known_trunk_state", None)
            b1_top_slug = (
                b1_kts[1][0].definition.slug
                if (b1_kts and b1_kts[1])
                else ""
            )
            if rem_hand[best_dump_i].definition.slug != b1_top_slug:
                child = _hydrogen_copy_game_for_play(state, player_index)
                child.ais = [
                    _OsmiumOverrideEvalAI(discard_idx_override=best_dump_i)
                ] * 4
                try:
                    if known_cost is not None and is_hsim:
                        child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined]
                    else:
                        child.play_card(player_index, hand_idx)
                    _update_trunk_sim_state_after_play(
                        state,
                        child,
                        player_index,
                        title,
                        consumer_id,
                        pre_played_len,
                        0,
                        pre_known_top,
                        pre_discard_len,
                        prev_keep_p,
                        root_plen,
                        state_pf,
                        is_hsim,
                    )
                    return [b1, child]
                except Exception:
                    pass
                finally:
                    child.ais = _HYDROGEN_SIM_AIS
            return [b1]

        return [self._simulate_play_step(state, player_index, hand_idx, known_cost)]

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
        if (
            type(self)._search_best_day_plan_with_table
            is OsmiumAI._search_best_day_plan_with_table
        ):
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
                _need_table=False,
            )
            return best_action, best_val
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
        _need_table: bool = True,
    ) -> tuple[int | None, float, dict[int | None, tuple[float, Game]]]:
        """Adaptive day-scaled beam search with in-search sub-decision branching."""
        played_count = len(base_sim.players[player_index].played_today)
        is_hsim_root = type(base_sim) is _OSMIUM_SIM_GAME_CLS_CACHED
        base_sim._root_played_len = played_count  # type: ignore[attr-defined]
        if is_hsim_root:
            if not hasattr(base_sim, "_known_trunk_state"):
                base_sim._known_trunk_state = None  # type: ignore[attr-defined]
            if not hasattr(base_sim, "_trunk_keep_prob"):
                base_sim._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
            if not hasattr(base_sim, "_known_trunk_discard_len"):
                base_sim._known_trunk_discard_len = len(base_sim.discard)  # type: ignore[attr-defined]
            if (
                base_sim._known_trunk_state is not None  # type: ignore[attr-defined]
                and not _has_sim_trunk_consumer(base_sim, player_index)
            ):
                base_sim._known_trunk_state = None  # type: ignore[attr-defined]
                base_sim._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
        elif (
            getattr(base_sim, "_known_trunk_state", None) is not None
            and not _has_sim_trunk_consumer(base_sim, player_index)
        ):
            base_sim._known_trunk_state = None  # type: ignore[attr-defined]
            base_sim._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
        is_live_play = max_plays is None and self._evaluation_depth == 0
        need_table = is_live_play or _need_table
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
        is_canon_root = is_hsim_root and base_sim._all_canonical_defs  # type: ignore[attr-defined]
        meta_by_slug = _HYDROGEN_META_BY_SLUG
        mset_tid = _OSMIUM_TITLE_MSET_BIT_BY_SLUG
        b_ps = base_sim.players
        base_player = b_ps[player_index]
        no_cond_char_root = (
            is_canon_root
            and base_sim.daily_condition is None
            and base_player.character is None
        )
        base_pf = base_sim._p_flags[player_index] if is_hsim_root else 0  # type: ignore[attr-defined]
        base_awake_opps = (
            (0 if player_index == 0 or b_ps[0].asleep else 1)
            + (0 if player_index == 1 or b_ps[1].asleep else 1)
            + (0 if player_index == 2 or b_ps[2].asleep else 1)
            + (0 if player_index == 3 or b_ps[3].asleep else 1)
        )
        b_sc = base_sim.suitcase
        sc_base_costs: tuple[int | None, ...] | None = None
        sc_base_metas: tuple | None = None
        sc_plain_specs: tuple | None = None
        if (
            is_canon_root
            and b_sc
            and not is_day6
            and all(
                (not c.markers or "_copied_definition" not in c.markers)
                for c in b_sc
            )
        ):
            sc_base_metas = tuple(meta_by_slug[c.definition.slug] for c in b_sc)
            if not (base_pf & _PF_HAS_PLAYED_COST_MOD):
                if all(not m.has_own_cost_mod for m in sc_base_metas):
                    sc_base_costs = tuple(base_sim.cards_energy_costs(player_index, b_sc))  # type: ignore[attr-defined]
                else:
                    sc_base_costs = tuple(
                        None
                        if m.has_own_cost_mod
                        else base_sim.energy_cost(player_index, sc_c)
                        for sc_c, m in zip(b_sc, sc_base_metas)
                    )
                if no_cond_char_root and _hydrogen_mod._ALL_CANONICAL_META_TAGS_OK:
                    plain_list: list[tuple | None] = []
                    any_plain = False
                    for i, (sc_c, m) in enumerate(zip(b_sc, sc_base_metas)):
                        cm = sc_c.markers
                        if (
                            not m.has_own_cost_mod
                            and (
                                not cm
                                or (
                                    ("_all_tags" not in cm or not cm["_all_tags"])
                                    and (
                                        "_extra_tags" not in cm
                                        or not cm["_extra_tags"]
                                    )
                                )
                            )
                            and m.title != "Istanbul"
                            and m.quick_sleep_or_late != 2
                            and m.cost < 8
                            and m.prof.conditional_energy_gain <= 0
                            and m.c_qcode in _SC_PLAIN_CQ_CODES
                            and (
                                m.quick_syn_kind == _QSYN_NONE
                                or m.quick_syn_kind == _QSYN_NEG_SMB
                            )
                            and (
                                m.quick_global_kind == _QGLOB_NONE
                                or m.quick_global_kind == _QGLOB_DOCK
                                or m.quick_global_kind == _QGLOB_COUCH_TUBE
                            )
                            and m.h_qcode == _HQ_NONE
                        ):
                            c = sc_base_costs[i]
                            assert c is not None
                            any_plain = True
                            base_v = (m.q_fun_term - c * 0.78) + m.q_imm_e_term
                            if m.has_quick_draw:
                                base_v += (
                                    m.quick_draw_cheap
                                    if c <= 2
                                    else m.quick_draw_exp
                                )
                            if m.quick_sleep_or_late == 1:
                                base_v -= 1.6
                            c_qc = m.c_qcode
                            if c_qc == _CQ_GROUP_DINNER:
                                base_v -= 1.85
                            elif c_qc == _CQ_BOUGIE_COFFEE_MACHINE:
                                base_v += 1.35
                            elif c_qc == _CQ_CHEESY_PHONE_GAME:
                                base_v += 1.05
                            elif (
                                m.conditional_fun_bonus > 0
                                and not m.is_multi_scaler
                            ):
                                cfb = m.conditional_fun_bonus
                                base_v += (cfb if cfb < 5 else 5) * 0.55
                            if m.quick_penalty != 0.0:
                                base_v -= m.quick_penalty
                            if m.quick_syn_kind == _QSYN_NEG_SMB:
                                base_v += m.quick_neg_smb_pen
                            if m.quick_global_kind == _QGLOB_DOCK:
                                base_v += 3.5
                            elif m.quick_global_kind == _QGLOB_COUCH_TUBE:
                                base_v += 3.1
                            if m.goes_to_tomorrow:
                                base_v += m.quick_tom_base
                                if m.quick_tom_extra != 0.0:
                                    base_v += m.quick_tom_extra
                            if c <= 1 and m.base_fun >= 0:
                                base_v += 0.65 if c == 0 else 0.25
                            plain_list.append(
                                (
                                    base_v,
                                    c <= 1,
                                    m.must_play_first,
                                    m.tag_mask,
                                    m.sorted_tag_bits,
                                )
                            )
                        elif (
                            not cm
                            and m.q_const_eligible
                            and sc_base_costs[i] == m.cost
                        ):
                            any_plain = True
                            plain_list.append(("Q", m))
                        else:
                            plain_list.append(None)
                    if any_plain:
                        sc_plain_specs = tuple(plain_list)
        ftb_ctx: tuple = (
            b_ps[0],
            b_ps[1],
            b_ps[2],
            b_ps[3],
            base_awake_opps,
            {},
            b_sc,
            tuple(c.instance_id for c in b_sc),
            tuple(c.instance_id for c in base_player.tomorrow_cards),
            sc_base_costs,
            sc_base_metas,
            sc_plain_specs,
            {},
            {},
            base_player.tomorrow_cards,
        )
        best_by_action: dict[int | None, tuple[float, Game]] = {}
        turns_by_action: dict[int | None, int] = {}

        tid_by_slug = _OSMIUM_TITLE_ID_BY_SLUG
        base_tom = base_player.tomorrow_cards
        use_inline_sig = (
            is_canon_root
            and type(self)._state_signature is OsmiumAI._state_signature
        )
        root_tom_sig: int | tuple = 0
        if is_canon_root and base_tom:
            nt_r = len(base_tom)
            if nt_r == 1:
                root_tom_sig = tid_by_slug[base_tom[0].definition.slug]
            else:
                root_tom_sig = tuple(
                    tid_by_slug[c_t.definition.slug] for c_t in base_tom
                )
        mset_res = _OSMIUM_MSET_BIT_BY_SLUG
        root_played = base_player.played_today
        root_hand = base_player.hand
        root_h_tid = -1
        root_p_tid = -1
        root_h_res = -1
        if is_canon_root:
            if len(root_hand) <= 30:
                rht = 0
                rhr = 0
                for c_h in root_hand:
                    hs = c_h.definition.slug
                    rht += mset_tid[hs]
                    rhr += mset_res[hs]
                root_h_tid = rht
                root_h_res = rhr
            if len(root_played) <= 30:
                rpt = 0
                for c_p in root_played:
                    rpt += mset_tid[c_p.definition.slug]
                root_p_tid = rpt
        use_fast_term = (
            type(self)._evaluate_terminal_day is OsmiumAI._evaluate_terminal_day
        )

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
            bed_val = (
                self._evaluate_terminal_day(
                    bed_now,
                    player_index,
                    turns_from_root=bed_turns,
                    root_skipped_turns=root_skipped,
                    _ftb_ctx=ftb_ctx,
                    _h_mset=root_h_tid,
                    _p_mset=root_p_tid,
                    _h_res_mset=root_h_res,
                    _ctx_ok=True,
                    _sc_ok=True,
                )
                if use_fast_term
                else self._evaluate_terminal_day(
                    bed_now,
                    player_index,
                    turns_from_root=bed_turns,
                    root_skipped_turns=root_skipped,
                    _ftb_ctx=ftb_ctx,
                )
            )
            best_value = bed_val
            best_first_actions: list[int | None] = [None]
            if need_table:
                best_by_action[None] = (bed_val, bed_now)
                turns_by_action[None] = bed_turns
        else:
            best_value = float("-inf")
            best_first_actions = []

        frontier: list[
            tuple[int, object, float, int, bool, int, int, int, int, int, int, int, int]
        ] = []
        seen_best: dict[tuple, tuple[float, int]] = {}
        next_idx_by_sig: dict[tuple, int] = {}
        seen_pre_pure: dict[tuple[int, int, int], tuple[float, object, int, int | None]] = {}
        seen_u_defs: set[int] = set()
        need_depth_1 = max_plays > 1
        use_fast_cont = (
            type(self)._continuation_potential is HydrogenAI._continuation_potential
            and type(self)._extra_continuation_bonus
            is HydrogenAI._extra_continuation_bonus
        )
        cont_cache: dict[tuple[int, int, int], float] = {}
        cont_special_mask = _OSMIUM_CONT_SPECIAL_MSET_MASK
        cont_after_mask = _OSMIUM_CONT_AFTER_MSET_MASK
        cont_neutral3_mask = _OSMIUM_NEUTRAL3_MSET_MASK
        if not cont_special_mask or not cont_after_mask:
            _ensure_osmium_caches()
            cont_special_mask = _OSMIUM_CONT_SPECIAL_MSET_MASK
            cont_after_mask = _OSMIUM_CONT_AFTER_MSET_MASK
            cont_neutral3_mask = _OSMIUM_NEUTRAL3_MSET_MASK
        use_inplace_play = (
            is_canon_root
            and type(self)._simulate_play_step is OsmiumAI._simulate_play_step
            and type(self)._simulate_play_branches is OsmiumAI._simulate_play_branches
        )
        next_prefix_id = 1
        next_world_id = 1
        inplace_op_get = _INPLACE_OP_BY_SLUG.get

        def _credit_actions(
            mask: int, val: float, st: object, turns: int
        ) -> None:
            nonlocal best_value, best_first_actions
            if not need_table:
                if val > best_value + 1e-9:
                    best_value = val
                    best_first_actions = [mask.bit_length() - 1]
                elif abs(val - best_value) <= 1e-9:
                    fa = mask.bit_length() - 1
                    if fa not in best_first_actions:
                        best_first_actions.append(fa)
                return
            if mask & (mask - 1) == 0:
                fa = mask.bit_length() - 1
                prev = best_by_action.get(fa)
                if prev is None or val > prev[0]:
                    best_by_action[fa] = (val, st)  # type: ignore[assignment]
                    turns_by_action[fa] = turns
                if val > best_value + 1e-9:
                    best_value = val
                    best_first_actions = [fa]
                elif (
                    abs(val - best_value) <= 1e-9
                    and fa not in best_first_actions
                ):
                    best_first_actions.append(fa)
                return
            m = mask
            while m:
                lsb = m & -m
                fa = lsb.bit_length() - 1
                m ^= lsb
                prev = best_by_action.get(fa)
                if prev is None or val > prev[0]:
                    best_by_action[fa] = (val, st)  # type: ignore[assignment]
                    turns_by_action[fa] = turns
                if val > best_value + 1e-9:
                    best_value = val
                    best_first_actions = [fa]
                elif (
                    abs(val - best_value) <= 1e-9
                    and fa not in best_first_actions
                ):
                    best_first_actions.append(fa)

        def _scan_hand_flags(h_cards: Sequence[CardInstance]) -> int:
            hf = 0
            min_bs = 99
            for hc in h_cards:
                hm_m = hc.markers
                if hm_m and "_copied_definition" in hm_m:
                    return 7
                hs = hc.definition.slug
                if hs == "bohnanza":
                    hf |= 1
                hm = meta_by_slug[hs]
                if hm.is_adj_prev_hand:
                    hf |= 2
                if hm.is_before_scaler:
                    hf |= 4
                    if hm.cost < min_bs:
                        min_bs = hm.cost
            return (min_bs << 3) | hf

        def _process_evaluated_child(
            child_st: Game,
            child_token: object,
            child_wid: int,
            stop_val: float,
            target_frontier: list[
                tuple[
                    int,
                    object,
                    float,
                    int,
                    bool,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                ]
            ],
            fa_mask: int,
            step_turns: int,
            cand_slug: str,
            cand_meta: _HydrogenDefnMeta | None,
            cand_pure: bool,
            can_go_deeper: bool,
            prev_hflags: int,
            prev_h_len: int,
            prev_suit_len: int,
            may_gain: bool,
            parent_st: Game,
            child_extra: bool,
            pure_pid_arg: int,
            pure_mset_arg: int,
            prev_blocked: int,
            c_h_tid: int,
            c_p_tid: int,
            c_h_res: int,
        ) -> tuple[int | None, object]:
            nonlocal next_prefix_id
            if need_table or stop_val >= best_value - 1e-8:
                _credit_actions(fa_mask, stop_val, child_token, step_turns)
            added_nf_idx: int | None = None
            stored_token = child_token
            c_player = child_st.players[player_index]
            c_hand = c_player.hand
            if not (can_go_deeper and not c_player.asleep and c_hand):
                return added_nf_idx, stored_token
            c_pf = child_st._p_flags[player_index] if is_hsim_root else 0  # type: ignore[attr-defined]
            if (
                no_cond_char_root
                and not (c_pf & (_PF_HAS_PLAYED_COST_MOD | _PF_HAS_TOM_COST_MOD))
            ):
                c_e = c_player.energy
                has_any_play = False
                for hc in c_hand:
                    hm = meta_by_slug[hc.definition.slug]
                    if (
                        not hm.has_own_cost_mod
                        and hm.cost <= c_e
                        and not hm.has_can_play
                    ):
                        has_any_play = True
                        break
                    if hc.markers:
                        has_any_play = True
                        break
                    hc_cost = (
                        _inline_own_cost_mod(
                            child_st,
                            c_player,
                            hc,
                            hm,
                            hm.cost,
                            c_pf,
                            True,
                            meta_by_slug,
                            True,
                        )
                        if hm.has_own_cost_mod
                        else hm.cost
                    )
                    if hc_cost <= c_e and (
                        not hm.has_can_play
                        or hm.beh.can_play(child_st, c_player, hc)
                    ):
                        has_any_play = True
                        break
                if not has_any_play:
                    return added_nf_idx, stored_token

            if not is_canon_root:
                c_hflags = 7
            elif (
                (
                    may_gain
                    and child_st is not parent_st
                    and (
                        len(c_hand) >= prev_h_len
                        or (
                            child_st.trunk is not parent_st.trunk
                            and cand_slug not in ("poker", "quacks")
                        )
                        or len(child_st.suitcase) != prev_suit_len
                        or (
                            not child_st._has_card_ref  # type: ignore[attr-defined]
                            and child_st.suitcase is not parent_st.suitcase
                        )
                    )
                )
                or (
                    (prev_hflags & 3)
                    and cand_slug
                    in (
                        "bohnanza",
                        "beach-conversation",
                        "swimming",
                        "carcassonne",
                    )
                )
                or (
                    (prev_hflags & 4)
                    and (cand_meta is None or cand_meta.is_before_scaler)
                )
            ):
                c_hflags = _scan_hand_flags(c_hand)
            else:
                c_hflags = prev_hflags

            c_played = c_player.played_today
            c_players = child_st.players
            c_opps_same = child_wid == 0 or (
                (player_index == 0 or c_players[0] is b_ps[0])
                and (player_index == 1 or c_players[1] is b_ps[1])
                and (player_index == 2 or c_players[2] is b_ps[2])
                and (player_index == 3 or c_players[3] is b_ps[3])
            )
            c_tail_adj = (
                True
                if not is_canon_root
                else (
                    cand_meta.is_adj_next_tail
                    if (child_st is parent_st and cand_meta is not None)
                    else (
                        bool(c_played)
                        and meta_by_slug[c_played[-1].definition.slug].is_adj_next_tail
                    )
                )
            )
            if cand_pure:
                c_blocked_mask = prev_blocked
            elif is_canon_root and not (c_pf & _SAFE_COMM_PREFIX_EXCLUDE):
                c_blocked_mask = (
                    _compute_blocked_tag_mask(c_player, c_pf, meta_by_slug)
                    if (c_pf & _BLOCKED_MASK_TRIGGERS)
                    else 0
                )
            else:
                c_blocked_mask = 0xFF
            c_has_kts = (
                child_st._known_trunk_state is not None  # type: ignore[attr-defined]
                if is_hsim_root
                else getattr(child_st, "_known_trunk_state", None) is not None
            )
            if (
                no_cond_char_root
                and c_blocked_mask == 0
                and not (c_hflags & 3)
                and not c_tail_adj
                and not child_st._has_nonplain_hand  # type: ignore[attr-defined]
                and not (c_pf & _CANON_SIG_EXCLUDE_PF)
                and len(c_hand) <= 30
                and len(c_played) <= 30
            ):
                if c_h_tid < 0:
                    c_h_tid = 0
                    for hc in c_hand:
                        c_h_tid += mset_tid[hc.definition.slug]
                if c_p_tid < 0:
                    c_p_tid = 0
                    for pc in c_played:
                        c_p_tid += mset_tid[pc.definition.slug]
                f_slug = (
                    c_played[0].definition.slug
                    if (c_played and (c_pf & _PF_HAS_FIRST_CARD_TOM))
                    else ""
                )
                c_kt = _get_sim_known_trunk_top(child_st) if c_has_kts else ()
                sig: tuple
                if c_kt:
                    sig = (
                        "C",
                        child_wid,
                        c_player.energy,
                        c_player.fun,
                        c_p_tid,
                        c_h_tid,
                        f_slug,
                        c_pf & _PF_GAVE_ENERGY,
                        tuple(kc.instance_id for kc in c_kt),
                        round(getattr(child_st, "_trunk_keep_prob", 1.0), 3),
                    )
                else:
                    sig = (
                        "C",
                        child_wid,
                        c_player.energy,
                        c_player.fun,
                        c_p_tid,
                        c_h_tid,
                        f_slug,
                        c_pf & _PF_GAVE_ENERGY,
                    )
            elif use_inline_sig:
                nh_s = len(c_hand)
                if nh_s == 0:
                    hand_sig_s: int | tuple = 0
                elif c_h_tid >= 0 and nh_s <= 30:
                    hand_sig_s = c_h_tid
                elif nh_s == 1:
                    hand_sig_s = mset_tid[c_hand[0].definition.slug]
                elif nh_s == 2:
                    hand_sig_s = (
                        mset_tid[c_hand[0].definition.slug]
                        + mset_tid[c_hand[1].definition.slug]
                    )
                elif nh_s == 3:
                    hand_sig_s = (
                        mset_tid[c_hand[0].definition.slug]
                        + mset_tid[c_hand[1].definition.slug]
                        + mset_tid[c_hand[2].definition.slug]
                    )
                elif nh_s == 4:
                    hand_sig_s = (
                        mset_tid[c_hand[0].definition.slug]
                        + mset_tid[c_hand[1].definition.slug]
                        + mset_tid[c_hand[2].definition.slug]
                        + mset_tid[c_hand[3].definition.slug]
                    )
                elif nh_s <= 30:
                    acc_hs = 0
                    for c_h_s in c_hand:
                        acc_hs += mset_tid[c_h_s.definition.slug]
                    hand_sig_s = acc_hs
                else:
                    h_ids_s = [tid_by_slug[c_h_s.definition.slug] for c_h_s in c_hand]
                    h_ids_s.sort()
                    hand_sig_s = tuple(h_ids_s)
                if c_h_tid < 0 and nh_s <= 30:
                    c_h_tid = hand_sig_s  # type: ignore[assignment]

                np_s = len(c_played)
                if np_s == 0:
                    played_sig_s: int | tuple = 0
                elif not (c_pf & _PF_HAS_SIG_MARKER):
                    acc_ps = 0
                    for idx_ps in range(np_s):
                        acc_ps |= tid_by_slug[c_played[idx_ps].definition.slug] << (
                            idx_ps << 3
                        )
                    played_sig_s = acc_ps
                else:
                    has_real_sig_s = False
                    played_list_s: list = []
                    sig_marker_keys_s = _OSMIUM_SIG_MARKER_KEYS
                    for c_p_s in c_played:
                        ms_p = c_p_s.markers
                        if (
                            not ms_p
                            or sig_marker_keys_s.isdisjoint(ms_p)
                            or (
                                "_blind_cutlery_ev" not in ms_p
                                and "_trunk_risk_adj" not in ms_p
                                and "_copied_definition" not in ms_p
                                and not ms_p.get("energy_cubes")
                                and not ms_p.get("_ski_on_shoulders_energy_cube")
                                and not ms_p.get("discarded_card")
                                and ms_p.get("discarded_item") is None
                                and ms_p.get("target_card") is None
                                and ms_p.get("target_player_index") is None
                            )
                        ):
                            played_list_s.append(tid_by_slug[c_p_s.definition.slug])
                        else:
                            has_real_sig_s = True
                            played_list_s.append(
                                (
                                    tid_by_slug[c_p_s.definition.slug],
                                    _effective_card_title(c_p_s),
                                    ms_p.get("energy_cubes", 0),
                                    getattr(ms_p.get("target_card"), "instance_id", None),
                                    ms_p.get("target_player_index"),
                                    bool(ms_p.get("_ski_on_shoulders_energy_cube")),
                                    bool(ms_p.get("discarded_card")),
                                    getattr(ms_p.get("discarded_item"), "instance_id", None),
                                    ms_p.get("_blind_cutlery_ev"),
                                    ms_p.get("_trunk_risk_adj"),
                                )
                            )
                    if not has_real_sig_s:
                        acc_ps = 0
                        for idx_ps, tid_ps in enumerate(played_list_s):
                            acc_ps |= tid_ps << (idx_ps << 3)
                        played_sig_s = acc_ps
                    else:
                        played_sig_s = tuple(played_list_s)

                tom_s = c_player.tomorrow_cards
                if tom_s is base_tom:
                    tom_sig_s: int | tuple = root_tom_sig
                else:
                    nt_s = len(tom_s)
                    if nt_s == 0:
                        tom_sig_s = 0
                    elif nt_s == 1:
                        tom_sig_s = tid_by_slug[tom_s[0].definition.slug]
                    else:
                        tom_sig_s = tuple(
                            tid_by_slug[c_t_s.definition.slug] for c_t_s in tom_s
                        )
                if c_has_kts:
                    kt_s = _get_sim_known_trunk_top(child_st)
                    if kt_s:
                        sig = (
                            c_player.energy,
                            c_player.fun,
                            played_sig_s,
                            hand_sig_s,
                            tom_sig_s,
                            tuple(kc.instance_id for kc in kt_s),
                            round(getattr(child_st, "_trunk_keep_prob", 1.0), 3),
                        )
                    else:
                        sig = (
                            c_player.energy,
                            c_player.fun,
                            played_sig_s,
                            hand_sig_s,
                            tom_sig_s,
                        )
                else:
                    sig = (
                        c_player.energy,
                        c_player.fun,
                        played_sig_s,
                        hand_sig_s,
                        tom_sig_s,
                    )
            else:
                sig = self._state_signature(child_st, player_index)

            prev_sb = seen_best.get(sig)
            if (
                prev_sb is None
                or stop_val > prev_sb[0] + 1e-9
                or (
                    abs(stop_val - prev_sb[0]) <= 1e-9
                    and step_turns < prev_sb[1]
                )
            ):
                seen_best[sig] = (stop_val, step_turns)
                if use_fast_cont:
                    if (
                        is_canon_root
                        and _hydrogen_mod._ALL_CANONICAL_META_TAGS_OK
                        and not child_st._has_nonplain_hand  # type: ignore[attr-defined]
                        and (not (c_hflags & 4) or c_player.energy < (c_hflags >> 3))
                        and not (c_pf & _PF_HAS_CONT_AFTER)
                    ):
                        ce_b = c_player.energy
                        cont_val = stop_val + (ce_b if ce_b < 7 else 7) * 0.72
                    else:
                        c_key: tuple[int, int, int] | None = None
                        cached_cont: float | None = None
                        if (
                            is_canon_root
                            and _hydrogen_mod._ALL_CANONICAL_META_TAGS_OK
                            and not child_st._has_nonplain_hand  # type: ignore[attr-defined]
                            and not (c_pf & _CONT_CACHE_EXCL_PF)
                            and len(c_hand) <= 30
                            and len(c_played) <= 30
                        ):
                            if c_h_tid < 0:
                                c_h_tid = 0
                                for hc in c_hand:
                                    c_h_tid += mset_tid[hc.definition.slug]
                            if c_p_tid < 0:
                                c_p_tid = 0
                                for pc in c_played:
                                    c_p_tid += mset_tid[pc.definition.slug]
                            if (c_p_tid & cont_special_mask) == 0:
                                has_bs_e = bool(c_hflags & 4) and c_player.energy >= (
                                    c_hflags >> 3
                                )
                                p_key_c = (
                                    c_p_tid if has_bs_e else (c_p_tid & cont_after_mask)
                                )
                                if (c_h_tid & cont_neutral3_mask) == 0:
                                    c_key = (c_player.energy, c_h_tid, p_key_c)
                                else:
                                    if c_h_res < 0:
                                        c_h_res = 0
                                        for hc in c_hand:
                                            c_h_res += mset_res[hc.definition.slug]
                                    c_key = (-1 - c_player.energy, c_h_res, p_key_c)
                                cached_cont = cont_cache.get(c_key)
                        if cached_cont is not None:
                            cont_val = stop_val + cached_cont
                        else:
                            cont_opp_cache = ftb_ctx[5] if c_opps_same else None
                            cv = self._continuation_potential(
                                child_st,
                                player_index,
                                _has_before_scaler_in_hand=bool(c_hflags & 4),
                                _cur_day_fun=(
                                    self._last_eval_day_fun
                                    if self._last_eval_day_fun_sim is child_st
                                    else None
                                ),
                                _opp_cache=cont_opp_cache,
                            )
                            if c_key is not None:
                                cont_cache[c_key] = cv
                            cont_val = stop_val + cv
                else:
                    cont_val = stop_val + self._continuation_potential(
                        child_st, player_index
                    )
                if cand_pure:
                    child_pid = pure_pid_arg
                    child_mset = pure_mset_arg
                else:
                    if (
                        is_canon_root
                        and c_blocked_mask != 0xFF
                        and not child_st._has_nonplain_hand  # type: ignore[attr-defined]
                        and not (c_hflags & 3)
                        and not (c_pf & _SAFE_COMM_PREFIX_EXCLUDE)
                        and not (
                            not c_played and (c_pf & _PF_HAS_FIRST_CARD_TOM)
                        )
                        and not c_tail_adj
                    ):
                        child_pid = next_prefix_id
                        next_prefix_id += 1
                    else:
                        child_pid = -1
                    child_mset = 0
                node = (
                    fa_mask,
                    child_token,
                    cont_val,
                    step_turns,
                    child_extra,
                    child_pid,
                    child_mset,
                    c_hflags,
                    c_blocked_mask,
                    child_wid,
                    c_h_tid,
                    c_p_tid,
                    c_h_res,
                )
                existing_idx = next_idx_by_sig.get(sig)
                if existing_idx is not None:
                    target_frontier[existing_idx] = node
                    added_nf_idx = existing_idx
                else:
                    added_nf_idx = len(target_frontier)
                    next_idx_by_sig[sig] = added_nf_idx
                    target_frontier.append(node)
            elif (
                need_table
                and abs(stop_val - prev_sb[0]) <= 1e-9
                and step_turns == prev_sb[1]
            ):
                existing_idx = next_idx_by_sig.get(sig)
                if existing_idx is not None:
                    pn = target_frontier[existing_idx]
                    target_frontier[existing_idx] = (
                        pn[0] | fa_mask,
                        pn[1],
                        pn[2],
                        pn[3],
                        pn[4],
                        pn[5],
                        pn[6],
                        pn[7],
                        pn[8],
                        pn[9],
                        pn[10],
                        pn[11],
                        pn[12],
                    )
                    added_nf_idx = existing_idx
                    stored_token = pn[1]
            return added_nf_idx, stored_token

        def _materialize_frontier(
            flist: list[
                tuple[
                    int,
                    object,
                    float,
                    int,
                    bool,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                ]
            ],
        ) -> None:
            for f_i, fn in enumerate(flist):
                st_t = fn[1]
                changed = False
                if type(st_t) is tuple:
                    if len(st_t) == 5:
                        mat_g: Game = _materialize_plain_child(
                            st_t[0],
                            player_index,
                            st_t[1],
                            st_t[2],
                            st_t[3],
                            st_t[4],
                        )
                    else:
                        mat_g = _materialize_plain_child(
                            st_t[0],
                            player_index,
                            st_t[1],
                            st_t[2],
                            st_t[3],
                            st_t[4],
                            st_t[5],
                            st_t[6],
                            st_t[7],
                        )
                    changed = True
                else:
                    mat_g = st_t  # type: ignore[assignment]
                fh = fn[10]
                fp = fn[11]
                fr = fn[12]
                if is_canon_root and (fh < 0 or fp < 0 or fr < 0):
                    mp = mat_g.players[player_index]
                    mh = mp.hand
                    m_pl = mp.played_today
                    if len(mh) <= 30 and (fh < 0 or fr < 0):
                        nh_t = 0
                        nh_r = 0
                        for c_h in mh:
                            hs = c_h.definition.slug
                            nh_t += mset_tid[hs]
                            nh_r += mset_res[hs]
                        fh = nh_t
                        fr = nh_r
                        changed = True
                    if len(m_pl) <= 30 and fp < 0:
                        np_t = 0
                        for c_p in m_pl:
                            np_t += mset_tid[c_p.definition.slug]
                        fp = np_t
                        changed = True
                if changed:
                    flist[f_i] = (
                        fn[0],
                        mat_g,
                        fn[2],
                        fn[3],
                        fn[4],
                        fn[5],
                        fn[6],
                        fn[7],
                        fn[8],
                        fn[9],
                        fh,
                        fp,
                        fr,
                    )

        root_hflags = _scan_hand_flags(base_player.hand) if is_canon_root else 7
        base_suit_len = len(b_sc)

        init_limit = (
            self.DAY6_BRANCH_LIMIT
            if (is_live_play and is_day6)
            else (14 if is_live_play else max(10, self.BRANCH_LIMIT + 1))
        )
        if is_hsim_root and not need_table:
            raw_playable: Sequence[tuple[int, int | None]] = (
                base_sim.hand_playable_with_costs(player_index)  # type: ignore[attr-defined]
            )
            if allowed_first_indices is not None:
                allowed_set = set(allowed_first_indices)
                raw_playable = [
                    (idx, cost) for idx, cost in raw_playable if idx in allowed_set
                ]
                if not raw_playable and allowed_first_indices:
                    raw_playable = [
                        (idx, None) for idx in allowed_first_indices[:init_limit]
                    ]
            if len(raw_playable) <= init_limit:
                initial_pairs: Sequence[tuple[int, int | None]] = raw_playable
            else:
                initial_with_costs = self._ordered_candidate_indices_with_costs(
                    base_sim, player_index, init_limit
                )
                if allowed_first_indices is not None:
                    allowed_set = set(allowed_first_indices)
                    filtered_pairs = [
                        (idx, cost)
                        for idx, cost in initial_with_costs
                        if idx in allowed_set
                    ]
                    if not filtered_pairs and allowed_first_indices:
                        filtered_pairs = [
                            (idx, None)
                            for idx in allowed_first_indices[:init_limit]
                        ]
                    initial_pairs = filtered_pairs
                else:
                    initial_pairs = initial_with_costs
        else:
            initial_with_costs = self._ordered_candidate_indices_with_costs(
                base_sim, player_index, init_limit
            )
            if allowed_first_indices is not None:
                allowed_set = set(allowed_first_indices)
                filtered_pairs = [
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

        if len(initial_pairs) > 1:
            dedup_pairs: list[tuple[int, int | None]] = []
            seen_u_defs.clear()
            b_hand = base_player.hand
            for idx, k_cost in initial_pairs:
                c_i = b_hand[idx]
                if not c_i.markers and not c_i.is_tomorrow:
                    did = id(c_i.definition)
                    if did in seen_u_defs:
                        continue
                    seen_u_defs.add(did)
                dedup_pairs.append((idx, k_cost))
            initial_pairs = dedup_pairs

        root_blocked_mask = (
            (
                _compute_blocked_tag_mask(base_player, base_pf, meta_by_slug)
                if (base_pf & _BLOCKED_MASK_TRIGGERS)
                else 0
            )
            if (is_canon_root and not (base_pf & _SAFE_COMM_PREFIX_EXCLUDE))
            else 0xFF
        )
        root_no_allows_e = not (base_pf & _PF_HAS_ALLOWS_E)
        root_has_kts = (
            base_sim._known_trunk_state is not None  # type: ignore[attr-defined]
            if is_hsim_root
            else getattr(base_sim, "_known_trunk_state", None) is not None
        )
        root_safe_anchor = (
            is_canon_root
            and root_blocked_mask != 0xFF
            and not base_sim._has_nonplain_hand  # type: ignore[attr-defined]
            and not (root_hflags & 3)
            and not (base_pf & _SAFE_COMM_PREFIX_EXCLUDE)
            and not (not root_played and (base_pf & _PF_HAS_FIRST_CARD_TOM))
            and not (
                root_played
                and meta_by_slug[root_played[-1].definition.slug].is_adj_next_tail
            )
        )
        root_h_len = len(root_hand)
        root_p_len = len(root_played)
        root_inplace_ok = (
            use_inplace_play
            and not root_has_kts
            and base_sim._fun_at_start_of_day is not None  # type: ignore[attr-defined]
            and not (base_pf & _INPLACE_PF_EXCLUDE)
            and not (
                root_played and root_played[-1].definition.slug == "beach-volleyball"
            )
        )
        root_has_euchre = bool(base_pf & _PF_HAS_EUCHRE_AWARDS)
        root_has_photo = bool(base_pf & _PF_HAS_PHOTO_SHOOT)
        bg_opps_root = 0
        for o_i, opp_r in enumerate(b_ps):
            if o_i != player_index:
                for oc_r in opp_r.played_today:
                    if "Board Game" in oc_r.tags:
                        bg_opps_root += 1
                        break
        root_crew_trig = bg_opps_root * 2 >= (len(b_ps) - 1)
        first_step_turns = (
            1
            if in_extra_play
            else (base_turns_offset if start_extra_active else base_turns_offset + 1)
        )
        for hand_idx, known_cost in initial_pairs:
            cand_card = root_hand[hand_idx]
            cand_slug = cand_card.definition.slug
            cand_meta = (
                meta_by_slug[cand_slug]
                if (is_canon_root and not cand_card.markers)
                else None
            )
            cand_pure = False
            if (
                root_safe_anchor
                and cand_meta is not None
                and not cand_card.is_tomorrow
                and not (cand_meta.tag_mask & root_blocked_mask)
            ):
                ck = cand_meta.comm_kind
                if ck == _COMM_PLAIN or (ck == _COMM_GAIN_E and root_no_allows_e):
                    cand_pure = True
            cand_mset = mset_tid[cand_slug] if cand_pure else 0
            may_gain_root = (
                not is_canon_root
                or cand_meta is None
                or cand_meta.has_on_play
                or bool(base_pf & (_PF_HAS_CARD_PLAY | _PF_HAS_TOM_PLAY))
            )
            fa_mask = 1 << hand_idx
            child_extra = (
                in_extra_play
                or start_extra_active
                or (
                    (cand_meta is None or cand_meta.has_allows_extra)
                    and _card_allows_extra_plays(base_sim, base_player, cand_card)
                )
            )
            inplace_gain_e = -1
            ip_extra_pf = 0
            ip_wingspan = False
            ip_set_asleep = False
            ip_markers: dict[str, object] | None = None
            ip_kc = known_cost
            if (
                root_inplace_ok
                and cand_meta is not None
                and not cand_card.is_tomorrow
                and not (cand_meta.played_board_flags & _PF_ALL_TAGS)
                and not (root_has_euchre and (cand_meta.tag_mask & _EUCHRE_TAG_MASK))
                and not (root_has_photo and (cand_meta.tag_mask & _PHOTO_TAG_MASK))
            ):
                orig_e_r = base_player.energy
                if ip_kc is None:
                    ip_kc = base_sim.energy_cost(player_index, cand_card)
                    legal_fallback = ip_kc <= orig_e_r and (
                        not cand_meta.has_can_play
                        or cand_meta.beh.can_play(base_sim, base_player, cand_card)
                    )
                else:
                    legal_fallback = True
                if legal_fallback:
                    if not cand_meta.has_on_play:
                        inplace_gain_e = 0
                    elif cand_meta.comm_kind == _COMM_GAIN_E and root_no_allows_e:
                        inplace_gain_e = cand_meta.comm_e_gain
                    else:
                        op = inplace_op_get(cand_slug, 0)
                        if op:
                            ip_res = _resolve_inplace_on_play(
                                op,
                                base_sim,
                                player_index,
                                base_player,
                                root_hand,
                                root_played,
                                base_pf,
                                root_no_allows_e,
                                orig_e_r,
                                ip_kc,
                                cand_card,
                                cand_meta,
                                meta_by_slug,
                                b_ps,
                                root_crew_trig,
                            )
                            if ip_res is not None:
                                (
                                    inplace_gain_e,
                                    ip_extra_pf,
                                    ip_wingspan,
                                    ip_set_asleep,
                                    ip_markers,
                                ) = ip_res

            if inplace_gain_e >= 0:
                assert cand_meta is not None and ip_kc is not None
                orig_e = base_player.energy
                orig_fun = base_player.fun
                orig_asleep = base_player.asleep
                c_h_tid = root_h_tid - mset_tid[cand_slug] if root_h_tid >= 0 else -1
                c_p_tid = (
                    root_p_tid + mset_tid[cand_slug]
                    if (root_p_tid >= 0 and root_p_len < 30)
                    else -1
                )
                c_h_res = root_h_res - mset_res[cand_slug] if root_h_res >= 0 else -1
                c_pf_new = (
                    base_pf
                    | cand_meta.played_board_flags
                    | ip_extra_pf
                    | (_PF_GAVE_ENERGY if inplace_gain_e else 0)
                )
                base_player.energy = orig_e - ip_kc + inplace_gain_e
                del root_hand[hand_idx]
                base_player.played_today.append(cand_card)
                base_sim._p_flags[player_index] = c_pf_new  # type: ignore[attr-defined]
                if ip_set_asleep:
                    base_player.asleep = True
                if ip_markers is not None:
                    cand_card.markers.update(ip_markers)
                if inplace_gain_e:
                    cand_card.markers["_gave_energy"] = True
                gain_f = 0
                try:
                    if ip_wingspan:
                        gain_f = base_sim.card_fun(player_index, cand_card)
                        if gain_f:
                            base_player.fun = orig_fun + gain_f
                    try:
                        stop_val = (
                            self._evaluate_terminal_day(
                                base_sim,
                                player_index,
                                turns_from_root=first_step_turns,
                                root_skipped_turns=root_skipped,
                                _ftb_ctx=ftb_ctx,
                                _h_mset=c_h_tid,
                                _p_mset=c_p_tid,
                                _h_res_mset=c_h_res,
                                _ctx_ok=True,
                                _sc_ok=True,
                            )
                            if use_fast_term
                            else self._evaluate_terminal_day(
                                base_sim,
                                player_index,
                                turns_from_root=first_step_turns,
                                root_skipped_turns=root_skipped,
                                _ftb_ctx=ftb_ctx,
                            )
                        )
                    except Exception:
                        stop_val = None
                    if stop_val is not None:
                        if gain_f or ip_set_asleep or ip_markers is not None:
                            deferred: tuple = (
                                base_sim,
                                hand_idx,
                                ip_kc,
                                inplace_gain_e,
                                c_pf_new,
                                gain_f,
                                ip_set_asleep,
                                ip_markers,
                            )
                        else:
                            deferred = (
                                base_sim,
                                hand_idx,
                                ip_kc,
                                inplace_gain_e,
                                c_pf_new,
                            )
                        _process_evaluated_child(
                            base_sim,
                            deferred,
                            0,
                            stop_val,
                            frontier,
                            fa_mask,
                            first_step_turns,
                            cand_slug,
                            cand_meta,
                            cand_pure,
                            need_depth_1,
                            root_hflags,
                            root_h_len,
                            base_suit_len,
                            may_gain_root,
                            base_sim,
                            child_extra,
                            0,
                            cand_mset,
                            root_blocked_mask,
                            c_h_tid,
                            c_p_tid,
                            c_h_res,
                        )
                finally:
                    root_hand.insert(hand_idx, cand_card)
                    base_player.played_today.pop()
                    base_player.energy = orig_e
                    if ip_wingspan:
                        base_player.fun = orig_fun
                    if ip_set_asleep:
                        base_player.asleep = orig_asleep
                    base_sim._p_flags[player_index] = base_pf  # type: ignore[attr-defined]
                    if inplace_gain_e or ip_markers is not None:
                        cand_card.markers.clear()
                    self._last_eval_day_fun_sim = None
                continue

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
                c_ps_w = child.players
                same_world_r = (
                    child.suitcase is base_sim.suitcase
                    and child.trunk is base_sim.trunk
                    and child.discard is base_sim.discard
                    and (player_index == 0 or c_ps_w[0] is b_ps[0])
                    and (player_index == 1 or c_ps_w[1] is b_ps[1])
                    and (player_index == 2 or c_ps_w[2] is b_ps[2])
                    and (player_index == 3 or c_ps_w[3] is b_ps[3])
                )
                try:
                    stop_val = (
                        self._evaluate_terminal_day(
                            child,
                            player_index,
                            turns_from_root=first_step_turns,
                            root_skipped_turns=root_skipped,
                            _ftb_ctx=ftb_ctx,
                            _ctx_ok=same_world_r,
                            _sc_ok=same_world_r,
                        )
                        if use_fast_term
                        else self._evaluate_terminal_day(
                            child,
                            player_index,
                            turns_from_root=first_step_turns,
                            root_skipped_turns=root_skipped,
                            _ftb_ctx=ftb_ctx,
                        )
                    )
                except Exception:
                    continue
                if same_world_r:
                    child_wid = 0
                else:
                    child_wid = next_world_id
                    next_world_id += 1
                _process_evaluated_child(
                    child,
                    child,
                    child_wid,
                    stop_val,
                    frontier,
                    fa_mask,
                    first_step_turns,
                    cand_slug,
                    cand_meta,
                    cand_pure,
                    need_depth_1,
                    root_hflags,
                    root_h_len,
                    base_suit_len,
                    may_gain_root,
                    base_sim,
                    child_extra,
                    0,
                    cand_mset,
                    root_blocked_mask,
                    -1,
                    -1,
                    -1,
                )

        if frontier:
            frontier.sort(key=lambda item: item[2], reverse=True)
            frontier = frontier[:beam_width]
            _materialize_frontier(frontier)

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
            next_frontier: list[
                tuple[
                    int,
                    object,
                    float,
                    int,
                    bool,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                ]
            ] = []
            next_idx_by_sig.clear()
            seen_pre_pure.clear()
            for (
                fa_mask,
                state_obj,
                _,
                turns_so_far,
                extra_active,
                pure_prefix_id,
                pure_mset,
                s_hflags,
                s_blocked_mask,
                s_world_id,
                s_h_tid,
                s_p_tid,
                s_h_res,
            ) in frontier:
                state: Game = state_obj  # type: ignore[assignment]
                s_player = state.players[player_index]
                s_hand = s_player.hand
                s_h_len = len(s_hand)
                s_played = s_player.played_today
                s_p_len = len(s_played)
                s_pf = state._p_flags[player_index] if is_hsim_root else 0  # type: ignore[attr-defined]
                need_dedup = False
                if (
                    no_cond_char_root
                    and type(state) is _OSMIUM_SIM_GAME_CLS_CACHED
                    and not state._has_nonplain_hand  # type: ignore[attr-defined]
                    and not (
                        s_pf
                        & (
                            _PF_ALL_TAGS
                            | _PF_HAS_PLAYED_COST_MOD
                            | _PF_HAS_TOM_COST_MOD
                        )
                    )
                    and not s_player.asleep
                    and s_h_len
                ):
                    s_e = s_player.energy
                    dedup_c: list[tuple[int, int]] = []
                    seen_u_defs.clear()
                    raw_cnt = 0
                    for idx, c_i in enumerate(s_hand):
                        c_meta = meta_by_slug[c_i.definition.slug]
                        if c_meta.has_can_play and not c_meta.beh.can_play(
                            state, s_player, c_i
                        ):
                            continue
                        cost = c_meta.cost
                        if cost < 0:
                            cost = 0
                        if c_meta.has_own_cost_mod:
                            cost = _inline_own_cost_mod(
                                state,
                                s_player,
                                c_i,
                                c_meta,
                                cost,
                                s_pf,
                                True,
                                meta_by_slug,
                                True,
                            )
                            if cost < 0:
                                cost = 0
                        if cost <= s_e:
                            raw_cnt += 1
                            if raw_cnt > branch_limit:
                                candidates = self._ordered_candidate_indices_with_costs(
                                    state, player_index, branch_limit
                                )
                                need_dedup = len(candidates) > 1
                                break
                            if not c_i.markers and not c_i.is_tomorrow:
                                did = id(c_i.definition)
                                if did in seen_u_defs:
                                    continue
                                seen_u_defs.add(did)
                            dedup_c.append((idx, cost))
                    else:
                        candidates = dedup_c
                else:
                    if is_hsim_root and type(state) is _OSMIUM_SIM_GAME_CLS_CACHED:
                        raw_cands = state.hand_playable_with_costs(player_index)  # type: ignore[attr-defined]
                        if len(raw_cands) <= branch_limit:
                            candidates = raw_cands
                        else:
                            candidates = self._ordered_candidate_indices_with_costs(
                                state, player_index, branch_limit
                            )
                    else:
                        candidates = self._ordered_candidate_indices_with_costs(
                            state, player_index, branch_limit
                        )
                    need_dedup = len(candidates) > 1
                if need_dedup:
                    dedup_c = []
                    seen_u_defs.clear()
                    for idx, k_cost in candidates:
                        c_i = s_hand[idx]
                        if not c_i.markers and not c_i.is_tomorrow:
                            did = id(c_i.definition)
                            if did in seen_u_defs:
                                continue
                            seen_u_defs.add(did)
                        dedup_c.append((idx, k_cost))
                    candidates = dedup_c
                s_suit_len = len(state.suitcase)
                s_tom_card_play = bool(s_pf & (_PF_HAS_CARD_PLAY | _PF_HAS_TOM_PLAY))
                s_no_allows_e = not (s_pf & _PF_HAS_ALLOWS_E)
                s_has_kts = (
                    state._known_trunk_state is not None  # type: ignore[attr-defined]
                    if is_hsim_root
                    else getattr(state, "_known_trunk_state", None) is not None
                )
                s_inplace_ok = (
                    use_inplace_play
                    and not s_has_kts
                    and state._fun_at_start_of_day is not None  # type: ignore[attr-defined]
                    and not (s_pf & _INPLACE_PF_EXCLUDE)
                    and not (
                        s_played and s_played[-1].definition.slug == "beach-volleyball"
                    )
                )
                s_has_euchre = bool(s_pf & _PF_HAS_EUCHRE_AWARDS)
                s_has_photo = bool(s_pf & _PF_HAS_PHOTO_SHOOT)
                s_ps = state.players
                step_turns = turns_so_far if extra_active else turns_so_far + 1
                for hand_idx, known_cost in candidates:
                    cand_card = s_hand[hand_idx]
                    cand_slug = cand_card.definition.slug
                    cand_meta = (
                        meta_by_slug[cand_slug]
                        if (is_canon_root and not cand_card.markers)
                        else None
                    )
                    cand_pure = False
                    next_mset = 0
                    pre_key: tuple[int, int, int] | None = None
                    if (
                        pure_prefix_id >= 0
                        and cand_meta is not None
                        and not cand_card.is_tomorrow
                        and not (cand_meta.tag_mask & s_blocked_mask)
                    ):
                        ck = cand_meta.comm_kind
                        if ck == _COMM_PLAIN:
                            cand_pure = True
                            e_after = s_player.energy - known_cost
                        elif ck == _COMM_GAIN_E and s_no_allows_e:
                            cand_pure = True
                            e_after = (
                                s_player.energy - known_cost + cand_meta.comm_e_gain
                            )
                        else:
                            e_after = 0
                        if cand_pure:
                            next_mset = pure_mset + mset_tid[cand_slug]
                            pre_key = (pure_prefix_id, e_after, next_mset)
                            cached_pure = seen_pre_pure.get(pre_key)
                            if cached_pure is not None:
                                p_val, p_child, p_turns, p_nf_idx = cached_pure
                                if need_table or p_val >= best_value - 1e-8:
                                    _credit_actions(fa_mask, p_val, p_child, p_turns)
                                if (
                                    need_table
                                    and p_nf_idx is not None
                                    and next_frontier[p_nf_idx][1] is p_child
                                ):
                                    pn = next_frontier[p_nf_idx]
                                    next_frontier[p_nf_idx] = (
                                        pn[0] | fa_mask,
                                        pn[1],
                                        pn[2],
                                        pn[3],
                                        pn[4],
                                        pn[5],
                                        pn[6],
                                        pn[7],
                                        pn[8],
                                        pn[9],
                                        pn[10],
                                        pn[11],
                                        pn[12],
                                    )
                                continue

                    may_gain = (
                        not is_canon_root
                        or cand_meta is None
                        or cand_meta.has_on_play
                        or s_tom_card_play
                    )
                    child_extra = extra_active or (
                        (cand_meta is None or cand_meta.has_allows_extra)
                        and _card_allows_extra_plays(state, s_player, cand_card)
                    )
                    inplace_gain_e = -1
                    ip_extra_pf = 0
                    ip_wingspan = False
                    ip_set_asleep = False
                    ip_markers = None
                    if (
                        s_inplace_ok
                        and cand_meta is not None
                        and not cand_card.is_tomorrow
                        and not (cand_meta.played_board_flags & _PF_ALL_TAGS)
                        and not (
                            s_has_euchre and (cand_meta.tag_mask & _EUCHRE_TAG_MASK)
                        )
                        and not (
                            s_has_photo and (cand_meta.tag_mask & _PHOTO_TAG_MASK)
                        )
                    ):
                        if not cand_meta.has_on_play:
                            inplace_gain_e = 0
                        elif cand_meta.comm_kind == _COMM_GAIN_E and s_no_allows_e:
                            inplace_gain_e = cand_meta.comm_e_gain
                        else:
                            op = inplace_op_get(cand_slug, 0)
                            if op:
                                ip_res = _resolve_inplace_on_play(
                                    op,
                                    state,
                                    player_index,
                                    s_player,
                                    s_hand,
                                    s_played,
                                    s_pf,
                                    s_no_allows_e,
                                    s_player.energy,
                                    known_cost,
                                    cand_card,
                                    cand_meta,
                                    meta_by_slug,
                                    b_ps,
                                    root_crew_trig,
                                )
                                if ip_res is not None:
                                    (
                                        inplace_gain_e,
                                        ip_extra_pf,
                                        ip_wingspan,
                                        ip_set_asleep,
                                        ip_markers,
                                    ) = ip_res

                    if inplace_gain_e >= 0:
                        assert cand_meta is not None
                        orig_e = s_player.energy
                        orig_fun = s_player.fun
                        orig_asleep = s_player.asleep
                        c_h_tid = s_h_tid - mset_tid[cand_slug] if s_h_tid >= 0 else -1
                        c_p_tid = (
                            s_p_tid + mset_tid[cand_slug]
                            if (s_p_tid >= 0 and s_p_len < 30)
                            else -1
                        )
                        c_h_res = s_h_res - mset_res[cand_slug] if s_h_res >= 0 else -1
                        c_pf_new = (
                            s_pf
                            | cand_meta.played_board_flags
                            | ip_extra_pf
                            | (_PF_GAVE_ENERGY if inplace_gain_e else 0)
                        )
                        s_player.energy = orig_e - known_cost + inplace_gain_e
                        del s_hand[hand_idx]
                        s_player.played_today.append(cand_card)
                        state._p_flags[player_index] = c_pf_new  # type: ignore[attr-defined]
                        if ip_set_asleep:
                            s_player.asleep = True
                        if ip_markers is not None:
                            cand_card.markers.update(ip_markers)
                        if inplace_gain_e:
                            cand_card.markers["_gave_energy"] = True
                        gain_f = 0
                        try:
                            if ip_wingspan:
                                gain_f = state.card_fun(player_index, cand_card)
                                if gain_f:
                                    s_player.fun = orig_fun + gain_f
                            try:
                                stop_val = (
                                    self._evaluate_terminal_day(
                                        state,
                                        player_index,
                                        turns_from_root=step_turns,
                                        root_skipped_turns=root_skipped,
                                        _ftb_ctx=ftb_ctx,
                                        _h_mset=c_h_tid,
                                        _p_mset=c_p_tid,
                                        _h_res_mset=c_h_res,
                                        _ctx_ok=(s_world_id == 0),
                                        _sc_ok=(s_world_id == 0),
                                    )
                                    if use_fast_term
                                    else self._evaluate_terminal_day(
                                        state,
                                        player_index,
                                        turns_from_root=step_turns,
                                        root_skipped_turns=root_skipped,
                                        _ftb_ctx=ftb_ctx,
                                    )
                                )
                            except Exception:
                                stop_val = None
                            if stop_val is not None:
                                if gain_f or ip_set_asleep or ip_markers is not None:
                                    deferred = (
                                        state,
                                        hand_idx,
                                        known_cost,
                                        inplace_gain_e,
                                        c_pf_new,
                                        gain_f,
                                        ip_set_asleep,
                                        ip_markers,
                                    )
                                else:
                                    deferred = (
                                        state,
                                        hand_idx,
                                        known_cost,
                                        inplace_gain_e,
                                        c_pf_new,
                                    )
                                added_nf_idx, stored_token = _process_evaluated_child(
                                    state,
                                    deferred,
                                    s_world_id,
                                    stop_val,
                                    next_frontier,
                                    fa_mask,
                                    step_turns,
                                    cand_slug,
                                    cand_meta,
                                    cand_pure,
                                    need_next,
                                    s_hflags,
                                    s_h_len,
                                    s_suit_len,
                                    may_gain,
                                    state,
                                    child_extra,
                                    pure_prefix_id,
                                    next_mset,
                                    s_blocked_mask,
                                    c_h_tid,
                                    c_p_tid,
                                    c_h_res,
                                )
                                if pre_key is not None and pre_key not in seen_pre_pure:
                                    seen_pre_pure[pre_key] = (
                                        stop_val,
                                        stored_token,
                                        step_turns,
                                        added_nf_idx,
                                    )
                        finally:
                            s_hand.insert(hand_idx, cand_card)
                            s_player.played_today.pop()
                            s_player.energy = orig_e
                            if ip_wingspan:
                                s_player.fun = orig_fun
                            if ip_set_asleep:
                                s_player.asleep = orig_asleep
                            state._p_flags[player_index] = s_pf  # type: ignore[attr-defined]
                            if inplace_gain_e or ip_markers is not None:
                                cand_card.markers.clear()
                            self._last_eval_day_fun_sim = None
                        continue

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
                        c_ps_w = child.players
                        same_world_c = (
                            child.suitcase is state.suitcase
                            and child.trunk is state.trunk
                            and child.discard is state.discard
                            and (player_index == 0 or c_ps_w[0] is s_ps[0])
                            and (player_index == 1 or c_ps_w[1] is s_ps[1])
                            and (player_index == 2 or c_ps_w[2] is s_ps[2])
                            and (player_index == 3 or c_ps_w[3] is s_ps[3])
                        )
                        same_root_w = same_world_c and s_world_id == 0
                        try:
                            stop_val = (
                                self._evaluate_terminal_day(
                                    child,
                                    player_index,
                                    turns_from_root=step_turns,
                                    root_skipped_turns=root_skipped,
                                    _ftb_ctx=ftb_ctx,
                                    _ctx_ok=same_root_w,
                                    _sc_ok=same_root_w,
                                )
                                if use_fast_term
                                else self._evaluate_terminal_day(
                                    child,
                                    player_index,
                                    turns_from_root=step_turns,
                                    root_skipped_turns=root_skipped,
                                    _ftb_ctx=ftb_ctx,
                                )
                            )
                        except Exception:
                            continue
                        if same_world_c:
                            child_wid = s_world_id
                        else:
                            child_wid = next_world_id
                            next_world_id += 1
                        added_nf_idx, stored_child = _process_evaluated_child(
                            child,
                            child,
                            child_wid,
                            stop_val,
                            next_frontier,
                            fa_mask,
                            step_turns,
                            cand_slug,
                            cand_meta,
                            cand_pure,
                            need_next,
                            s_hflags,
                            s_h_len,
                            s_suit_len,
                            may_gain,
                            state,
                            child_extra,
                            pure_prefix_id,
                            next_mset,
                            s_blocked_mask,
                            -1,
                            -1,
                            -1,
                        )
                        if pre_key is not None and pre_key not in seen_pre_pure:
                            seen_pre_pure[pre_key] = (
                                stop_val,
                                stored_child,
                                step_turns,
                                added_nf_idx,
                            )

            if need_next:
                next_frontier.sort(key=lambda item: item[2], reverse=True)
                frontier = next_frontier[:active_beam]
                _materialize_frontier(frontier)

        if need_table and best_by_action:
            mat_cache: dict[int, Game] = {}
            for b_fa, (b_val, b_st) in best_by_action.items():
                if type(b_st) is tuple:
                    bid = id(b_st)
                    m_st = mat_cache.get(bid)
                    if m_st is None:
                        if len(b_st) == 5:
                            m_st = _materialize_plain_child(
                                b_st[0],
                                player_index,
                                b_st[1],
                                b_st[2],
                                b_st[3],
                                b_st[4],
                            )
                        else:
                            m_st = _materialize_plain_child(
                                b_st[0],
                                player_index,
                                b_st[1],
                                b_st[2],
                                b_st[3],
                                b_st[4],
                                b_st[5],
                                b_st[6],
                                b_st[7],
                            )
                        mat_cache[bid] = m_st
                    best_by_action[b_fa] = (b_val, m_st)

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
        if live_player.asleep:
            if game.day >= DAY_COUNT:
                return 0
            sim = _hydrogen_copy_game(
                self._prepare_observable_game(game, player_index)
            )
            sim_p = sim.players[player_index]
            today_fun = game.fun_scored_today(live_player)
            next_tom: list[CardInstance] = []
            for c in live_player.played_today:
                eff_t = _effective_card_title(c)
                if eff_t == "Chuck a Frisbee" or (
                    eff_t == "Nos Shirt" and today_fun > 7
                ):
                    sim_p.hand.append(CardInstance(c.instance_id, c.definition))
                    continue
                if not c.effective_behavior.has_tomorrow_action:
                    continue
                if eff_t == "Date's First Nos" and today_fun < 2:
                    continue
                if eff_t == "Epic Prank":
                    t_idx = (
                        c.markers.get("target_player_index")
                        if c.markers
                        else None
                    )
                    if (
                        isinstance(t_idx, int)
                        and today_fun < game.fun_scored_today(game.players[t_idx])
                    ):
                        continue
                next_tom.append(_clone_card(c))
            days_left_eod = DAY_COUNT - game.day
            if getattr(sim, "_all_canonical_defs", False) and any(
                c.definition.slug not in _OSMIUM_SPEC_BY_SLUG
                or _OSMIUM_SPEC_BY_SLUG[c.definition.slug].defn is not c.definition
                for c in (*sim_p.hand, *options)
            ):
                sim._all_canonical_defs = False  # type: ignore[attr-defined]
            best_val_eod = float("-inf")
            best_indices_eod: list[int] = [0]
            for idx, option in enumerate(options):
                opt_copy = CardInstance(option.instance_id, option.definition)
                sim_p.hand.append(opt_copy)
                try:
                    val = self._evaluate_tomorrow_and_reserve_fast(
                        sim, player_index, next_tom, days_left=days_left_eod
                    )
                finally:
                    sim_p.hand.pop()
                val += _suitcase_denial_value(
                    game, player_index, option, pr_markers, scout_markers
                )
                if own_markers and any(m in option.markers for m in own_markers):
                    val -= 3.0
                if val > best_val_eod + 1e-9:
                    best_val_eod = val
                    best_indices_eod = [idx]
                elif abs(val - best_val_eod) <= 1e-9:
                    best_indices_eod.append(idx)
            return self.rng.choice(best_indices_eod)

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

        def _eval_draft_option_sample(
            opt_idx: int,
            opt_card: CardInstance,
            base_sim: Game,
            future_pools: list[tuple[CardInstance, ...]],
        ) -> float:
            sim = _hydrogen_copy_game(base_sim)
            sim.rng = _clone_random(base_sim.rng)
            if (
                opt_idx < len(sim.suitcase)
                and sim.suitcase[opt_idx].instance_id == opt_card.instance_id
            ):
                picked = sim.suitcase.pop(opt_idx)
            else:
                picked = _clone_card(opt_card)
            sim.give_card(player_index, picked)
            sim.players[player_index].picked_cards[picked.title] += 1
            sim.record_card_pick_or_draw(player_index, picked)
            if sim.trunk or sim.discard:
                try:
                    sim.suitcase.insert(opt_idx, sim._draw_from_trunk())
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
                sim_ctx = None
                if sim_is_canon:
                    seen_ok_slug: dict[str, bool] = {}
                    meta_by_slug = _HYDROGEN_META_BY_SLUG
                    sim_pl = sim.players[player_index]
                    can_use_neutral_q = (
                        type(sim) is _OSMIUM_SIM_GAME_CLS_CACHED
                        and sim_pl.character is None
                        and sim.daily_condition is None
                        and _hydrogen_mod._ALL_CANONICAL_META_TAGS_OK
                        and not sim._has_nonplain_hand  # type: ignore[attr-defined]
                        and not (
                            sim._p_flags[player_index]  # type: ignore[attr-defined]
                            & (
                                _PF_HAS_COPIED
                                | _PF_ALL_TAGS
                                | _PF_MAY_HAVE_TAG_MARKERS
                                | _PF_HAS_PLAYED_COST_MOD
                                | _PF_HAS_TOM_COST_MOD
                            )
                        )
                        and len(sim_pl.hand) < 250
                        and len(sim_pl.played_today) < 250
                        and all(not tc.markers for tc in sim_pl.tomorrow_cards)
                    )
                    neutral_q_ok = (
                        (_NEUTRAL_Q_OK_D0 if sim.day >= DAY_COUNT else _NEUTRAL_Q_OK_D1)
                        if can_use_neutral_q
                        else None
                    )
                    for i in range(len(trunk_list) - 1, -1, -1):
                        tc = trunk_list[i]
                        if known_ids is not None and tc.instance_id in known_ids:
                            continue
                        if not tc.markers:
                            s = tc.definition.slug
                            if neutral_q_ok is not None:
                                ok_nq = neutral_q_ok.get(s)
                                if ok_nq is not None:
                                    if ok_nq:
                                        chosen_trunk_indices.append(i)
                                        if len(chosen_trunk_indices) == needed:
                                            break
                                    continue
                            ok = seen_ok_slug.get(s)
                            if ok is None:
                                if sim_ctx is None:
                                    sim_ctx = _build_quick_zone_ctx(
                                        sim_pl, sim_is_canon
                                    )
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
                            if sim_ctx is None:
                                sim_ctx = _build_quick_zone_ctx(
                                    sim_pl, sim_is_canon
                                )
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
                    sim_ctx = _build_quick_zone_ctx(
                        sim.players[player_index], sim_is_canon
                    )
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

            prev_pending = self._draft_opp_pending_draws
            if (
                not live_player.played_today
                and game._is_setup
                and game.stats is not _get_shared_sim_stats()
                and game.day < DAY_COUNT
            ):
                my_pos = (player_index - game.starting_player) % 4
                n_fut = min(len(future_pools), 2)
                self._draft_opp_pending_draws = tuple(
                    0
                    if o_idx == player_index
                    else n_fut
                    + (
                        1
                        if ((o_idx - game.starting_player) % 4) > my_pos
                        else 0
                    )
                    for o_idx in range(4)
                )
            else:
                self._draft_opp_pending_draws = None
            try:
                _, sample_val = self._search_best_day_plan(
                    sim,
                    player_index,
                    beam_width=draft_beam,
                    max_plays=draft_max_plays,
                )
            finally:
                self._draft_opp_pending_draws = prev_pending
            return sample_val

        groups_by_key: dict[tuple, int] = {}
        groups: list[tuple[int, CardInstance, float, list[int]]] = []
        syn_sim = samples[0][0]
        for idx, option in enumerate(options):
            denial_val = _suitcase_denial_value(
                game, player_index, option, pr_markers, scout_markers
            )
            has_own_m = bool(
                own_markers and any(m in option.markers for m in own_markers)
            )
            synergy_val = (
                self._draft_synergy_adjustment(
                    syn_sim, player_index, option, wanted_tags
                )
                + denial_val
                - (3.0 if has_own_m else 0.0)
            )
            opt_key = (
                (idx,)
                if (option.markers or option.is_tomorrow or has_custom_draft_syn)
                else (id(option.definition), has_own_m, denial_val)
            )
            g_idx = groups_by_key.get(opt_key)
            if g_idx is None:
                groups_by_key[opt_key] = len(groups)
                groups.append((idx, option, synergy_val, [idx]))
            else:
                groups[g_idx][3].append(idx)

        plan_sums = [0.0] * len(groups)
        for g_i, (rep_idx, rep_option, _, _) in enumerate(groups):
            s_total = 0.0
            for base_sim, future_pools in samples:
                s_total += _eval_draft_option_sample(
                    rep_idx, rep_option, base_sim, future_pools
                )
            plan_sums[g_i] = s_total

        n_samples = len(samples)
        best_indices: list[int] = []
        best_value = float("-inf")
        for g_i, (_, _, syn_val, group_indices) in enumerate(groups):
            total_val = (plan_sums[g_i] / n_samples) + syn_val
            if total_val > best_value + 1e-9:
                best_value = total_val
                best_indices = list(group_indices)
            elif abs(total_val - best_value) <= 1e-9:
                best_indices.extend(group_indices)

        best_indices.sort()
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
        is_trunk_place = last_played in _HAND_TO_TRUNK_TITLES
        if (
            len(options) <= 1
            or self._evaluation_depth >= 1
            or game.stats is _get_shared_sim_stats()
            or (is_trunk_place and len(game.trunk) + len(game.discard) < 4)
            or len(player.played_today) >= 9
        ):
            return _OSMIUM_EVAL_AI_SINGLETON.choose_card_to_discard(
                game, player_index, options
            )

        self._ensure_live_game(game)
        base_sim = self._prepare_observable_game(game, player_index)
        sim_p0 = base_sim.players[player_index]
        sim_hand_ids = {hc.instance_id for hc in sim_p0.hand}
        if not all(opt.instance_id in sim_hand_ids for opt in options):
            return _OSMIUM_EVAL_AI_SINGLETON.choose_card_to_discard(
                game, player_index, options
            )

        groups: list[tuple[CardInstance, list[int]]] = []
        unmarked_group_idx: dict[int, int] = {}
        for opt_i, opt in enumerate(options):
            if not opt.markers and not opt.is_tomorrow:
                did = id(opt.definition)
                g_pos = unmarked_group_idx.get(did)
                if g_pos is None:
                    unmarked_group_idx[did] = len(groups)
                    groups.append((opt, [opt_i]))
                else:
                    groups[g_pos][1].append(opt_i)
            else:
                groups.append((opt, [opt_i]))

        if is_trunk_place:
            has_ed = any(
                _effective_card_title(pc) == "Epic Duels"
                for pc in player.played_today
            )
            p_keep = _trunk_top_keep_prob(
                game, player_index, has_epic_duels=has_ed
            )
            has_bcm_played = any(
                _effective_card_title(pc) == "Bougie Coffee Machine"
                for pc in player.played_today
            )
            has_opp_consumer = any(
                _effective_card_title(oc) in _TRUNK_CONSUMER_TITLES
                for o_i, opp in enumerate(game.players)
                if o_i != player_index
                for oc in opp.played_today
            )
            wanted_tags_live = _wanted_tags_for_player_fast(player)
        else:
            p_keep = 1.0
            has_bcm_played = False
            has_opp_consumer = False
            wanted_tags_live = {}

        best_val = float("-inf")
        best_indices: list[int] = []
        prev_depth = self._evaluation_depth
        self._evaluation_depth = 1
        try:
            for opt, group_indices in groups:
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
                if is_trunk_place:
                    try:
                        rem_prev = _get_sim_known_trunk_top(sim)
                        target_card = sim_player.hand.pop(hand_idx)
                        target_card.is_tomorrow = False
                        target_card.markers.clear()
                        sim.trunk.append(target_card)
                        sim._known_trunk_state = (  # type: ignore[attr-defined]
                            len(sim.trunk),
                            (target_card,) + rem_prev,
                        )
                        sim._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                        if sim_player.played_today:
                            sim_source = sim_player.played_today[-1]
                            for t_src in tuple(sim_player.tomorrow_cards):
                                t_src.effective_behavior.on_tomorrow_card_play(
                                    sim, sim_player, t_src, sim_source
                                )
                            for p_src in tuple(sim_player.played_today):
                                p_src.effective_behavior.on_card_play(
                                    sim, sim_player, p_src, sim_source
                                )
                            _, val_kept = self._search_best_day_plan(
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
                            _, val_kept = self._search_best_day_plan(
                                sim,
                                player_index,
                                beam_width=8,
                                max_plays=6,
                                allow_bed_now=True,
                            )
                    except Exception:
                        continue
                    if p_keep >= 0.999:
                        val = val_kept
                    else:
                        gift_pen = (
                            0.20
                            * max(
                                0.0,
                                _hydrogen_quick_card_desirability(
                                    game,
                                    player_index,
                                    opt,
                                    wanted_tags_live,
                                    _in_hand=False,
                                ),
                            )
                            if opt.effective_cost <= 3
                            else 0.0
                        )
                        has_rem_consumer = any(
                            hc.definition.title in _TRUNK_CONSUMER_TITLES
                            or (has_bcm_played and hc.definition.title == "Leftovers")
                            or (
                                has_opp_consumer
                                and hc.definition.title == "Wedding Anniversary"
                            )
                            for hc in sim_player.hand
                        )
                        if has_rem_consumer:
                            try:
                                sim_lost = _hydrogen_copy_game(base_sim)
                                sim_lost.rng = _clone_random(base_sim.rng)
                                sim_lost_player = sim_lost.players[player_index]
                                sim_lost.discard_from_hand(player_index, hand_idx)
                                sim_lost._known_trunk_state = None  # type: ignore[attr-defined]
                                sim_lost._trunk_keep_prob = 1.0  # type: ignore[attr-defined]
                                if sim_lost_player.played_today:
                                    lost_source = sim_lost_player.played_today[-1]
                                    for t_src in tuple(sim_lost_player.tomorrow_cards):
                                        t_src.effective_behavior.on_tomorrow_card_play(
                                            sim_lost,
                                            sim_lost_player,
                                            t_src,
                                            lost_source,
                                        )
                                    for p_src in tuple(sim_lost_player.played_today):
                                        p_src.effective_behavior.on_card_play(
                                            sim_lost,
                                            sim_lost_player,
                                            p_src,
                                            lost_source,
                                        )
                                    _, val_lost = self._search_best_day_plan(
                                        sim_lost,
                                        player_index,
                                        beam_width=6,
                                        max_plays=5,
                                        allow_bed_now=True,
                                        base_turns_offset=1,
                                        root_skipped_override=sim_p0.skipped_turns,
                                        start_extra_active=_card_allows_extra_plays(
                                            sim_lost, sim_lost_player, lost_source
                                        ),
                                    )
                                else:
                                    _, val_lost = self._search_best_day_plan(
                                        sim_lost,
                                        player_index,
                                        beam_width=6,
                                        max_plays=5,
                                        allow_bed_now=True,
                                    )
                            except Exception:
                                val_lost = val_kept
                            val = p_keep * val_kept + (1.0 - p_keep) * (
                                val_lost - gift_pen
                            )
                        else:
                            val = val_kept - (1.0 - p_keep) * gift_pen
                else:
                    try:
                        sim.discard_from_hand(player_index, hand_idx)
                        if sim_player.played_today:
                            sim_source = sim_player.played_today[-1]
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
                    best_indices = list(group_indices)
                elif abs(val - best_val) <= 1e-9:
                    best_indices.extend(group_indices)
        finally:
            self._evaluation_depth = prev_depth

        if not best_indices:
            return _OSMIUM_EVAL_AI_SINGLETON.choose_card_to_discard(
                game, player_index, options
            )
        best_indices.sort()
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
        self._ensure_live_game(game)
        known_top = self._get_live_known_trunk_top(game, player_index)
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
            - (1.35 if options[i].definition.title == "Group Dinner" else 0.0)
            + (
                3.0
                if known_top and options[i].definition.title in _TRUNK_CONSUMER_TITLES
                else 0.0
            ),
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
                sim.ais = _OSMIUM_SIM_AIS
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
                    sim.ais = _HYDROGEN_SIM_AIS
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
                finally:
                    sim.ais = _HYDROGEN_SIM_AIS
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
                top_c = known_top[0]
                if "Outdoors" in top_c.tags:
                    return False
                reach_e = _fast_reachable_energy(game, player_index)
                has_ac = any(
                    hc.definition.title == "Assorted Cutlery"
                    and game.energy_cost(player_index, hc) <= reach_e
                    for hc in player.hand
                )
                if (
                    has_ac
                    and _osmium_free_play_value(game, player_index, top_c) >= 5.5
                ):
                    return False
                return True
            unseen_counts, unseen_total = _unseen_tag_counts_and_total(
                game, player_index, known_top
            )
            p_outdoors = unseen_counts.get("Outdoors", 50) / max(1, unseen_total)
            mult = self._quacks_multiplier(game, player_index)
            return p_outdoors < (mult / (1.0 + draws))

        return super().choose_optional_action(game, player_index, action)


Osmium = OsmiumAI
OsmiumBrain = OsmiumAI
OsmiumbrainAI = OsmiumAI
