"""Opponent-aware, data-calibrated lookahead policy (CarbonAI)."""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING
import random
import weakref

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
    BerylliumBrain,
    _ACQUIRER_TITLES,
    _HIGH_COPY_FOOD_TITLES,
    _NEUTRAL_DRAFT_DEFNS,
    _NOOP_COUNTER,
    _STOCHASTIC_PLAY_TITLES,
    _TAGGED_NEUTRAL_3_BY_TAG,
    _allows_energy_gain_today,
    _beryllium_build_profile as _raw_beryllium_build_profile,
    _beryllium_free_play_value,
    _contextual_copy_value,
    _effective_card_title as _raw_effective_card_title,
    _effective_source_definition,
    _evaluate_istanbul_discards,
    _get_beryllium_sim_game_cls,
    _istanbul_energy_per_discard,
    _istanbul_extra_energy_support,
    _select_bohnanza_discards,
    _select_istanbul_discards,
)
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.models import PlayerState

if TYPE_CHECKING:
    from thenos.game import Game


_CARBON_PROFILE_BY_DEFN_ID: dict[int, CardSynergyProfile] = {}
_CARBON_RESERVE_EVAL_CACHE: dict[tuple, float] = {}
_ALL_TAG_BONUS_KEYS: tuple[str, ...] = (
    "Relax",
    "Food",
    "Exercise",
    "Social",
    "Outdoors",
    "Indoors",
    "Board Game",
    "Item",
)


def _ensure_carbon_caches() -> None:
    if _CARBON_PROFILE_BY_DEFN_ID:
        return
    from thenos.cards.catalog import CARD_REGISTRY

    for defn in (
        *CARD_REGISTRY.values(),
        *_NEUTRAL_DRAFT_DEFNS,
        *_TAGGED_NEUTRAL_3_BY_TAG.values(),
    ):
        did = id(defn)
        if did not in _CARBON_PROFILE_BY_DEFN_ID:
            _CARBON_PROFILE_BY_DEFN_ID[did] = _raw_beryllium_build_profile(
                CardInstance(-1, defn)
            )


def _beryllium_build_profile(card: CardInstance) -> CardSynergyProfile:
    m = card.markers
    if not m or "_copied_definition" not in m:
        did = id(card.definition)
        if did in _CARBON_PROFILE_BY_DEFN_ID:
            return _CARBON_PROFILE_BY_DEFN_ID[did]
        _ensure_carbon_caches()
        if did in _CARBON_PROFILE_BY_DEFN_ID:
            return _CARBON_PROFILE_BY_DEFN_ID[did]
    return _raw_beryllium_build_profile(card)


def _effective_card_title(card: CardInstance) -> str:
    m = card.markers
    if not m or "_copied_definition" not in m:
        return card.definition.title
    return _raw_effective_card_title(card)


def _wanted_tags_for_player(player: PlayerState) -> dict[str, float]:
    """Compute signed tag synergy weights using cached definition profiles."""
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

    prof_by_id = _CARBON_PROFILE_BY_DEFN_ID
    for card in player.tomorrow_cards:
        m = card.markers
        un_copied = not m or "_copied_definition" not in m
        did = id(card.definition)
        prof = (
            prof_by_id[did]
            if (un_copied and did in prof_by_id)
            else _beryllium_build_profile(card)
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
        c_beh = (
            card.definition.behavior if un_copied else card.effective_behavior
        )
        if c_beh.tomorrow_all_tags:
            for t in _ALL_TAG_BONUS_KEYS:
                weights[t] = weights[t] + 0.35 if t in weights else 0.35

    for card in player.played_today:
        m = card.markers
        did = id(card.definition)
        prof = (
            prof_by_id[did]
            if ((not m or "_copied_definition" not in m) and did in prof_by_id)
            else _beryllium_build_profile(card)
        )
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
        did = id(card.definition)
        prof = (
            prof_by_id[did]
            if ((not m or "_copied_definition" not in m) and did in prof_by_id)
            else _beryllium_build_profile(card)
        )
        if prof.is_multi_scaler or prof.is_tomorrow_scaler:
            delta = prof.signed_match_benefit * 0.90
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta
        elif prof.is_single_buff:
            delta = prof.signed_match_benefit * 0.30
            for t in prof.target_tags:
                weights[t] = weights[t] + delta if t in weights else delta
        c_tags = card.definition.tags if not m else card.tags
        for t in c_tags:
            weights[t] = weights[t] + 0.15 if t in weights else 0.15
    return weights


_ISTANBUL_SIMPLE_MARKER_KEYS: frozenset[str] = frozenset(
    {"_gave_energy", "_all_tags"}
)
_ISTANBUL_ROLLOUT_CACHE: dict[tuple, tuple[int, ...]] = {}


def _select_istanbul_discards_fast(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> tuple[int, ...]:
    prof_by_id = _CARBON_PROFILE_BY_DEFN_ID
    if not prof_by_id:
        _ensure_carbon_caches()
    player = game.players[player_index]
    can_cache = (
        len(player.hand) == len(options)
        and all(
            not c.markers and hc is c and id(c.definition) in prof_by_id
            for hc, c in zip(player.hand, options)
        )
        and all(
            not tc.markers and id(tc.definition) in prof_by_id
            for tc in player.tomorrow_cards
        )
        and all(
            (not pc.markers or pc.markers.keys() <= _ISTANBUL_SIMPLE_MARKER_KEYS)
            and id(pc.definition) in prof_by_id
            for pc in player.played_today
        )
        and all(
            id(oc.definition) in prof_by_id
            and (
                not oc.markers
                or "_copied_definition" not in oc.markers
                or id(_effective_source_definition(oc)) in prof_by_id
            )
            for idx, opp in enumerate(game.players)
            if idx != player_index
            for oc in opp.played_today
        )
    )
    if not can_cache:
        return _select_istanbul_discards(game, player_index, options)

    opp_key = tuple(
        (
            opp.fun,
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
        for idx, opp in enumerate(game.players)
        if idx != player_index
    )
    key = (
        game.day,
        game.daily_condition.title if game.daily_condition is not None else None,
        player_index,
        player.character.title if player.character is not None else None,
        player.energy,
        player.fun,
        opp_key,
        tuple(tc.definition.slug for tc in player.tomorrow_cards),
        tuple(
            (pc.definition.slug, bool(pc.markers.get("_gave_energy")))
            if pc.markers
            else pc.definition.slug
            for pc in player.played_today
        ),
        tuple(c.definition.slug for c in options),
    )
    cached = _ISTANBUL_ROLLOUT_CACHE.get(key)
    if cached is not None:
        return cached
    res = _select_istanbul_discards(game, player_index, options)
    if len(_ISTANBUL_ROLLOUT_CACHE) > 8192:
        _ISTANBUL_ROLLOUT_CACHE.clear()
    _ISTANBUL_ROLLOUT_CACHE[key] = res
    return res


def _carbon_own_suitcase_markers(player: PlayerState) -> set[str]:
    """Return Suitcase marker keys placed by `player` today (including copied Scout/Puerto Rico)."""
    markers: set[str] = set()
    for c in player.played_today:
        pr = c.markers.get("suitcase_target_marker")
        if isinstance(pr, str) and pr:
            markers.add(pr)
        if _effective_card_title(c) == "Scout Other Cottages":
            markers.add(f"_scout_energy_cube_{c.instance_id}")
    return markers


def _opponent_suitcase_markers(
    game: Game, player_index: int
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Return (puerto_rico_marker -> owner_idx, scout_markers) placed by opponents."""
    pr_markers: dict[str, int] = {}
    scout_markers: list[str] = []
    for idx, opp in enumerate(game.players):
        if idx == player_index:
            continue
        for c in opp.played_today:
            pr = c.markers.get("suitcase_target_marker")
            if isinstance(pr, str) and pr:
                pr_markers[pr] = idx
            if _effective_card_title(c) == "Scout Other Cottages":
                mk = f"_scout_energy_cube_{c.instance_id}"
                if mk not in scout_markers:
                    scout_markers.append(mk)
    return pr_markers, tuple(scout_markers)


def _get_known_trunk_top_any(
    game: Game, player_index: int
) -> tuple[CardInstance, ...]:
    """Return known top-of-Trunk cards for `player_index` on either sim or live `Game`."""
    if game.stats is _get_shared_sim_stats():
        return _get_sim_known_trunk_top(game)
    if 0 <= player_index < len(game.ais):
        ai = game.ais[player_index]
        if isinstance(ai, CarbonAI):
            return ai._get_live_known_trunk_top(game, player_index)
    return ()


def _suitcase_denial_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    pr_markers: dict[str, int] | None = None,
    scout_markers: Sequence[str] | None = None,
) -> float:
    """Zero-sum value gained from denying opponent Suitcase markers or hate-drafting."""
    bonus = 0.0
    player = game.players[player_index]
    if card.markers:
        if pr_markers is None or scout_markers is None:
            pr_markers, scout_markers = _opponent_suitcase_markers(
                game, player_index
            )
        for m, owner_idx in pr_markers.items():
            if card.markers.get(m):
                opp_fun = game.players[owner_idx].fun
                # Denying 3 Fun to an opponent is worth ~1.1-1.5 relative Fun
                bonus += 1.45 if opp_fun >= player.fun - 3 else 1.10
        for sm in scout_markers:
            if card.markers.get(sm):
                bonus += 0.95

    # Hate-drafting during morning Draw Phase when an opponent has a strong visible engine
    if not player.played_today:
        c_tags = card.definition.tags if not card.markers else card.tags
        c_cost = card.definition.cost if not card.markers else card.effective_cost
        c_title = card.definition.title
        # Next opponent in turn order gets first crack at unpicked Suitcase cards
        next_opp_idx = (player_index + 1) % len(game.players)
        next_opp = game.players[next_opp_idx]
        opp_threat = 0.0
        for tc in next_opp.tomorrow_cards:
            t_title = _effective_card_title(tc)
            if t_title == "Photo Shoot" and (c_tags & {"Social", "Relax"}) and c_cost <= 2:
                opp_threat += 0.45
            elif t_title == "Date's First Nos" and c_cost <= 1:
                opp_threat += 0.35
            else:
                tc_prof = _beryllium_build_profile(tc)
                if tc_prof.signed_match_benefit >= 1.5 and (
                    c_tags & tc_prof.target_tags
                ):
                    opp_threat += 0.25
        if c_title in {"Colouring", "Euchre Awards Ceremony", "Butter Tarts"}:
            opp_threat += 0.25
        bonus += min(opp_threat, 0.75)

    return bonus


def _carbon_extra_energy_support(
    game: Game,
    player_index: int,
    card: CardInstance,
    *,
    extra_fodder: int = 0,
    ignore_today_block: bool = False,
    include_future_tomorrow: bool = False,
) -> int:
    """Compute unplayed net Energy ramp + unplayed tag discounts in `player.hand`.

    Active `tomorrow_cards` and `played_today` cost modifiers/start-of-day Energy are
    already reflected in `game.energy_cost(player_index, card)` and `player.energy`,
    so they are never re-added for today. When `include_future_tomorrow=True` (used
    for future-day `7 + support` checks on Days 1-5), Tomorrow Energy generators
    played today or held in hand are also credited.
    """
    player = game.players[player_index]
    eff_cost = card.definition.cost if not card.markers else card.effective_cost
    c_tags = card.definition.tags if not card.markers else card.tags
    can_gain_e = (
        ignore_today_block
        or include_future_tomorrow
        or _allows_energy_gain_today(game, player)
    )
    support = 0

    if include_future_tomorrow:
        for pc in player.played_today:
            if pc is card or pc.is_tomorrow:
                continue
            p_prof = _beryllium_build_profile(pc)
            if p_prof.goes_to_tomorrow and p_prof.tomorrow_start_energy > 0:
                support += p_prof.tomorrow_start_energy

    has_sunscreen = any(
        _effective_card_title(hc) == "Sunscreen"
        for hc in (
            player.hand
            if (ignore_today_block or include_future_tomorrow)
            else (*player.hand, *player.played_today)
        )
        if hc is not card
    )
    for hc in player.hand:
        if hc is card:
            continue
        h_title = _effective_card_title(hc)
        h_prof = _beryllium_build_profile(hc)
        h_cost = hc.definition.cost if not hc.markers else hc.effective_cost

        if h_title == "Shady Spot" and "Outdoors" in c_tags:
            shady_cost = max(0, h_cost - (1 if has_sunscreen else 0))
            support += max(0, (eff_cost - (eff_cost // 2)) - shady_cost)
            continue
        if h_title == "Forced Family Fun" and "Board Game" in c_tags:
            support += max(0, (eff_cost - (eff_cost // 2)) - h_cost)
            continue
        if h_title == "Boat Ride" and "Item" in c_tags:
            support += max(0, eff_cost - h_cost)
            continue
        if h_title == "Treat Cereal" and "Exercise" in c_tags:
            support += max(0, 4 - h_cost)
            continue

        if can_gain_e:
            if h_prof.immediate_energy_gain > h_cost and h_title not in {
                "Sleep In",
                "Pudding Chômeur",
            }:
                support += h_prof.immediate_energy_gain - h_cost
            elif h_title == "Pudding Chômeur" and len(player.hand) >= 2:
                support += 2
            elif (
                h_prof.conditional_energy_gain > h_cost
                and h_title
                not in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}
            ):
                support += h_prof.conditional_energy_gain - h_cost
            elif h_title == "Potato Pancakes":
                relax_zone = (
                    player.hand
                    if (ignore_today_block or include_future_tomorrow)
                    else (*player.played_today, *player.hand)
                )
                relax_cnt = sum(
                    1
                    for oc in relax_zone
                    if oc is not hc and oc is not card and "Relax" in oc.tags
                )
                support += max(0, 2 * relax_cnt - h_cost)
            elif h_title == "Würstchen":
                est_hand = len(player.hand) + extra_fodder - 1
                support += max(0, est_hand - h_cost)
            elif h_title == "Butter Tarts":
                support += 4
            elif (
                include_future_tomorrow
                and h_prof.tomorrow_start_energy > 0
                and game.day < DAY_COUNT
            ):
                support += max(0, h_prof.tomorrow_start_energy - 1)
        elif (
            include_future_tomorrow
            and h_prof.tomorrow_start_energy > 0
            and game.day < DAY_COUNT
        ):
            support += max(0, h_prof.tomorrow_start_energy - 1)

        if (
            h_prof.energy_delta < 0
            and not h_prof.is_single_buff
            and (not h_prof.target_tags or (c_tags & h_prof.target_tags))
        ):
            support += max(0, abs(h_prof.energy_delta) - max(0, h_cost - 1))

    support += _istanbul_extra_energy_support(
        game,
        player_index,
        card,
        extra_fodder=extra_fodder,
        ignore_today_block=ignore_today_block or include_future_tomorrow,
    )
    return support


def _carbon_free_play_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float] | None = None,
) -> float:
    """Accurate, PYTHONHASHSEED-deterministic value of playing `card` for 0 Energy."""
    val = _beryllium_free_play_value(game, player_index, card, {})
    player = game.players[player_index]
    if wanted_tags is None:
        wanted_tags = _wanted_tags_for_player(player)
    zero_tag = (
        player.character.zero_fun_tag
        if player.character is not None
        else ""
    )
    if wanted_tags:
        c_tags = card.tags
        for t in (c_tags if len(c_tags) <= 1 else sorted(c_tags)):
            w = wanted_tags.get(t, 0.0)
            if w > 0.0:
                val += w * 0.85
            elif w < 0.0 and t != zero_tag:
                val += w * 0.50
    title = _effective_card_title(card)
    if zero_tag and title == "Colouring":
        return -8.0
    if title == "Group Dinner":
        val -= 1.35
    elif title == "Tell a Story":
        val -= 1.25
    elif title == "Adventure Race" and not (zero_tag and zero_tag in card.tags):
        val = max(val, 18.0)
    elif title == "Date's First Nos" and game.day < DAY_COUNT:
        val += 3.5
    elif title == "Civilization the Boardgame":
        has_ct = any(
            _effective_card_title(c) == "Couch Tube"
            for c in player.hand
        )
        if has_ct:
            val += 8.5
    return val


_QUICK_BRANCH_TITLES: frozenset[str] = frozenset(
    {
        "Nos Book",
        "Group Dinner",
        "Tell a Story",
        "Assorted Cutlery",
        "Bougie Coffee Machine",
        "Medical Advice",
        "Scout Other Cottages",
        "Bohnanza",
        "Ski on Shoulders",
        "Cheesy Phone Game",
        "Wakesurf",
        "Slalom Start",
    }
)
_QUICK_PENALTY_BY_TITLE: dict[str, float] = {
    "High-End White": 0.55,
    "Euchre": 0.30,
    "Solo": 0.70,
    "Hold the Baby": 0.45,
}
_TERMINAL_AWAKE_OPP_TITLES: frozenset[str] = frozenset(
    {
        "Fit to Print",
        "Play With the Kids",
        "Puerto Rico",
        "Scout Other Cottages",
        "San Juan",
        "Cheesy Phone Game",
    }
)
_RESERVE_FLAT_EXTRA: dict[str, float] = {
    "Wedding Anniversary": 2.1,
    "Euchre Awards Ceremony": 1.8,
}
_RESERVE_EMPIRICAL_FLAT: dict[str, float] = {
    "Cheesy Phone Game": -0.85,
    "Bohnanza": 1.1,
    "Group Dinner": -1.2,
}
_RESERVE_DYNAMIC_EXTRA_1: frozenset[str] = frozenset(
    {
        "Leftovers",
        "Paddleboat",
        "Potato Pancakes",
        "Keeper",
        "Camp Crossroads Dessert",
        "Pudding Chômeur",
    }
)
_RESERVE_DYNAMIC_EXTRA_2: frozenset[str] = frozenset(
    {
        "Assorted Cutlery",
        "Ski on Shoulders",
        "Adventure Race",
        "Würstchen",
        "Medical Advice",
        "Wakesurf",
        "Slalom Start",
    }
)


def _carbon_quick_card_desirability(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
    *,
    cost: int | None = None,
    prof: CardSynergyProfile | None = None,
    remaining_draw_picks: int = 0,
) -> float:
    """Data-calibrated heuristic card desirability."""
    m = card.markers
    uncopied = not m or "_copied_definition" not in m
    if prof is None:
        prof = (
            _CARBON_PROFILE_BY_DEFN_ID.get(id(card.definition))
            if uncopied
            else None
        )
        if prof is None:
            prof = _beryllium_build_profile(card)
    player = game.players[player_index]
    if cost is None:
        cost = game.energy_cost(player_index, card)
    days_left = DAY_COUNT - game.day
    c_defn = card.definition if uncopied else card.effective_definition
    title = c_defn.title
    eff_cost = c_defn.cost
    eff_base_fun = c_defn.base_fun
    c_tags = c_defn.tags if not m else card.tags
    zero_tag = (
        player.character.zero_fun_tag
        if player.character is not None
        else ""
    )
    is_zeroed = bool(zero_tag and zero_tag in c_tags)

    if title == "Würstchen":
        proj_hand = len(player.hand) + remaining_draw_picks
        if card not in player.hand:
            proj_hand += (1 if days_left == 0 else 2)
        imm_energy = max(2, proj_hand)
    else:
        imm_energy = prof.immediate_energy_gain

    scored_base_fun = 0.0 if is_zeroed else float(eff_base_fun)
    if (
        not is_zeroed
        and game.daily_condition is not None
        and game.daily_condition.fun_tag
        and game.daily_condition.fun_tag in c_tags
    ):
        cond_weight = 1.0 if (days_left == 0 or player.played_today) else 0.55
        scored_base_fun += game.daily_condition.fun_delta * cond_weight

    imm_fun_gain = 0.0 if is_zeroed else prof.immediate_fun_gain

    val = (
        scored_base_fun
        + imm_fun_gain * 1.15
        - cost * 0.78
        + imm_energy * 1.30
    )

    if title == "Istanbul" and _allows_energy_gain_today(game, player, card):
        e_per = _istanbul_energy_per_discard(card)
        other_cnt = (
            sum(1 for hc in player.hand if hc is not card) + remaining_draw_picks
        )
        has_e_or_empty_sink = any(
            _effective_card_title(hc)
            in {"Kayak", "Decaf", "Wakesurf", "Adventure Race", "Wakeboard"}
            for hc in player.hand
            if hc is not card
        ) or any(
            _effective_card_title(pc) in {"Decaf", "Wakesurf"}
            for pc in player.played_today
        )
        disc_est = min(
            3,
            other_cnt if has_e_or_empty_sink else max(0, other_cnt - 1),
        )
        if disc_est > 0:
            other_cost = sum(
                game.energy_cost(player_index, hc)
                for hc in player.hand
                if hc is not card
                and hc.effective_behavior.can_play(game, player, hc)
            )
            has_sink = has_e_or_empty_sink or other_cost > max(
                3, player.energy - cost
            )
            sac_cost = 0.70 if days_left == 0 else 1.45
            net_per = max(0.0, e_per * (1.20 if has_sink else 0.75) - sac_cost)
            val += disc_est * net_per

    if (
        prof.immediate_cards_gained > 0
        and not prof.penalizes_before
        and not prof.penalizes_after
        and not prof.is_multi_scaler
    ):
        val += min(prof.immediate_cards_gained, 3) * (
            1.05 if cost <= 2 else 0.25
        )

    if title == "Sleep In":
        val -= 1.6
    elif (
        prof.must_play_late
        and prof.immediate_energy_gain > 0
        and len(player.played_today) < 2
    ):
        val -= 0.75

    # Accurate high-cost funding check without double-counting active discounts
    if eff_cost >= 8 and cost > player.energy:
        today_sup = _carbon_extra_energy_support(
            game, player_index, card, extra_fodder=remaining_draw_picks
        )
        today_unfunded = max(0, cost - (player.energy + today_sup))
        if days_left >= 1:
            extra_fodder = remaining_draw_picks + (
                2 if card not in player.hand else 0
            )
            future_sup = _carbon_extra_energy_support(
                game,
                player_index,
                card,
                extra_fodder=extra_fodder,
                include_future_tomorrow=True,
            )
            char_cost = (
                max(0, player.character.modify_energy_cost(c_tags, eff_cost))
                if player.character is not None
                else eff_cost
            )
            future_unfunded = max(0, char_cost - (7 + future_sup))
            unfunded = min(today_unfunded, future_unfunded)
            support_for_slack = max(today_sup, future_sup)
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
            eff_unfunded = max(0, unfunded - slack)
            val -= eff_unfunded * (1.50 if days_left >= 1 else 1.95)

    if prof.conditional_energy_gain > 0:
        if title in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}:
            t_tags = prof.target_tags
            match_cnt = 0
            for other in player.played_today:
                if other is not card and (
                    (other.definition.tags if not other.markers else other.tags)
                    & t_tags
                ):
                    match_cnt += 1
            for other in player.hand:
                if other is not card and (
                    (other.definition.tags if not other.markers else other.tags)
                    & t_tags
                ):
                    match_cnt += 1
            per_match = 2 if title == "Potato Pancakes" else 1
            future_matches = 0.8 if (days_left >= 1 and title == "Potato Pancakes") else 0.3
            val += (min(match_cnt, 3) + future_matches) * per_match * 0.95
        else:
            t_tags = prof.target_tags
            has_cond = _condition_met_in_played(player, card, prof)
            if not has_cond:
                for other in player.hand:
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

    # Calibrated card-specific adjustments (grounded in 20k-game mechanics)
    if title in _QUICK_BRANCH_TITLES:
        if title == "Nos Book":
            if player.played_today or len(player.hand) >= 2 or game.day < DAY_COUNT:
                val += 2.45
        elif title == "Group Dinner":
            val -= 1.85
        elif title == "Tell a Story":
            has_same_turn_combo = (
                any(
                    other.definition.title
                    in {"Assorted Cutlery", "Bougie Coffee Machine"}
                    for other in player.hand
                )
                and (
                    any(
                        _effective_card_title(pc) == "Epic Duels"
                        for pc in player.played_today
                    )
                    or any(
                        other.definition.title == "Epic Duels"
                        for other in player.hand
                    )
                    or all(
                        opp.asleep
                        for idx, opp in enumerate(game.players)
                        if idx != player_index
                    )
                )
            )
            if not has_same_turn_combo:
                val -= 1.45
        elif title == "Assorted Cutlery":
            has_combo = bool(_get_known_trunk_top_any(game, player_index))
            if not has_combo:
                for other in player.hand:
                    if other is not card and other.definition.title in {
                        "Read the Radar",
                        "Fishing Boat",
                        "Ski Boat",
                        "Grocery Store Run",
                        "Boat Ride",
                        "Fancy Craft",
                    }:
                        has_combo = True
                        break
            val += 2.6 if has_combo else 0.55
        elif title == "Bougie Coffee Machine":
            val += 1.35
        elif title == "Medical Advice":
            if days_left >= 1 and len(player.played_today) <= 1:
                pb_bonus = (
                    4.5
                    if any(
                        _effective_card_title(c) == "Paddleboat"
                        for c in (*player.played_today, *player.hand)
                    )
                    else 0.0
                )
                val += 4.6 + pb_bonus - 0.9 * len(player.played_today)
        elif title == "Scout Other Cottages":
            if days_left >= 1 and len(player.played_today) <= 1:
                val += 3.8 - 0.9 * len(player.played_today)
        elif title == "Bohnanza":
            fodder_cnt = 0
            for zone in (player.played_today, player.hand):
                for c in zone:
                    if (
                        c is not card
                        and (
                            c.definition.base_fun
                            if not c.markers
                            else c.effective_base_fun
                        )
                        <= 0
                        and not _beryllium_build_profile(c).goes_to_tomorrow
                    ):
                        fodder_cnt += 1
            val += 1.15 + 0.65 * min(fodder_cnt, 3)
        elif title == "Ski on Shoulders":
            if game.day >= 2:
                opps_strictly_ahead = 0
                for idx, opp in enumerate(game.players):
                    if idx != player_index and opp.fun > player.fun:
                        opps_strictly_ahead += 1
                if opps_strictly_ahead >= 2:
                    val += 2.1
                elif days_left >= 1 and opps_strictly_ahead == 1:
                    val += 0.4
                else:
                    val -= 0.5
            else:
                val += 0.3
        elif title == "Cheesy Phone Game":
            val += 1.05
        elif title == "Wakesurf":
            if not is_zeroed:
                val += 2.20
                if len(player.hand) >= 4 and not any(
                    other.definition.title
                    in {"Paddleboard", "Navy SEALing", "Istanbul", "Scrabble"}
                    for other in player.hand
                ):
                    val -= 0.65 * (len(player.hand) - 3)
        elif title == "Slalom Start":
            if not is_zeroed:
                val += 2.00
                if len(player.hand) >= 4 and not any(
                    other.definition.title
                    in {"Paddleboard", "Navy SEALing", "Istanbul", "Scrabble"}
                    for other in player.hand
                ):
                    val -= 0.65 * (len(player.hand) - 3)
        if is_zeroed and title in {"Wakesurf", "Slalom Start"} and len(player.hand) >= 4:
            if not any(
                other.definition.title
                in {"Paddleboard", "Navy SEALing", "Istanbul", "Scrabble"}
                for other in player.hand
            ):
                val -= 0.65 * (len(player.hand) - 3)
    elif (
        prof.conditional_fun_bonus > 0
        and not prof.is_multi_scaler
        and not is_zeroed
    ):
        val += min(prof.conditional_fun_bonus, 5) * 0.55

    flat_pen = _QUICK_PENALTY_BY_TITLE.get(title)
    if flat_pen is not None:
        val -= flat_pen

    if wanted_tags:
        tag_match = 0.0
        iter_tags = c_tags if len(c_tags) <= 1 else sorted(c_tags)
        for t in iter_tags:
            if zero_tag and t == zero_tag:
                if eff_base_fun > 0 or prof.conditional_fun_bonus > 0:
                    tag_match -= 0.75
                continue
            w = wanted_tags.get(t, 0.0)
            if w != 0.0:
                tag_match += w * 0.78
                if w > 0 and cost <= 1:
                    tag_match += 0.38 * min(w, 2.5)
        if prof.must_play_first and tag_match > 0.0:
            tag_match *= 0.30
        val += tag_match

    pure_zero_tag_fun_buff = bool(
        zero_tag
        and prof.target_tags == frozenset({zero_tag})
        and prof.energy_delta >= 0
    )

    if prof.is_multi_scaler and not pure_zero_tag_fun_buff:
        t_tags = prof.target_tags
        matching_in_hand = 0
        for c in player.hand:
            if c is not card and (
                (c.definition.tags if not c.markers else c.tags) & t_tags
            ):
                matching_in_hand += 1
        matching_played = 0
        for c in player.played_today:
            if c is not card and (
                (c.definition.tags if not c.markers else c.tags) & t_tags
            ):
                matching_played += 1
        abundance = _tag_abundance(t_tags)
        future_factor = min(days_left, 3) * 0.50 * abundance
        val += (
            matching_in_hand * 0.95
            + matching_played * 0.50
            + future_factor
        ) * prof.signed_match_benefit
    elif prof.is_tomorrow_scaler and not pure_zero_tag_fun_buff:
        if days_left >= 1:
            t_tags = prof.target_tags
            matching_in_hand = 0
            for c in player.hand:
                if c is not card and (
                    (c.definition.tags if not c.markers else c.tags) & t_tags
                ):
                    matching_in_hand += 1
            abundance = _tag_abundance(t_tags)
            val += (
                matching_in_hand * 0.65 + 0.45 * abundance
            ) * prof.signed_match_benefit
    elif prof.is_single_buff and not pure_zero_tag_fun_buff:
        t_tags = prof.target_tags
        has_match = False
        for c in player.hand:
            if c is not card and (
                not t_tags
                or ((c.definition.tags if not c.markers else c.tags) & t_tags)
            ):
                has_match = True
                break
        abundance = _tag_abundance(t_tags) if t_tags else 1.0
        val += prof.signed_match_benefit * (
            0.78 if has_match else 0.25 * abundance
        )
    elif prof.signed_match_benefit < 0.0 and title not in {
        "Medical Advice",
        "Scout Other Cottages",
    }:
        val += prof.signed_match_benefit * 1.35

    if prof.is_global_after_scaler:
        if zero_tag:
            val -= 8.0
        else:
            cheap_cards = 0
            for c in player.hand:
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
                + min(days_left, 2) * 0.50
            )
    elif title == "Bring a Friend":
        zero_or_sink = 0
        for c in player.hand:
            if c is not card and (
                (c.definition.cost if not c.markers else c.effective_cost) == 0
                or c.definition.title
                in {
                    "Assorted Cutlery",
                    "Address the Food",
                    "Bougie Coffee Machine",
                }
            ):
                zero_or_sink += 1
        val += zero_or_sink * 1.65
    elif title == "Evening on the Dock":
        val += 3.5
    elif title == "Couch Tube":
        val += 3.1

    if prof.goes_to_tomorrow and days_left >= 1:
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

    return val


def _carbon_hand_keep_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
) -> float:
    """Value of keeping `card` in hand right now using Carbon calibrations."""
    player = game.players[player_index]
    can_play_now = card.effective_behavior.can_play(game, player, card)
    cost = game.energy_cost(player_index, card)
    if game.day >= DAY_COUNT:
        if not can_play_now:
            return -5.0
        extra_e = _carbon_extra_energy_support(game, player_index, card)
        if cost > player.energy + extra_e and not any(
            _effective_card_title(c) in {"Wrestle the Kids", "Play With the Kids"}
            for c in (*player.played_today, *player.hand)
        ):
            return -4.0
    val = _carbon_quick_card_desirability(
        game, player_index, card, wanted_tags, cost=cost
    )
    if not can_play_now:
        val -= 1.25
    elif cost <= player.energy:
        val += 0.45
    return val


class _CarbonEvalAI:
    """High-accuracy tactical rollout policy used inside CarbonAI simulations & sub-decisions."""

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
        best_idx = 0
        best_score = float("-inf")
        for idx, card in enumerate(options):
            cost = game.energy_cost(player_index, card)
            score = _carbon_quick_card_desirability(
                game, player_index, card, wanted_tags, cost=cost
            )
            if player.played_today and cost <= player.energy:
                if card.effective_behavior.can_play(game, player, card):
                    score += 0.65
                elif game.day >= DAY_COUNT:
                    score -= 3.0
            if (
                own_markers
                and card.markers
                and any(m in card.markers for m in own_markers)
            ):
                score -= 3.0
            score += _suitcase_denial_value(
                game, player_index, card, pr_markers, scout_markers
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
            prof = _beryllium_build_profile(card)
            est = _carbon_quick_card_desirability(
                game, player_index, card, wanted_tags, prof=prof
            )
            bucket = prof.play_order_bucket
            if card.definition.title == "Nos Book" and not player.played_today:
                bucket = 6
                est -= 6.0
            key = (bucket, -est)
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
        if not options:
            raise ValueError("No discard option was supplied")
        player = game.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played == "Tell a Story" and len(options) >= 2:
            # Only set up Trunk top for Assorted Cutlery / Bougie Coffee Machine when safe from opponent theft
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
                        fp_val = _carbon_free_play_value(
                            game, player_index, cand, wanted_tags
                        )
                        if not safe_to_setup:
                            fp_val *= 0.45
                        cand_cost = game.energy_cost(player_index, cand)
                        cand_des = _carbon_hand_keep_value(
                            game, player_index, cand, wanted_tags
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
                        fp_val = _carbon_free_play_value(
                            game, player_index, cand, wanted_tags
                        )
                        cand_cost = game.energy_cost(player_index, cand)
                        cand_des = _carbon_hand_keep_value(
                            game, player_index, cand, wanted_tags
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

        return min(
            range(len(options)),
            key=lambda i: _carbon_hand_keep_value(
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
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played == "Bohnanza":
            return _select_bohnanza_discards(game, player_index, options)
        if last_played == "Istanbul":
            return _select_istanbul_discards_fast(game, player_index, options)
        if last_played == "Navy SEALing":
            wanted_tags = _wanted_tags_for_player(player)
            ranked = sorted(
                range(len(options)),
                key=lambda i: _carbon_hand_keep_value(
                    game, player_index, options[i], wanted_tags
                ),
            )
            is_doubled = any(
                _effective_card_title(c) == "Bring a Friend"
                for c in player.played_today[:-1]
            ) or (
                len(player.played_today) >= 2
                and _effective_card_title(player.played_today[-2]) == "Work Call"
            )
            thresh = 4.2 if is_doubled else 2.2
            return tuple(
                i
                for i in ranked
                if _carbon_hand_keep_value(
                    game, player_index, options[i], wanted_tags
                )
                < thresh
            )
        if last_played in {"Scrabble", "Last Year's Shirt"}:
            if not game.suitcase:
                return ()
            wanted_tags = _wanted_tags_for_player(player)
            own_markers = _carbon_own_suitcase_markers(player)
            pr_markers, scout_markers = _opponent_suitcase_markers(
                game, player_index
            )
            paddleboat_bonus = 2.0 * sum(
                1
                for c in player.played_today
                if _effective_card_title(c) == "Paddleboat"
            )
            want_empty_hand = any(
                _effective_card_title(c) in {"Wakesurf", "Slalom Start"}
                for c in player.played_today
            )

            def sc_value(sc: CardInstance) -> float:
                sc_cost = game.energy_cost(player_index, sc)
                v = _carbon_quick_card_desirability(
                    game, player_index, sc, wanted_tags, cost=sc_cost
                )
                if own_markers and any(m in sc.markers for m in own_markers):
                    v -= 3.0
                v += _suitcase_denial_value(
                    game, player_index, sc, pr_markers, scout_markers
                )
                can_play_now = sc_cost <= player.energy and sc.effective_behavior.can_play(
                    game, player, sc
                )
                if can_play_now:
                    v += 0.55 + (1.8 if want_empty_hand else 0.0)
                elif game.day >= DAY_COUNT:
                    v = -1.5
                return v + paddleboat_bonus

            def hand_card_value(hc: CardInstance) -> float:
                hc_cost = game.energy_cost(player_index, hc)
                v = _carbon_quick_card_desirability(
                    game, player_index, hc, wanted_tags, cost=hc_cost
                )
                can_play_now = hc_cost <= player.energy and hc.effective_behavior.can_play(
                    game, player, hc
                )
                if can_play_now:
                    v += 0.55 + (1.8 if want_empty_hand else 0.0)
                elif game.day >= DAY_COUNT and not any(
                    _effective_card_title(c) in {"Wrestle the Kids", "Play With the Kids"}
                    for c in (*player.played_today, *player.hand)
                ):
                    v = -1.8
                return v

            suitcase_vals = sorted(
                (sc_value(sc) for sc in game.suitcase),
                reverse=True,
            )
            hand_ranked = sorted(
                range(len(options)),
                key=lambda i: hand_card_value(options[i]),
            )
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
                h_val = hand_card_value(options[i])
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

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if not options:
            raise ValueError("No copy option was supplied")
        player = game.players[player_index]
        source_card = player.played_today[-1] if player.played_today else None
        if source_card is None:
            return 0
        pay_cost = (
            type(source_card.effective_behavior).__name__ != "LeftoversBehavior"
        )

        def copy_score(i: int) -> float:
            opt = options[i]
            v = _contextual_copy_value(
                game, player_index, opt, source_card, pay_cost=pay_cost
            )
            if _effective_card_title(opt) == "Group Dinner":
                v -= 1.35
            return v

        return max(range(len(options)), key=copy_score)

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
        if last_played == "Christmas Name Draw":
            p_energy = player.energy
            return max(
                range(len(options)),
                key=lambda i: sum(
                    2.4
                    if (
                        c.effective_behavior.can_play(game, player, c)
                        and game.energy_cost(player_index, c) <= p_energy
                    )
                    else 0.5
                    for c in player.hand
                    if c.tags & options[i].tags
                )
                + sum(
                    0.45 for sc in game.suitcase if sc.tags & options[i].tags
                ),
            )
        if last_played == "Nos Book":
            zero_tag = (
                player.character.zero_fun_tag
                if player.character is not None
                else None
            )
            has_bohnanza = any(
                _effective_card_title(c) == "Bohnanza" for c in player.hand
            )
            return max(
                range(len(options)),
                key=lambda i: (
                    0 if (zero_tag and zero_tag in options[i].tags) else 1,
                    0
                    if (
                        has_bohnanza
                        and game.card_fun(player_index, options[i]) <= 1
                        and not _beryllium_build_profile(options[i]).goes_to_tomorrow
                    )
                    else 1,
                    game.card_fun(player_index, options[i]),
                ),
            )
        if last_played == "Address the Food":
            wanted_tags = _wanted_tags_for_player(player)
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
                        _carbon_free_play_value(
                            game, player_index, opt, wanted_tags
                        )
                        + denial
                        + own_pen
                    )
                return (
                    _carbon_quick_card_desirability(
                        game, player_index, opt, wanted_tags
                    )
                    - 2.0
                    + denial
                    + own_pen
                )

            return max(range(len(options)), key=food_target_val)

        wanted_tags = _wanted_tags_for_player(player)
        return max(
            range(len(options)),
            key=lambda i: _carbon_quick_card_desirability(
                game, player_index, options[i], wanted_tags
            ),
        )

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        # Mark the Suitcase card least attractive to awake opponents using public card properties
        opp_wanted: dict[str, float] = {}
        for idx, opp in enumerate(game.players):
            if idx == player_index or opp.asleep:
                continue
            ow = _wanted_tags_for_player_public(opp)
            for t, w in ow.items():
                if w > 0:
                    opp_wanted[t] = opp_wanted.get(t, 0.0) + w * 0.45

        def opponent_pick_attractiveness(sc: CardInstance) -> tuple[float, float]:
            prof = _beryllium_build_profile(sc)
            eff_cost = sc.effective_cost
            sc_tags = sc.tags if len(sc.tags) <= 1 else sorted(sc.tags)
            neutral_des = (
                float(sc.effective_base_fun)
                + prof.immediate_fun_gain * 1.15
                - eff_cost * 0.85
                + prof.immediate_energy_gain * 1.30
                + min(prof.immediate_cards_gained, 3) * (1.0 if eff_cost <= 2 else 0.25)
                + sum(opp_wanted.get(t, 0.0) * 0.75 for t in sc_tags)
            )
            if eff_cost >= 8:
                neutral_des -= (eff_cost - 6) * 1.25
            sweep_risk = (1.0 if "Food" in sc.tags else 0.0) + (
                0.6 if "Relax" in sc.tags else 0.0
            )
            cheap_risk = 0.8 if eff_cost <= 1 else 0.0
            return (neutral_des + sweep_risk + cheap_risk, -eff_cost)

        return min(
            range(len(options)),
            key=lambda i: opponent_pick_attractiveness(options[i]),
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
        best_score = -999.0
        for p_idx in eligible_player_indices:
            cand = game.players[p_idx]
            has_bg = any("Board Game" in c.tags for c in cand.played_today)
            if has_bg:
                score = -20.0
            elif cand.asleep:
                score = 15.0
            elif p_idx == player_index:
                other_bgs = sum(
                    1
                    for c in cand.hand
                    if "Board Game" in c.tags and c.title != "San Juan"
                )
                score = 12.0 if other_bgs == 0 else -5.0 - other_bgs * 3.0
            else:
                bg_affinity = 0.0
                if (
                    cand.character is not None
                    and cand.character.fun_tag == "Board Game"
                ):
                    bg_affinity += 5.0
                for tc in (*cand.tomorrow_cards, *cand.played_today):
                    if "Board Game" in _beryllium_build_profile(tc).target_tags:
                        bg_affinity += 4.0
                score = (
                    8.0
                    - cand.energy * 0.45
                    - len(cand.hand) * 0.65
                    + len(cand.played_today) * 0.5
                    - bg_affinity
                )
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
        last_played = (
            _effective_card_title(player.played_today[-1])
            if player.played_today
            else ""
        )
        if last_played == "Poker":
            known_top = _get_sim_known_trunk_top(game)
            if known_top and known_top[0].tags:
                for t in options:
                    if t in known_top[0].tags:
                        return t
            unseen_counts, _ = _unseen_tag_counts_and_total(
                game, player_index, known_top
            )
            return max(
                options, key=lambda t: (unseen_counts.get(t, 0), t == "Outdoors")
            )
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
            if last_card is None:
                return False
            draws = int(last_card.markers.get("_quacks_draws", 0))
            if game.stats is _get_shared_sim_stats():
                last_card.markers["_quacks_draws"] = draws + 1
            known_top = _get_sim_known_trunk_top(game)
            if known_top:
                return "Outdoors" not in known_top[0].tags
            unseen_counts, unseen_total = _unseen_tag_counts_and_total(
                game, player_index, known_top
            )
            p_outdoors = unseen_counts.get("Outdoors", 31) / max(1, unseen_total)
            # Exact risk-neutral EV threshold: p < 2 / (3 + 2 * draws) + continuation option
            ev_thresh = (2.0 / (3.0 + 2.0 * draws)) + (
                0.015 if draws <= 1 else 0.0
            )
            return p_outdoors < ev_thresh

        if "Cities and Knights" in action or "Epic Prank" in action:
            eligible = (
                [c for c in player.hand if "Item" in c.tags]
                if "Epic Prank" in action
                else list(player.hand)
            )
            if not eligible:
                return False
            if (
                any(
                    _effective_card_title(c) == "Slalom Start"
                    for c in player.played_today
                )
                and len(player.hand) == 1
            ):
                return False
            wanted_tags = _wanted_tags_for_player(player)
            worst_val = min(
                _carbon_hand_keep_value(game, player_index, c, wanted_tags)
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
                    _effective_card_title(c) == "Slalom Start"
                    for c in player.played_today
                )
                and len(player.hand) == 2
            ):
                base_thresh += 2.6
            return worst_val < base_thresh
        return True

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> bool:
        if not options or len(game.trunk) + len(game.discard) < 2:
            return False
        return game.players[player_index].energy >= 2

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        if not cards:
            return ()
        player = game.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)

        has_cutlery = any(
            c.title == "Assorted Cutlery"
            and game.energy_cost(player_index, c) <= player.energy
            for c in player.hand
        )
        has_bcm = any(
            c.title == "Bougie Coffee Machine"
            and game.energy_cost(player_index, c) <= player.energy
            for c in player.hand
        )
        has_quacks = any(
            c.title == "Quacks"
            and game.energy_cost(player_index, c) <= player.energy
            for c in player.hand
        )

        indices = list(range(len(cards)))
        if has_cutlery:
            best_top = max(
                indices,
                key=lambda i: _carbon_free_play_value(
                    game, player_index, cards[i], wanted_tags
                ),
            )
            rest = [i for i in indices if i != best_top]
            if has_quacks:
                rest.sort(
                    key=lambda i: (
                        "Outdoors" in cards[i].tags,
                        _carbon_quick_card_desirability(
                            game, player_index, cards[i], wanted_tags
                        ),
                    )
                )
            else:
                rest.sort(
                    key=lambda i: _carbon_quick_card_desirability(
                        game, player_index, cards[i], wanted_tags
                    ),
                    reverse=True,
                )
            order = [best_top, *rest]
        elif has_bcm:
            order = sorted(
                indices,
                key=lambda i: (
                    _carbon_free_play_value(
                        game, player_index, cards[i], wanted_tags
                    )
                    + 2.5
                    if "Food" in cards[i].tags
                    else _carbon_quick_card_desirability(
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
                    _carbon_quick_card_desirability(
                        game, player_index, cards[i], wanted_tags
                    ),
                ),
            )
        else:
            order = sorted(
                indices,
                key=lambda i: _carbon_quick_card_desirability(
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
        wanted_tags = _wanted_tags_for_player(game.players[player_index])
        return tuple(
            i
            for i, c in enumerate(cards)
            if _carbon_quick_card_desirability(
                game, player_index, c, wanted_tags
            )
            < 2.0
        )

    def order_daily_conditions(
        self,
        game: Game,
        player_index: int,
        conditions: Sequence[Any],
    ) -> Sequence[int]:
        if len(conditions) <= 1:
            return tuple(range(len(conditions)))
        player = game.players[player_index]
        wanted_tags = _wanted_tags_for_player(player)
        zero_tag = (
            player.character.zero_fun_tag
            if player.character is not None
            else ""
        )

        def score_condition(cond: Any) -> float:
            val = float(getattr(cond, "starting_energy_delta", 0)) * 0.35
            f_tag = getattr(cond, "fun_tag", "")
            f_delta = int(getattr(cond, "fun_delta", 0))
            if f_tag and f_delta != 0:
                if zero_tag and f_tag == zero_tag:
                    # Food modifiers do not affect us when our Food Fun is finalized to 0;
                    # negative Food modifiers hurt opponents relative to us!
                    val -= f_delta * 1.4
                else:
                    hand_matches = sum(1 for c in player.hand if f_tag in c.tags)
                    my_aff = (
                        hand_matches * 0.9
                        + max(0.0, wanted_tags.get(f_tag, 0.0)) * 0.6
                    )
                    opp_aff = 1.0 + (
                        sum(
                            0.5
                            for idx, opp in enumerate(game.players)
                            if idx != player_index
                            and opp.character is not None
                            and opp.character.fun_tag == f_tag
                        )
                    )
                    val += f_delta * (my_aff - opp_aff * 0.65)
            if getattr(cond, "hot_day", False):
                indoors = sum(1 for c in player.hand if "Indoors" in c.tags)
                outdoors = sum(1 for c in player.hand if "Outdoors" in c.tags)
                val += (indoors - outdoors) * 0.85
            return val

        return tuple(
            sorted(
                range(len(conditions)),
                key=lambda i: (score_condition(conditions[i]), -i),
                reverse=True,
            )
        )


def _wanted_tags_for_player_public(player: PlayerState) -> dict[str, float]:
    """Compute tag weights for an opponent using ONLY public zones (character, tomorrow, played_today)."""
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
        prof = _beryllium_build_profile(card)
        if prof.signed_match_benefit != 0.0:
            for t in sorted(prof.target_tags):
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.15
                )
        chosen_tag = card.markers.get("chosen_tag")
        if isinstance(chosen_tag, str):
            weights[chosen_tag] = weights.get(chosen_tag, 0.0) + 1.5

    for card in player.played_today:
        prof = _beryllium_build_profile(card)
        if (
            prof.is_multi_scaler and not prof.is_before_scaler
        ) or prof.is_tomorrow_scaler:
            for t in sorted(prof.target_tags):
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.05
                )
    return weights


_CARBON_EVAL_AI_SINGLETON = _CarbonEvalAI()
_CARBON_SIM_AIS: list[Any] = [
    _CARBON_EVAL_AI_SINGLETON,
    _CARBON_EVAL_AI_SINGLETON,
    _CARBON_EVAL_AI_SINGLETON,
    _CARBON_EVAL_AI_SINGLETON,
]


_CARBON_SIM_GAME_CLS: type | None = None


def _get_carbon_sim_game_cls() -> type:
    global _CARBON_SIM_GAME_CLS
    if _CARBON_SIM_GAME_CLS is not None:
        return _CARBON_SIM_GAME_CLS
    base_cls = _get_beryllium_sim_game_cls()

    class _CarbonSimGame(base_cls):  # type: ignore[valid-type, misc]
        """Simulation Game subclass with copy-on-write zones and batch scoring."""

        def _unshare_opp(self, opp_idx: int) -> PlayerState:
            self._shared_opp_mask &= ~(1 << opp_idx)
            src_p = self.players[opp_idx]
            card_remap: dict[int, CardInstance] | None = (
                {} if self._has_card_ref and src_p.played_today else None
            )
            new_played: list[CardInstance] = []
            for c in src_p.played_today:
                m = c.markers
                nc = CardInstance(
                    c.instance_id,
                    c.definition,
                    c.is_tomorrow,
                    dict(m) if m else {},
                )
                if card_remap is not None:
                    card_remap[id(c)] = nc
                new_played.append(nc)
            new_p = PlayerState(
                name=src_p.name,
                energy=src_p.energy,
                fun=src_p.fun,
                character=src_p.character,
                hand=src_p.hand,
                played_today=new_played,
                tomorrow_cards=src_p.tomorrow_cards,
                asleep=src_p.asleep,
                skipped_turns=src_p.skipped_turns,
                acquired_cards=_NOOP_COUNTER,
                picked_cards=_NOOP_COUNTER,
            )
            self.players[opp_idx] = new_p
            if card_remap is not None:
                for idx in range(4):
                    own_p = self.players[idx]
                    is_shared = bool(self._shared_opp_mask & (1 << idx))
                    for oc_i, oc in enumerate(own_p.played_today):
                        om = oc.markers
                        if om:
                            tc = om.get("target_card")
                            if tc is not None:
                                mapped = card_remap.get(id(tc))
                                if mapped is not None:
                                    if is_shared:
                                        own_p = self._unshare_opp(idx)
                                        is_shared = False
                                    own_p.played_today[oc_i].markers["target_card"] = mapped
            return new_p

        def go_to_bed(self, player_index: int) -> None:
            if self._shared_opp_mask & (1 << player_index):
                self._unshare_opp(player_index)
            self.players[player_index].asleep = True

        def skip_next_turn(self, player: PlayerState) -> None:
            if self._shared_opp_mask:
                for opp_idx in range(4):
                    if (
                        self._shared_opp_mask & (1 << opp_idx)
                    ) and self.players[opp_idx] is player:
                        player = self._unshare_opp(opp_idx)
                        break
            player.skipped_turns += 1

        def gain_energy(
            self,
            player: PlayerState,
            amount: int,
            source: CardInstance | None = None,
        ) -> None:
            if amount > 0:
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
                if self._shared_opp_mask:
                    for opp_idx in range(4):
                        if (
                            self._shared_opp_mask & (1 << opp_idx)
                        ) and self.players[opp_idx] is player:
                            player = self._unshare_opp(opp_idx)
                            break
                player.energy += amount
                if source is not None:
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

        def choose_card_to_copy(
            self,
            player_index: int,
            eligible_cards: Sequence[CardInstance],
        ) -> CardInstance:
            target = super().choose_card_to_copy(player_index, eligible_cards)
            if self._shared_opp_mask:
                for opp_idx in range(4):
                    if self._shared_opp_mask & (1 << opp_idx):
                        opp_played = self.players[opp_idx].played_today
                        for pos, oc in enumerate(opp_played):
                            if oc is target:
                                new_opp = self._unshare_opp(opp_idx)
                                return new_opp.played_today[pos]
            return target

        def choose_card_target(
            self,
            player_index: int,
            eligible_cards: Sequence[CardInstance],
        ) -> CardInstance:
            target = super().choose_card_target(player_index, eligible_cards)
            if self._shared_opp_mask:
                for opp_idx in range(4):
                    if self._shared_opp_mask & (1 << opp_idx):
                        opp_played = self.players[opp_idx].played_today
                        for pos, oc in enumerate(opp_played):
                            if oc is target:
                                new_opp = self._unshare_opp(opp_idx)
                                return new_opp.played_today[pos]
            return target

        def _draw_from_trunk(self) -> CardInstance:
            if not self.trunk:
                if not self.discard:
                    from thenos.game import DeckExhaustedError

                    raise DeckExhaustedError(
                        "Both the Trunk and discard pile are empty"
                    )
                self.trunk = (
                    self.discard.copy()
                    if self._discard_shared
                    else self.discard
                )
                self.discard = []
                self._trunk_shared = False
                self._discard_shared = False
                self.rng = _clone_random(self.rng)
                self.rng.shuffle(self.trunk)
            elif self._trunk_shared:
                self.trunk = self.trunk.copy()
                self._trunk_shared = False
            card = self.trunk.pop()
            return CardInstance(
                card.instance_id,
                card.definition,
                card.is_tomorrow,
                dict(card.markers) if card.markers else {},
            )

        def return_cards_to_trunk_top(
            self,
            player_index: int,
            cards: Sequence[CardInstance],
        ) -> list[CardInstance]:
            if cards and self._trunk_shared:
                self.trunk = self.trunk.copy()
                self._trunk_shared = False
            return super().return_cards_to_trunk_top(player_index, cards)

        def discard_card(self, card: CardInstance) -> None:
            if self._discard_shared:
                self.discard = self.discard.copy()
                self._discard_shared = False
            if card.markers or card.is_tomorrow:
                card.is_tomorrow = False
                card.markers.clear()
            self.discard.append(card)

        def arrange_daily_conditions(self, player_index: int) -> None:
            if self._cond_shared:
                self._condition_deck = self._condition_deck.copy()
                self._condition_knowledge = self._condition_knowledge.copy()
                self._cond_shared = False
            super().arrange_daily_conditions(player_index)

        @staticmethod
        def _tomorrow_grants_all_tags(player: PlayerState) -> bool:
            for source in player.tomorrow_cards:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.tomorrow_all_tags:
                    return True
            return False

        @staticmethod
        def _grants_all_tags(player: PlayerState, card: CardInstance) -> bool:
            for source in player.tomorrow_cards:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.tomorrow_all_tags:
                    return True
            for source in player.played_today:
                if source is card:
                    break
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.after_all_tags:
                    return True
            return False

        @staticmethod
        def _sync_played_today_tags(player: PlayerState) -> None:
            seen_after_all = False
            for source in player.tomorrow_cards:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.tomorrow_all_tags:
                    seen_after_all = True
                    break
            for played_card in player.played_today:
                m = played_card.markers
                if seen_after_all:
                    m["_all_tags"] = True
                elif m and "_all_tags" in m:
                    m.pop("_all_tags", None)
                s_beh = (
                    played_card.definition.behavior
                    if (not m or "_copied_definition" not in m)
                    else played_card.effective_behavior
                )
                if s_beh.after_all_tags:
                    seen_after_all = True

        def give_card(self, player_index: int, card: CardInstance) -> None:
            if card.markers or card.is_tomorrow:
                self._has_marked_hand = True
            player = self.players[player_index]
            player.hand.append(card)
            for source in player.played_today:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if type(s_beh).on_card_acquire is not CardBehavior.on_card_acquire:
                    s_beh.on_card_acquire(self, player, source, card)

        def record_card_pick_or_draw(
            self,
            player_index: int,
            card: CardInstance,
        ) -> None:
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

        def _finish_play_card(
            self,
            player: PlayerState,
            hand_index: int,
            raw_card: CardInstance,
            c_beh: CardBehavior,
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
            if (
                raw_card.definition.title
                if (not rm or "_copied_definition" not in rm)
                else _effective_card_title(raw_card)
            ) == "Last Year's Shirt" and player.hand:
                player.hand = [
                    CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(c.markers) if c.markers else {},
                    )
                    for c in player.hand
                ]
            player.played_today.append(card)
            self._sync_played_today_tags(player)
            if type(c_beh).on_play is not CardBehavior.on_play:
                c_beh.on_play(self, player, card)
            if player.tomorrow_cards:
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
            if cm and ("target_card" in cm or "suitcase_target" in cm):
                self._has_card_ref = True
            return card

        def play_card(self, player_index: int, hand_index: int) -> CardInstance:
            player = self.players[player_index]
            if player.asleep:
                raise ValueError(f"{player.name} has already gone to bed")
            if hand_index < 0 or hand_index >= len(player.hand):
                raise ValueError(f"Invalid hand index: {hand_index}")
            raw_card = player.hand[hand_index]
            rm = raw_card.markers
            c_beh = (
                raw_card.definition.behavior
                if (not rm or "_copied_definition" not in rm)
                else raw_card.effective_behavior
            )
            if (
                type(c_beh).can_play is not CardBehavior.can_play
                and not c_beh.can_play(self, player, raw_card)
            ):
                raise ValueError(f"{raw_card.title} cannot legally be played")
            cost = self.energy_cost(player_index, raw_card)
            if cost > player.energy:
                raise ValueError(
                    f"Not enough Energy to play {raw_card.title}"
                )
            return self._finish_play_card(
                player, hand_index, raw_card, c_beh, cost
            )

        def play_card_known_cost(
            self, player_index: int, hand_index: int, cost: int
        ) -> CardInstance:
            player = self.players[player_index]
            raw_card = player.hand[hand_index]
            rm = raw_card.markers
            c_beh = (
                raw_card.definition.behavior
                if (not rm or "_copied_definition" not in rm)
                else raw_card.effective_behavior
            )
            return self._finish_play_card(
                player, hand_index, raw_card, c_beh, cost
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
            return played

        def copy_card_effect(
            self,
            player_index: int,
            source: CardInstance,
            destination: CardInstance,
            *,
            pay_source_cost: bool = True,
        ) -> None:
            super().copy_card_effect(
                player_index,
                source,
                destination,
                pay_source_cost=pay_source_cost,
            )
            dm = destination.markers
            if dm and ("target_card" in dm or "suitcase_target" in dm):
                self._has_card_ref = True

        def energy_cost(self, player_index: int, card: CardInstance) -> int:
            player = self.players[player_index]
            m = card.markers
            has_sources = bool(player.played_today or player.tomorrow_cards)
            temporary_all_tags = (
                has_sources
                and (not m or "_all_tags" not in m)
                and self._grants_all_tags(player, card)
            )
            if temporary_all_tags:
                m["_all_tags"] = True
            try:
                c_defn = (
                    card.definition
                    if (not m or "_copied_definition" not in m)
                    else card.effective_definition
                )
                cost = c_defn.cost
                c_tags = (
                    card.tags
                    if (temporary_all_tags or (m and "_all_tags" in m))
                    else c_defn.tags
                )
                if self.daily_condition is not None:
                    cost = self.daily_condition.modify_energy_cost(c_tags, cost)
                if player.character is not None:
                    cost = player.character.modify_energy_cost(c_tags, cost)
                c_beh = c_defn.behavior
                if (
                    type(c_beh).modify_own_energy_cost
                    is not CardBehavior.modify_own_energy_cost
                ):
                    cost = c_beh.modify_own_energy_cost(
                        self, player, card, cost
                    )
                if has_sources:
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
            finally:
                if temporary_all_tags:
                    m.pop("_all_tags", None)

        def hand_playable_with_costs(
            self, player_index: int
        ) -> list[tuple[int, int]]:
            player = self.players[player_index]
            hand = player.hand
            if player.asleep or not hand:
                return []
            p_energy = player.energy
            cond = self.daily_condition
            char = player.character
            grants_all = False
            tom_mods: list[tuple[CardInstance, CardBehavior]] = []
            for source in player.tomorrow_cards:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.tomorrow_all_tags:
                    grants_all = True
                if (
                    type(s_beh).modify_tomorrow_energy_cost
                    is not CardBehavior.modify_tomorrow_energy_cost
                ):
                    tom_mods.append((source, s_beh))
            today_mods: list[tuple[CardInstance, CardBehavior]] = []
            for source in player.played_today:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.after_all_tags:
                    grants_all = True
                if (
                    type(s_beh).modify_energy_cost
                    is not CardBehavior.modify_energy_cost
                ):
                    today_mods.append((source, s_beh))

            out: list[tuple[int, int]] = []
            plain_fast = (
                not grants_all
                and not tom_mods
                and not today_mods
                and cond is None
                and char is None
            )
            for index, card in enumerate(hand):
                m = card.markers
                c_defn = (
                    card.definition
                    if (not m or "_copied_definition" not in m)
                    else card.effective_definition
                )
                c_beh = c_defn.behavior
                if (
                    type(c_beh).can_play is not CardBehavior.can_play
                    and not c_beh.can_play(self, player, card)
                ):
                    continue
                if (
                    plain_fast
                    and type(c_beh).modify_own_energy_cost
                    is CardBehavior.modify_own_energy_cost
                ):
                    cost = c_defn.cost
                    if cost < 0:
                        cost = 0
                else:
                    temp_all = grants_all and (not m or "_all_tags" not in m)
                    if temp_all:
                        m["_all_tags"] = True
                    try:
                        cost = c_defn.cost
                        c_tags = (
                            card.tags
                            if (temp_all or (m and "_all_tags" in m))
                            else c_defn.tags
                        )
                        if cond is not None:
                            cost = cond.modify_energy_cost(c_tags, cost)
                        if char is not None:
                            cost = char.modify_energy_cost(c_tags, cost)
                        if (
                            type(c_beh).modify_own_energy_cost
                            is not CardBehavior.modify_own_energy_cost
                        ):
                            cost = c_beh.modify_own_energy_cost(
                                self, player, card, cost
                            )
                        for source, s_beh in tom_mods:
                            cost = s_beh.modify_tomorrow_energy_cost(
                                self, player, source, card, cost
                            )
                        for source, s_beh in today_mods:
                            cost = s_beh.modify_energy_cost(
                                self, player, source, card, cost
                            )
                        if cost < 0:
                            cost = 0
                    finally:
                        if temp_all:
                            m.pop("_all_tags", None)
                if cost <= p_energy:
                    out.append((index, cost))
            return out

        def score_played_today_fast(
            self, player_index: int
        ) -> tuple[int, list[CardInstance]]:
            player = self.players[player_index]
            played = player.played_today
            if not played:
                return 0, []
            cond = self.daily_condition
            char = player.character
            seen_after_all = False
            tom_mods: list[tuple[CardInstance, CardBehavior]] = []
            for source in player.tomorrow_cards:
                sm = source.markers
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.tomorrow_all_tags:
                    seen_after_all = True
                if (
                    type(s_beh).modify_tomorrow_fun
                    is not CardBehavior.modify_tomorrow_fun
                ):
                    tom_mods.append((source, s_beh))
            today_mods: list[tuple[CardInstance, CardBehavior]] = []
            next_tomorrow_cards: list[CardInstance] = []
            for source in played:
                sm = source.markers
                if seen_after_all:
                    sm["_all_tags"] = True
                elif sm and "_all_tags" in sm:
                    sm.pop("_all_tags", None)
                s_beh = (
                    source.definition.behavior
                    if (not sm or "_copied_definition" not in sm)
                    else source.effective_behavior
                )
                if s_beh.after_all_tags:
                    seen_after_all = True
                if type(s_beh).modify_fun is not CardBehavior.modify_fun:
                    today_mods.append((source, s_beh))
                if s_beh.has_tomorrow_action:
                    next_tomorrow_cards.append(source)
            scored_today = 0
            for target in played:
                if not target.is_tomorrow:
                    m = target.markers
                    t_defn = (
                        target.definition
                        if (not m or "_copied_definition" not in m)
                        else target.effective_definition
                    )
                    t_beh = t_defn.behavior
                    if type(t_beh).fun_value is CardBehavior.fun_value:
                        value = t_defn.base_fun
                    else:
                        value = t_beh.fun_value(self, player, target)
                    t_tags = (
                        target.tags if (m and "_all_tags" in m) else t_defn.tags
                    )
                    if cond is not None:
                        value = cond.modify_fun(t_tags, value)
                    if char is not None:
                        value = char.modify_fun(t_tags, value)
                    for source, s_beh in tom_mods:
                        value = s_beh.modify_tomorrow_fun(
                            self, player, source, target, value
                        )
                    for source, s_beh in today_mods:
                        value = s_beh.modify_fun(
                            self, player, source, target, value
                        )
                    if char is not None:
                        value = char.finalize_fun(t_tags, value)
                    scored_today += value
            return scored_today, next_tomorrow_cards

        def end_day(self) -> None:
            if self._shared_opp_mask:
                for opp_idx in range(4):
                    if self._shared_opp_mask & (1 << opp_idx):
                        self._unshare_opp(opp_idx)
            if self._cow_acting_idx >= 0:
                acting_idx = self._cow_acting_idx
                has_marked = self._has_marked_hand
                for p_idx, p in enumerate(self.players):
                    if p_idx != acting_idx:
                        p.hand = (
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
                            if has_marked
                            else p.hand.copy()
                        )
                    if p.tomorrow_cards:
                        p.tomorrow_cards = [
                            CardInstance(
                                c.instance_id,
                                c.definition,
                                c.is_tomorrow,
                                dict(c.markers) if c.markers else {},
                            )
                            for c in p.tomorrow_cards
                        ]
                self._cow_acting_idx = -1
            players = self.players
            self._fun_at_start_of_scoring = (
                players[0].fun,
                players[1].fun,
                players[2].fun,
                players[3].fun,
            )
            for p_i, player in enumerate(players):
                if player.played_today:
                    has_on_score = False
                    for card in player.played_today:
                        m = card.markers
                        c_beh = (
                            card.definition.behavior
                            if (not m or "_copied_definition" not in m)
                            else card.effective_behavior
                        )
                        if type(c_beh).on_score is not CardBehavior.on_score:
                            has_on_score = True
                            break
                    if not has_on_score:
                        scored, _ = self.score_played_today_fast(p_i)
                        player.fun += scored
                    else:
                        for card in player.played_today:
                            player.fun += self.card_fun(p_i, card)
                            card.effective_behavior.on_score(self, player, card)
            self._fun_at_start_of_scoring = None

            for player in players:
                if player.played_today:
                    for card in tuple(player.played_today):
                        m = card.markers
                        c_beh = (
                            card.definition.behavior
                            if (not m or "_copied_definition" not in m)
                            else card.effective_behavior
                        )
                        if type(c_beh).on_end_day is not CardBehavior.on_end_day:
                            c_beh.on_end_day(self, player, card)

            for player in players:
                if player.played_today:
                    for card in tuple(player.played_today):
                        m = card.markers
                        c_beh = (
                            card.definition.behavior
                            if (not m or "_copied_definition" not in m)
                            else card.effective_behavior
                        )
                        if (
                            type(c_beh).on_after_end_day
                            is not CardBehavior.on_after_end_day
                        ):
                            c_beh.on_after_end_day(self, player, card)

            for player in players:
                previous_tomorrow = player.tomorrow_cards
                player.tomorrow_cards = []
                for card in previous_tomorrow:
                    self.discard_card(card)

                if player.played_today:
                    for card in tuple(player.played_today):
                        m = card.markers
                        c_beh = (
                            card.definition.behavior
                            if (not m or "_copied_definition" not in m)
                            else card.effective_behavior
                        )
                        if c_beh.has_tomorrow_action:
                            card.is_tomorrow = True
                            player.tomorrow_cards.append(card)
                        else:
                            self.discard_card(card)
                    player.played_today = []
            self._has_card_ref = False
            self._has_opp_suitcase_ref = False

    _CARBON_SIM_GAME_CLS = _CarbonSimGame
    return _CarbonSimGame


def _carbon_copy_game(src: Game) -> Game:
    sim_cls = _CARBON_SIM_GAME_CLS or _get_carbon_sim_game_cls()
    is_carbon_sim = type(src) is sim_cls
    sim = object.__new__(sim_cls)
    sim.rng = src.rng if is_carbon_sim else _clone_random(src.rng)
    sim.ais = _CARBON_SIM_AIS
    sim._trunk_shared = False
    sim._discard_shared = False
    sim._cond_shared = False
    sim._shared_opp_mask = 0
    sim._cow_acting_idx = -1
    sim._all_canonical_defs = getattr(src, "_all_canonical_defs", False)
    sim._has_opp_suitcase_ref = (
        src._has_opp_suitcase_ref  # type: ignore[attr-defined]
        if is_carbon_sim
        else any(
            oc.markers
            and ("suitcase_target" in oc.markers or "target_card" in oc.markers)
            for opp in src.players
            for oc in opp.played_today
        )
    )

    cards_with_refs: list[CardInstance] = []
    card_map: dict[int, CardInstance] | None = None

    # Check if any marked card in played_today / tomorrow_cards has target_card or suitcase_target
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
    sim._has_marked_hand = any(
        bool(c.markers) or c.is_tomorrow for p in src.players for c in p.hand
    )

    if need_map:
        card_map = {}

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
                card_map[id(c)] = nc  # type: ignore[index]
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
                    if is_carbon_sim
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
                    if is_carbon_sim
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

    if is_carbon_sim:
        sim.trunk = src.trunk.copy()
        sim.discard = src.discard.copy()
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


def _carbon_copy_game_for_play(src: Game, player_index: int) -> Game:
    """Fast copy-on-write clone for single-player `play_card` / `go_to_bed` search nodes."""
    sim_cls = _CARBON_SIM_GAME_CLS or _get_carbon_sim_game_cls()
    if type(src) is not sim_cls or src._has_opp_suitcase_ref:  # type: ignore[attr-defined]
        return _carbon_copy_game(src)

    sim = object.__new__(sim_cls)
    sim.rng = src.rng
    sim.ais = _CARBON_SIM_AIS
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

    src_players = src.players
    p = src_players[player_index]

    new_played = [
        CardInstance(
            c.instance_id,
            c.definition,
            c.is_tomorrow,
            dict(c.markers) if c.markers else {},
        )
        for c in p.played_today
    ]
    new_suitcase = [
        CardInstance(
            c.instance_id,
            c.definition,
            c.is_tomorrow,
            dict(c.markers) if c.markers else {},
        )
        for c in src.suitcase
    ]

    if has_ref:
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
    sim._known_trunk_state = getattr(src, "_known_trunk_state", None)
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


class CarbonAI(BerylliumBrain):
    """Opponent-aware, data-calibrated lookahead policy."""

    BEAM_WIDTH = 24
    DRAFT_BEAM_WIDTH = 10
    BRANCH_LIMIT = 8
    MAX_TURN_PLAYS = 14
    RESERVE_TOMORROW_WEIGHT = 0.76
    FIRST_TO_BED_BONUS = 0.65
    DRAFT_SAMPLES = 3
    PLAY_SAMPLES = 2

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)
        _ensure_carbon_caches()
        self._reserve_eval_cache = _CARBON_RESERVE_EVAL_CACHE
        self._live_game_ref: weakref.ReferenceType[Game] | None = None
        self._extra_draw_picks_state: tuple[int, int, int, int] | None = None

    def _ensure_live_game(self, game: Game) -> None:
        if game.stats is _get_shared_sim_stats():
            return
        if self._live_game_ref is None or self._live_game_ref() is not game:
            self._live_game_ref = weakref.ref(game)
            self._known_trunk_state = None
            self._quacks_card_key = None
            self._draw_phase_tracker = None
            self._extra_draw_picks_state = None
            self._pending_play_choice = None

    def _get_live_known_trunk_top(
        self, game: Game, player_index: int
    ) -> tuple[CardInstance, ...]:
        self._ensure_live_game(game)
        return super()._get_live_known_trunk_top(game, player_index)

    def _remaining_draw_picks_today(
        self, game: Game, player_index: int
    ) -> int:
        player = game.players[player_index]
        if player.played_today or not game._is_setup:
            return 0
        if game.stats is _get_shared_sim_stats():
            return 0
        self._ensure_live_game(game)
        gid = id(game)
        total_picks = sum(player.picked_cards.values())
        if (
            self._draw_phase_tracker is None
            or self._draw_phase_tracker[0] != gid
            or self._draw_phase_tracker[1] != game.day
            or self._draw_phase_tracker[2] != player_index
            or total_picks < self._draw_phase_tracker[3]
        ):
            day_start_picks = total_picks
            self._draw_phase_tracker = (
                gid,
                game.day,
                player_index,
                day_start_picks,
            )
        else:
            day_start_picks = self._draw_phase_tracker[3]
        extra_picks = (
            self._extra_draw_picks_state[3]
            if (
                self._extra_draw_picks_state is not None
                and self._extra_draw_picks_state[:3]
                == (gid, game.day, player_index)
            )
            else 0
        )
        picks_so_far = total_picks - day_start_picks
        return max(0, 2 + extra_picks - picks_so_far)

    def _prepare_observable_game(self, game: Game, player_index: int) -> Game:
        _ensure_carbon_caches()
        self._ensure_live_game(game)
        sim = super()._prepare_observable_game(game, player_index)
        sim.__class__ = _CARBON_SIM_GAME_CLS or _get_carbon_sim_game_cls()
        sim.ais = _CARBON_SIM_AIS
        sim._trunk_shared = False  # type: ignore[attr-defined]
        sim._discard_shared = False  # type: ignore[attr-defined]
        sim._cond_shared = False  # type: ignore[attr-defined]
        sim._shared_opp_mask = 0  # type: ignore[attr-defined]
        sim._cow_acting_idx = -1  # type: ignore[attr-defined]
        prof_by_id = _CARBON_PROFILE_BY_DEFN_ID
        sim._all_canonical_defs = (  # type: ignore[attr-defined]
            all(
                id(c.definition) in prof_by_id
                for p in sim.players
                for zone in (p.hand, p.played_today, p.tomorrow_cards)
                for c in zone
            )
            and all(id(c.definition) in prof_by_id for c in sim.suitcase)
            and all(id(c.definition) in prof_by_id for c in sim.trunk)
            and all(id(c.definition) in prof_by_id for c in sim.discard)
        )
        own_played_ids = {id(c) for c in sim.players[player_index].played_today}
        sim._has_opp_suitcase_ref = any(  # type: ignore[attr-defined]
            oc.markers
            and (
                "suitcase_target" in oc.markers
                or (
                    "target_card" in oc.markers
                    and id(oc.markers.get("target_card")) in own_played_ids
                )
            )
            for idx, opp in enumerate(sim.players)
            if idx != player_index
            for oc in opp.played_today
        )
        sim._has_card_ref = any(  # type: ignore[attr-defined]
            c.markers
            and ("target_card" in c.markers or "suitcase_target" in c.markers)
            for p in sim.players
            for c in p.played_today
        )
        sim._has_marked_hand = any(  # type: ignore[attr-defined]
            bool(c.markers) or c.is_tomorrow
            for p in sim.players
            for c in p.hand
        )
        return sim

    def _simulate_play_step(
        self,
        state: Game,
        player_index: int,
        hand_idx: int,
        known_cost: int | None = None,
    ) -> Game:
        # Keep _CARBON_SIM_AIS active DURING play_card so Bohnanza, Leftovers,
        # Wedding Anniversary, Tell a Story, Scrabble, etc. use Carbon's logic!
        child = _carbon_copy_game_for_play(state, player_index)
        card_title = child.players[player_index].hand[hand_idx].definition.title
        known_top = _get_sim_known_trunk_top(child)
        use_fast = known_cost is not None and type(child) is _CARBON_SIM_GAME_CLS

        if not known_top:
            if card_title == "Assorted Cutlery" and child.trunk:
                if getattr(child, "_trunk_shared", False):
                    child.trunk = child.trunk.copy()
                    child._trunk_shared = False  # type: ignore[attr-defined]
                best_t_idx = len(child.trunk) - 1
                best_diff = float("inf")
                wanted_tags = _wanted_tags_for_player(
                    child.players[player_index]
                )
                seen_defn_ids: set[int] = set()
                for t_i, tc in enumerate(child.trunk):
                    if not tc.markers:
                        did = id(tc.definition)
                        if did in seen_defn_ids:
                            continue
                        seen_defn_ids.add(did)
                        t_prof = _beryllium_build_profile(tc)
                    else:
                        t_prof = _beryllium_build_profile(tc)
                    if (
                        t_prof.ends_day_immediately
                        or t_prof.penalizes_before
                        or t_prof.penalizes_after
                    ):
                        continue
                    diff = abs(
                        _carbon_free_play_value(
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
                return child

        if use_fast:
            child.play_card_known_cost(player_index, hand_idx, known_cost)  # type: ignore[attr-defined,arg-type]
        else:
            child.play_card(player_index, hand_idx)

        # If Tell a Story placed a card on Trunk while >= 2 opponents are awake
        # without Epic Duels, opponents may draw/disrupt the Trunk before our next turn.
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

    # ------------------------------------------------------------------
    # Suitcase Drafting: Opponent-Aware + Realistic Depletion + Fast Pruning
    # ------------------------------------------------------------------

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if self._evaluation_depth >= 1 or len(options) <= 1:
            return _CARBON_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        _ensure_carbon_caches()
        self._ensure_live_game(game)
        live_player = game.players[player_index]
        own_markers = _carbon_own_suitcase_markers(live_player)
        pr_markers, scout_markers = _opponent_suitcase_markers(
            game, player_index
        )
        remaining_draw_picks = self._remaining_draw_picks_today(
            game, player_index
        )
        wanted_tags = _wanted_tags_for_player(live_player)

        if live_player.played_today and (
            len(live_player.played_today) >= 8
            or len(live_player.hand) >= 10
            or len(options) < 2
        ):
            return _CARBON_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        # Quick pre-score to prune hopeless options on multi-sample draft searches
        total_future_my_picks = remaining_draw_picks
        quick_scores: list[float] = []
        for option in options:
            qs = (
                _carbon_quick_card_desirability(
                    game,
                    player_index,
                    option,
                    wanted_tags,
                    remaining_draw_picks=total_future_my_picks,
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
            elif remaining_draw_picks == 1 or has_stochastic_option:
                num_samples = 2
            else:
                num_samples = 1
            draft_beam = self.DRAFT_BEAM_WIDTH
            draft_max_plays = 9
        else:
            num_samples = 2 if has_stochastic_option else 1
            draft_beam = 5 if len(live_player.played_today) < 6 else 4
            draft_max_plays = 6 if len(live_player.played_today) < 6 else 4

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

            # Prune clearly hopeless options to 1 sample + narrower beam
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
                sim = _carbon_copy_game(base_sim)
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
                    pick_i = _CARBON_EVAL_AI_SINGLETON.choose_suitcase_card(
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
                    seen_ok: dict[int, bool] = {}
                    trunk_list = sim.trunk
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
                                    <= _carbon_quick_card_desirability(
                                        sim, player_index, tc, {}
                                    )
                                    <= 1.6
                                )
                                seen_ok[did] = ok
                        else:
                            ok = (
                                0.2
                                <= _carbon_quick_card_desirability(
                                    sim, player_index, tc, {}
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
            game, player_index, card, {}
        )
        player = game.players[player_index]
        days_left = DAY_COUNT - game.day
        title = card.title
        cost = game.energy_cost(player_index, card)

        if wanted_tags:
            prof = _beryllium_build_profile(card)
            c_tags = card.tags
            tag_align = 0.0
            for t in (c_tags if len(c_tags) <= 1 else sorted(c_tags)):
                w = wanted_tags.get(t, 0.0)
                if w != 0.0:
                    tag_align += 0.38 * w
                    if w > 0.0 and cost <= 1:
                        tag_align += 0.28 * min(w, 3.0)
            if prof.must_play_first and tag_align > 0.0:
                tag_align *= 0.25
            bonus += tag_align

        # Undo ArgonBrain's double-penalty on 8+ cost payoffs ONLY when real discount/ramp exists
        if card.effective_cost >= 8 and cost > player.energy:
            if not (
                any(c.title == "Assorted Cutlery" for c in player.hand)
                and any(c.title == "Tell a Story" for c in player.hand)
            ):
                from thenos.ais.argonbrain import _extra_energy_support_for_card

                old_sup = _extra_energy_support_for_card(player, card)
                old_unfunded = max(0, card.effective_cost - (6 + old_sup))
                if old_unfunded > 0:
                    bonus += old_unfunded * 1.45  # cancel Argon's raw penalty

                today_sup = _carbon_extra_energy_support(
                    game, player_index, card
                )
                today_unfunded = max(0, cost - (player.energy + today_sup))
                if days_left >= 1:
                    future_sup = _carbon_extra_energy_support(
                        game,
                        player_index,
                        card,
                        extra_fodder=2,
                        include_future_tomorrow=True,
                    )
                    char_cost = (
                        max(
                            0,
                            player.character.modify_energy_cost(
                                card.tags, card.effective_cost
                            ),
                        )
                        if player.character is not None
                        else card.effective_cost
                    )
                    future_unfunded = max(0, char_cost - (7 + future_sup))
                    base_unfunded = min(today_unfunded, future_unfunded)
                    new_sup = max(today_sup, future_sup)
                else:
                    base_unfunded = today_unfunded
                    new_sup = today_sup

                if title == "Adventure Race":
                    slack = (
                        1
                        if (
                            days_left >= 2
                            and (new_sup >= 2 or days_left >= 3)
                        )
                        else 0
                    )
                    unfunded = max(0, base_unfunded - slack)
                    if new_sup >= 3:
                        bonus += 1.75 - unfunded * 0.95
                    else:
                        bonus += 0.65 - unfunded * 1.15
                elif (
                    title in {"Wakeboard", "Civilization the Boardgame"}
                    and new_sup >= 2
                ):
                    bonus += 0.85 - base_unfunded * 1.10
                else:
                    bonus -= base_unfunded * 1.15

        # Calibrated card adjustments grounded in 20,000-game Beryllium sample (n~25,000/card)
        if title == "Medical Advice":
            if days_left >= 1 and len(player.played_today) <= 1:
                bonus += 3.35
                if any(
                    _effective_card_title(c) == "Paddleboat"
                    for c in (*player.hand, *player.played_today)
                ):
                    bonus += 2.00
            else:
                bonus -= 1.0
        elif title == "Scout Other Cottages" and days_left >= 1:
            bonus += 1.25
        elif title == "Würstchen":
            proj_cards = len(player.hand) + (2 if days_left >= 1 else 0)
            bonus += 1.45 + 0.35 * max(0, proj_cards - 4)
        elif title == "Date's First Nos":
            if days_left >= 1:
                bonus += 1.45 if len(player.hand) >= 2 else 0.75
            else:
                bonus -= 2.0
        elif title == "Potato Pancakes":
            relax_owned = sum(
                1
                for c in (*player.hand, *player.tomorrow_cards, *player.played_today)
                if c is not card and "Relax" in c.tags
            )
            bonus += 0.95 + 0.65 * min(relax_owned, 3)
        elif title == "Bohnanza":
            fodder = sum(
                1
                for c in (*player.hand, *player.played_today)
                if c is not card
                and c.effective_base_fun <= 0
                and not _beryllium_build_profile(c).goes_to_tomorrow
            )
            bonus += 0.95 + 0.45 * min(fodder, 3)
        elif title == "Bougie Coffee Machine":
            bonus += 0.95
        elif title == "Address the Food":
            bonus += 0.55
        elif title == "Istanbul" and days_left >= 1:
            bonus += 0.85
        elif title == "Morning Run" and days_left >= 1 and not player.played_today:
            bonus += 0.80
        elif title == "Wrestle the Kids" and (len(player.hand) >= 3 or days_left >= 2):
            bonus += 0.70
        elif title == "Stay Up Late":
            bonus += 1.10 if days_left == 0 else 0.50
        elif title == "Zero Gravity Chair":
            bonus += 0.55
        elif title == "Bring a Friend":
            zero_or_free = sum(
                1
                for c in player.hand
                if c is not card
                and (
                    c.effective_cost == 0
                    or c.title
                    in {"Assorted Cutlery", "Address the Food", "Decaf"}
                )
            )
            if zero_or_free >= 1 or days_left >= 2:
                bonus += 0.75 + 0.50 * min(zero_or_free, 3)
        elif title == "Work Call":
            zero_tag = (
                player.character.zero_fun_tag
                if player.character is not None
                else ""
            )
            has_big_hit = any(
                c is not card
                and c.effective_base_fun >= 5
                and not (zero_tag and zero_tag in c.tags)
                for c in player.hand
            )
            if has_big_hit:
                bonus += 1.35
            elif days_left >= 2:
                bonus += 0.45
        elif title == "Shady Spot":
            if any(
                c is not card and "Outdoors" in c.tags and c.effective_cost >= 5
                for c in player.hand
            ):
                bonus += 1.80
        elif title == "Treat Cereal":
            if any(
                c is not card and "Exercise" in c.tags and c.effective_cost >= 5
                for c in player.hand
            ):
                bonus += 1.50
        elif title in {"Wakesurf", "Slalom Start"}:
            has_dump = any(
                c.title in {"Istanbul", "Navy SEALing", "Paddleboard"}
                for c in player.hand
            )
            if not has_dump:
                bonus -= 0.85
        elif title == "Group Dinner":
            bonus -= 1.55
        elif title == "Tell a Story":
            has_safe_combo = any(
                c.title in {"Assorted Cutlery", "Bougie Coffee Machine"}
                for c in player.hand
            ) and any(c.title == "Epic Duels" for c in player.hand)
            if not has_safe_combo:
                bonus -= 1.35
        elif title == "Cheesy Phone Game":
            bonus -= 0.75
        elif title == "High-End White":
            bonus -= 0.65
        elif title == "Euchre":
            bonus -= 0.50
        elif title in {"Cheap Red", "Kneeboard", "Teach Kid to Ski", "Morning Walk", "Solo"}:
            bonus -= 0.50
        elif title == "Nos Book":
            bonus -= 0.50
        elif title == "Ski on Shoulders":
            opps_ahead = sum(
                1
                for idx, opp in enumerate(game.players)
                if idx != player_index and opp.fun > player.fun
            )
            if opps_ahead < 2:
                bonus -= 1.20
        elif title == "Assorted Cutlery":
            has_real_setup = bool(
                _get_known_trunk_top_any(game, player_index)
            ) or any(
                c is not card
                and c.title
                in {
                    "Read the Radar",
                    "Fishing Boat",
                    "Ski Boat",
                    "Grocery Store Run",
                    "Boat Ride",
                    "Fancy Craft",
                }
                for c in player.hand
            )
            if not has_real_setup:
                bonus -= 0.90
        elif title == "Hold the Baby":
            bonus -= 0.55

        return bonus

    # ------------------------------------------------------------------
    # Playing Phase: Efficient Contender Multi-Sampling & Opponent Timing
    # ------------------------------------------------------------------

    def _has_active_stochastic_play(
        self, game: Game, player_index: int, playable_indices: Sequence[int]
    ) -> bool:
        """Only trigger multi-sample play verification when stochasticity can actually occur today."""
        player = game.players[player_index]
        if game.daily_conditions and game.day < DAY_COUNT:
            known_dc = game.known_daily_conditions(player_index)
            if not known_dc or known_dc[0] is None:
                return True
        if any(
            _effective_card_title(c) == "Photo Shoot"
            for c in player.tomorrow_cards
        ):
            return True
        if any(
            _effective_card_title(c) == "Euchre Awards Ceremony"
            for c in player.played_today
        ):
            return True
        for idx in playable_indices:
            if idx < len(player.hand) and player.hand[idx].title in _STOCHASTIC_PLAY_TITLES:
                return True
        # Check if a stochastic card in hand could become playable after ramp
        max_reach = player.energy + 4
        for hc in player.hand:
            if hc.title in _STOCHASTIC_PLAY_TITLES and hc.effective_cost <= max_reach:
                return True
        return False

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
            return _CARBON_EVAL_AI_SINGLETON.choose_card_to_play(
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
        return _CARBON_EVAL_AI_SINGLETON.choose_card_to_play(
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
            return _CARBON_EVAL_AI_SINGLETON.choose_extra_card_to_play(
                game, player_index, playable_hand_indices
            )
        best_action = self._select_root_play_action(
            game,
            player_index,
            playable_hand_indices,
            allow_bed_now=True,
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
        )
        if not use_multi or len(table_0) <= 1:
            return first_action_0

        # Select only genuine contenders from sample 0 (avoiding Beryllium's 31x redundant search!)
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
        # Add 1 extra sample when the top action itself is stochastic
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
        n_total = 1 + len(extra_samples)

        for cand in contenders:
            total = table_0[cand][0]
            valid = True
            for sim in extra_samples:
                if cand is None:
                    bed_sim = _carbon_copy_game_for_play(sim, player_index)
                    bed_sim.go_to_bed(player_index)
                    total += self._evaluate_terminal_day(bed_sim, player_index)
                else:
                    try:
                        child = self._simulate_play_step(
                            sim, player_index, cand
                        )
                        if child.players[player_index].asleep:
                            val = self._evaluate_terminal_day(
                                child, player_index
                            )
                        else:
                            _, val = self._search_best_day_plan(
                                child,
                                player_index,
                                beam_width=9,
                                max_plays=7,
                                allow_bed_now=True,
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
        return (
            p.energy,
            p.fun,
            tuple(
                [
                    c.definition.title
                    if (
                        not c.markers
                        or (
                            "_copied_definition" not in c.markers
                            and not c.markers.get("energy_cubes")
                            and c.markers.get("target_card") is None
                            and c.markers.get("target_player_index") is None
                        )
                    )
                    else (
                        c.definition.title,
                        _effective_card_title(c),
                        c.markers.get("energy_cubes", 0),
                        getattr(
                            c.markers.get("target_card"), "instance_id", None
                        ),
                        c.markers.get("target_player_index"),
                    )
                    for c in p.played_today
                ]
            ),
            (
                tuple(sorted([c.definition.title for c in hand]))
                if len(hand) > 1
                else ((hand[0].definition.title,) if hand else ())
            ),
            tuple([c.definition.title for c in p.tomorrow_cards])
            if p.tomorrow_cards
            else (),
        )

    def _ordered_candidate_indices(
        self,
        sim: Game,
        player_index: int,
        limit: int,
    ) -> list[int]:
        return [
            idx
            for idx, _ in self._ordered_candidate_indices_with_costs(
                sim, player_index, limit
            )
        ]

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
        if type(sim) is _CARBON_SIM_GAME_CLS:
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

        wanted_tags = _wanted_tags_for_player(player)
        last_card = player.played_today[-1] if player.played_today else None
        last_prof = (
            _beryllium_build_profile(last_card)
            if last_card is not None
            else None
        )
        last_title = (
            _effective_card_title(last_card) if last_card is not None else ""
        )
        last_tags = (
            (
                last_card.definition.tags
                if not last_card.markers
                else last_card.tags
            )
            if last_card is not None
            else frozenset()
        )
        known_top = _get_sim_known_trunk_top(sim)

        players = sim.players
        awake_opps = 0
        for idx in range(4):
            if idx != player_index and not players[idx].asleep:
                awake_opps += 1

        has_epic_duels_played = False
        has_early_food_played = False
        for pc in player.played_today:
            pc_title = _effective_card_title(pc)
            if pc_title == "Epic Duels":
                has_epic_duels_played = True
            if (
                not has_early_food_played
                and "Food" in (pc.definition.tags if not pc.markers else pc.tags)
                and pc_title
                in {
                    "Rouladen",
                    "High-End Red",
                    "High-End White",
                    "Morning Coffee",
                    "Afternoon Coffee",
                    "Fajitas",
                    "Butter Tarts",
                    "Würstchen",
                    "Calzones",
                    "Ice Cream Sandwich",
                    "Doritos",
                    "Potato Pancakes",
                }
            ):
                has_early_food_played = True

        has_trunk_consumer = False
        has_trunk_setter = False
        for c in player.hand:
            ct = c.definition.title
            if ct in {
                "Assorted Cutlery",
                "Quacks",
                "Poker",
                "Bougie Coffee Machine",
                "Booby Prize",
            }:
                has_trunk_consumer = True
            if ct in _TRUNK_REORDER_TITLES:
                has_trunk_setter = True

        def sort_key(hand_idx: int, cost: int) -> tuple[int, float, int]:
            card = player.hand[hand_idx]
            prof = _beryllium_build_profile(card)
            desirability = _carbon_quick_card_desirability(
                sim, player_index, card, wanted_tags, cost=cost, prof=prof
            )
            bucket = prof.play_order_bucket
            title = card.definition.title
            c_tags = card.definition.tags if not card.markers else card.tags
            eff_base_fun = (
                card.definition.base_fun
                if not card.markers
                else card.effective_base_fun
            )

            if (
                prof.conditional_energy_gain > 0
                and prof.immediate_energy_gain == 0
            ):
                if _condition_met_in_played(player, card, prof):
                    bucket = 1
                    desirability += prof.conditional_energy_gain * 1.35
                else:
                    bucket = 4

            if (
                title == "Colouring"
                and player.character is not None
                and player.character.zero_fun_tag
            ):
                bucket = 6
                desirability -= 8.0
            elif title == "Nos Book":
                if not player.played_today:
                    bucket = 6
                    desirability -= 6.0
            elif (
                title == "Epic Duels"
                and awake_opps > 0
                and not has_epic_duels_played
                and has_trunk_setter
                and has_trunk_consumer
            ):
                # Play Epic Duels BEFORE Trunk-setup combos so opponents cannot steal the top card!
                bucket = 1
                desirability += 3.8
            elif title == "Group Dinner":
                # If we play Group Dinner at all, play it early before an opponent sleeps and locks it out
                bucket = 1
            elif title == "Sing Song" and awake_opps > 0 and cost >= 2:
                # Delay Sing Song while opponents are awake so their Social plays discount it
                bucket = 5
            elif title == "The Crew" and awake_opps > 0:
                bg_opps = 0
                for idx in range(4):
                    if idx != player_index:
                        for oc in players[idx].played_today:
                            if "Board Game" in oc.tags:
                                bg_opps += 1
                                break
                if bg_opps < 2:
                    bucket = 5
            elif title == "Wedding Anniversary":
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
                        bucket = 0 if not player.played_today else 2
                        desirability += best_cv
                    elif awake_opps > 0:
                        bucket = 5
                else:
                    bucket = 6
                    desirability -= 4.0
            elif title == "Medical Advice":
                if sim.day < DAY_COUNT and len(player.played_today) <= 1:
                    bucket = 2
                    desirability += 3.5
                else:
                    bucket = 6
                    desirability -= 3.0
            elif title == "Tell a Story":
                safe = has_epic_duels_played or awake_opps == 0
                cutlery_cards = [
                    c
                    for c in player.hand
                    if c.definition.title == "Assorted Cutlery"
                ]
                bcm_cards = [
                    c
                    for c in player.hand
                    if c.definition.title == "Bougie Coffee Machine"
                ]
                if safe and cutlery_cards:
                    c_cost = sim.energy_cost(player_index, cutlery_cards[0])
                    if cost + c_cost <= player.energy and any(
                        other is not card
                        and other is not cutlery_cards[0]
                        and _carbon_free_play_value(
                            sim, player_index, other, wanted_tags
                        )
                        >= 4.5
                        for other in player.hand
                    ):
                        bucket = 2
                        desirability += 5.0
                elif safe and bcm_cards:
                    b_cost = sim.energy_cost(player_index, bcm_cards[0])
                    if cost + b_cost <= player.energy and any(
                        other is not card
                        and other is not bcm_cards[0]
                        and "Food" in other.tags
                        and _carbon_free_play_value(
                            sim, player_index, other, wanted_tags
                        )
                        >= 4.0
                        for other in player.hand
                    ):
                        bucket = 2
                        desirability += 4.5
                else:
                    bucket = 5
            elif (
                title
                in {
                    "Read the Radar",
                    "Grocery Store Run",
                    "Fishing Boat",
                    "Ski Boat",
                }
                and has_trunk_consumer
            ):
                bucket = 2
                desirability += 2.5
            elif title == "Assorted Cutlery":
                if known_top:
                    top_val = _carbon_free_play_value(
                        sim, player_index, known_top[0], wanted_tags
                    )
                    if top_val >= 3.5:
                        has_item_buff = any(
                            other.definition.title
                            in {"Boat Ride", "Fancy Craft", "Canoe"}
                            and cost + sim.energy_cost(player_index, other)
                            <= player.energy
                            for other in player.hand
                        )
                        bucket = 3 if has_item_buff else 2
                        desirability += top_val
                    elif top_val < 2.0 and cost >= 2:
                        bucket = 6
                        desirability -= 4.0
            elif title == "Quacks" and known_top:
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
            elif title == "Poker" and known_top and known_top[0].tags:
                bucket = 2
                desirability += 3.5
            elif title in {"Last Year's Shirt", "Scrabble"}:
                has_pb_played = any(
                    _effective_card_title(c) == "Paddleboat"
                    for c in player.played_today
                )
                has_pb_hand = any(
                    c.definition.title == "Paddleboat"
                    and cost + sim.energy_cost(player_index, c) <= player.energy
                    for c in player.hand
                )
                if len(player.hand) >= 2 and sim.suitcase:
                    bucket = 3 if has_pb_hand else 2
                    if has_pb_played:
                        desirability += 3.5
                elif title == "Last Year's Shirt":
                    bucket = 6
                    desirability -= 4.0
            elif title == "Solo":
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
                        for other in player.hand
                    )
                    bucket = 2 if has_bg_setup else 1
                    desirability += 2.2
            elif title == "San Juan":
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
            elif title == "Cities and Knights" and len(player.hand) >= 2:
                bucket = 5
            elif title == "Address the Food":
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
            elif title in {"Treat Store Run", "Fancy Floatie"}:
                tag_needed = (
                    "Food" if title == "Treat Store Run" else "Relax"
                )
                matches_in_sc = 0
                for sc in sim.suitcase:
                    if tag_needed in sc.tags:
                        matches_in_sc += 1
                if matches_in_sc >= 2:
                    bucket = 2
                    desirability += matches_in_sc * 1.5
                elif matches_in_sc == 0:
                    desirability -= 2.0
            elif title == "Paddleboard":
                if len(player.hand) >= 3:
                    bucket = 5
                    desirability -= 2.0
            elif title == "Istanbul":
                if not _allows_energy_gain_today(sim, player, card):
                    bucket = 5
                    desirability -= 3.0
                elif len(player.hand) >= 2:
                    other_cost = sum(
                        sim.energy_cost(player_index, other)
                        for other in player.hand
                        if other is not card
                        and other.effective_behavior.can_play(sim, player, other)
                    )
                    if (
                        other_cost > player.energy - cost
                        or any(
                            _effective_card_title(other)
                            in {"Kayak", "Decaf", "Wakesurf", "Slalom Start"}
                            for other in player.hand
                            if other is not card
                        )
                        or any(
                            _effective_card_title(pc)
                            in {"Decaf", "Wakesurf", "Slalom Start"}
                            for pc in player.played_today
                        )
                    ):
                        bucket = 2
                        desirability += 2.4
            elif title == "Shady Spot":
                # Play Shady Spot BEFORE Sunscreen only? Wait: Sunscreen (1E) makes Shady Spot cost 1E!
                has_sunscreen_hand = any(
                    other.definition.title == "Sunscreen"
                    and sim.energy_cost(player_index, other) == 1
                    for other in player.hand
                )
                bucket = 3 if has_sunscreen_hand else 2

            if last_card is not None and last_prof is not None:
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
                        for other in player.hand
                    )
                ):
                    bucket = 2
                    desirability += 4.0
                elif (
                    last_prof.is_single_buff
                    and last_prof.target_tags
                    and (c_tags & last_prof.target_tags)
                    and not prof.is_multi_scaler
                    and not prof.is_single_buff
                    and (
                        title != "Nos Book"
                        or len(player.played_today) >= 1
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

            return (bucket, -desirability, cost)

        keyed = [
            (sort_key(idx, cost), idx, cost)
            for idx, cost in playable_with_costs
        ]
        keyed.sort(key=lambda item: item[0])
        if has_early_food_played and len(keyed) > 1:
            for pos, item in enumerate(keyed):
                if player.hand[item[1]].definition.title == "Leftovers":
                    if pos > 0:
                        keyed.insert(0, keyed.pop(pos))
                    break
        if len(keyed) <= limit:
            return [(idx, cost) for _, idx, cost in keyed]
        selected: list[tuple[int, int]] = []
        selected_indices: set[int] = set()
        seen_buckets: set[int] = set()
        for key_tup, idx, cost in keyed:
            b = key_tup[0]
            if b not in seen_buckets:
                seen_buckets.add(b)
                selected.append((idx, cost))
                selected_indices.add(idx)
        for _, idx, cost in keyed:
            if len(selected) >= limit:
                break
            if idx not in selected_indices:
                selected.append((idx, cost))
                selected_indices.add(idx)
        return selected

    def _search_best_day_plan_with_table(
        self,
        base_sim: Game,
        player_index: int,
        *,
        beam_width: int = 12,
        max_plays: int | None = None,
        allow_bed_now: bool = True,
        allowed_first_indices: Sequence[int] | None = None,
    ) -> tuple[int | None, float, dict[int | None, tuple[float, Game]]]:
        """Multi-card beam search using _carbon_copy_game_for_play and upgraded Day d+1 rollout."""
        played_count = len(base_sim.players[player_index].played_today)
        is_live_play = max_plays is None and self._evaluation_depth == 0
        if max_plays is None:
            if played_count >= 12:
                max_plays = 6
                beam_width = 10
            elif played_count >= 7:
                max_plays = 10
                beam_width = 17
            else:
                max_plays = self.MAX_TURN_PLAYS
                beam_width = self.BEAM_WIDTH

        best_by_action: dict[int | None, tuple[float, Game]] = {}

        if allow_bed_now:
            bed_now = _carbon_copy_game_for_play(base_sim, player_index)
            bed_now.go_to_bed(player_index)
            bed_val = self._evaluate_terminal_day(bed_now, player_index)
            best_value = bed_val
            best_first_actions: list[int | None] = [None]
            best_by_action[None] = (bed_val, bed_now)
        else:
            best_value = float("-inf")
            best_first_actions = []

        frontier: list[tuple[int, Game, float]] = []
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

        for hand_idx, known_cost in initial_pairs:
            try:
                child = self._simulate_play_step(
                    base_sim, player_index, hand_idx, known_cost
                )
                stop_val = self._evaluate_terminal_day(child, player_index)
            except Exception:
                continue
            prev = best_by_action.get(hand_idx)
            if prev is None or stop_val > prev[0]:
                best_by_action[hand_idx] = (stop_val, child)
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
                    frontier.append((hand_idx, child, cont_val))

        if frontier:
            frontier.sort(key=lambda item: item[2], reverse=True)
            frontier = frontier[:beam_width]

        for depth in range(1, max_plays):
            if not frontier:
                break
            need_next = depth + 1 < max_plays
            if is_live_play:
                active_beam = (
                    beam_width if depth < 7 else max(9, beam_width - 6)
                )
                branch_limit = 12 if depth < 6 else 9
            else:
                active_beam = (
                    beam_width if depth < 5 else max(4, beam_width - 3)
                )
                branch_limit = (
                    self.BRANCH_LIMIT
                    if depth < 5
                    else max(5, self.BRANCH_LIMIT - 2)
                )
            next_frontier: list[tuple[int, Game, float]] = []
            for first_action, state, _ in frontier:
                candidates = self._ordered_candidate_indices_with_costs(
                    state, player_index, branch_limit
                )
                for hand_idx, known_cost in candidates:
                    try:
                        child = self._simulate_play_step(
                            state, player_index, hand_idx, known_cost
                        )
                        stop_val = self._evaluate_terminal_day(
                            child, player_index
                        )
                    except Exception:
                        continue
                    prev = best_by_action.get(first_action)
                    if prev is None or stop_val > prev[0]:
                        best_by_action[first_action] = (stop_val, child)
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
                                (first_action, child, cont_val)
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
                base_sim, player_index, best_by_action, best_value
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

    def _refine_with_next_day_rollout(
        self,
        base_sim: Game,
        player_index: int,
        best_by_action: dict[int | None, tuple[float, Game]],
        max_fast_val: float,
    ) -> tuple[list[int | None], float]:
        """Run Day d+1 play search using Carbon's own _BerylliumSimGame & _CARBON_SIM_AIS."""
        contenders = [
            (act, f_val, term_sim)
            for act, (f_val, term_sim) in best_by_action.items()
            if f_val >= max_fast_val - 2.8
        ]
        if len(contenders) <= 1:
            return [act for act, _, _ in contenders], max_fast_val

        configs = {
            (
                tuple(sorted(c.title for c in t_sim.players[player_index].hand)),
                tuple(
                    _effective_card_title(c)
                    for c in t_sim.players[player_index].played_today
                    if c.effective_behavior.has_tomorrow_action
                    or _effective_card_title(c)
                    in {"Chuck a Frisbee", "Scout Other Cottages"}
                ),
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

        other_asleep_now = any(
            p.asleep
            for idx, p in enumerate(base_sim.players)
            if idx != player_index
        )
        first_bed_bonus = (
            (0.80 if base_sim.day == DAY_COUNT - 1 else self.FIRST_TO_BED_BONUS)
            if not other_asleep_now
            else 0.0
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
            next_sim = _carbon_copy_game(term_sim)
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
            end_day_fun = (
                float(p_next.fun)
                - 1.35 * gd_cnt
                - (0.45 * min(awake_opps, 3) if scout_cnt and awake_opps else 0.0)
            )
            cache_key = (
                tuple(sorted(c.title for c in p_next.hand)),
                tuple(
                    (_effective_card_title(c), c.markers.get("chosen_tag"))
                    for c in p_next.tomorrow_cards
                ),
            )
            delta_tomorrow = rollout_cache.get(cache_key)
            if delta_tomorrow is None:
                try:
                    next_sim.day += 1
                    next_sim.start_day()
                    wanted_next = _wanted_tags_for_player(p_next)
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

                    prev_depth = self._evaluation_depth
                    self._evaluation_depth = 1
                    try:
                        # Use self._search_best_day_plan (NOT ArgonBrain super!) so Istanbul/Bohnanza/Carbon eval apply
                        _, day2_total = self._search_best_day_plan(
                            next_sim,
                            player_index,
                            beam_width=6,
                            max_plays=8,
                            allow_bed_now=True,
                        )
                    finally:
                        self._evaluation_depth = prev_depth
                    delta_tomorrow = (day2_total - float(p_next.fun)) - 7.0
                    rollout_cache[cache_key] = delta_tomorrow
                except Exception:
                    delta_tomorrow = f_val - end_day_fun - first_bed_bonus

            rollout_total = end_day_fun + first_bed_bonus + delta_tomorrow
            blended = 0.55 * f_val + 0.45 * rollout_total
            best_by_action[act] = (blended, term_sim)
            if blended > best_blended + 1e-9:
                best_blended = blended
                best_acts = [act]
            elif abs(blended - best_blended) <= 1e-9:
                best_acts.append(act)

        return best_acts, best_blended

    def _continuation_potential(self, sim: Game, player_index: int) -> float:
        """PYTHONHASHSEED-deterministic continuation upside for active/pending scalers."""
        player = sim.players[player_index]
        if player.asleep or not player.hand:
            return 0.0

        prof_by_id = _CARBON_PROFILE_BY_DEFN_ID
        played_today = player.played_today
        active_after_tags: dict[str, float] = {}
        global_after_bonus = 0.0
        has_after_all_tags = False
        for played in played_today:
            pm = played.markers
            p_uncopied = not pm or "_copied_definition" not in pm
            did = id(played.definition)
            prof = (
                prof_by_id[did]
                if (p_uncopied and did in prof_by_id)
                else _beryllium_build_profile(played)
            )
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
        energy_budget = player.energy
        for hc in player.hand:
            hm = hc.markers
            h_uncopied = not hm or "_copied_definition" not in hm
            hdid = id(hc.definition)
            h_prof = (
                prof_by_id[hdid]
                if (h_uncopied and hdid in prof_by_id)
                else _beryllium_build_profile(hc)
            )
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
            for card in player.hand:
                cm = card.markers
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

    def _evaluate_terminal_day(
        self, terminal_sim: Game, player_index: int
    ) -> float:
        """Opponent-aware terminal-day evaluation."""
        players = terminal_sim.players
        player = players[player_index]

        other_asleep = False
        awake_opps = 0
        for idx in range(4):
            if idx != player_index:
                if players[idx].asleep:
                    other_asleep = True
                else:
                    awake_opps += 1
        opponents_awake = awake_opps > 0

        if terminal_sim.day < DAY_COUNT and not other_asleep:
            first_to_bed_value = (
                0.80
                if terminal_sim.day == DAY_COUNT - 1
                else self.FIRST_TO_BED_BONUS
            )
        else:
            first_to_bed_value = 0.0

        played_today = player.played_today
        needs_full_end_day = False
        has_loop2 = False
        has_loop3 = False
        for c in played_today:
            m = c.markers
            uncopied = not m or "_copied_definition" not in m
            c_beh = c.definition.behavior if uncopied else c.effective_behavior
            b_type = type(c_beh)
            if (
                b_type.on_score is not CardBehavior.on_score
                or b_type.on_end_day is not CardBehavior.on_end_day
                or b_type.on_after_end_day is not CardBehavior.on_after_end_day
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
            scoring = _carbon_copy_game(terminal_sim)
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
            if type(terminal_sim) is _CARBON_SIM_GAME_CLS:
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
                    # Discount simulated 4-of-4 Suitcase survival when opponents are still awake
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
        """PYTHONHASHSEED-deterministic reserve & tomorrow valuation with Carbon calibrations."""
        player = sim.players[player_index]
        hand = player.hand
        if not hand and not next_tomorrow_cards:
            return 0.0

        prof_by_id = _CARBON_PROFILE_BY_DEFN_ID
        if not prof_by_id:
            _ensure_carbon_caches()
        is_canonical = getattr(sim, "_all_canonical_defs", False)
        if hand:
            if is_canonical:
                hand_slugs = tuple(
                    [hc.definition.slug for hc in hand if not hc.markers]
                )
            else:
                hand_slugs = tuple(
                    [
                        hc.definition.slug
                        for hc in hand
                        if not hc.markers and id(hc.definition) in prof_by_id
                    ]
                )
            can_cache = len(hand_slugs) == len(hand)
            has_ski_shoulders = can_cache and (
                "ski-on-shoulders" in hand_slugs
            )
        else:
            hand_slugs = ()
            can_cache = True
            has_ski_shoulders = False

        tom_key: tuple = ()
        zero_e = False
        if can_cache and next_tomorrow_cards:
            has_lawn = False
            tom_key_list = []
            for tc in next_tomorrow_cards:
                if not is_canonical and id(tc.definition) not in prof_by_id:
                    can_cache = False
                    break
                tm = tc.markers
                if (
                    tm
                    and "_copied_definition" in tm
                    and id(_effective_source_definition(tc)) not in prof_by_id
                ):
                    can_cache = False
                    break
                t_title = (
                    tc.definition.title
                    if (not tm or "_copied_definition" not in tm)
                    else _effective_card_title(tc)
                )
                if t_title == "Lawn Maintenance":
                    has_lawn = True
                tom_key_list.append(
                    (
                        tc.definition.slug,
                        t_title,
                        tm.get("chosen_tag") if tm else None,
                    )
                )
            tom_key = tuple(tom_key_list)
            zero_e = has_lawn and player.energy == 0

        cache = self._reserve_eval_cache
        cache_key: tuple | None = None
        ski_ahead_ge2: bool | None = None
        if can_cache:
            if has_ski_shoulders:
                p_fun = player.fun
                opps_ahead_cnt = 0
                for idx, opp in enumerate(sim.players):
                    if idx != player_index and opp.fun > p_fun:
                        opps_ahead_cnt += 1
                ski_ahead_ge2 = opps_ahead_cnt >= 2
            cache_key = (
                days_left,
                self.RESERVE_TOMORROW_WEIGHT,
                player.character,
                zero_e,
                ski_ahead_ge2,
                tom_key,
                hand_slugs,
            )
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val

        tom_fun = 0.0
        tom_energy_delta = 0
        tom_tag_bonuses: dict[str, float] = {}
        tom_all_tags = False
        has_colouring = False
        has_photo_shoot = False
        has_dates_first = False
        tom_pos_energy_sum = 0

        for tc in next_tomorrow_cards:
            tm = tc.markers
            tc_uncopied = not tm or "_copied_definition" not in tm
            t_prof = (
                prof_by_id.get(id(tc.definition)) if tc_uncopied else None
            )
            if t_prof is None:
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
            if t_title == "Lawn Maintenance" and player.energy == 0:
                tom_fun += 3.0
            elif t_title == "Wash the Dishes" and len(hand) <= 1:
                tom_fun += 2.0
            elif t_title == "Photo Shoot":
                tom_fun += 3.5
            elif t_title == "Date's First Nos":
                tom_fun += 2.2

            if t_prof.signed_match_benefit != 0.0:
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
            val = tom_fun + tom_energy_value
        else:
            char = player.character
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

            held_scaler_benefit_by_tag: dict[str, float] = {}
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

            paired_partner_counts: dict[str, int] = {}
            seen_must_play_first = False
            surpluses: list[float] = []

            if len(hand) <= 1:
                ordered_hand: Sequence[CardInstance] = hand
            elif can_cache:
                ordered_hand = sorted(
                    hand,
                    key=lambda c: (c.definition.cost, -c.definition.base_fun),
                )
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
                c_tags = c_defn.tags if not cm else c.tags
                ordered_c_tags = (
                    c_tags if len(c_tags) <= 1 else sorted(c_tags)
                )
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
                    imm_e = max(3.0, float(len(hand)))
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
                    if pos_tom_all > 0.0:
                        raw_val += pos_tom_all * 1.25
                        is_synergy_piece = True
                elif tom_tag_bonuses:
                    for t in ordered_c_tags:
                        tb = tom_tag_bonuses.get(t, 0.0)
                        if tb > 0.0:
                            raw_val += tb * 1.25
                            is_synergy_piece = True

                best_partner_tag: str | None = None
                best_partner_benefit = 0.0
                if held_scaler_benefit_by_tag:
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

                if c_prof.is_multi_scaler:
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
                    is_synergy_piece = True
                    wait_factor = 2.10 if days_left >= 2 else 1.35
                    raw_val += wait_factor * c_prof.signed_match_benefit
                elif c_title == "Couch Tube" and days_left >= 1:
                    is_synergy_piece = True
                    raw_val += 3.40 if days_left >= 2 else 2.40
                elif c_title == "Cheesy Phone Game" and days_left >= 1:
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
                elif c_title == "Ski on Shoulders" and days_left >= 1:
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
                elif c_prof.is_tomorrow_scaler:
                    if days_left >= 2:
                        raw_val += 0.50 * c_prof.signed_match_benefit
                elif c_prof.is_single_buff and days_left >= 1:
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

            if len(surpluses) > 1:
                surpluses.sort(reverse=True)
            has_combo_or_energy = (
                bool(held_multi_scalers) or tom_energy_delta >= 2
            )
            slot_weights = (
                (1.0, 1.0, 0.85, 0.55)
                if has_combo_or_energy
                else (1.0, 0.92, 0.50, 0.25)
            )
            held_value = 0.0
            for i, s in enumerate(surpluses):
                held_value += s * (slot_weights[i] if i < 4 else 0.08)
            val = (
                tom_fun
                + tom_energy_value
                + self.RESERVE_TOMORROW_WEIGHT * held_value
            )

        extra = 0.0
        if has_colouring:
            if (
                player.character is not None
                and player.character.zero_fun_tag is not None
            ):
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

        if has_dates_first:
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

        rtw = self.RESERVE_TOMORROW_WEIGHT
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
                            and other.definition.title in _HIGH_COPY_FOOD_TITLES
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
                    relax_cnt = 0
                    for other in hand:
                        if other is not hc and "Relax" in (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        ):
                            relax_cnt += 1
                    if relax_cnt >= 1:
                        extra += min(relax_cnt, 3) * 1.6 * rtw
                elif h_title == "Keeper":
                    out_cnt = 0
                    for other in hand:
                        if other is not hc and "Outdoors" in (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        ):
                            out_cnt += 1
                    if out_cnt >= 2:
                        extra += min(out_cnt, 3) * 0.9 * rtw
                elif h_title == "Camp Crossroads Dessert":
                    ex_cnt = 0
                    for other in hand:
                        if other is not hc and "Exercise" in (
                            other.definition.tags
                            if not other.markers
                            else other.tags
                        ):
                            ex_cnt += 1
                    if ex_cnt >= 2:
                        extra += min(ex_cnt, 3) * 0.9 * rtw
                elif h_title == "Pudding Chômeur" and len(hand) >= 2:
                    extra += 1.4 * rtw

            # Carbon empirical adjustments
            flat2 = _RESERVE_EMPIRICAL_FLAT.get(h_title)
            if flat2 is not None:
                extra += flat2 * rtw
            elif h_title in _RESERVE_DYNAMIC_EXTRA_2:
                if h_title == "Assorted Cutlery":
                    has_real_combo = False
                    for other in hand:
                        if other is not hc and other.definition.title in {
                            "Read the Radar",
                            "Fishing Boat",
                            "Ski Boat",
                            "Grocery Store Run",
                            "Boat Ride",
                            "Fancy Craft",
                        }:
                            has_real_combo = True
                            break
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
                            sup += 3
                        elif op.immediate_energy_gain > other.definition.cost:
                            sup += op.immediate_energy_gain - other.definition.cost
                    if sup >= 3:
                        extra += 3.0 * rtw
                    elif sup >= 2 or days_left >= 3:
                        extra += 1.6 * rtw
                    elif sup == 0 and days_left <= 1:
                        extra -= 1.2 * rtw
                elif h_title == "Würstchen" and len(hand) >= 3:
                    extra += 1.8 * rtw
                elif h_title == "Medical Advice" and days_left >= 2:
                    extra += 1.8 * rtw
                elif h_title in {"Wakesurf", "Slalom Start"} and len(hand) >= 3:
                    extra -= 0.9 * rtw

        res_val = val + extra
        if cache_key is not None:
            if len(cache) > 131072:
                cache.clear()
            cache[cache_key] = res_val
        return res_val

    # ------------------------------------------------------------------
    # Tactical Sub-Decisions Routed Through Carbon's Upgraded Policy
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
            return _select_bohnanza_discards(game, player_index, options)
        if last_played == "Istanbul":
            ranked = _evaluate_istanbul_discards(game, player_index, options)
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
                    sim = _carbon_copy_game(base_sim)
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
        return _CARBON_EVAL_AI_SINGLETON.choose_cards_to_discard(
            game, player_index, options
        )

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _CARBON_EVAL_AI_SINGLETON.choose_card_to_discard(
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
            return _CARBON_EVAL_AI_SINGLETON.choose_card_to_copy(
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
            return _CARBON_EVAL_AI_SINGLETON.choose_card_to_copy(
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
                sim = _carbon_copy_game(base_sim)
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
                    # Use Carbon's own _search_best_day_plan (not ArgonBrain super!)
                    _, val = self._search_best_day_plan(
                        sim, player_index, beam_width=5, max_plays=5
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
        return _CARBON_EVAL_AI_SINGLETON.choose_card_target(
            game, player_index, options
        )

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _CARBON_EVAL_AI_SINGLETON.choose_suitcase_target(
            game, player_index, options
        )

    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        return _CARBON_EVAL_AI_SINGLETON.choose_player(
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
        return _CARBON_EVAL_AI_SINGLETON.choose_tag(
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
                sim = _carbon_copy_game(base_sim)
                sim_player = sim.players[player_index]
                sim_source = sim_player.played_today[-1]
                sim_player.energy -= spend
                sim_source.markers["energy_cubes"] = spend
                for t_src in tuple(sim_player.tomorrow_cards):
                    t_src.effective_behavior.on_tomorrow_card_play(
                        sim, sim_player, t_src, sim_source
                    )
                for p_src in tuple(sim_player.played_today):
                    p_src.effective_behavior.on_card_play(
                        sim, sim_player, p_src, sim_source
                    )
                _, val = self._search_best_day_plan(
                    sim, player_index, beam_width=3, max_plays=3
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
        return _CARBON_EVAL_AI_SINGLETON.choose_optional_action(
            game, player_index, action
        )

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        self._ensure_live_game(game)
        order = _CARBON_EVAL_AI_SINGLETON.order_cards_for_trunk(
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
        return _CARBON_EVAL_AI_SINGLETON.choose_cards_for_trunk(
            game, player_index, cards
        )

    def order_daily_conditions(
        self,
        game: Game,
        player_index: int,
        conditions: Sequence[Any],
    ) -> Sequence[int]:
        return _CARBON_EVAL_AI_SINGLETON.order_daily_conditions(
            game, player_index, conditions
        )

    def _record_extra_pick_if_live(self, game: Game, player_index: int) -> None:
        if game.stats is _get_shared_sim_stats():
            return
        self._ensure_live_game(game)
        gid = id(game)
        key = (gid, game.day, player_index)
        prev = (
            self._extra_draw_picks_state[3]
            if (
                self._extra_draw_picks_state is not None
                and self._extra_draw_picks_state[:3] == key
            )
            else 0
        )
        self._extra_draw_picks_state = (*key, prev + 1)

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
        wanted_tags = _wanted_tags_for_player(player)
        if game.day < DAY_COUNT:
            best_des = max(
                (
                    _carbon_quick_card_desirability(
                        game, player_index, c, wanted_tags
                    )
                    for c in options
                ),
                default=0.0,
            )
            take = player.energy >= 2 or best_des >= 2.5
            if take:
                self._record_extra_pick_if_live(game, player_index)
            return take

        # On Day 6, unplayed hand cards score 0; only pay 1 Energy when 2nd-best visible option is profitable
        if len(options) < 2:
            return False
        pr_markers, scout_markers = _opponent_suitcase_markers(
            game, player_index
        )
        scored = sorted(
            (
                (
                    _carbon_quick_card_desirability(
                        game, player_index, c, wanted_tags
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
        s_prof = _beryllium_build_profile(second_card)
        net_e_pos = (
            s_prof.immediate_energy_gain > second_cost
            or second_card.title
            in {"Würstchen", "Istanbul", "Potato Pancakes", "Pudding Chômeur"}
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


Carbon = CarbonAI
CarbonBrain = CarbonAI
CarbonbrainAI = CarbonAI
