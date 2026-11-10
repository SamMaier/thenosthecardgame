"""Enhanced compute-scaled lookahead policy (BerylliumBrain)."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence, TYPE_CHECKING
import random

from thenos.ais.argonbrain import (
    ArgonBrain,
    CardSynergyProfile,
    DAY_COUNT,
    SUITCASE_SIZE,
    _ACTIVE_PROBE_KEYS,
    _DEFN_ID_CACHE,
    _EVAL_AI_SINGLETON,
    _EvalPlayerAI,
    _TRUNK_REORDER_TITLES,
    _build_profile,
    _clone_card,
    _clone_random,
    _condition_met_in_played,
    _extra_energy_support_for_card,
    _fast_copy_game,
    _free_play_card_value,
    _get_shared_sim_stats,
    _get_sim_known_trunk_top,
    _own_suitcase_markers,
    _quick_card_desirability,
    _seeded_random,
    _tag_abundance,
    _unseen_tag_counts_and_total,
)
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.models import PlayerState

if TYPE_CHECKING:
    from thenos.game import Game

_STOCHASTIC_PLAY_TITLES: frozenset[str] = frozenset(
    {
        "5 10 15",
        "Assorted Cutlery",
        "Booby Prize",
        "Bougie Coffee Machine",
        "Euchre Awards Ceremony",
        "Euchre Tournament",
        "Fishing Boat",
        "Grocery Store Run",
        "Last Year's Shirt",
        "Long Distance Visitors",
        "Make Plans",
        "Medical Advice",
        "Paddleboard",
        "Poker",
        "Quacks",
        "Scrabble",
        "Ski Boat",
        "Throw a Baseball",
    }
)

_ACQUIRER_TITLES: frozenset[str] = frozenset(
    {
        "5 10 15",
        "Booby Prize",
        "Bougie Coffee Machine",
        "Euchre Awards Ceremony",
        "Euchre Tournament",
        "Fancy Floatie",
        "Fishing Boat",
        "Grocery Store Run",
        "Last Year's Shirt",
        "Long Distance Visitors",
        "Make Plans",
        "Medical Advice",
        "Paddleboard",
        "Photo Shoot",
        "Scout Other Cottages",
        "Scrabble",
        "Ski Boat",
        "Throw a Baseball",
        "Treat Store Run",
    }
)

_HIGH_COPY_FOOD_TITLES: frozenset[str] = frozenset(
    {
        "Afternoon Coffee",
        "Beaver Burgers",
        "Bougie Coffee Machine",
        "Butter Tarts",
        "Calzones",
        "Fajitas",
        "High-End Red",
        "High-End White",
        "Ice Cream Sandwich",
        "Ice Wine",
        "Morning Coffee",
        "Potato Pancakes",
        "Rouladen",
        "Würstchen",
    }
)

_NEUTRAL_DRAFT_DEFNS: tuple[CardDefinition, ...] = (
    CardDefinition(
        slug="_beryllium_neutral_1",
        title="_BerylliumNeutral1",
        tags=frozenset(),
        cost=1,
        base_fun=2,
        behavior=CardBehavior(),
    ),
    CardDefinition(
        slug="_beryllium_neutral_2",
        title="_BerylliumNeutral2",
        tags=frozenset(),
        cost=2,
        base_fun=3,
        behavior=CardBehavior(),
    ),
    CardDefinition(
        slug="_beryllium_neutral_3",
        title="_BerylliumNeutral3",
        tags=frozenset(),
        cost=2,
        base_fun=2,
        behavior=CardBehavior(),
    ),
)

_TAGGED_NEUTRAL_3_BY_TAG: dict[str, CardDefinition] = {
    tag: CardDefinition(
        slug=f"_beryllium_neutral_3_{tag.lower().replace(' ', '_')}",
        title="_BerylliumNeutral3",
        tags=frozenset({tag}),
        cost=2,
        base_fun=2,
        behavior=CardBehavior(),
    )
    for tag in (
        "Relax",
        "Food",
        "Exercise",
        "Social",
        "Outdoors",
        "Indoors",
        "Board Game",
        "Item",
    )
}


_BEHAVIOR_ID_TO_DEFN: dict[int, CardDefinition] = {}
_CANONICAL_DEFN_IDS: set[int] = set()
_BERYLLIUM_DEFN_CACHE: dict[int, CardSynergyProfile] = {}
_ISTANBUL_ENERGY_CACHE: dict[int, int] = {}


class _IstanbulProbeAI:
    def choose_cards_to_discard(
        self,
        game: Any,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> Sequence[int]:
        return (0,) if options else ()


def _istanbul_energy_per_discard(card: CardInstance) -> int:
    """Return how much Energy `card`'s behavior grants per discarded card."""
    beh = card.effective_behavior
    if not _BEHAVIOR_ID_TO_DEFN:
        _ensure_behavior_defn_map()
    bid = id(beh)
    is_canonical = bid in _BEHAVIOR_ID_TO_DEFN
    if is_canonical:
        cached = _ISTANBUL_ENERGY_CACHE.get(bid)
        if cached is not None:
            return cached
    from thenos.ais.random_ai import RandomAI
    from thenos.game import Game

    ais: list[Any] = [RandomAI(_seeded_random(9910 + i)) for i in range(4)]
    ais[0] = _IstanbulProbeAI()
    probe = Game([], ais, _seeded_random(9909))
    p0 = probe.players[0]
    p0.energy = 0
    src = CardInstance(-1, card.effective_definition)
    p0.hand.append(CardInstance(-2, _NEUTRAL_DRAFT_DEFNS[0]))
    p0.played_today.append(src)
    try:
        beh.on_play(probe, p0, src)
        per_discard = max(0, p0.energy)
    except Exception:
        per_discard = 3
    if is_canonical:
        _ISTANBUL_ENERGY_CACHE[bid] = per_discard
    return per_discard


def _allows_energy_gain_today(
    game: Game,
    player: PlayerState,
    source_card: CardInstance | None = None,
) -> bool:
    return all(
        c.effective_behavior.allows_energy_gain(game, player, c, source_card)
        for c in player.played_today
    )


def _istanbul_extra_energy_support(
    game: Game,
    player_index: int,
    target: CardInstance,
    *,
    extra_fodder: int = 0,
    ignore_today_block: bool = False,
) -> int:
    """Estimate net ramp Energy `Istanbul` in hand can contribute toward `target`."""
    player = game.players[player_index]
    extra = 0
    rem_fodder = sum(1 for hc in player.hand if hc is not target) + extra_fodder
    for hc in player.hand:
        if hc is target or _effective_card_title(hc) != "Istanbul":
            continue
        if not ignore_today_block:
            if not hc.effective_behavior.can_play(game, player, hc):
                continue
            if not _allows_energy_gain_today(game, player, hc):
                continue
        rem_fodder -= 1
        discards = min(3, max(0, rem_fodder))
        if discards > 0:
            rem_fodder -= discards
            e_per = _istanbul_energy_per_discard(hc)
            hc_cost = (
                hc.effective_cost
                if ignore_today_block
                else game.energy_cost(player_index, hc)
            )
            extra += max(0, e_per * discards - hc_cost)
    return extra


def _ensure_behavior_defn_map() -> None:
    if _BEHAVIOR_ID_TO_DEFN:
        return
    from thenos.cards.catalog import CARD_REGISTRY

    for defn in CARD_REGISTRY.values():
        _CANONICAL_DEFN_IDS.add(id(defn))
        if type(defn.behavior) is not CardBehavior:
            _BEHAVIOR_ID_TO_DEFN[id(defn.behavior)] = defn
    for defn in _NEUTRAL_DRAFT_DEFNS:
        _CANONICAL_DEFN_IDS.add(id(defn))
    for defn in _TAGGED_NEUTRAL_3_BY_TAG.values():
        _CANONICAL_DEFN_IDS.add(id(defn))


def _effective_source_definition(card: CardInstance) -> CardDefinition:
    """Return the canonical source CardDefinition for copied cards, or card.definition."""
    markers = card.markers
    if markers:
        copied = markers.get("_copied_definition")
        if isinstance(copied, CardDefinition):
            if not _BEHAVIOR_ID_TO_DEFN:
                _ensure_behavior_defn_map()
            return _BEHAVIOR_ID_TO_DEFN.get(id(copied.behavior), copied)
    return card.definition


def _effective_card_title(card: CardInstance) -> str:
    """Return the exact effective card title, resolving copied effects via _effective_source_definition."""
    markers = card.markers
    if markers:
        copied = markers.get("_copied_definition")
        if isinstance(copied, CardDefinition):
            if not _BEHAVIOR_ID_TO_DEFN:
                _ensure_behavior_defn_map()
            return _BEHAVIOR_ID_TO_DEFN.get(id(copied.behavior), copied).title
    return card.definition.title


def _beryllium_build_profile(card: CardInstance) -> CardSynergyProfile:
    """Return the behavioral profile of card's effective source definition without marker cache collisions."""
    markers = card.markers
    if not markers:
        defn = card.definition
        fast = _BERYLLIUM_DEFN_CACHE.get(id(defn))
        if fast is not None:
            return fast
        if not _CANONICAL_DEFN_IDS:
            _ensure_behavior_defn_map()
        prof = _build_profile(card)
        if not _ACTIVE_PROBE_KEYS and id(defn) in _CANONICAL_DEFN_IDS:
            _BERYLLIUM_DEFN_CACHE[id(defn)] = prof
        return prof
    if not _BEHAVIOR_ID_TO_DEFN:
        _ensure_behavior_defn_map()
    src_defn = _effective_source_definition(card)
    if id(src_defn) in _CANONICAL_DEFN_IDS:
        fast = _BERYLLIUM_DEFN_CACHE.get(id(src_defn))
        if fast is not None:
            return fast
        prof = _build_profile(CardInstance(card.instance_id, src_defn))
        if not _ACTIVE_PROBE_KEYS:
            _BERYLLIUM_DEFN_CACHE[id(src_defn)] = prof
        return prof
    return _build_profile(
        CardInstance(card.instance_id, src_defn, False, {"_transient": True})
    )


def _wanted_tags_for_player(player: PlayerState) -> dict[str, float]:
    """Compute signed tag synergy weights using exact effective definitions for copied cards."""
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
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.15
                )
        chosen_tag = card.markers.get("chosen_tag")
        if isinstance(chosen_tag, str):
            weights[chosen_tag] = weights.get(chosen_tag, 0.0) + 1.5
        if bool(getattr(card.effective_behavior, "tomorrow_all_tags", False)):
            for t in (
                "Relax",
                "Food",
                "Exercise",
                "Social",
                "Outdoors",
                "Indoors",
                "Board Game",
                "Item",
            ):
                weights[t] = weights.get(t, 0.0) + 0.35

    for card in player.played_today:
        prof = _beryllium_build_profile(card)
        if (
            prof.is_multi_scaler and not prof.is_before_scaler
        ) or prof.is_tomorrow_scaler:
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.05
                )
        elif prof.penalizes_after or prof.signed_match_benefit < 0:
            for t in prof.target_tags:
                weights[t] = weights.get(t, 0.0) + prof.signed_match_benefit

    for card in player.hand:
        prof = _beryllium_build_profile(card)
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
        c_tags = card.definition.tags if not card.markers else card.tags
        for t in c_tags:
            weights[t] = weights.get(t, 0.0) + 0.15
    return weights


def _beryllium_quick_card_desirability(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
    *,
    cost: int | None = None,
    prof: CardSynergyProfile | None = None,
) -> float:
    """Fast heuristic valuation of a card including signed tag synergies."""
    if prof is None:
        if not card.markers:
            prof = _BERYLLIUM_DEFN_CACHE.get(id(card.definition))
        if prof is None:
            prof = _build_profile(card)
    player = game.players[player_index]
    if cost is None:
        cost = game.energy_cost(player_index, card)
    days_left = DAY_COUNT - game.day
    c_defn = card.definition if not card.markers else card.effective_definition
    title = c_defn.title
    eff_cost = c_defn.cost
    eff_base_fun = c_defn.base_fun
    c_tags = c_defn.tags if not card.markers else card.tags

    imm_energy = (
        max(2, len(player.hand))
        if title == "Würstchen"
        else prof.immediate_energy_gain
    )
    val = (
        float(eff_base_fun)
        + prof.immediate_fun_gain * 1.15
        - cost * 0.78
        + imm_energy * 1.30
    )
    if title == "Istanbul" and _allows_energy_gain_today(game, player, card):
        e_per = _istanbul_energy_per_discard(card)
        other_cnt = sum(1 for hc in player.hand if hc is not card)
        has_e_or_empty_sink = any(
            _effective_card_title(hc) in {"Kayak", "Decaf", "Wakesurf"}
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
            sac_cost = 0.75 if days_left == 0 else 1.55
            net_per = max(0.0, e_per * (1.15 if has_sink else 0.70) - sac_cost)
            val += disc_est * net_per
    if (
        prof.immediate_cards_gained > 0
        and not prof.penalizes_before
        and not prof.penalizes_after
        and not prof.is_multi_scaler
    ):
        val += min(prof.immediate_cards_gained, 3) * (
            0.95 if cost <= 1 else 0.15
        )

    if title == "Sleep In":
        val -= 1.8
    elif (
        prof.must_play_late
        and prof.immediate_energy_gain > 0
        and len(player.played_today) < 2
    ):
        val -= 0.85

    if eff_cost >= 8 and cost > player.energy:
        support = _extra_energy_support_for_card(
            player, card
        ) + _istanbul_extra_energy_support(
            game,
            player_index,
            card,
            extra_fodder=2 if (days_left >= 1 and card not in player.hand) else 0,
        )
        unfunded = max(0, eff_cost - (6 + support))
        if unfunded > 0:
            val -= unfunded * 1.65

    if prof.conditional_energy_gain > 0:
        if title in {"Potato Pancakes", "Keeper", "Camp Crossroads Dessert"}:
            t_tags = prof.target_tags
            match_cnt = sum(
                1
                for other in (*player.played_today, *player.hand)
                if other is not card
                and (
                    (
                        other.definition.tags
                        if not other.markers
                        else other.tags
                    )
                    & t_tags
                )
            )
            per_match = 2 if title == "Potato Pancakes" else 1
            val += min(match_cnt, 3) * per_match * 0.85
        else:
            t_tags = prof.target_tags
            has_cond = _condition_met_in_played(player, card, prof) or any(
                other is not card
                and (
                    (
                        other.definition.tags
                        if not other.markers
                        else other.tags
                    )
                    & t_tags
                )
                for other in player.hand
            )
            val += prof.conditional_energy_gain * (0.95 if has_cond else 0.35)

    if title == "Nos Book":
        if (
            player.played_today
            or len(player.hand) >= 2
            or game.day < DAY_COUNT
        ):
            val += 3.0
    elif title == "Assorted Cutlery":
        has_combo = any(
            other is not card
            and (
                other.definition.title in _TRUNK_REORDER_TITLES
                or other.definition.title
                in {"Boat Ride", "Fancy Craft", "Canoe"}
            )
            for other in player.hand
        )
        val += 3.2 if has_combo else 2.2
    elif title == "Ski on Shoulders":
        if game.day >= 2:
            opps_strictly_ahead = sum(
                1
                for idx, opp in enumerate(game.players)
                if idx != player_index and opp.fun > player.fun
            )
            if opps_strictly_ahead >= 2:
                val += 3.4
            elif days_left >= 1 and opps_strictly_ahead >= 1:
                val += 1.6
        elif days_left >= 1:
            val += 1.8
    elif prof.conditional_fun_bonus > 0 and not prof.is_multi_scaler:
        val += min(prof.conditional_fun_bonus, 5) * 0.55

    if title in {"Slalom Start", "Wakesurf"} and len(player.hand) >= 5:
        if not any(
            other.definition.title
            in {"Paddleboard", "Navy SEALing", "Istanbul", "Scrabble"}
            for other in player.hand
        ):
            val -= 0.55 * (len(player.hand) - 4)

    if wanted_tags:
        tag_match = 0.0
        for t in c_tags:
            w = wanted_tags.get(t, 0.0)
            if w != 0.0:
                tag_match += w * 0.78
                if w > 0 and cost <= 1:
                    tag_match += 0.38 * min(w, 2.5)
        if prof.must_play_first and tag_match > 0.0:
            tag_match *= 0.30
        val += tag_match

    if prof.is_multi_scaler:
        t_tags = prof.target_tags
        matching_in_hand = sum(
            1
            for c in player.hand
            if c is not card
            and ((c.definition.tags if not c.markers else c.tags) & t_tags)
        )
        matching_played = sum(
            1
            for c in player.played_today
            if c is not card
            and ((c.definition.tags if not c.markers else c.tags) & t_tags)
        )
        abundance = _tag_abundance(t_tags)
        future_factor = min(days_left, 3) * 0.50 * abundance
        val += (
            matching_in_hand * 0.95
            + matching_played * 0.50
            + future_factor
        ) * prof.signed_match_benefit
    elif prof.is_tomorrow_scaler:
        if days_left >= 1:
            t_tags = prof.target_tags
            matching_in_hand = sum(
                1
                for c in player.hand
                if c is not card
                and ((c.definition.tags if not c.markers else c.tags) & t_tags)
            )
            abundance = _tag_abundance(t_tags)
            val += (
                matching_in_hand * 0.65 + 0.40 * abundance
            ) * prof.signed_match_benefit
    elif prof.is_single_buff:
        t_tags = prof.target_tags
        has_match = any(
            c is not card
            and ((c.definition.tags if not c.markers else c.tags) & t_tags)
            for c in player.hand
        )
        abundance = _tag_abundance(t_tags)
        val += prof.signed_match_benefit * (
            0.78 if has_match else 0.25 * abundance
        )
    elif prof.signed_match_benefit < 0.0:
        val += prof.signed_match_benefit * 1.35

    if prof.is_global_after_scaler:
        cheap_cards = sum(
            1
            for c in player.hand
            if (c.definition.cost if not c.markers else c.effective_cost) <= 1
        )
        val += (
            prof.signed_match_benefit * 0.85
            + cheap_cards * 0.55
            + min(days_left, 2) * 0.50
        )
    elif title == "Bring a Friend":
        zero_cards = sum(
            1
            for c in player.hand
            if c is not card
            and (c.definition.cost if not c.markers else c.effective_cost) == 0
        )
        val += zero_cards * 1.45
    elif title == "Evening on the Dock":
        val += 3.5
    elif title == "Couch Tube":
        val += 2.6

    if prof.goes_to_tomorrow and days_left >= 1:
        val += (
            prof.tomorrow_start_fun * 0.9 + prof.tomorrow_start_energy * 1.15
        )
        if title == "Photo Shoot":
            val += 3.5

    if cost == 0 and eff_base_fun >= 0:
        val += 0.65
    elif cost == 1 and eff_base_fun >= 0:
        val += 0.25

    return val


_quick_card_desirability = _beryllium_quick_card_desirability


def _fast_argon_wanted_tags_for_player(player: PlayerState) -> dict[str, float]:
    """Exact ArgonBrain _wanted_tags_for_player avoiding property getter overhead."""
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
        prof = (
            _BERYLLIUM_DEFN_CACHE.get(id(card.definition))
            if not card.markers
            else None
        )
        if prof is None:
            prof = _build_profile(card)
        if prof.signed_match_benefit != 0.0:
            for t in prof.target_tags:
                weights[t] = (
                    weights.get(t, 0.0) + prof.signed_match_benefit * 1.15
                )
        if card.markers:
            chosen_tag = card.markers.get("chosen_tag")
            if isinstance(chosen_tag, str):
                weights[chosen_tag] = weights.get(chosen_tag, 0.0) + 1.5

    for card in player.played_today:
        prof = (
            _BERYLLIUM_DEFN_CACHE.get(id(card.definition))
            if not card.markers
            else None
        )
        if prof is None:
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
        prof = (
            _BERYLLIUM_DEFN_CACHE.get(id(card.definition))
            if not card.markers
            else None
        )
        if prof is None:
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
        c_tags = card.definition.tags if not card.markers else card.tags
        for t in c_tags:
            weights[t] = weights.get(t, 0.0) + 0.15
    return weights


def _exact_played_today_fun(game: Game, player_index: int) -> int:
    """Compute exact total Fun of all cards currently in played_today."""
    player = game.players[player_index]
    prev_scores = game._fun_at_start_of_scoring
    game._fun_at_start_of_scoring = tuple(p.fun for p in game.players)
    try:
        total = sum(
            game.card_fun(player_index, c) for c in player.played_today
        )
    finally:
        game._fun_at_start_of_scoring = prev_scores
    return total


def _beryllium_free_play_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float] | None = None,
) -> float:
    """Estimate value of playing `card` from the Trunk for free, fixing Wakeboard."""
    val = _free_play_card_value(game, player_index, card, wanted_tags)
    if card.title == "Wakeboard":
        player = game.players[player_index]
        char = player.character
        if not (
            char is not None
            and char.zero_fun_tag
            and char.zero_fun_tag in card.tags
        ):
            val += float(
                sum(c.definition.cost for c in player.played_today) + 3
            )
    return val


def _contextual_hand_keep_value(
    game: Game,
    player_index: int,
    card: CardInstance,
    wanted_tags: dict[str, float],
) -> float:
    """Value of keeping `card` in hand right now (recognizing dead Day 6 cards)."""
    player = game.players[player_index]
    can_play_now = card.effective_behavior.can_play(game, player, card)
    cost = game.energy_cost(player_index, card)
    if game.day >= DAY_COUNT:
        if not can_play_now:
            return -5.0
        extra_e = 0
        if _allows_energy_gain_today(game, player):
            extra_e = sum(
                max(
                    0,
                    _beryllium_build_profile(hc).immediate_energy_gain
                    - game.energy_cost(player_index, hc),
                )
                for hc in player.hand
                if hc is not card
                and hc.effective_behavior.can_play(game, player, hc)
                and game.energy_cost(player_index, hc) <= player.energy
            )
            if any(
                hc is not card
                and _effective_card_title(hc) == "Istanbul"
                and hc.effective_behavior.can_play(game, player, hc)
                and game.energy_cost(player_index, hc) <= player.energy + extra_e
                for hc in player.hand
            ):
                extra_e += _istanbul_extra_energy_support(
                    game, player_index, card
                )
        if cost > player.energy + extra_e and not any(
            c.title in {"Wrestle the Kids", "Play With the Kids"}
            for c in (*player.played_today, *player.hand)
        ):
            return -4.0
    val = _quick_card_desirability(game, player_index, card, wanted_tags)
    if not can_play_now:
        val -= 1.25
    elif cost <= player.energy:
        val += 0.45
    return val


def _contextual_copy_value(
    game: Game,
    player_index: int,
    option: CardInstance,
    source_card: CardInstance,
    *,
    pay_cost: bool,
) -> float:
    """Fast, board-aware valuation of copying `option` with `source_card` (Leftovers / Wedding Anniversary)."""
    player = game.players[player_index]
    prof = _beryllium_build_profile(option)
    title = _effective_card_title(option)
    days_left = DAY_COUNT - game.day

    if title in {"Evening Chat", "Early Bedtime"} and (
        player.energy >= 3 or days_left == 0
    ):
        return -8.0
    if title == "Sleep In" and player.energy >= 2:
        return -5.0
    if title in {"Medical Advice", "Scout Other Cottages"} and len(
        player.played_today
    ) >= 3:
        return -4.0

    cost_paid = option.definition.cost if pay_cost else 0
    rem_energy = player.energy - cost_paid
    if rem_energy < 0:
        return -99.0

    prior_cards = game.cards_played_before(player, source_card)

    # 1. Exact immediate/board Fun of the copied card at source_card's position
    char = player.character
    zeroed = bool(
        char is not None
        and char.zero_fun_tag
        and char.zero_fun_tag in source_card.tags
    )
    if zeroed:
        board_fun = 0.0
    else:
        board_fun = float(option.effective_base_fun + prof.immediate_fun_gain)
        if title == "Ice Wine":
            board_fun = 1.0 + 2.0 * sum(
                1 for c in prior_cards if "Food" in c.tags
            )
        elif title == "Cliff Climbing":
            board_fun = 2.0 + 2.0 * sum(
                1 for c in prior_cards if "Outdoors" in c.tags
            )
        elif title == "Outdoor Movie":
            board_fun = 1.0 + 2.0 * sum(
                1 for c in prior_cards if "Relax" in c.tags
            )
        elif title == "Evening on the Dock":
            board_fun = float(len(prior_cards))
        elif title == "Couch Tube":
            board_fun = float(sum(c.definition.cost for c in prior_cards))
        elif title == "New Nos Book Entry":
            board_fun = 5.0 if len(prior_cards) >= 4 else 1.0
        elif title == "Nos Book":
            board_fun = 3.0 if len(prior_cards) >= 1 else 0.0
        elif title == "Campfire":
            board_fun = 2.0 * len(player.tomorrow_cards)
        elif title == "Bubly":
            board_fun = (
                2.0
                if any(c.markers.get("_gave_energy") for c in prior_cards)
                else 0.0
            )
        elif title == "Paint Rocks":
            board_fun = (
                4.0
                if sum(1 for c in prior_cards if "Outdoors" in c.tags) >= 2
                else 2.0
            )
        elif title == "Swimming":
            board_fun = (
                6.0
                if (prior_cards and "Exercise" in prior_cards[-1].tags)
                else 3.0
            )
        elif title == "Trekking Through History":
            board_fun = 2.0 * sum(
                1 for c in prior_cards if "Board Game" in c.tags
            )

    # 2. Immediate Energy gain (valued by whether we have playable cards left in hand)
    if not _allows_energy_gain_today(game, player, source_card):
        imm_e = 0.0
    elif title == "Würstchen":
        imm_e = float(len(player.hand))
    elif title == "Istanbul":
        e_per = _istanbul_energy_per_discard(option)
        has_e_sink = any(
            _effective_card_title(hc) in {"Kayak", "Decaf", "Wakesurf"}
            for hc in player.hand
        ) or any(
            _effective_card_title(pc) in {"Decaf", "Wakesurf"}
            for pc in player.played_today
        )
        disc_est = min(
            3,
            len(player.hand) if has_e_sink else max(0, len(player.hand) - 1),
        )
        imm_e = float(disc_est) * max(
            0.0, float(e_per) - (0.45 if days_left == 0 else 1.15)
        )
    elif title == "Butter Tarts":
        imm_e = float(rem_energy)
    elif title == "Potato Pancakes":
        imm_e = 2.0 * sum(1 for c in prior_cards if "Relax" in c.tags)
    elif title == "Keeper":
        imm_e = float(sum(1 for c in prior_cards if "Outdoors" in c.tags))
    elif title == "Camp Crossroads Dessert":
        imm_e = float(sum(1 for c in prior_cards if "Exercise" in c.tags))
    elif title == "Pudding Chômeur":
        imm_e = 3.0 if not player.hand else 2.0
    elif prof.conditional_energy_gain > 0:
        imm_e = (
            float(prof.conditional_energy_gain)
            if _condition_met_in_played(player, option, prof)
            else 0.0
        )
    else:
        imm_e = float(prof.immediate_energy_gain)

    unspent_hand_cost = sum(
        game.energy_cost(player_index, hc) for hc in player.hand
    )
    energy_need_weight = (
        1.35
        if unspent_hand_cost > rem_energy
        else (
            1.10
            if any(
                _effective_card_title(hc)
                in {"Kayak", "Decaf", "Make Plans", "Long Distance Visitors"}
                for hc in player.hand
            )
            or any(
                _effective_card_title(pc) == "Decaf"
                for pc in player.played_today
            )
            else 0.35
        )
    )
    e_val = imm_e * energy_need_weight

    # 3. Future cards in hand buffed by "After" scalers or single buffs
    after_val = 0.0
    eff_energy = rem_energy + imm_e
    if (
        prof.is_multi_scaler
        and not prof.is_before_scaler
        and prof.signed_match_benefit > 0
    ):
        matching_playable = sum(
            1
            for hc in player.hand
            if (hc.tags & prof.target_tags)
            and game.energy_cost(player_index, hc) <= eff_energy
        )
        after_val += matching_playable * prof.signed_match_benefit * 1.15
    elif prof.is_global_after_scaler:
        playable_after = sum(
            1
            for hc in player.hand
            if max(0, game.energy_cost(player_index, hc) + prof.energy_delta)
            <= eff_energy
        )
        after_val += playable_after * prof.signed_match_benefit * 1.15
    elif prof.is_single_buff and prof.signed_match_benefit > 0:
        has_next = any(
            (not prof.target_tags or (hc.tags & prof.target_tags))
            and max(0, game.energy_cost(player_index, hc) + prof.energy_delta)
            <= eff_energy
            for hc in player.hand
        )
        if has_next:
            after_val += prof.signed_match_benefit * 1.10

    # 4. Card draw / Suitcase pick value
    draw_val = 0.0
    if (
        prof.immediate_cards_gained > 0
        and not prof.penalizes_before
        and not prof.penalizes_after
    ):
        draw_val += min(prof.immediate_cards_gained, 3) * 1.15
        if any(c.title == "Paddleboat" for c in player.played_today):
            draw_val += prof.immediate_cards_gained * 2.0

    # 5. Tomorrow value
    tom_val = 0.0
    if prof.goes_to_tomorrow and days_left >= 1:
        tom_val += (
            prof.tomorrow_start_fun * 0.95 + prof.tomorrow_start_energy * 1.20
        )
        if title == "Photo Shoot":
            tom_val += 5.2
        elif prof.signed_match_benefit != 0.0:
            tom_val += prof.signed_match_benefit * 1.15

    return (
        board_fun + e_val + after_val + draw_val + tom_val - cost_paid * 0.85
    )


def _select_bohnanza_discards(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> tuple[int, ...]:
    """Select cards to trash with Bohnanza by measuring exact board-wide Fun + tomorrow impact."""
    if not options:
        return ()
    player = game.players[player_index]
    sim = _fast_copy_game(game)
    sim_player = sim.players[player_index]
    bohnanza_card = (
        sim_player.played_today[-1] if sim_player.played_today else None
    )
    if bohnanza_card is None or len(sim_player.played_today) <= 1:
        return ()

    days_left = DAY_COUNT - game.day
    hand_tags = {t for hc in player.hand for t in hc.tags}

    def future_aura_value(card: CardInstance) -> float:
        prof = _beryllium_build_profile(card)
        aura = 0.0
        if prof.goes_to_tomorrow and days_left >= 1:
            t_val = prof.tomorrow_start_fun + prof.tomorrow_start_energy * 1.20
            if card.title == "Photo Shoot":
                t_val += 5.5
            elif card.title == "Date's First Nos":
                t_val += 6.0
            elif prof.signed_match_benefit != 0.0:
                t_val += prof.signed_match_benefit * 1.4
            aura += max(0.0, t_val)
        if card.title == "Chuck a Frisbee" and days_left >= 1:
            aura += 2.2
        elif card.title == "Scout Other Cottages" and days_left >= 1:
            aura += 3.2
        if player.hand and player.energy > 0:
            if prof.is_multi_scaler and not prof.is_before_scaler:
                if prof.target_tags & hand_tags:
                    aura += max(0.0, prof.signed_match_benefit * 1.5)
            elif prof.is_global_after_scaler:
                aura += max(0.0, prof.signed_match_benefit * 1.8)
            elif card.title == "Christmas Name Draw":
                aura += 2.5
            elif card.title == "Paddleboat" and any(
                hc.title in _ACQUIRER_TITLES for hc in player.hand
            ):
                aura += 3.0
        for hc in player.hand:
            if sim.energy_cost(player_index, hc) > player.energy:
                continue
            if hc.title == "Couch Tube":
                aura += card.definition.cost * 0.9
            elif hc.title == "Evening on the Dock":
                aura += 0.9
            elif (
                hc.title == "New Nos Book Entry"
                and len(sim_player.played_today) <= 5
            ):
                aura += 0.8
            elif hc.title == "Cliff Climbing" and "Outdoors" in card.tags:
                aura += 1.8
            elif hc.title == "Ice Wine" and "Food" in card.tags:
                aura += 1.8
            elif hc.title == "Outdoor Movie" and "Relax" in card.tags:
                aura += 1.8
            elif hc.title == "Bubly" and card.markers.get("_gave_energy"):
                aura += 1.8
        return aura

    chosen: list[int] = []
    option_clones = list(
        sim_player.played_today[
            : min(len(options), len(sim_player.played_today) - 1)
        ]
    )
    base_board_fun = _exact_played_today_fun(sim, player_index)

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
        new_board_fun = _exact_played_today_fun(sim, player_index)
        net_gain = (new_board_fun - base_board_fun) - aura
        if net_gain > 0.25:
            chosen.append(idx)
            base_board_fun = new_board_fun
        else:
            sim_player.played_today.insert(pos, clone)
            bohnanza_card.markers["energy_cubes"] = (
                int(bohnanza_card.markers.get("energy_cubes", 0)) - 1
            )

    return tuple(chosen)


def _evaluate_istanbul_discards(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> list[tuple[float, tuple[int, ...]]]:
    """Rank candidate discard index tuples for Istanbul by today's playable plan + tomorrow reserve."""
    if not options:
        return [(0.0, ())]
    sim = _fast_copy_game(game)
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
    wanted_tags = _wanted_tags_for_player(player)
    n = len(options)

    sim_opts = [
        player.hand[i]
        if i < len(player.hand)
        and player.hand[i].instance_id == options[i].instance_id
        else _clone_card(options[i])
        for i in range(n)
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

    max_opp_hand = max(
        (
            len(opp.hand)
            for idx, opp in enumerate(sim.players)
            if idx != player_index
        ),
        default=0,
    )
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
        m = 1.0
        for bp in baf_positions:
            if bp < pos:
                m *= 2.0
        if pos >= 1 and _effective_card_title(orig_played[pos - 1]) == "Work Call":
            m *= 2.0
        return m

    future_mult = float(2 ** len(baf_positions))
    decaf_played = sum(
        _prior_card_mult(pos, c)
        for pos, c in enumerate(orig_played)
        if _effective_card_title(c) == "Decaf"
    )
    wrestle_played = sum(
        _prior_card_mult(pos, c)
        for pos, c in enumerate(orig_played)
        if _effective_card_title(c) == "Wrestle the Kids"
    )
    pwtk_played = sum(
        _prior_card_mult(pos, c)
        for pos, c in enumerate(orig_played)
        if _effective_card_title(c) == "Play With the Kids"
    )
    wakesurf_played = sum(
        _prior_card_mult(pos, c)
        for pos, c in enumerate(orig_played)
        if _effective_card_title(c) == "Wakesurf"
    )
    slalom_played = sum(
        _prior_card_mult(pos, c)
        for pos, c in enumerate(orig_played)
        if _effective_card_title(c) == "Slalom Start"
    )
    paddleboat_played = sum(
        _prior_card_mult(pos, c)
        for pos, c in enumerate(orig_played)
        if _effective_card_title(c) == "Paddleboat"
    )
    prior_gave_energy = pending_post_e > 0 or any(
        c.markers.get("_gave_energy") for c in orig_played
    )

    post_draws = 0
    s_tags = source_card.tags
    for pc in orig_played[:-1]:
        if _effective_card_title(pc) == "Euchre Awards Ceremony" and (
            s_tags & {"Food", "Board Game"}
        ):
            post_draws += 1
    for tc in player.tomorrow_cards:
        if _effective_card_title(tc) == "Photo Shoot" and (
            s_tags & {"Social", "Relax"}
        ):
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
    try:
        player.energy = 0
        player.hand = neutral_pair
        base_prior_fun = sum(sim.card_fun(player_index, pc) for pc in orig_played)

        for c in sim_opts:
            prof = _beryllium_build_profile(c)
            title = _effective_card_title(c)
            player.hand = sim_hand
            player.energy = eff_start_energy
            can_p = c.effective_behavior.can_play(sim, player, c)
            cost_c = sim.energy_cost(player_index, c)

            player.energy = max(eff_start_energy, 30)
            f_des = _beryllium_quick_card_desirability(
                sim, player_index, c, wanted_tags, cost=cost_c, prof=prof
            )

            had_all_tags = "_all_tags" in c.markers
            player.played_today.append(c)
            try:
                player.energy = 0
                player.hand = neutral_pair
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
                if title not in {"Wakesurf", "Slalom Start"}:
                    player.hand = []
                own_fun = sim.card_fun(player_index, c)
                base_board_fun = float(own_fun + prior_delta)
            finally:
                player.played_today.pop()
                if not had_all_tags:
                    c.markers.pop("_all_tags", None)
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
                    for idx, opp in enumerate(sim.players)
                    if idx != player_index
                ):
                    g_val += 2.2 * c_mult
            elif title in {"Quacks", "Poker"}:
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

    def ramp_gain_for(idx: int, cur_e: int, rem_cards: int) -> int:
        if not allows_e:
            return 0
        t = titles[idx]
        p = profs[idx]
        if t == "Würstchen":
            return max(0, rem_cards - 1)
        if t == "Butter Tarts":
            return max(0, cur_e - costs[idx])
        if t == "Pudding Chômeur":
            return 3
        if p.conditional_energy_gain > 0 and p.immediate_energy_gain == 0:
            if t == "Potato Pancakes":
                return 2 * sum(
                    1 for pc in orig_played if "Relax" in pc.tags
                )
            if t == "Keeper":
                return sum(
                    1 for pc in orig_played if "Outdoors" in pc.tags
                )
            if t == "Camp Crossroads Dessert":
                return sum(
                    1 for pc in orig_played if "Exercise" in pc.tags
                )
            return (
                p.conditional_energy_gain
                if _condition_met_in_played(player, sim_opts[idx], p)
                else 0
            )
        return p.immediate_energy_gain

    candidate_tuples: list[tuple[int, ...]] = []
    if n <= 6:
        for k in range(n + 1):
            candidate_tuples.extend(combinations(range(n), k))
    else:
        seen_tuples: set[tuple[int, ...]] = set()

        def _add_cand(cand: tuple[int, ...]) -> None:
            if cand not in seen_tuples:
                seen_tuples.add(cand)
                candidate_tuples.append(cand)

        for i in range(n):
            _add_cand((i,))
        if n <= 8:
            for cand in combinations(range(n), 2):
                _add_cand(cand)

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

            for cost_w in (0.18, 0.52, 1.05):
                def rank_key(i: int, cw: float = cost_w) -> float:
                    h_val = hold_surpluses[i] * 0.75
                    if can_plays[i] and costs[i] <= reach_e:
                        rg = ramp_gain_for(i, e_k, m_kept)
                        net_e_bonus = max(0, rg - costs[i]) * 1.15
                        t = titles[i]
                        cm = card_mults[i]
                        extra_sink = 0.0
                        if t == "Kayak":
                            extra_sink = min(max(0, e_k - costs[i]), 8) * 0.85 * cm
                        elif t == "Decaf":
                            extra_sink = min(max(0, e_k - costs[i]), 8) * 0.80 * cm
                        elif (
                            t == "Bubly"
                            and not prior_gave_energy
                            and k > 0
                            and e_per > 0
                        ):
                            extra_sink = 2.0 * cm
                        p_val = (
                            gross_play_vals[i]
                            + extra_sink
                            + net_e_bonus
                            - costs[i] * cw
                        )
                        return max(h_val, p_val)
                    if days_left >= 1:
                        return h_val
                    return -10.0 + 0.01 * funded_des[i]

                ordered = sorted(range(n), key=rank_key)
                _add_cand(tuple(sorted(ordered[:k])))

    slot_weights = (1.0, 0.92, 0.50, 0.25)

    def score_discard_tuple(disc_tuple: tuple[int, ...]) -> float:
        k = len(disc_tuple)
        disc_set = set(disc_tuple)
        base_kept = [i for i in range(n) if i not in disc_set]
        start_e = eff_start_energy + k * e_per
        start_gave_e = prior_gave_energy or (k > 0 and e_per > 0)

        best_mode_score = float("-inf")
        for peak_first in (False, True):
            cur_e = start_e
            cur_gave_e = start_gave_e
            unplayed = list(base_kept)
            today_val = 0.0
            decaf_w = decaf_played
            wrestle_w = wrestle_played
            pwtk_w = pwtk_played
            wakesurf_w = wakesurf_played
            slalom_w = slalom_played
            kayak_w = 0.0
            extra_draws = post_draws

            # 1. Play strictly net-Energy-positive ramps when profitable or needed
            progressed = True
            while progressed and unplayed:
                progressed = False
                other_cost_sum = sum(
                    costs[j] for j in unplayed if can_plays[j]
                )
                has_sink_card = (
                    decaf_w > 0
                    or any(titles[j] in {"Kayak", "Decaf"} for j in unplayed)
                    or extra_draws > 0
                )
                for idx in list(unplayed):
                    if not can_plays[idx] or costs[idx] > cur_e:
                        continue
                    rg = ramp_gain_for(idx, cur_e, len(unplayed))
                    if rg <= costs[idx]:
                        continue
                    need_e = (other_cost_sum > cur_e) or has_sink_card
                    h_keep = hold_surpluses[idx] * 0.55
                    if not need_e and gross_play_vals[idx] < h_keep:
                        continue
                    if titles[idx] == "Pudding Chômeur":
                        others = [j for j in unplayed if j != idx]
                        if others:
                            fodder = min(others, key=lambda j: hold_surpluses[j])
                            if (
                                not need_e
                                and gross_play_vals[idx]
                                < h_keep + hold_surpluses[fodder] * 0.55
                            ):
                                continue
                            unplayed.remove(fodder)
                    cur_e = cur_e - costs[idx] + rg
                    cur_gave_e = True
                    today_val += gross_play_vals[idx]
                    extra_draws += min(profs[idx].immediate_cards_gained, 2)
                    unplayed.remove(idx)
                    progressed = True
                    break

            # 2. Pack remaining playable cards in `unplayed`
            def play_Card_effective_val(idx: int, avail_e: int) -> float:
                v = gross_play_vals[idx]
                t = titles[idx]
                cm = card_mults[idx]
                if t == "Bubly" and cur_gave_e and not prior_gave_energy:
                    v += 2.0 * cm
                elif t == "Kayak":
                    v += min(max(0, avail_e - costs[idx]), 10) * 0.85 * cm
                elif t == "Decaf":
                    v += min(max(0, avail_e - costs[idx]), 10) * 0.80 * cm
                return v

            playable_rem = [j for j in unplayed if can_plays[j]]
            if peak_first:
                playable_rem.sort(
                    key=lambda j: play_Card_effective_val(j, cur_e)
                    - hold_surpluses[j] * 0.55,
                    reverse=True,
                )
            else:
                playable_rem.sort(
                    key=lambda j: (
                        play_Card_effective_val(j, cur_e)
                        - hold_surpluses[j] * 0.55
                    )
                    / max(1, costs[j]),
                    reverse=True,
                )

            for idx in playable_rem:
                if idx not in unplayed or costs[idx] > cur_e:
                    continue
                t = titles[idx]
                cm = card_mults[idx]
                c_val = gross_play_vals[idx]
                if t == "Bubly" and cur_gave_e and not prior_gave_energy:
                    c_val += 2.0 * cm
                proj_sink = 0.0
                if t == "Kayak":
                    proj_sink = max(0, cur_e - costs[idx]) * cm
                elif t == "Decaf":
                    proj_sink = max(0, cur_e - costs[idx]) * cm
                elif t == "Wrestle the Kids":
                    proj_sink = max(0, len(unplayed) - 1) * cm
                elif t == "Play With the Kids" and (len(unplayed) - 1) > max_opp_hand:
                    proj_sink = 3.0 * cm
                elif t == "Wakesurf" and (len(unplayed) - 1) == 0:
                    proj_sink = 5.0 * cm
                elif t == "Slalom Start" and (len(unplayed) - 1) == 1:
                    proj_sink = 4.0 * cm

                # Account for losing 1 point per active Wrestle / Decaf cost
                eod_hand_cost = wrestle_w
                eod_decaf_cost = (
                    costs[idx] * decaf_w if decaf_w > 0 else 0.0
                )
                hold_opp = (
                    hold_surpluses[idx] * 0.55 if days_left >= 1 else 0.0
                )
                if (
                    c_val + proj_sink - eod_hand_cost - eod_decaf_cost
                    <= hold_opp - 1e-6
                ):
                    continue

                cur_e -= costs[idx]
                today_val += c_val
                unplayed.remove(idx)
                extra_draws += min(profs[idx].immediate_cards_gained, 2)
                if t == "Kayak":
                    kayak_w = max(kayak_w, cm)
                elif t == "Decaf":
                    decaf_w += cm
                elif t == "Wrestle the Kids":
                    wrestle_w += cm
                elif t == "Play With the Kids":
                    pwtk_w += cm
                elif t == "Wakesurf":
                    wakesurf_w += cm
                elif t == "Slalom Start":
                    slalom_w += cm

            # 3. Leftover Energy sinks (drawn Suitcase/Trunk cards, Kayak, Decaf)
            rem_sink_e = cur_e
            if extra_draws > 0 and rem_sink_e > 0:
                used_draw_e = min(rem_sink_e, 2 * extra_draws)
                today_val += used_draw_e * 1.05
                rem_sink_e -= used_draw_e
            if rem_sink_e > 0 and (kayak_w > 0 or decaf_w > 0):
                per_e = max(kayak_w, decaf_w)
                today_val += rem_sink_e * per_e

            # 4. End-of-day hand-count bonuses & tomorrow reserve value
            m_rem = len(unplayed)
            if wrestle_w > 0:
                today_val += wrestle_w * m_rem
            if pwtk_w > 0 and m_rem > max_opp_hand:
                today_val += pwtk_w * 3.0
            if wakesurf_w > 0 and m_rem == 0:
                today_val += wakesurf_w * 5.0
            if slalom_w > 0 and m_rem == 1:
                today_val += slalom_w * 4.0

            reserve_val = 0.0
            if days_left >= 1 and unplayed:
                rem_surpluses = sorted(
                    (hold_surpluses[j] for j in unplayed), reverse=True
                )
                reserve_val = 0.75 * sum(
                    s * (slot_weights[pos] if pos < len(slot_weights) else 0.08)
                    for pos, s in enumerate(rem_surpluses)
                )

            mode_score = today_val + reserve_val - 0.02 * k
            if mode_score > best_mode_score:
                best_mode_score = mode_score

        return best_mode_score

    scored = [(score_discard_tuple(cand), cand) for cand in candidate_tuples]
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return scored


def _select_istanbul_discards(
    game: Game,
    player_index: int,
    options: Sequence[CardInstance],
) -> tuple[int, ...]:
    """Select cards to discard for Istanbul by optimizing post-discard playable value + reserve."""
    return _evaluate_istanbul_discards(game, player_index, options)[0][1]


class _BerylliumEvalAI(_EvalPlayerAI):
    """Upgraded rollout policy used inside BerylliumBrain simulation clones."""

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
            cutlery_cards = [
                c for c in options if c.title == "Assorted Cutlery"
            ]
            if cutlery_cards:
                cutlery = cutlery_cards[0]
                cutlery_cost = game.energy_cost(player_index, cutlery)
                if cutlery_cost <= player.energy:
                    rem_e = player.energy - cutlery_cost
                    best_i: int | None = None
                    best_net = float("-inf")
                    for i, cand in enumerate(options):
                        if cand is cutlery and len(cutlery_cards) == 1:
                            continue
                        fp_val = _beryllium_free_play_value(
                            game, player_index, cand, wanted_tags
                        )
                        cand_cost = game.energy_cost(player_index, cand)
                        cand_des = _contextual_hand_keep_value(
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
                            opp_cost = max(0.0, (cand_des - 0.5) * 0.45)
                        net = fp_val - opp_cost
                        if net > best_net:
                            best_net = net
                            best_i = i
                    if best_i is not None and best_net >= max(
                        1.5, cutlery_cost * 0.90
                    ):
                        return best_i

            bcm_cards = [
                c for c in options if c.title == "Bougie Coffee Machine"
            ]
            if bcm_cards:
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
                        fp_val = _beryllium_free_play_value(
                            game, player_index, cand, wanted_tags
                        )
                        cand_cost = game.energy_cost(player_index, cand)
                        cand_des = _contextual_hand_keep_value(
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
                                else max(0.0, (cand_des - 0.5) * 0.45)
                            )
                        )
                        net = fp_val - opp_cost
                        if net > best_f_net:
                            best_f_net = net
                            best_f_i = i
                    if best_f_i is not None and best_f_net >= 1.5:
                        return best_f_i

        return min(
            range(len(options)),
            key=lambda i: _contextual_hand_keep_value(
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
            return _select_istanbul_discards(game, player_index, options)

        if last_played == "Navy SEALing":
            wanted_tags = _wanted_tags_for_player(player)
            ranked = sorted(
                range(len(options)),
                key=lambda i: _contextual_hand_keep_value(
                    game, player_index, options[i], wanted_tags
                ),
            )
            is_doubled = any(
                c.title == "Bring a Friend"
                for c in player.played_today[:-1]
            ) or (
                len(player.played_today) >= 2
                and player.played_today[-2].title == "Work Call"
            )
            thresh = 4.2 if is_doubled else 2.2
            return tuple(
                i
                for i in ranked
                if _contextual_hand_keep_value(
                    game, player_index, options[i], wanted_tags
                )
                < thresh
            )

        return super().choose_cards_to_discard(game, player_index, options)

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
            return super().choose_card_to_copy(game, player_index, options)
        pay_cost = (
            type(source_card.effective_behavior).__name__ != "LeftoversBehavior"
        )
        return max(
            range(len(options)),
            key=lambda i: _contextual_copy_value(
                game,
                player_index,
                options[i],
                source_card,
                pay_cost=pay_cost,
            ),
        )

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        player = game.players[player_index]
        if "Cities and Knights" in action or "Epic Prank" in action:
            eligible = (
                [c for c in player.hand if "Item" in c.tags]
                if "Epic Prank" in action
                else list(player.hand)
            )
            if not eligible:
                return False
            if (
                any(c.title == "Slalom Start" for c in player.played_today)
                and len(player.hand) == 1
            ):
                return False
            wanted_tags = _wanted_tags_for_player(player)
            worst_val = min(
                _contextual_hand_keep_value(game, player_index, c, wanted_tags)
                for c in eligible
            )
            is_doubled = any(
                c.title == "Bring a Friend" for c in player.played_today[:-1]
            ) or (
                len(player.played_today) >= 2
                and player.played_today[-2].title == "Work Call"
            )
            base_thresh = 4.5 if "Epic Prank" in action else 2.5
            if is_doubled:
                base_thresh *= 1.8
            if any(c.title == "Wakesurf" for c in player.played_today) or (
                any(c.title == "Slalom Start" for c in player.played_today)
                and len(player.hand) == 2
            ):
                base_thresh += 2.5
            return worst_val < base_thresh
        return super().choose_optional_action(game, player_index, action)


class _FastArgonEvalAI(_EvalPlayerAI):
    """Exact _EvalPlayerAI with fast-pathed tag and desirability evaluation."""

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> Sequence[int]:
        if options:
            player = game.players[player_index]
            if (
                player.played_today
                and _effective_card_title(player.played_today[-1]) == "Istanbul"
            ):
                return _select_istanbul_discards(game, player_index, options)
        return super().choose_cards_to_discard(game, player_index, options)

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if not options:
            raise ValueError("No suitcase card option was supplied")
        player = game.players[player_index]
        wanted_tags = _fast_argon_wanted_tags_for_player(player)
        own_markers = _own_suitcase_markers(player)
        best_idx = 0
        best_score = float("-inf")
        for idx, card in enumerate(options):
            score = _beryllium_quick_card_desirability(
                game, player_index, card, wanted_tags
            )
            if own_markers and any(m in card.markers for m in own_markers):
                score -= 2.8
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
        wanted_tags = _fast_argon_wanted_tags_for_player(player)
        best_idx = playable_hand_indices[0]
        best_key = (99, 0.0)
        for hand_idx in playable_hand_indices:
            card = player.hand[hand_idx]
            prof = (
                _BERYLLIUM_DEFN_CACHE.get(id(card.definition))
                if not card.markers
                else None
            )
            if prof is None:
                prof = _build_profile(card)
            est = _beryllium_quick_card_desirability(
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


_FAST_ARGON_EVAL_SINGLETON = _FastArgonEvalAI()
_BERYLLIUM_EVAL_AI_SINGLETON = _BerylliumEvalAI()
_BERYLLIUM_SIM_AIS = [
    _BERYLLIUM_EVAL_AI_SINGLETON,
    _BERYLLIUM_EVAL_AI_SINGLETON,
    _BERYLLIUM_EVAL_AI_SINGLETON,
    _BERYLLIUM_EVAL_AI_SINGLETON,
]
_ARGON_EVAL_SIM_AIS = [
    _FAST_ARGON_EVAL_SINGLETON,
    _FAST_ARGON_EVAL_SINGLETON,
    _FAST_ARGON_EVAL_SINGLETON,
    _FAST_ARGON_EVAL_SINGLETON,
]


class _NoOpCounter:
    """Zero-allocation dummy counter for simulation clones."""

    __slots__ = ()

    def __getitem__(self, key: Any) -> int:
        return 0

    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def copy(self) -> _NoOpCounter:
        return self


_NOOP_COUNTER: Any = _NoOpCounter()
_BERYLLIUM_SIM_GAME_CLS: type | None = None


def _get_beryllium_sim_game_cls() -> type:
    global _BERYLLIUM_SIM_GAME_CLS
    if _BERYLLIUM_SIM_GAME_CLS is not None:
        return _BERYLLIUM_SIM_GAME_CLS
    from thenos.game import Game

    class _BerylliumSimGame(Game):
        """Simulation Game subclass with copy-on-draw and fast-path cost/fun evaluation."""

        def _draw_from_trunk(self) -> CardInstance:
            if not self.trunk:
                if not self.discard:
                    from thenos.game import DeckExhaustedError

                    raise DeckExhaustedError(
                        "Both the Trunk and discard pile are empty"
                    )
                self.trunk = self.discard
                self.discard = []
                self.rng = _clone_random(self.rng)
                self.rng.shuffle(self.trunk)
            card = self.trunk.pop()
            return CardInstance(
                card.instance_id,
                card.definition,
                card.is_tomorrow,
                dict(card.markers) if card.markers else {},
            )

        def play_card(self, player_index: int, hand_index: int) -> CardInstance:
            player = self.players[player_index]
            if player.asleep:
                raise ValueError(f"{player.name} has already gone to bed")
            if hand_index < 0 or hand_index >= len(player.hand):
                raise ValueError(f"Invalid hand index: {hand_index}")
            raw_card = player.hand[hand_index]
            c_beh = (
                raw_card.definition.behavior
                if not raw_card.markers
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

            player.energy -= cost
            player.hand.pop(hand_index)
            card = CardInstance(
                raw_card.instance_id,
                raw_card.definition,
                raw_card.is_tomorrow,
                dict(raw_card.markers) if raw_card.markers else {},
            )
            if (
                raw_card.definition.title
                if not raw_card.markers
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
                    s_beh = (
                        source.definition.behavior
                        if not source.markers
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
                s_beh = (
                    source.definition.behavior
                    if not source.markers
                    else source.effective_behavior
                )
                if type(s_beh).on_card_play is not CardBehavior.on_card_play:
                    s_beh.on_card_play(self, player, source, card)
            return card

        def play_card_for_effect(
            self,
            player_index: int,
            card: CardInstance,
            *,
            cost_adjustment: int = 0,
            pay_energy: bool = False,
            ignore_restrictions: bool = False,
        ) -> CardInstance:
            player = self.players[player_index]
            is_lys = (
                card.definition.title
                if not card.markers
                else _effective_card_title(card)
            ) == "Last Year's Shirt"
            if card in player.hand:
                idx = player.hand.index(card)
                if is_lys:
                    player.hand = [
                        CardInstance(
                            c.instance_id,
                            c.definition,
                            c.is_tomorrow,
                            dict(c.markers) if c.markers else {},
                        )
                        for c in player.hand
                    ]
                    card = player.hand[idx]
                else:
                    cloned = CardInstance(
                        card.instance_id,
                        card.definition,
                        card.is_tomorrow,
                        dict(card.markers) if card.markers else {},
                    )
                    player.hand[idx] = cloned
                    card = cloned
            elif is_lys and player.hand:
                player.hand = [
                    CardInstance(
                        c.instance_id,
                        c.definition,
                        c.is_tomorrow,
                        dict(c.markers) if c.markers else {},
                    )
                    for c in player.hand
                ]
            return super().play_card_for_effect(
                player_index,
                card,
                cost_adjustment=cost_adjustment,
                pay_energy=pay_energy,
                ignore_restrictions=ignore_restrictions,
            )

        def copy_card_effect(
            self,
            player_index: int,
            source: CardInstance,
            destination: CardInstance,
            *,
            pay_source_cost: bool = True,
        ) -> None:
            player = self.players[player_index]
            if (
                source.definition.title
                if not source.markers
                else _effective_card_title(source)
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
            super().copy_card_effect(
                player_index,
                source,
                destination,
                pay_source_cost=pay_source_cost,
            )

        def energy_cost(self, player_index: int, card: CardInstance) -> int:
            player = self.players[player_index]
            temporary_all_tags = (
                "_all_tags" not in card.markers
                and self._grants_all_tags(player, card)
            )
            if temporary_all_tags:
                card.markers["_all_tags"] = True
            try:
                c_defn = (
                    card.definition
                    if not card.markers
                    else card.effective_definition
                )
                cost = c_defn.cost
                c_tags = (
                    card.tags
                    if (temporary_all_tags or card.markers)
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
                for source in player.tomorrow_cards:
                    s_beh = (
                        source.definition.behavior
                        if not source.markers
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
                    s_beh = (
                        source.definition.behavior
                        if not source.markers
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
                    card.markers.pop("_all_tags", None)

        def playable_hand_indices(self, player_index: int) -> list[int]:
            player = self.players[player_index]
            if player.asleep:
                return []
            p_energy = player.energy
            res: list[int] = []
            for index, card in enumerate(player.hand):
                c_beh = (
                    card.definition.behavior
                    if not card.markers
                    else card.effective_behavior
                )
                if (
                    type(c_beh).can_play is not CardBehavior.can_play
                    and not c_beh.can_play(self, player, card)
                ):
                    continue
                if self.energy_cost(player_index, card) <= p_energy:
                    res.append(index)
            return res

        def card_fun(self, player_index: int, target: CardInstance) -> int:
            if target.is_tomorrow:
                return 0
            player = self.players[player_index]
            if any(
                played_card is target for played_card in player.played_today
            ):
                self._sync_played_today_tags(player)
            t_defn = (
                target.definition
                if not target.markers
                else target.effective_definition
            )
            t_beh = t_defn.behavior
            if type(t_beh).fun_value is CardBehavior.fun_value:
                value = t_defn.base_fun
            else:
                value = t_beh.fun_value(self, player, target)
            t_tags = target.tags if target.markers else t_defn.tags
            if self.daily_condition is not None:
                value = self.daily_condition.modify_fun(t_tags, value)
            if player.character is not None:
                value = player.character.modify_fun(t_tags, value)
            for source in player.tomorrow_cards:
                s_beh = (
                    source.definition.behavior
                    if not source.markers
                    else source.effective_behavior
                )
                if (
                    type(s_beh).modify_tomorrow_fun
                    is not CardBehavior.modify_tomorrow_fun
                ):
                    value = s_beh.modify_tomorrow_fun(
                        self, player, source, target, value
                    )
            for source in player.played_today:
                s_beh = (
                    source.definition.behavior
                    if not source.markers
                    else source.effective_behavior
                )
                if type(s_beh).modify_fun is not CardBehavior.modify_fun:
                    value = s_beh.modify_fun(
                        self, player, source, target, value
                    )
            if player.character is not None:
                value = player.character.finalize_fun(t_tags, value)
            return value

    _BERYLLIUM_SIM_GAME_CLS = _BerylliumSimGame
    return _BerylliumSimGame


def _beryllium_fast_copy_game(src: Game, ais: list[Any]) -> Game:
    sim_cls = _BERYLLIUM_SIM_GAME_CLS or _get_beryllium_sim_game_cls()
    sim = object.__new__(sim_cls)
    sim.rng = src.rng if type(src) is sim_cls else _clone_random(src.rng)
    sim.ais = ais
    card_map: dict[int, CardInstance] = {}
    cards_with_markers: list[CardInstance] = []

    def clone_tracked(card: CardInstance) -> CardInstance:
        markers = card.markers
        if markers:
            cloned = CardInstance(
                card.instance_id,
                card.definition,
                card.is_tomorrow,
                dict(markers),
            )
            card_map[id(card)] = cloned
            cards_with_markers.append(cloned)
            return cloned
        cloned = CardInstance(
            card.instance_id,
            card.definition,
            card.is_tomorrow,
            {},
        )
        card_map[id(card)] = cloned
        return cloned

    is_beryllium_sim = type(src) is sim_cls
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
                if is_beryllium_sim
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
            played_today=[clone_tracked(c) for c in p.played_today],
            tomorrow_cards=[clone_tracked(c) for c in p.tomorrow_cards],
            asleep=p.asleep,
            skipped_turns=p.skipped_turns,
            acquired_cards=_NOOP_COUNTER,
            picked_cards=_NOOP_COUNTER,
        )
        for p in src.players
    ]
    sim.suitcase = [clone_tracked(c) for c in src.suitcase]
    if is_beryllium_sim:
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
    for cloned in cards_with_markers:
        m = cloned.markers
        tc = m.get("target_card")
        if isinstance(tc, CardInstance):
            mapped = card_map.get(id(tc))
            if mapped is not None:
                m["target_card"] = mapped
        sc = m.get("suitcase_target")
        if isinstance(sc, CardInstance):
            mapped = card_map.get(id(sc))
            if mapped is not None:
                m["suitcase_target"] = mapped
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


def _beryllium_copy_game(src: Game) -> Game:
    return _beryllium_fast_copy_game(src, _BERYLLIUM_SIM_AIS)


class BerylliumBrain(ArgonBrain):
    """Compute-scaled lookahead policy with full draw-phase & next-day rollouts."""

    BEAM_WIDTH = 22
    DRAFT_BEAM_WIDTH = 8
    BRANCH_LIMIT = 8
    MAX_TURN_PLAYS = 14
    RESERVE_TOMORROW_WEIGHT = 0.75
    FIRST_TO_BED_BONUS = 0.45
    DRAFT_SAMPLES = 3
    PLAY_SAMPLES = 3

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        _evaluation_depth: int = 0,
    ) -> None:
        super().__init__(rng, _evaluation_depth=_evaluation_depth)
        self._draw_phase_tracker: tuple[int, int, int, int] | None = None
        self._reserve_eval_cache: dict[tuple, float] = {}

    def _prepare_observable_game(self, game: Game, player_index: int) -> Game:
        sim = super()._prepare_observable_game(game, player_index)
        sim.__class__ = _BERYLLIUM_SIM_GAME_CLS or _get_beryllium_sim_game_cls()
        sim.ais = _BERYLLIUM_SIM_AIS
        for p in sim.players:
            p.acquired_cards = _NOOP_COUNTER
            p.picked_cards = _NOOP_COUNTER
        return sim

    def _simulate_play_step(
        self, state: Game, player_index: int, hand_idx: int
    ) -> Game:
        child = _beryllium_fast_copy_game(state, _ARGON_EVAL_SIM_AIS)
        card_title = child.players[player_index].hand[hand_idx].definition.title
        known_top = _get_sim_known_trunk_top(child)
        if not known_top:
            if card_title == "Assorted Cutlery" and child.trunk:
                best_t_idx = len(child.trunk) - 1
                best_diff = float("inf")
                wanted_tags: dict[str, float] | None = None
                seen_defn_ids: set[int] = set()
                for t_i, tc in enumerate(child.trunk):
                    if not tc.markers:
                        did = id(tc.definition)
                        if did in seen_defn_ids:
                            continue
                        seen_defn_ids.add(did)
                        t_prof = _BERYLLIUM_DEFN_CACHE.get(did)
                        if t_prof is None:
                            t_prof = _build_profile(tc)
                    else:
                        t_prof = _build_profile(tc)
                    if (
                        t_prof.ends_day_immediately
                        or t_prof.penalizes_before
                        or t_prof.penalizes_after
                    ):
                        continue
                    if wanted_tags is None:
                        wanted_tags = _fast_argon_wanted_tags_for_player(
                            child.players[player_index]
                        )
                    diff = abs(
                        _free_play_card_value(
                            child, player_index, tc, wanted_tags
                        )
                        - 2.4
                    )
                    if diff < best_diff:
                        best_diff = diff
                        best_t_idx = t_i
                child.trunk[-1], child.trunk[best_t_idx] = (
                    child.trunk[best_t_idx],
                    child.trunk[-1],
                )
                child.play_card(player_index, hand_idx)
                child.ais = _BERYLLIUM_SIM_AIS
                return child
            if card_title == "Poker":
                child.play_card(player_index, hand_idx)
                p_card = child.players[player_index].played_today[-1]
                p_card.markers["poker_success"] = False
                p_card.markers["_blind_poker"] = True
                child.ais = _BERYLLIUM_SIM_AIS
                return child
            if card_title == "Quacks":
                child.play_card(player_index, hand_idx)
                q_card = child.players[player_index].played_today[-1]
                q_card.markers["_quacks_outdoors"] = False
                q_card.markers["energy_cubes"] = 1
                q_card.markers["_blind_quacks"] = True
                child.ais = _BERYLLIUM_SIM_AIS
                return child
        child.play_card(player_index, hand_idx)
        child.ais = _BERYLLIUM_SIM_AIS
        return child

    def _remaining_draw_picks_today(
        self, game: Game, player_index: int
    ) -> int:
        """Return how many normal morning Draw Phase picks remain after the current pick."""
        player = game.players[player_index]
        if player.played_today or not game._is_setup:
            return 0
        if game.stats is _get_shared_sim_stats():
            return 0
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
        picks_so_far = total_picks - day_start_picks
        return max(0, 2 - picks_so_far)

    def _state_signature(self, sim: Game, player_index: int) -> tuple:
        p = sim.players[player_index]
        return (
            p.energy,
            p.fun,
            tuple(
                (
                    c.definition.title,
                    _effective_card_title(c),
                    c.markers.get("energy_cubes", 0),
                    getattr(c.markers.get("target_card"), "instance_id", None),
                    c.markers.get("target_player_index"),
                )
                if c.markers
                else (c.definition.title, c.definition.title, 0, None, None)
                for c in p.played_today
            ),
            tuple(sorted(c.definition.title for c in p.hand)),
            tuple(c.definition.title for c in p.tomorrow_cards),
        )

    # ------------------------------------------------------------------
    # Suitcase Drafting: Multi-Sample + Full Draw-Phase Completion
    # ------------------------------------------------------------------

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        if self._evaluation_depth >= 1 or len(options) <= 1:
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_suitcase_card(
                game, player_index, options
            )

        live_player = game.players[player_index]
        own_markers = _own_suitcase_markers(live_player)
        remaining_draw_picks = self._remaining_draw_picks_today(
            game, player_index
        )

        if live_player.played_today and (
            len(live_player.played_today) >= 10
            or len(live_player.hand) >= 11
            or len(options) < 2
        ):
            wanted_tags = _wanted_tags_for_player(live_player)
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
                if own_markers and any(m in option.markers for m in own_markers):
                    score -= 3.0
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        has_stochastic_option = any(
            opt.title in _STOCHASTIC_PLAY_TITLES for opt in options
        )
        if not live_player.played_today:
            num_samples = (
                self.DRAFT_SAMPLES
                if remaining_draw_picks >= 1 or has_stochastic_option
                else 1
            )
            draft_beam = self.DRAFT_BEAM_WIDTH
            draft_max_plays = 9
        else:
            num_samples = 2 if has_stochastic_option else 1
            draft_beam = 6 if len(live_player.played_today) < 6 else 4
            draft_max_plays = 6 if len(live_player.played_today) < 6 else 4

        samples: list[tuple[Game, list[tuple[CardInstance, ...]]]] = []
        for _ in range(num_samples):
            base_sim = self._prepare_observable_game(game, player_index)
            future_pools: list[tuple[CardInstance, ...]] = []
            if remaining_draw_picks > 0 and len(base_sim.trunk) >= SUITCASE_SIZE * (
                remaining_draw_picks + 1
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
                for _ in range(remaining_draw_picks):
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

        wanted_tags = _wanted_tags_for_player(live_player)
        best_indices: list[int] = []
        best_value = float("-inf")

        for idx, option in enumerate(options):
            plan_sum = 0.0
            for base_sim, future_pools in samples:
                sim = _beryllium_copy_game(base_sim)
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
                    pick_i = _BERYLLIUM_EVAL_AI_SINGLETON.choose_suitcase_card(
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

                if not live_player.played_today and len(sim.trunk) >= SUITCASE_SIZE:
                    known_ids = {
                        kc.instance_id for kc in _get_sim_known_trunk_top(sim)
                    }
                    neutral_pool = [
                        i
                        for i, tc in enumerate(sim.trunk)
                        if tc.instance_id not in known_ids
                        and 0.2
                        <= _quick_card_desirability(sim, player_index, tc, {})
                        <= 1.5
                    ]
                    for slot_i in range(len(sim.suitcase)):
                        if not neutral_pool:
                            break
                        t_idx = neutral_pool.pop()
                        sim.suitcase[slot_i], sim.trunk[t_idx] = (
                            _clone_card(sim.trunk[t_idx]),
                            sim.suitcase[slot_i],
                        )

                _, sample_val = self._search_best_day_plan(
                    sim,
                    player_index,
                    beam_width=draft_beam,
                    max_plays=draft_max_plays,
                )
                plan_sum += sample_val

            plan_val = plan_sum / len(samples)
            synergy_val = self._draft_synergy_adjustment(
                samples[0][0], player_index, option, wanted_tags
            )
            if own_markers and any(m in option.markers for m in own_markers):
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
        bonus = super()._draft_synergy_adjustment(
            game, player_index, card, wanted_tags
        )
        player = game.players[player_index]
        days_left = DAY_COUNT - game.day
        title = card.title

        if title == "Euchre Awards Ceremony":
            cheap_cnt = sum(
                1 for c in player.hand if c is not card and c.effective_cost <= 2
            )
            bonus += 1.45 + 0.35 * min(cheap_cnt, 4)
        elif title == "Photo Shoot" and days_left >= 1:
            bonus += 1.40
        elif title == "Rouladen":
            bonus += 1.15
        elif title == "Leftovers":
            good_foods = sum(
                1
                for c in player.hand
                if c is not card and c.title in _HIGH_COPY_FOOD_TITLES
            )
            if good_foods >= 1:
                bonus += 1.65 + 0.45 * min(good_foods, 3)
            elif days_left >= 2:
                bonus += 0.65
        elif title == "Wedding Anniversary":
            if not player.played_today:
                bonus += 2.35
            else:
                bonus += 0.85
        elif title == "Paddleboat":
            acquirers = sum(
                1
                for c in (*player.hand, *player.tomorrow_cards)
                if c is not card and c.title in _ACQUIRER_TITLES
            )
            if acquirers >= 1:
                bonus += 1.60 + 0.70 * min(acquirers, 3)
            elif days_left >= 2:
                bonus += 0.60
        elif title in _ACQUIRER_TITLES and any(
            c is not card and c.title == "Paddleboat"
            for c in (*player.hand, *player.played_today)
        ):
            bonus += 1.35
        elif title == "Colouring":
            if (
                player.character is not None
                and player.character.zero_fun_tag is not None
            ):
                bonus -= 6.0
            else:
                tag_scalers = sum(
                    1
                    for c in player.hand
                    if c is not card
                    and (
                        _beryllium_build_profile(c).is_multi_scaler
                        or _beryllium_build_profile(c).is_before_scaler
                        or c.title
                        in {
                            "Newfangled Tres Fute",
                            "Potato Pancakes",
                            "Keeper",
                            "Camp Crossroads Dessert",
                        }
                    )
                )
                bonus += 1.25 + 1.10 * min(tag_scalers, 3)
        elif title == "Scout Other Cottages" and days_left >= 1:
            if not player.played_today and not any(
                _beryllium_build_profile(c).must_play_first
                for c in player.hand
            ):
                bonus += 1.65
        elif title == "Weird Chip Flavour":
            has_food_or_bubly = any(
                c is not card
                and (
                    c.title
                    in {
                        "Bubly",
                        "High-End Red",
                        "High-End White",
                        "Ice Wine",
                        "Doritos",
                        "Ice Cream Sandwich",
                    }
                )
                for c in player.hand
            )
            bonus += 1.15 if has_food_or_bubly else 0.45
        elif title == "Bubly":
            has_energy_card = any(
                c is not card
                and (
                    _beryllium_build_profile(c).immediate_energy_gain > 0
                    or _beryllium_build_profile(c).conditional_energy_gain > 0
                    or c.title in {"Würstchen", "Butter Tarts", "Istanbul"}
                )
                for c in player.hand
            )
            if has_energy_card:
                bonus += 0.90
        elif title == "Group Dinner":
            bonus -= 0.55
        elif title in {"Cities and Knights", "Poker", "Solo", "Beach Volleyball"}:
            bonus -= 0.65

        return bonus

    # ------------------------------------------------------------------
    # Playing Phase: Multi-Sample & Deep Beam Search + Day d+1 Rollout
    # ------------------------------------------------------------------

    def _has_stochastic_play_options(
        self, game: Game, player_index: int, playable_indices: Sequence[int]
    ) -> bool:
        player = game.players[player_index]
        if any(c.title == "Photo Shoot" for c in player.tomorrow_cards):
            return True
        if any(c.title == "Euchre Awards Ceremony" for c in player.played_today):
            return True
        for idx in playable_indices:
            if idx < len(player.hand):
                if player.hand[idx].title in _STOCHASTIC_PLAY_TITLES:
                    return True
        for hc in player.hand:
            if hc.title in _STOCHASTIC_PLAY_TITLES:
                return True
        return False

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
            played_count < 9
            and self._has_stochastic_play_options(
                game, player_index, playable_hand_indices
            )
        )
        num_samples = self.PLAY_SAMPLES if use_multi else 1
        if num_samples <= 1:
            base_sim = self._prepare_observable_game(game, player_index)
            best_action, _ = self._search_best_day_plan(
                base_sim,
                player_index,
                beam_width=self.BEAM_WIDTH,
                allow_bed_now=allow_bed_now,
                allowed_first_indices=playable_hand_indices,
            )
            return best_action

        samples = [
            self._prepare_observable_game(game, player_index)
            for _ in range(num_samples)
        ]
        first_action_0, _ = self._search_best_day_plan(
            samples[0],
            player_index,
            beam_width=self.BEAM_WIDTH,
            allow_bed_now=allow_bed_now,
            allowed_first_indices=playable_hand_indices,
        )
        candidates: list[int | None] = [
            idx
            for idx in self._ordered_candidate_indices(
                samples[0], player_index, 10
            )
            if idx in playable_hand_indices
        ]
        if (
            first_action_0 is not None
            and first_action_0 in playable_hand_indices
            and first_action_0 not in candidates
        ):
            candidates.insert(0, first_action_0)
        if allow_bed_now:
            candidates.append(None)

        best_actions: list[int | None] = []
        best_mean = float("-inf")
        for cand in candidates:
            total = 0.0
            valid = True
            for s_idx, sim in enumerate(samples):
                if cand is None:
                    bed_sim = _beryllium_copy_game(sim)
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
                                beam_width=12 if s_idx == 0 else 8,
                                max_plays=9 if s_idx == 0 else 6,
                                allow_bed_now=True,
                            )
                        total += val
                    except Exception:
                        valid = False
                        break
            if not valid:
                continue
            mean_val = total / len(samples)
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

        best_action = self._select_root_play_action(
            game,
            player_index,
            playable_hand_indices,
            allow_bed_now=True,
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
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_card_to_play(
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
        return _BERYLLIUM_EVAL_AI_SINGLETON.choose_card_to_play(
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
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_extra_card_to_play(
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

    def _ordered_candidate_indices(
        self,
        sim: Game,
        player_index: int,
        limit: int,
    ) -> list[int]:
        """Return legal hand indices ordered by context-aware synergy role using Beryllium valuations."""
        player = sim.players[player_index]
        if player.asleep or not player.hand:
            return []
        p_energy = player.energy
        playable_with_costs: list[tuple[int, int]] = []
        for index, card in enumerate(player.hand):
            c_beh = (
                card.definition.behavior
                if not card.markers
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
            return [playable_with_costs[0][0]]

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
        known_top = _get_sim_known_trunk_top(sim)

        has_trunk_consumer = any(
            c.definition.title
            in {
                "Assorted Cutlery",
                "Quacks",
                "Poker",
                "Bougie Coffee Machine",
                "Booby Prize",
            }
            for c in player.hand
        )
        has_early_food_played = any(
            "Food" in (c.definition.tags if not c.markers else c.tags)
            and _effective_card_title(c)
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
            for c in player.played_today
        )

        def sort_key(hand_idx: int, cost: int) -> tuple[int, float, int]:
            card = player.hand[hand_idx]
            prof = _beryllium_build_profile(card)
            desirability = _beryllium_quick_card_desirability(
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

            if title == "Nos Book":
                if not player.played_today:
                    bucket = 6
                    desirability -= 6.0
            elif title == "Tell a Story":
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
                if cutlery_cards:
                    c_cost = sim.energy_cost(player_index, cutlery_cards[0])
                    if cost + c_cost <= player.energy and any(
                        other is not card
                        and other is not cutlery_cards[0]
                        and _beryllium_free_play_value(
                            sim, player_index, other, wanted_tags
                        )
                        >= 4.0
                        for other in player.hand
                    ):
                        bucket = 2
                        desirability += 5.0
                elif bcm_cards:
                    b_cost = sim.energy_cost(player_index, bcm_cards[0])
                    if cost + b_cost <= player.energy and any(
                        other is not card
                        and other is not bcm_cards[0]
                        and "Food" in other.tags
                        and _beryllium_free_play_value(
                            sim, player_index, other, wanted_tags
                        )
                        >= 3.5
                        for other in player.hand
                    ):
                        bucket = 2
                        desirability += 4.5
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
                    top_val = _beryllium_free_play_value(
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
                    for idx, opp in enumerate(sim.players)
                    if idx != player_index
                    for oc in opp.played_today
                )
                opp_awake = any(
                    not opp.asleep
                    for idx, opp in enumerate(sim.players)
                    if idx != player_index
                )
                if not opp_played_bg and opp_awake:
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
                    for idx, opp in enumerate(sim.players)
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
                matches_in_sc = sum(
                    1 for sc in sim.suitcase if tag_needed in sc.tags
                )
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
                        desirability += 2.2

            if last_card is not None and last_prof is not None:
                last_tags = (
                    last_card.definition.tags
                    if not last_card.markers
                    else last_card.tags
                )
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
                        card.definition.cost * 0.8
                        + (
                            eff_base_fun + prof.conditional_fun_bonus
                        )
                        * 0.6
                    )

            return (bucket, -desirability, cost)

        keyed = [(sort_key(idx, cost), idx) for idx, cost in playable_with_costs]
        keyed.sort(key=lambda item: item[0])
        if has_early_food_played and len(keyed) > 1:
            for pos, item in enumerate(keyed):
                if player.hand[item[1]].definition.title == "Leftovers":
                    if pos > 0:
                        keyed.insert(0, keyed.pop(pos))
                    break
        if len(keyed) <= limit:
            return [idx for _, idx in keyed]
        selected: list[int] = []
        seen_buckets: set[int] = set()
        for key_tup, idx in keyed:
            b = key_tup[0]
            if b not in seen_buckets:
                seen_buckets.add(b)
                selected.append(idx)
        for _, idx in keyed:
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
        beam_width: int = 12,
        max_plays: int | None = None,
        allow_bed_now: bool = True,
        allowed_first_indices: Sequence[int] | None = None,
    ) -> tuple[int | None, float]:
        best_action, best_val, _ = self._search_best_day_plan_with_table(
            base_sim,
            player_index,
            beam_width=beam_width,
            max_plays=max_plays,
            allow_bed_now=allow_bed_now,
            allowed_first_indices=allowed_first_indices,
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
    ) -> tuple[int | None, float, dict[int | None, tuple[float, Game]]]:
        """Expanded multi-card beam search with deterministic Day d+1 rollout verification."""
        played_count = len(base_sim.players[player_index].played_today)
        is_live_play = max_plays is None and self._evaluation_depth == 0
        if max_plays is None:
            if played_count >= 12:
                max_plays = 6
                beam_width = 10
            elif played_count >= 7:
                max_plays = 10
                beam_width = 16
            else:
                max_plays = 14
                beam_width = 22

        best_by_action: dict[int | None, tuple[float, Game]] = {}

        if allow_bed_now:
            bed_now = _beryllium_copy_game(base_sim)
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

        init_limit = 16 if is_live_play else max(10, self.BRANCH_LIMIT + 2)
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
            try:
                child = self._simulate_play_step(
                    base_sim, player_index, hand_idx
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
                active_beam = (
                    beam_width if depth < 7 else max(8, beam_width - 6)
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
                candidates = self._ordered_candidate_indices(
                    state, player_index, branch_limit
                )
                for hand_idx in candidates:
                    try:
                        child = self._simulate_play_step(
                            state, player_index, hand_idx
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
                (initial_candidates[0] if initial_candidates else None),
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
        """Run a zero-noise Day d+1 play search on the top contending Day d terminal states."""
        contenders = [
            (act, f_val, term_sim)
            for act, (f_val, term_sim) in best_by_action.items()
            if f_val >= max_fast_val - 3.0
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
        rollout_cache: dict[tuple, float] = {}
        best_blended = float("-inf")
        best_acts: list[int | None] = []

        contenders.sort(key=lambda item: item[1], reverse=True)
        for act, f_val, term_sim in contenders[:6]:
            next_sim = _beryllium_copy_game(term_sim)
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
            end_day_fun = float(p_next.fun)
            first_bed_bonus = (
                self.FIRST_TO_BED_BONUS if not other_asleep_now else 0.0
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
                    # Add 3 deterministic baseline drafted cards (Cost 1/2/2, Fun 2/3/2 = 7.0 baseline Fun)
                    wanted_next = _wanted_tags_for_player(p_next)
                    top_tag = max(
                        wanted_next.items(),
                        key=lambda kv: (kv[1], kv[0]),
                        default=("", 0.0),
                    )
                    for n_i, n_defn in enumerate(_NEUTRAL_DRAFT_DEFNS):
                        if (
                            n_i == 2
                            and top_tag[1] >= 1.5
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
                        _, day2_total = super()._search_best_day_plan(
                            next_sim,
                            player_index,
                            beam_width=4,
                            max_plays=6,
                            allow_bed_now=True,
                        )
                    finally:
                        self._evaluation_depth = prev_depth
                    delta_tomorrow = (day2_total - end_day_fun) - 7.0
                    rollout_cache[cache_key] = delta_tomorrow
                except Exception:
                    delta_tomorrow = f_val - end_day_fun - first_bed_bonus

            rollout_total = end_day_fun + first_bed_bonus + delta_tomorrow
            blended = 0.60 * f_val + 0.40 * rollout_total
            best_by_action[act] = (blended, term_sim)
            if blended > best_blended + 1e-9:
                best_blended = blended
                best_acts = [act]
            elif abs(blended - best_blended) <= 1e-9:
                best_acts.append(act)

        return best_acts, best_blended

    def _continuation_potential(self, sim: Game, player_index: int) -> float:
        """Estimate upside from continuing to play cards today using effective definitions for copied cards."""
        player = sim.players[player_index]
        if player.asleep or not player.hand:
            return 0.0

        active_after_tags: dict[str, float] = {}
        global_after_bonus = 0.0
        has_after_all_tags = False
        for played in player.played_today:
            prof = _beryllium_build_profile(played)
            if getattr(played.effective_behavior, "after_all_tags", False):
                has_after_all_tags = True
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

        pending_before_bonus = 0.0
        energy_budget = player.energy
        for hc in player.hand:
            h_prof = _beryllium_build_profile(hc)
            h_cost = (
                hc.definition.cost if not hc.markers else hc.effective_cost
            )
            if h_prof.is_before_scaler and h_cost <= energy_budget:
                if h_prof.target_tags:
                    t_tags = h_prof.target_tags
                    matches_played = sum(
                        1
                        for p in player.played_today
                        if (
                            (
                                p.definition.tags
                                if not p.markers
                                else p.tags
                            )
                            & t_tags
                        )
                    )
                    pending_before_bonus += (
                        matches_played * h_prof.signed_match_benefit * 0.90
                    )
                    for t in t_tags:
                        active_after_tags[t] = (
                            active_after_tags.get(t, 0.0)
                            + h_prof.signed_match_benefit * 0.70
                        )
                elif hc.definition.title == "Evening on the Dock":
                    pending_before_bonus += len(player.played_today) * 0.90
                elif hc.definition.title == "Couch Tube":
                    pending_before_bonus += (
                        sum(
                            (
                                p.definition.cost
                                if not p.markers
                                else p.effective_cost
                            )
                            for p in player.played_today
                        )
                        * 0.85
                    )

        matching_playable = 0.0
        if active_after_tags or global_after_bonus != 0.0:
            for card in player.hand:
                cost = (
                    card.definition.cost
                    if not card.markers
                    else card.effective_cost
                )
                if cost <= energy_budget:
                    matching_playable += global_after_bonus
                    if has_after_all_tags:
                        matching_playable += sum(active_after_tags.values())
                    else:
                        c_tags = (
                            card.definition.tags
                            if not card.markers
                            else card.tags
                        )
                        for t in c_tags:
                            matching_playable += active_after_tags.get(t, 0.0)
        return (
            min(player.energy, 7) * 0.72
            + matching_playable * 0.95
            + pending_before_bonus
        )

    def _evaluate_terminal_day(
        self, terminal_sim: Game, player_index: int
    ) -> float:
        """Evaluate a state where player_index goes to bed today using effective definitions."""
        players = terminal_sim.players
        player = players[player_index]

        other_asleep = False
        opponents_awake = False
        for idx, p in enumerate(players):
            if idx != player_index:
                if p.asleep:
                    other_asleep = True
                else:
                    opponents_awake = True
        first_to_bed_value = (
            self.FIRST_TO_BED_BONUS
            if (terminal_sim.day < DAY_COUNT and not other_asleep)
            else 0.0
        )

        needs_full_end_day = False
        for c in player.played_today:
            if _beryllium_build_profile(c).has_end_day_hook:
                needs_full_end_day = True
                break
        eval_sim = terminal_sim
        if needs_full_end_day:
            scoring = _beryllium_copy_game(terminal_sim)
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
            terminal_sim._fun_at_start_of_scoring = (
                players[0].fun,
                players[1].fun,
                players[2].fun,
                players[3].fun,
            )
            scored_today = 0
            next_tomorrow_cards = []
            for c in player.played_today:
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

        for c in player.played_today:
            if c.markers:
                t = c.definition.title
                if t == "Poker" and c.markers.get("_blind_poker"):
                    today_total_fun += 1.35
                elif t == "Quacks" and c.markers.get("_blind_quacks"):
                    today_total_fun += 0.08

        if opponents_awake:
            for c in player.played_today:
                eff_title = _effective_card_title(c)
                if (
                    eff_title == "Fit to Print"
                    and len(player.played_today) <= 4
                ):
                    if all(
                        len(player.played_today) > len(opp.played_today)
                        for idx, opp in enumerate(players)
                        if idx != player_index
                    ):
                        today_total_fun -= (
                            3.5 if len(player.played_today) <= 3 else 1.8
                        )
                elif (
                    eff_title == "Play With the Kids" and len(player.hand) <= 2
                ):
                    if all(
                        len(player.hand) > len(opp.hand)
                        for idx, opp in enumerate(players)
                        if idx != player_index
                    ):
                        today_total_fun -= 2.2
                elif eff_title == "Puerto Rico":
                    today_total_fun -= 1.2
                elif eff_title == "San Juan":
                    t_idx = c.markers.get("target_player_index")
                    if (
                        isinstance(t_idx, int)
                        and t_idx != player_index
                        and not players[t_idx].asleep
                    ):
                        today_total_fun -= 1.4
                elif eff_title == "Cheesy Phone Game":
                    relax_opps = sum(
                        1
                        for idx, opp in enumerate(players)
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
        """Enhanced reserve & tomorrow evaluation fixing Colouring, Photo Shoot, Date's First Nos, and combo pieces."""
        player = sim.players[player_index]
        hand = player.hand
        if not hand and not next_tomorrow_cards:
            return 0.0

        can_cache = True
        for hc in hand:
            if hc.markers:
                can_cache = False
                break
        cache_key: tuple | None = None
        if can_cache:
            players = sim.players
            p_fun = player.fun
            cache_key = (
                days_left,
                player.character.title if player.character is not None else None,
                player.energy == 0 if next_tomorrow_cards else False,
                (
                    players[0].fun >= p_fun,
                    players[1].fun >= p_fun,
                    players[2].fun >= p_fun,
                    players[3].fun >= p_fun,
                ),
                tuple(
                    (tc.definition.slug, _effective_card_title(tc))
                    for tc in player.tomorrow_cards
                )
                if player.tomorrow_cards
                else (),
                tuple(
                    (
                        tc.definition.slug,
                        _effective_card_title(tc),
                        tc.markers.get("chosen_tag") if tc.markers else None,
                    )
                    for tc in next_tomorrow_cards
                )
                if next_tomorrow_cards
                else (),
                tuple(hc.definition.slug for hc in hand) if hand else (),
            )
            cached_val = self._reserve_eval_cache.get(cache_key)
            if cached_val is not None:
                return cached_val

        tom_fun = 0.0
        tom_energy_delta = 0
        tom_tag_bonuses: dict[str, float] = {}
        tom_all_tags = False
        has_colouring = False
        has_photo_shoot = False
        has_dates_first = False

        for tc in next_tomorrow_cards:
            t_prof = _beryllium_build_profile(tc)
            t_title = (
                tc.definition.title
                if not tc.markers
                else _effective_card_title(tc)
            )
            tom_fun += t_prof.tomorrow_start_fun
            tom_energy_delta += t_prof.tomorrow_start_energy
            if t_title == "Lawn Maintenance" and player.energy == 0:
                tom_fun += 3.0
            elif t_title == "Wash the Dishes" and len(hand) <= 1:
                tom_fun += 2.0
            elif t_title == "Photo Shoot":
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
            if tc.markers:
                chosen_tag = tc.markers.get("chosen_tag")
                if isinstance(chosen_tag, str):
                    tom_tag_bonuses[chosen_tag] = (
                        tom_tag_bonuses.get(chosen_tag, 0.0) + 1.2
                    )
            if bool(getattr(tc.effective_behavior, "tomorrow_all_tags", False)):
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
                c_prof = _beryllium_build_profile(c)
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

            for c in ordered_hand:
                c_prof = _beryllium_build_profile(c)
                c_defn = (
                    c.definition if not c.markers else c.effective_definition
                )
                c_title = c_defn.title
                c_tags = c_defn.tags if not c.markers else c.tags
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
                    support = (
                        max(0, tom_energy_delta)
                        + _extra_energy_support_for_card(player, c)
                        + _istanbul_extra_energy_support(
                            sim,
                            player_index,
                            c,
                            extra_fodder=3,
                            ignore_today_block=True,
                        )
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
                    pos_tom = sum(
                        v for v in tom_tag_bonuses.values() if v > 0
                    )
                    if pos_tom > 0:
                        raw_val += pos_tom * 1.25
                        is_synergy_piece = True
                elif tom_tag_bonuses:
                    for t in c_tags:
                        tb = tom_tag_bonuses.get(t, 0.0)
                        if tb > 0:
                            raw_val += tb * 1.25
                            is_synergy_piece = True

                best_partner_tag: str | None = None
                best_partner_benefit = 0.0
                if held_scaler_benefit_by_tag:
                    for t in c_tags:
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
                    opps_ahead = sum(
                        1
                        for idx, opp in enumerate(sim.players)
                        if idx != player_index and opp.fun >= player.fun
                    )
                    if opps_ahead >= 2:
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

            surpluses.sort(reverse=True)
            has_combo_or_energy = (
                bool(held_multi_scalers) or tom_energy_delta >= 2
            )
            slot_weights = (
                (1.0, 1.0, 0.85, 0.55)
                if has_combo_or_energy
                else (1.0, 0.92, 0.50, 0.25)
            )
            held_value = sum(
                s * (slot_weights[i] if i < len(slot_weights) else 0.08)
                for i, s in enumerate(surpluses)
            )
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
            cheap_held = sum(
                1
                for hc in hand
                if (hc.definition.cost if not hc.markers else hc.effective_cost)
                <= 2
            )
            extra += 1.8 + 0.65 * min(cheap_held, 3)
            if any(hc.definition.title == "Paddleboat" for hc in hand):
                extra += 3.5

        if has_dates_first:
            cheap_or_energy = sum(
                1
                for hc in hand
                if (hc.definition.cost if not hc.markers else hc.effective_cost)
                <= 1
                or _beryllium_build_profile(hc).immediate_energy_gain >= 2
            )
            extra += (cheap_or_energy - 1) * 1.15

        for hc in hand:
            title = hc.definition.title
            if title == "Leftovers":
                if any(
                    other is not hc
                    and other.definition.title in _HIGH_COPY_FOOD_TITLES
                    for other in hand
                ):
                    extra += 2.2 * self.RESERVE_TOMORROW_WEIGHT
            elif title == "Paddleboat":
                if has_photo_shoot or any(
                    other is not hc
                    and other.definition.title in _ACQUIRER_TITLES
                    for other in hand
                ):
                    extra += 2.5 * self.RESERVE_TOMORROW_WEIGHT
            elif title == "Wedding Anniversary":
                extra += 2.1 * self.RESERVE_TOMORROW_WEIGHT
            elif title == "Euchre Awards Ceremony":
                extra += 1.8 * self.RESERVE_TOMORROW_WEIGHT
            elif title == "Potato Pancakes":
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
                    extra += (
                        min(relax_cnt, 3) * 1.6 * self.RESERVE_TOMORROW_WEIGHT
                    )
            elif title == "Keeper":
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
                    extra += (
                        min(out_cnt, 3) * 0.9 * self.RESERVE_TOMORROW_WEIGHT
                    )
            elif title == "Camp Crossroads Dessert":
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
                    extra += min(ex_cnt, 3) * 0.9 * self.RESERVE_TOMORROW_WEIGHT
            elif title == "Pudding Chômeur" and len(hand) >= 2:
                extra += 1.4 * self.RESERVE_TOMORROW_WEIGHT

        res_val = val + extra
        if cache_key is not None:
            if len(self._reserve_eval_cache) > 8192:
                self._reserve_eval_cache.clear()
            self._reserve_eval_cache[cache_key] = res_val
        return res_val

    # ------------------------------------------------------------------
    # Search-Guided Tactical Sub-Decisions
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
        if last_played == "Navy SEALing":
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_cards_to_discard(
                game, player_index, options
            )
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
                    sim = _beryllium_copy_game(base_sim)
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
        return super().choose_cards_to_discard(game, player_index, options)

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        options: Sequence[CardInstance],
    ) -> int:
        return _BERYLLIUM_EVAL_AI_SINGLETON.choose_card_to_discard(
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
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_card_to_copy(
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
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_card_to_copy(
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
            ),
            reverse=True,
        )[:5]
        best_idx = ranked_indices[0]
        best_val = float("-inf")
        prev_depth = self._evaluation_depth
        self._evaluation_depth = 1
        try:
            for idx in ranked_indices:
                option = options[idx]
                sim = _beryllium_copy_game(base_sim)
                sim_player = sim.players[player_index]
                sim_source = sim_player.played_today[-1]
                try:
                    sim.copy_card_effect(
                        player_index,
                        _clone_card(option),
                        sim_source,
                        pay_source_cost=pay_cost,
                    )
                    _, val = super()._search_best_day_plan(
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
                sim = _beryllium_copy_game(base_sim)
                sim_player = sim.players[player_index]
                sim_player.energy -= spend
                sim_player.played_today[-1].markers["energy_cubes"] = spend
                _, val = super()._search_best_day_plan(
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
        if "Cities and Knights" in action or "Epic Prank" in action:
            return _BERYLLIUM_EVAL_AI_SINGLETON.choose_optional_action(
                game, player_index, action
            )
        return super().choose_optional_action(game, player_index, action)


Beryllium = BerylliumBrain
BerylliumAI = BerylliumBrain
BerylliumbrainAI = BerylliumBrain
