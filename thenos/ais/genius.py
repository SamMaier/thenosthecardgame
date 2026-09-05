"""A high-compute policy for competitive, fair play."""

from __future__ import annotations

import copy
from collections import Counter
import math
import random
from statistics import median
from typing import Sequence, TYPE_CHECKING

from thenos.ais.planner import PlannerAI
from thenos.cards.basic import (
    EpicPrankBehavior,
    SettlersCitiesAndKnightsBehavior,
)
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import CARD_REGISTRY
from thenos.cards.copy_effects import WeddingAnniversaryBehavior
from thenos.cards.exercise import NavySEALingBehavior
from thenos.cards.food import AddressTheFoodBehavior, PuddingChomeurBehavior
from thenos.cards.fun_effects import FunForNextCardBehavior

if TYPE_CHECKING:
    from thenos.game import Game


_PROJECTED_PLAY_DEFINITION = CardDefinition(
    slug="__genius_projected_play__",
    title="Projected opponent play",
    tags=frozenset(),
    cost=0,
    behavior=CardBehavior(),
)
_TYPICAL_CARD_COST = max(
    1,
    int(median(definition.cost for definition in CARD_REGISTRY.values())),
)


class GeniusAI(PlannerAI):
    """Deprecated high-compute policy retained for explicit comparisons.

    New simulations should use
    :class:`thenos.ais.galaxybrain.GalaxybrainAI`.
    ``GeniusAI`` remains available as an explicit ``Competitor`` factory for
    large comparative runs, but is no longer used by standard scripts or tests.

    The policy models only observable state. Before any hypothetical card can
    draw from the Trunk, hidden cards are sampled from the public catalog after
    removing this player's hand and public zones. The live Trunk order,
    opponent card identities, and game RNG therefore cannot influence a
    decision. Opponent evaluation uses only scores, visible cards, asleep
    state, hand sizes, and remaining Energy. A conservative estimate of
    opponents' remaining play capacity is included so relative-play bonuses
    are not awarded merely because opponents have not taken their later turns
    yet.

    Ordinary play search retains six partial plans for four plies. The final
    day expands to sixteen plans over six plies, when every leaf can be scored
    without a future approximation. Suitcase picks receive a separate search,
    and distinctive effect choices are resolved on engine copies. This is
    intentionally much more expensive than ``PlannerAI`` while bounded.
    """

    SEARCH_DEPTH = 4
    BEAM_WIDTH = 6
    BRANCH_WIDTH = 7
    ROOT_WIDTH = 12
    FUTURE_WEIGHT = 0.55
    SAME_DAY_VALUE_WEIGHT = 0.75
    DETERMINIZATIONS = 1

    @staticmethod
    def _projected_opponent_plays(
        game: Game,
        player_index: int,
        opponent_index: int,
    ) -> int:
        """Estimate future opponent plays from public state only.

        Opponent card identities are hidden. Their hand size and remaining
        Energy are public, so use the catalog's median card cost as a stable,
        card-agnostic capacity estimate. Rounding up is intentional: a cheap
        card in an unknown hand is plausible, and relative-play bonuses should
        not assume opponents stop before their remaining turns.
        """
        if opponent_index == player_index:
            return 0
        opponent = game.players[opponent_index]
        if opponent.asleep or not opponent.hand or opponent.energy <= 0:
            return 0
        return min(
            len(opponent.hand),
            (opponent.energy + _TYPICAL_CARD_COST - 1) // _TYPICAL_CARD_COST,
        )

    @classmethod
    def _add_projected_opponent_plays(
        cls,
        game: Game,
        player_index: int,
    ) -> None:
        """Add no-op plays to a scoring copy for observable future pressure."""
        for opponent_index, opponent in enumerate(game.players):
            projected_count = cls._projected_opponent_plays(
                game, player_index, opponent_index
            )
            for offset in range(projected_count):
                opponent.played_today.append(
                    CardInstance(
                        -1_000_000
                        - opponent_index * 1_000
                        - offset,
                        _PROJECTED_PLAY_DEFINITION,
                    )
                )

    def _portfolio_synergy(self, game: Game, player_index: int) -> float:
        """Estimate future pair value from this player's own known cards."""
        player = game.players[player_index]
        cards = [*player.tomorrow_cards, *player.hand]
        synergy = 0.0
        for source in cards:
            behavior = source.effective_behavior
            tag = getattr(behavior, "tag", None)
            if not isinstance(tag, str):
                continue
            matches = sum(
                tag in target.tags for target in player.hand if target is not source
            )
            if not matches:
                continue
            bonus = getattr(behavior, "bonus", 0)
            fun_bonus = getattr(behavior, "fun_bonus", 0)
            energy_delta = getattr(behavior, "energy_delta", 0)
            multiplier = getattr(behavior, "multiplier", 1)
            strength = max(
                0.0,
                float(bonus) if isinstance(bonus, int) else 0.0,
                float(fun_bonus) if isinstance(fun_bonus, int) else 0.0,
                float(-energy_delta)
                if isinstance(energy_delta, int) and energy_delta < 0
                else 0.0,
                float(multiplier - 1)
                if isinstance(multiplier, int) and multiplier > 1
                else 0.0,
            )
            if strength:
                synergy += min(matches, 3) * min(strength, 3.0)
        return synergy

    def _planning_copy(self, game: Game) -> Game:
        """Sample all hidden cards from public information, never live values."""
        own_index = next(
            index for index, ai in enumerate(game.ais) if ai is self
        )
        simulation = copy.deepcopy(game)
        seed = self.rng.getrandbits(64)
        planning_rng = random.Random(seed)
        simulation.rng = random.Random(seed)
        simulation.sample_daily_conditions(own_index, planning_rng)

        known_cards = [*simulation.suitcase, *simulation.discard]
        for player in simulation.players:
            known_cards.extend(player.played_today)
            known_cards.extend(player.tomorrow_cards)
        known_cards.extend(simulation.players[own_index].hand)
        known_titles = Counter(card.title for card in known_cards)
        unknown_definitions = [
            definition
            for definition in CARD_REGISTRY.values()
            if known_titles[definition.title] == 0
            and (simulation.daily_conditions or definition.slug != "read-the-radar")
        ]
        required = len(simulation.trunk) + sum(
            len(player.hand)
            for index, player in enumerate(simulation.players)
            if index != own_index
        )

        if len(unknown_definitions) >= required:
            planning_rng.shuffle(unknown_definitions)
            sampled = [
                CardInstance(10_000_000 + index, definition)
                for index, definition in enumerate(unknown_definitions[:required])
            ]
            cursor = 0
            for index, player in enumerate(simulation.players):
                if index == own_index:
                    continue
                count = len(player.hand)
                player.hand = sampled[cursor : cursor + count]
                cursor += count
            simulation.trunk = sampled[cursor : cursor + len(simulation.trunk)]
            planning_rng.shuffle(simulation.trunk)
        else:
            # Synthetic unit tests may not use the default unique-card deck.
            simulation.trunk.sort(key=lambda card: card.instance_id)
            planning_rng.shuffle(simulation.trunk)

        simulation.discard.sort(key=lambda card: card.instance_id)
        planning_rng.shuffle(simulation.discard)
        for ai in simulation.ais:
            if isinstance(ai, PlannerAI):
                ai._evaluation_depth = self._evaluation_depth + 1
        return simulation

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)

    def _same_day_hand_value(self, game: Game, player_index: int) -> float:
        """Value remaining plays, including generic on-play Energy gains."""
        player = game.players[player_index]
        if player.asleep or player.energy <= 0 or not player.hand:
            return 0.0

        # A pending immediately-next-card effect cannot be estimated by the
        # ordinary hand knapsack. That estimator considers each hand card in
        # isolation, which would make every candidate appear immediately next
        # and apply the one-shot effect repeatedly. Resolve the actual next
        # play first, then value the remaining hand after the effect is spent.
        if player.played_today and isinstance(
            player.played_today[-1].effective_behavior,
            FunForNextCardBehavior,
        ):
            best_value = 0.0
            for card in tuple(player.hand):
                if not card.effective_behavior.can_play(game, player, card):
                    continue
                if game.energy_cost(player_index, card) > player.energy:
                    continue

                simulation = copy.deepcopy(game)
                simulated_player = simulation.players[player_index]
                simulated_card = next(
                    candidate
                    for candidate in simulated_player.hand
                    if candidate.instance_id == card.instance_id
                )
                simulation.play_card(
                    player_index,
                    simulated_player.hand.index(simulated_card),
                )
                projected_fun = simulation.card_fun(
                    player_index, simulated_card
                )
                downstream_value = (
                    0.0
                    if simulated_player.asleep
                    else self._future_hand_value(simulation, player_index)
                )
                best_value = max(
                    best_value,
                    projected_fun + downstream_value,
                )
            return best_value

        best_value = self._future_hand_value(game, player_index)
        has_energy_bottleneck = any(
            card.effective_behavior.can_play(game, player, card)
            and game.energy_cost(player_index, card) > player.energy
            for card in player.hand
        )
        if not has_energy_bottleneck:
            return best_value

        for card in tuple(player.hand):
            if not card.effective_behavior.can_play(game, player, card):
                continue
            cost = game.energy_cost(player_index, card)
            if cost > player.energy:
                continue

            simulation = copy.deepcopy(game)
            simulated_ai = simulation.ais[player_index]
            if isinstance(simulated_ai, PlannerAI):
                simulated_ai._evaluation_depth = self._evaluation_depth + 1
            simulated_player = simulation.players[player_index]
            simulated_card = next(
                candidate
                for candidate in simulated_player.hand
                if candidate.instance_id == card.instance_id
            )
            hand_index = simulated_player.hand.index(simulated_card)
            energy_before = simulated_player.energy
            payment = simulation.energy_cost(player_index, simulated_card)
            fun_before = simulated_player.fun
            simulation.play_card(player_index, hand_index)
            energy_after = simulated_player.energy
            energy_gained = energy_after - (energy_before - payment)
            if energy_gained <= 0:
                continue

            projected_fun = simulation.card_fun(
                player_index, simulated_card
            )
            immediate_fun = simulated_player.fun - fun_before
            downstream_value = (
                0.0
                if simulated_player.asleep
                else self._future_hand_value(
                    simulation,
                    player_index,
                    capacity=energy_after,
                )
            )
            best_value = max(
                best_value,
                immediate_fun + projected_fun + downstream_value,
            )
        return best_value

    def _project_next_day_suitcase_picks(
        self,
        game: Game,
        player_index: int,
    ) -> None:
        """Project every player's guaranteed picks from public information.

        A real non-final day is followed by three rounds of Suitcase picks
        before anyone plays again.  Terminal search positions must therefore
        not treat an empty current hand as an empty next-day hand.  Opponents
        consume cards in public turn order using the same context-free card
        estimate as Genius; this neither inspects their hidden hands nor adds
        title-specific policy.  Replacement cards come from the planning
        copy's independently sampled Trunk.
        """
        from thenos.game import DAILY_PICKS

        for _ in range(DAILY_PICKS):
            for picker_index in game.player_order():
                # Normal games can refill every slot.  Synthetic unit states
                # sometimes omit a deck, so stop before an impossible refill.
                if not game.suitcase or (not game.trunk and not game.discard):
                    return
                choice = max(
                    range(len(game.suitcase)),
                    key=lambda index: (
                        self._card_value(game.suitcase[index]),
                        -game.suitcase[index].instance_id,
                    ),
                )
                game.pick_suitcase_cards(
                    picker_index,
                    (game.suitcase[choice],),
                )

    def _state_value(self, game: Game, player_index: int) -> float:
        """Blend long-term self value with the observable score margin."""
        scoring = copy.deepcopy(game)
        self._add_projected_opponent_plays(scoring, player_index)
        # This copy remains private to the evaluation. Read its same-day value
        # before scoring it instead of cloning the entire game a second time.
        same_day_value = self._same_day_hand_value(scoring, player_index)
        scoring.end_day()
        player = scoring.players[player_index]
        own_score = player.fun
        opponent_score = max(
            opponent.fun
            for index, opponent in enumerate(scoring.players)
            if index != player_index
        )

        from thenos.game import DAILY_ENERGY, DAYS_PER_GAME

        future_value = 0.0
        if game.day < DAYS_PER_GAME:
            scoring.day = game.day + 1
            player.energy = scoring.prepare_condition_forecast(player_index)
            player.asleep = False
            for card in player.tomorrow_cards:
                card.effective_behavior.on_start_day(scoring, player, card)
            self._project_next_day_suitcase_picks(scoring, player_index)
            immediate_future = player.fun - own_score
            energy_delta = player.energy - DAILY_ENERGY
            reserve_value = self._future_hand_value(scoring, player_index)
            # _future_hand_value restores its temporary hand/played changes.
            # Hide Tomorrow sources briefly rather than copying the whole
            # scoring state solely to clear this one list.
            tomorrow_cards = player.tomorrow_cards
            player.tomorrow_cards = []
            try:
                ordinary_reserve_value = self._future_hand_value(
                    scoring,
                    player_index,
                )
            finally:
                player.tomorrow_cards = tomorrow_cards
            tomorrow_reserve_bonus = reserve_value - ordinary_reserve_value
            future_value = (
                immediate_future
                + 0.65 * energy_delta
                + 0.20 * reserve_value
                # The reserve estimate is broadly discounted because most
                # future cards are available in either branch. A Tomorrow
                # modifier is caused by today's decision, however, so raise
                # its total coefficient to one before FUTURE_WEIGHT applies.
                + 0.80 * tomorrow_reserve_bonus
                + 0.18 * self._portfolio_synergy(scoring, player_index)
            )

        value = (
            own_score
            + self.SAME_DAY_VALUE_WEIGHT * same_day_value
            + self.FUTURE_WEIGHT * future_value
        )
        if (
            game.day < 6
            and game.players[player_index].asleep
            and not any(
                opponent.asleep
                for index, opponent in enumerate(game.players)
                if index != player_index
            )
        ):
            # ``first_to_bed`` lives inside Game.playing_phase, so isolated
            # play simulations cannot update starting_player themselves.
            value += 1.0
        if game.day >= 5:
            # Near the finish, converting a narrow deficit into a lead matters
            # more than adding points to an existing blowout. Only cumulative
            # Fun and currently visible cards contribute to this public margin.
            margin = own_score - opponent_score
            win_weight = 2.5 if game.day == 5 else 7.0
            value += win_weight * math.tanh(margin / 3.0)
        return value

    def _best_play(
        self,
        root: Game,
        player_index: int,
        playable: Sequence[int],
        *,
        depth: int | None = None,
        beam_width: int | None = None,
    ) -> tuple[int, float]:
        """Search today broadly, expanding further on the exact final day."""
        final_day_search = depth is None and root.day >= 6
        search_depth = (
            6 if final_day_search else (depth or self.SEARCH_DEPTH)
        )
        width = 16 if final_day_search else (beam_width or self.BEAM_WIDTH)
        branch_width = 12 if final_day_search else self.BRANCH_WIDTH
        root_width = 20 if final_day_search else self.ROOT_WIDTH

        def shortlist(state: Game, choices: Sequence[int]) -> list[int]:
            player = state.players[player_index]
            return sorted(
                choices,
                key=lambda index: self._card_value(player.hand[index]),
                reverse=True,
            )[:branch_width]

        root_plays = shortlist(root, playable)
        if len(root_plays) < root_width:
            ranked_remaining = sorted(
                (index for index in playable if index not in root_plays),
                key=lambda index: self._card_value(
                    root.players[player_index].hand[index]
                ),
                reverse=True,
            )
            root_plays.extend(
                ranked_remaining[: root_width - len(root_plays)]
            )

        nodes: list[tuple[Game, int, float]] = []
        for hand_index in root_plays:
            simulation = copy.deepcopy(root)
            simulation.play_card(player_index, hand_index)
            nodes.append(
                (
                    simulation,
                    hand_index,
                    self._state_value(simulation, player_index),
                )
            )
        nodes.sort(key=lambda node: node[2], reverse=True)
        nodes = nodes[:width]

        for _ in range(1, search_depth):
            candidates = list(nodes)
            for state, first_choice, _ in nodes:
                next_plays = state.playable_hand_indices(player_index)
                for hand_index in shortlist(state, next_plays):
                    simulation = copy.deepcopy(state)
                    simulation.play_card(player_index, hand_index)
                    candidates.append(
                        (
                            simulation,
                            first_choice,
                            self._state_value(simulation, player_index),
                        )
                    )
            candidates.sort(key=lambda node: node[2], reverse=True)
            nodes = candidates[:width]

        best_value = nodes[0][2]
        tied = [node for node in nodes if abs(node[2] - best_value) < 1e-9]
        best = self.rng.choice(tied)
        return best[1], best[2]

    def _suitcase_pick_value(
        self,
        root: Game,
        player_index: int,
        suitcase_index: int,
        *,
        extra_pick_cost: int = 0,
    ) -> float:
        simulation = copy.deepcopy(root)
        simulation.players[player_index].energy -= extra_pick_cost
        simulation.pick_suitcase_cards(
            player_index, (simulation.suitcase[suitcase_index],)
        )

        value = self._state_value(simulation, player_index)
        playable = simulation.playable_hand_indices(player_index)
        if playable:
            final_day = simulation.day >= 6
            _, planned_value = self._best_play(
                simulation,
                player_index,
                playable,
                depth=4 if final_day else 2,
                beam_width=10 if final_day else 4,
            )
            value = max(value, planned_value)
        return value

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
        if cached is not None:
            return cached
        if self._evaluation_depth:
            return super().choose_card_to_play(
                game, player_index, playable_hand_indices
            )

        results = [
            self._best_play(
                self._planning_copy(game),
                player_index,
                playable_hand_indices,
            )
            for _ in range(self.DETERMINIZATIONS)
        ]
        vote_counts = {
            choice: sum(result_choice == choice for result_choice, _ in results)
            for choice, _ in results
        }
        most_votes = max(vote_counts.values())
        finalists = [
            choice for choice, votes in vote_counts.items() if votes == most_votes
        ]
        if len(finalists) == 1:
            return finalists[0]
        average_values = {
            choice: sum(
                value for result_choice, value in results if result_choice == choice
            )
            / vote_counts[choice]
            for choice in finalists
        }
        best_average = max(average_values.values())
        return self.rng.choice(
            [
                choice
                for choice, value in average_values.items()
                if abs(value - best_average) < 1e-9
            ]
        )

    def choose_to_go_to_bed(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> bool:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        self._pending_play_choice = None
        if self._evaluation_depth:
            return super().choose_to_go_to_bed(
                game, player_index, playable_hand_indices
            )

        results: list[tuple[int, float, float]] = []
        for _ in range(self.DETERMINIZATIONS):
            root = self._planning_copy(game)
            stopped = copy.deepcopy(root)
            stopped.go_to_bed(player_index)
            stop_value = self._state_value(stopped, player_index)
            choice, play_value = self._best_play(
                root, player_index, playable_hand_indices
            )
            results.append((choice, play_value, stop_value))

        continuations = [
            (choice, play_value)
            for choice, play_value, stop_value in results
            if play_value > stop_value + 1e-9
        ]
        if 2 * len(continuations) <= self.DETERMINIZATIONS:
            return True

        vote_counts = {
            choice: sum(candidate == choice for candidate, _ in continuations)
            for choice, _ in continuations
        }
        most_votes = max(vote_counts.values())
        finalists = [
            choice
            for choice, votes in vote_counts.items()
            if votes == most_votes
        ]
        if len(finalists) == 1:
            choice = finalists[0]
        else:
            average_values = {
                candidate: sum(
                    value
                    for result_choice, value in continuations
                    if result_choice == candidate
                )
                / vote_counts[candidate]
                for candidate in finalists
            }
            best_average = max(average_values.values())
            choice = self.rng.choice(
                [
                    candidate
                    for candidate, value in average_values.items()
                    if abs(value - best_average) < 1e-9
                ]
            )

        self._cache_play_choice(
            game, player_index, playable_hand_indices, choice
        )
        return False

    def choose_extra_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        if self._evaluation_depth:
            return super().choose_extra_card_to_play(
                game, player_index, playable_hand_indices
            )

        votes: list[int | None] = []
        for _ in range(self.DETERMINIZATIONS):
            root = self._planning_copy(game)
            stop_value = self._state_value(root, player_index)
            choice, play_value = self._best_play(
                root, player_index, playable_hand_indices
            )
            votes.append(choice if play_value > stop_value + 1e-9 else None)
        stop_votes = sum(choice is None for choice in votes)
        if stop_votes * 2 >= self.DETERMINIZATIONS:
            return None
        play_votes = [choice for choice in votes if choice is not None]
        counts = {
            choice: play_votes.count(choice) for choice in set(play_votes)
        }
        most_votes = max(counts.values())
        return self.rng.choice(
            [choice for choice, count in counts.items() if count == most_votes]
        )

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        if not suitcase:
            raise ValueError("Cannot choose from an empty Suitcase")
        if self._evaluation_depth:
            return self._choose_best(
                [self._card_value(card) for card in suitcase]
            )

        totals = [0.0] * len(suitcase)
        for _ in range(self.DETERMINIZATIONS):
            root = self._planning_copy(game)
            for index in range(len(suitcase)):
                totals[index] += self._suitcase_pick_value(
                    root, player_index, index
                )
        return self._choose_best(totals)

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        if self._evaluation_depth or not suitcase:
            return False
        current_total = 0.0
        pick_total = 0.0
        for _ in range(self.DETERMINIZATIONS):
            root = self._planning_copy(game)
            current_total += self._state_value(root, player_index)
            pick_total += max(
                self._suitcase_pick_value(
                    root, player_index, index, extra_pick_cost=1
                )
                for index in range(len(suitcase))
            )
        return pick_total > current_total + 1e-9

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        """Resolve each public copy target on a fair simulation."""
        if self._evaluation_depth:
            return self._choose_best(
                [self._card_value(card) for card in eligible_cards]
            )

        root = self._planning_copy(game)
        destination = root.players[player_index].played_today[-1]
        values: list[float] = []
        for eligible in eligible_cards:
            simulation = copy.deepcopy(root)
            simulated_destination = simulation.players[
                player_index
            ].played_today[-1]
            simulated_target = next(
                card
                for player in simulation.players
                for card in player.visible_cards
                if card.instance_id == eligible.instance_id
            )
            pay_source_cost = isinstance(
                destination.effective_behavior,
                WeddingAnniversaryBehavior,
            )
            simulation.copy_card_effect(
                player_index,
                simulated_target,
                simulated_destination,
                pay_source_cost=pay_source_cost,
            )
            values.append(self._state_value(simulation, player_index))
        return self._choose_best(values)

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        if self._evaluation_depth:
            return super().choose_card_target(
                game, player_index, eligible_cards
            )
        played = game.players[player_index].played_today
        if not played or not isinstance(
            played[-1].effective_behavior,
            AddressTheFoodBehavior,
        ):
            return super().choose_card_target(
                game, player_index, eligible_cards
            )

        root = self._planning_copy(game)
        values: list[float] = []
        for eligible in eligible_cards:
            simulation = copy.deepcopy(root)
            target = next(
                card
                for card in simulation.suitcase
                if card.instance_id == eligible.instance_id
            )
            picked = simulation.pick_suitcase_cards(
                player_index, (target,)
            )[0]
            simulation.play_card_for_effect(
                player_index,
                picked,
                cost_adjustment=-1,
                pay_energy=True,
            )
            values.append(self._state_value(simulation, player_index))
        return self._choose_best(values)

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> int:
        if not hand:
            raise ValueError("Cannot choose a card from an empty hand")
        if self._evaluation_depth:
            return super().choose_card_to_discard(game, player_index, hand)

        played = game.players[player_index].played_today
        behavior = played[-1].effective_behavior if played else None
        exact_behaviors = (
            EpicPrankBehavior,
            PuddingChomeurBehavior,
            SettlersCitiesAndKnightsBehavior,
        )
        if not isinstance(behavior, exact_behaviors):
            return super().choose_card_to_discard(game, player_index, hand)

        root = self._planning_copy(game)
        values: list[float] = []
        for eligible in hand:
            simulation = copy.deepcopy(root)
            simulated_player = simulation.players[player_index]
            target = next(
                card
                for card in simulated_player.hand
                if card.instance_id == eligible.instance_id
            )
            simulation.discard_from_hand(
                player_index, simulated_player.hand.index(target)
            )
            source = simulated_player.played_today[-1]
            if isinstance(behavior, SettlersCitiesAndKnightsBehavior):
                source.markers["discarded_card"] = True
            elif isinstance(behavior, EpicPrankBehavior):
                source.markers["discarded_item"] = target
            values.append(self._state_value(simulation, player_index))
        return self._choose_best(values)

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> Sequence[int]:
        """Optimize Navy SEALing exactly; cycle only weak cards otherwise."""
        played = game.players[player_index].played_today
        is_navy_sealing = bool(played) and isinstance(
            played[-1].effective_behavior,
            NavySEALingBehavior,
        )
        if self._evaluation_depth:
            if is_navy_sealing:
                from thenos.game import DAYS_PER_GAME

                return tuple(
                    index
                    for index, card in enumerate(hand)
                    if game.day >= DAYS_PER_GAME
                    or self._card_value(card) < 4.5
                )
            return tuple(
                index
                for index, card in enumerate(hand)
                if self._card_value(card) < 1.0
            )

        if is_navy_sealing:
            ranked = sorted(
                range(len(hand)),
                key=lambda index: self._card_value(hand[index]),
            )
            root = self._planning_copy(game)
            values: list[float] = []
            for count in range(len(hand) + 1):
                simulation = copy.deepcopy(root)
                selected_ids = {
                    hand[index].instance_id for index in ranked[:count]
                }
                discard_indices = [
                    index
                    for index, card in enumerate(
                        simulation.players[player_index].hand
                    )
                    if card.instance_id in selected_ids
                ]
                simulation.discard_cards_from_hand(
                    player_index, discard_indices
                )
                simulation.players[player_index].played_today[-1].markers[
                    "energy_cubes"
                ] = count
                values.append(self._state_value(simulation, player_index))
            best_count = self._choose_best(values)
            return tuple(sorted(ranked[:best_count]))

        return tuple(
            index
            for index, card in enumerate(hand)
            if self._card_value(card) < 0.75
            and not card.effective_behavior.has_tomorrow_action
        )

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        """Trade Energy for Fun only after pricing the whole remaining hand."""
        player = game.players[player_index]
        best_totals: list[float] = []
        for spend in range(maximum + 1):
            capacity = maximum - spend
            best = [0.0] * (capacity + 1)
            for candidate in player.hand:
                cost = game.energy_cost(player_index, candidate)
                if cost > capacity:
                    continue
                value = max(0.0, self._card_value(candidate))
                for energy in range(capacity, cost - 1, -1):
                    best[energy] = max(
                        best[energy], best[energy - cost] + value
                    )
            best_totals.append(spend + best[capacity])
        best_value = max(best_totals)
        choices = [
            spend
            for spend, value in enumerate(best_totals)
            if abs(value - best_value) < 1e-9
        ]
        return max(choices)
