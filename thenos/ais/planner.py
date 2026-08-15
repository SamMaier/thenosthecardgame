"""A bounded lookahead policy that balances today against future setup."""

from __future__ import annotations

import copy
import random
from typing import Sequence, TYPE_CHECKING

from thenos.ais.greedy import GreedyAI
from thenos.cards.base import CardBehavior, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game


class PlannerAI(GreedyAI):
    """Plan a short sequence of plays and retain useful cards for later days.

    The policy searches at most three of its own plays with a four-state beam.
    Each candidate is resolved by the rules engine, so ordering, immediate
    effects, and visible-card interactions are included.  Its terminal value
    combines end-of-day Fun with a discounted estimate of the next day's
    Tomorrow effects, Energy, and playable hand value.

    Planning copies receive an independently shuffled Trunk.  This lets random
    card effects participate in evaluation without exposing the live hidden
    order or coupling policy randomness to the game's random generator.
    Opponent futures are not modeled and no opponent hand cards are inspected.
    """

    SEARCH_DEPTH = 3
    BEAM_WIDTH = 3
    BRANCH_WIDTH = 4
    ROOT_WIDTH = 7
    FUTURE_WEIGHT = 0.55

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)

    def _card_value(self, card: CardInstance) -> float:
        behavior_bonus = (
            1.25 if type(card.effective_behavior) is not CardBehavior else 0.0
        )
        tomorrow_bonus = (
            2.0 if card.effective_behavior.has_tomorrow_action else 0.0
        )
        return (
            card.effective_base_fun
            + behavior_bonus
            + tomorrow_bonus
            - 0.15 * card.effective_cost
        )

    def _planning_copy(self, game: Game) -> Game:
        """Copy public state while severing evaluation from hidden order."""
        simulation = copy.deepcopy(game)
        seed = self.rng.getrandbits(64)
        simulation.rng = random.Random(seed)
        planning_rng = random.Random(seed)
        simulation.trunk.sort(key=lambda card: card.instance_id)
        planning_rng.shuffle(simulation.trunk)
        simulation.discard.sort(key=lambda card: card.instance_id)
        planning_rng.shuffle(simulation.discard)
        for ai in simulation.ais:
            if isinstance(ai, PlannerAI):
                ai._evaluation_depth = self._evaluation_depth + 1
        return simulation

    def _future_hand_value(self, game: Game, player_index: int) -> float:
        """Estimate one future day's best printed-value Energy allocation."""
        player = game.players[player_index]
        capacity = max(0, player.energy)
        best = [0.0] * (capacity + 1)
        for card in player.hand:
            if not card.effective_behavior.can_play(game, player, card):
                continue
            cost = min(capacity, game.energy_cost(player_index, card))
            value = max(0.0, self._card_value(card))
            for energy in range(capacity, cost - 1, -1):
                best[energy] = max(best[energy], best[energy - cost] + value)
        return best[capacity]

    def _state_value(self, game: Game, player_index: int) -> float:
        scoring = copy.deepcopy(game)
        scoring.end_day()
        player = scoring.players[player_index]
        today_score = player.fun

        from thenos.game import DAILY_ENERGY, DAYS_PER_GAME

        if game.day >= DAYS_PER_GAME:
            return float(today_score)

        # Resolve only this player's already-visible Tomorrow setup. Current
        # implementations adjust Energy here and never consult hidden zones.
        player.energy = DAILY_ENERGY
        player.asleep = False
        for card in player.tomorrow_cards:
            card.effective_behavior.on_start_day(scoring, player, card)

        immediate_future = player.fun - today_score
        tomorrow_score = sum(
            scoring.card_fun(player_index, card)
            for card in player.tomorrow_cards
        )
        energy_delta = player.energy - DAILY_ENERGY
        reserve_value = self._future_hand_value(scoring, player_index)
        future_value = (
            immediate_future
            + tomorrow_score
            + 0.65 * energy_delta
            + 0.20 * reserve_value
        )
        return today_score + self.FUTURE_WEIGHT * future_value

    def _shortlist(
        self,
        game: Game,
        player_index: int,
        playable: Sequence[int],
    ) -> list[int]:
        player = game.players[player_index]
        ranked = sorted(
            playable,
            key=lambda index: self._card_value(player.hand[index]),
            reverse=True,
        )
        return ranked[: self.BRANCH_WIDTH]

    def _best_play(
        self,
        root: Game,
        player_index: int,
        playable: Sequence[int],
        *,
        depth: int | None = None,
        beam_width: int | None = None,
    ) -> tuple[int, float]:
        """Return the first move and value of the best bounded play sequence."""
        search_depth = depth or self.SEARCH_DEPTH
        width = beam_width or self.BEAM_WIDTH
        nodes: list[tuple[Game, int, float]] = []
        root_plays = self._shortlist(root, player_index, playable)
        if len(root_plays) < self.ROOT_WIDTH:
            ranked_remaining = sorted(
                (index for index in playable if index not in root_plays),
                key=lambda index: self._card_value(
                    root.players[player_index].hand[index]
                ),
                reverse=True,
            )
            root_plays.extend(
                ranked_remaining[: self.ROOT_WIDTH - len(root_plays)]
            )
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
            candidates = list(nodes)  # Stopping at the current horizon is valid.
            for state, first_choice, _ in nodes:
                next_plays = state.playable_hand_indices(player_index)
                for hand_index in self._shortlist(
                    state, player_index, next_plays
                ):
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

    def choose_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int:
        if not playable_hand_indices:
            raise ValueError("No playable card was supplied")
        if self._evaluation_depth:
            return super().choose_card_to_play(
                game, player_index, playable_hand_indices
            )
        root = self._planning_copy(game)
        choice, _ = self._best_play(root, player_index, playable_hand_indices)
        return choice

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
        root = self._planning_copy(game)
        stop_value = self._state_value(root, player_index)
        choice, play_value = self._best_play(
            root, player_index, playable_hand_indices
        )
        return choice if play_value > stop_value + 1e-9 else None

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
        picked = simulation.pick_suitcase_cards(
            player_index, (simulation.suitcase[suitcase_index],)
        )[0]
        value = self._state_value(simulation, player_index)
        player = simulation.players[player_index]
        if picked in player.hand:
            hand_index = player.hand.index(picked)
            if hand_index in simulation.playable_hand_indices(player_index):
                played = copy.deepcopy(simulation)
                played.play_card(player_index, hand_index)
                value = max(value, self._state_value(played, player_index))
        return value

    def choose_suitcase_card(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        if not suitcase:
            raise ValueError("Cannot choose from an empty Suitcase")
        if self._evaluation_depth:
            return self._choose_best([self._card_value(card) for card in suitcase])
        root = self._planning_copy(game)
        values = [
            self._suitcase_pick_value(root, player_index, index)
            for index in range(len(suitcase))
        ]
        return self._choose_best(values)

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        if self._evaluation_depth or not suitcase:
            return False
        root = self._planning_copy(game)
        current = self._state_value(root, player_index)
        best_pick = max(
            self._suitcase_pick_value(
                root, player_index, index, extra_pick_cost=1
            )
            for index in range(len(suitcase))
        )
        return best_pick > current + 1e-9

    def choose_player(
        self,
        game: Game,
        player_index: int,
        eligible_player_indices: Sequence[int],
    ) -> int:
        if not eligible_player_indices:
            raise ValueError("Cannot choose from an empty player selection")
        values = [
            (
                -any(
                    "Board Game" in card.tags
                    for card in game.players[index].played_today
                ),
                -len(game.players[index].played_today),
            )
            for index in eligible_player_indices
        ]
        best = max(values)
        choices = [
            index
            for index, value in zip(eligible_player_indices, values, strict=True)
            if value == best
        ]
        return self.rng.choice(choices)

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        return self._choose_best([self._card_value(card) for card in eligible_cards])

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        own_tags = [
            card.tags for card in game.players[player_index].visible_cards
        ]
        values = [
            self._card_value(card)
            + sum(bool(card.tags.intersection(tags)) for tags in own_tags)
            for card in eligible_cards
        ]
        return self._choose_best(values)

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        # Mark the least tempting card so it is more likely to survive.
        values = [-self._card_value(card) for card in suitcase]
        return self._choose_best(values)

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> int:
        if not hand:
            raise ValueError("Cannot choose a card from an empty hand")
        return self._choose_best([-self._card_value(card) for card in hand])

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> Sequence[int]:
        # Both current multi-discard effects reward cycling low-value cards.
        return tuple(
            index for index, card in enumerate(hand) if self._card_value(card) <= 1.0
        )

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        player = game.players[player_index]
        best_total = float("-inf")
        best_spends: list[int] = []
        for spend in range(maximum + 1):
            remaining = maximum - spend
            affordable_value = max(
                (
                    self._card_value(candidate)
                    for candidate in player.hand
                    if game.energy_cost(player_index, candidate) <= remaining
                ),
                default=0.0,
            )
            total = spend + max(0.0, affordable_value)
            if total > best_total + 1e-9:
                best_total = total
                best_spends = [spend]
            elif abs(total - best_total) < 1e-9:
                best_spends.append(spend)
        return self.rng.choice(best_spends)

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        return sorted(
            range(len(cards)),
            key=lambda index: self._card_value(cards[index]),
            reverse=True,
        )
