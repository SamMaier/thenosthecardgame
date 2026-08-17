"""A myopic player that maximizes its score if the day ended now."""

from __future__ import annotations

import copy
import random
from typing import Sequence, TYPE_CHECKING

from thenos.ais.random_ai import RandomAI
from thenos.cards.base import CardInstance

if TYPE_CHECKING:
    from thenos.game import Game


class GreedyAI(RandomAI):
    """Choose the action with the best one-step end-of-day score.

    Card plays are evaluated on deep copies of the game, so immediate effects,
    visible-card interactions, and end-of-day scoring all count. The policy does
    not search subsequent turns: after every real decision it evaluates again.
    Random tie-breaking keeps equivalent choices unbiased and reproducible.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)
        self._evaluation_depth = 0
        self._pending_play_choice: tuple[tuple[object, ...], int] | None = None

    @staticmethod
    def _play_decision_key(
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> tuple[object, ...]:
        """Identify one observable play decision without hidden card values."""
        player = game.players[player_index]
        public_players = tuple(
            (
                candidate.fun,
                candidate.energy,
                candidate.asleep,
                len(candidate.hand),
                tuple(card.instance_id for card in candidate.visible_cards),
            )
            for candidate in game.players
        )
        return (
            id(game),
            game.day,
            game.starting_player,
            player_index,
            player.energy,
            player.fun,
            tuple(playable_hand_indices),
            tuple(card.instance_id for card in player.hand),
            tuple(card.instance_id for card in game.suitcase),
            public_players,
        )

    def _cache_play_choice(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
        choice: int,
    ) -> None:
        self._pending_play_choice = (
            self._play_decision_key(game, player_index, playable_hand_indices),
            choice,
        )

    def _take_cached_play_choice(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        pending = self._pending_play_choice
        self._pending_play_choice = None
        if pending is None:
            return None
        key, choice = pending
        if key != self._play_decision_key(
            game, player_index, playable_hand_indices
        ):
            return None
        return choice

    def _choose_best(self, values: Sequence[int]) -> int:
        if not values:
            raise ValueError("Cannot choose from an empty selection")
        best = max(values)
        return self.rng.choice(
            [index for index, value in enumerate(values) if value == best]
        )

    @staticmethod
    def _printed_value(card: CardInstance) -> int:
        return card.effective_base_fun

    @staticmethod
    def _score_if_day_ended(game: Game, player_index: int) -> int:
        simulation = copy.deepcopy(game)
        simulation.end_day()
        return simulation.players[player_index].fun

    def _score_after_play(
        self,
        game: Game,
        player_index: int,
        hand_index: int,
    ) -> int:
        simulation = copy.deepcopy(game)
        simulated_ai = simulation.ais[player_index]
        if isinstance(simulated_ai, GreedyAI):
            simulated_ai._evaluation_depth = self._evaluation_depth + 1
        simulation.play_card(player_index, hand_index)
        simulation.end_day()
        return simulation.players[player_index].fun

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
            values = [
                self._printed_value(game.players[player_index].hand[index])
                for index in playable_hand_indices
            ]
        else:
            values = [
                self._score_after_play(game, player_index, index)
                for index in playable_hand_indices
            ]
        return playable_hand_indices[self._choose_best(values)]

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
            return False

        current_score = self._score_if_day_ended(game, player_index)
        projected_scores = [
            self._score_after_play(game, player_index, hand_index)
            for hand_index in playable_hand_indices
        ]
        best_position = self._choose_best(projected_scores)
        if projected_scores[best_position] <= current_score:
            return True
        self._cache_play_choice(
            game,
            player_index,
            playable_hand_indices,
            playable_hand_indices[best_position],
        )
        return False

    def choose_extra_card_to_play(
        self,
        game: Game,
        player_index: int,
        playable_hand_indices: Sequence[int],
    ) -> int | None:
        choice = self.choose_card_to_play(
            game, player_index, playable_hand_indices
        )
        if self._evaluation_depth:
            return choice
        current_score = self._score_if_day_ended(game, player_index)
        projected_score = self._score_after_play(game, player_index, choice)
        return choice if projected_score > current_score else None

    def _score_after_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase_index: int,
        *,
        extra_pick_cost: int = 0,
    ) -> int:
        simulation = copy.deepcopy(game)
        simulated_ai = simulation.ais[player_index]
        if isinstance(simulated_ai, GreedyAI):
            simulated_ai._evaluation_depth = self._evaluation_depth + 1
        player = simulation.players[player_index]
        player.energy -= extra_pick_cost
        picked = simulation.pick_suitcase_cards(
            player_index, (simulation.suitcase[suitcase_index],)
        )[0]
        if picked in player.hand:
            hand_index = player.hand.index(picked)
            if hand_index in simulation.playable_hand_indices(player_index):
                simulation.play_card(player_index, hand_index)
        simulation.end_day()
        return simulation.players[player_index].fun

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
                [self._printed_value(card) for card in suitcase]
            )
        return self._choose_best(
            [
                self._score_after_suitcase_pick(game, player_index, index)
                for index in range(len(suitcase))
            ]
        )

    def choose_extra_suitcase_pick(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> bool:
        if self._evaluation_depth or not suitcase:
            return False
        current_score = self._score_if_day_ended(game, player_index)
        best_pick_score = max(
            self._score_after_suitcase_pick(
                game, player_index, index, extra_pick_cost=1
            )
            for index in range(len(suitcase))
        )
        return best_pick_score > current_score

    def choose_card_to_copy(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        return self._choose_best(
            [self._printed_value(card) for card in eligible_cards]
        )

    def choose_card_target(
        self,
        game: Game,
        player_index: int,
        eligible_cards: Sequence[CardInstance],
    ) -> int:
        return self._choose_best(
            [self._printed_value(card) for card in eligible_cards]
        )

    def choose_suitcase_target(
        self,
        game: Game,
        player_index: int,
        suitcase: Sequence[CardInstance],
    ) -> int:
        return self._choose_best(
            [self._printed_value(card) for card in suitcase]
        )

    def choose_card_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> int:
        if not hand:
            raise ValueError("Cannot choose a card from an empty hand")
        # Preserve high-Fun and cheap cards; discard the least promising copy.
        values = [
            (-card.effective_base_fun, card.effective_cost)
            for card in hand
        ]
        best = max(values)
        return self.rng.choice(
            [index for index, value in enumerate(values) if value == best]
        )

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> Sequence[int]:
        # A generic immediate-value policy cannot prove that an unknown
        # replacement improves today's score, so it keeps its current hand.
        return ()

    def choose_optional_action(
        self,
        game: Game,
        player_index: int,
        action: str,
    ) -> bool:
        # Every currently modeled optional action awards immediate Fun.
        return True

    def choose_energy_to_spend(
        self,
        game: Game,
        player_index: int,
        card: CardInstance,
        maximum: int,
    ) -> int:
        # The currently modeled optional spend converts Energy directly to Fun.
        return maximum

    def order_cards_for_trunk(
        self,
        game: Game,
        player_index: int,
        cards: Sequence[CardInstance],
    ) -> Sequence[int]:
        return sorted(
            range(len(cards)),
            key=lambda index: cards[index].effective_base_fun,
            reverse=True,
        )
