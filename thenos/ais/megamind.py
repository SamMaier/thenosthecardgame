"""A fast, catalog-agnostic competitive search policy."""

from __future__ import annotations

import copy
import math
import random
from statistics import median
from typing import Sequence, TYPE_CHECKING

from thenos.ais.planner import PlannerAI
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import CARD_REGISTRY

if TYPE_CHECKING:
    from thenos.game import Game


_PROJECTED_PLAY_DEFINITION = CardDefinition(
    slug="__megamind_projected_play__",
    title="Projected opponent play",
    tags=frozenset(),
    cost=0,
    behavior=CardBehavior(),
)
_TYPICAL_CARD_COST = max(
    1,
    int(median(definition.cost for definition in CARD_REGISTRY.values())),
)


class MegamindAI(PlannerAI):
    """Search strong same-day plans without knowing any individual card.

    Megamind deliberately depends only on the public ``Game`` API and on the
    common properties exposed by every card.  In particular, it contains no
    card titles, slugs, behavior types, or catalog-derived lookup table.  Card
    text is evaluated by playing cards on private engine copies.

    Compared with :class:`GeniusAI`, the search spends less work evaluating
    each node and puts that budget into the choices that most often decide a
    game: ordering today's plays and converting Energy into final-day Fun.
    """

    SEARCH_DEPTH = 4
    BEAM_WIDTH = 5
    BRANCH_WIDTH = 6
    ROOT_WIDTH = 10
    FUTURE_WEIGHT = 0.50
    FUTURE_HAND_WEIGHT = 0.24

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)

    @staticmethod
    def _projected_opponent_plays(
        game: Game,
        player_index: int,
        opponent_index: int,
    ) -> int:
        """Estimate remaining opponent plays from public state only.

        Opponent card identities are hidden, but hand size, asleep state, and
        remaining Energy are public.  A catalog-wide median cost gives a
        stable, card-agnostic estimate of how many cards an unknown opponent
        could still play; rounding up keeps conditional effects conservative.
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
        """Add no-op plays to a scoring copy for public future pressure."""
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

    def _state_value(self, game: Game, player_index: int) -> float:
        """Value a public position with one scoring copy and cheap reserves."""
        scoring = copy.deepcopy(game)
        self._add_projected_opponent_plays(scoring, player_index)
        scoring.end_day()
        scored_player = scoring.players[player_index]
        own_score = scored_player.fun

        from thenos.game import DAILY_ENERGY, DAYS_PER_GAME

        future_value = 0.0
        if game.day < DAYS_PER_GAME:
            scored_player.energy = DAILY_ENERGY
            scored_player.asleep = False
            for card in scored_player.tomorrow_cards:
                card.effective_behavior.on_start_day(
                    scoring, scored_player, card
                )
            immediate_future = scored_player.fun - own_score
            energy_delta = scored_player.energy - DAILY_ENERGY
            future_value = (
                immediate_future
                + 0.65 * energy_delta
                + self.FUTURE_HAND_WEIGHT
                * self._future_hand_value(scoring, player_index)
            )

        value = own_score + self.FUTURE_WEIGHT * future_value

        # Scores are public.  Late in the game, prefer points which change the
        # win rather than treating every point in a blowout as equally useful.
        if game.day >= DAYS_PER_GAME - 1:
            best_opponent = max(
                opponent.fun
                for index, opponent in enumerate(scoring.players)
                if index != player_index
            )
            margin = own_score - best_opponent
            weight = 2.0 if game.day == DAYS_PER_GAME - 1 else 6.0
            value += weight * math.tanh(margin / 3.0)
        return value

    def _suitcase_pick_value(
        self,
        root: Game,
        player_index: int,
        suitcase_index: int,
        *,
        extra_pick_cost: int = 0,
    ) -> float:
        """Value a pick against a complete, bounded same-day allocation.

        The inherited evaluator tries leaving the picked card in hand or
        playing only that card.  Since Suitcase choices happen before the
        playing phase, that favors one large card over several smaller cards
        that fit the same Energy budget.  Add the existing rules-aware
        knapsack estimate for the whole hand.  This is linear in hand size and
        Energy capacity, so it captures multi-card allocation without running
        another beam search for every visible Suitcase card.

        The ordinary exact-play branch is retained for immediate card effects
        which the allocation estimate cannot model in isolation.
        """
        simulation = copy.deepcopy(root)
        simulation.players[player_index].energy -= extra_pick_cost
        picked = simulation.pick_suitcase_cards(
            player_index, (simulation.suitcase[suitcase_index],)
        )[0]

        value = self._state_value(simulation, player_index)
        player = simulation.players[player_index]
        allocation_value = self._future_hand_value(
            simulation,
            player_index,
            capacity=player.energy,
        )

        from thenos.game import DAYS_PER_GAME

        reserve_credit = (
            self.FUTURE_WEIGHT * self.FUTURE_HAND_WEIGHT
            if simulation.day < DAYS_PER_GAME
            else 0.0
        )
        value += (1.0 - reserve_credit) * allocation_value

        if picked in player.hand:
            hand_index = player.hand.index(picked)
            if hand_index in simulation.playable_hand_indices(player_index):
                played = copy.deepcopy(simulation)
                played.play_card(player_index, hand_index)
                value = max(value, self._state_value(played, player_index))
        return value

    def choose_cards_to_discard(
        self,
        game: Game,
        player_index: int,
        hand: Sequence[CardInstance],
    ) -> Sequence[int]:
        """Cycle weak reserves aggressively for any multi-discard effect."""
        from thenos.game import DAYS_PER_GAME

        if game.day >= DAYS_PER_GAME:
            return tuple(range(len(hand)))
        return tuple(
            index
            for index, card in enumerate(hand)
            if self._card_value(card) < 4.5
        )

    def _best_play(
        self,
        root: Game,
        player_index: int,
        playable: Sequence[int],
        *,
        depth: int | None = None,
        beam_width: int | None = None,
    ) -> tuple[int, float]:
        """Use a slightly wider exact-payoff search on the final day."""
        from thenos.game import DAYS_PER_GAME

        if depth is not None or root.day < DAYS_PER_GAME:
            return super()._best_play(
                root,
                player_index,
                playable,
                depth=depth,
                beam_width=beam_width,
            )

        original_depth = self.SEARCH_DEPTH
        original_width = self.BEAM_WIDTH
        original_branch = self.BRANCH_WIDTH
        original_root = self.ROOT_WIDTH
        try:
            self.SEARCH_DEPTH = 6
            self.BEAM_WIDTH = 9
            self.BRANCH_WIDTH = 9
            self.ROOT_WIDTH = 16
            return super()._best_play(root, player_index, playable)
        finally:
            self.SEARCH_DEPTH = original_depth
            self.BEAM_WIDTH = original_width
            self.BRANCH_WIDTH = original_branch
            self.ROOT_WIDTH = original_root
