"""A fast, catalog-agnostic competitive search policy."""

from __future__ import annotations

import copy
import math
import random
from typing import Sequence, TYPE_CHECKING

from thenos.ais.planner import PlannerAI
from thenos.cards.base import CardInstance

if TYPE_CHECKING:
    from thenos.game import Game


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

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(rng)

    def _state_value(self, game: Game, player_index: int) -> float:
        """Value a public position with one scoring copy and cheap reserves."""
        scoring = copy.deepcopy(game)
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
                + 0.24 * self._future_hand_value(scoring, player_index)
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
