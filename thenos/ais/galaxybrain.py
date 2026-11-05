"""A fast, public-information competitive search policy."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from thenos.ais.megamind import MegamindAI

if TYPE_CHECKING:
    from thenos.game import Game


class GalaxybrainAI(MegamindAI):
    """Search more selectively than Megamind on public-safe state samples.

    The policy has no card-name table and does not retain learned catalog
    rankings.  Card text is evaluated by resolving it on independent engine
    copies.  Before search, the live Trunk and hidden opponent hands are pooled
    and deterministically resampled with the policy RNG, so neither their live
    order nor their hidden allocation can affect a decision.
    """

    SEARCH_DEPTH = 3
    BEAM_WIDTH = 4
    BRANCH_WIDTH = 5
    ROOT_WIDTH = 9
    FUTURE_WEIGHT = 0.50
    FUTURE_HAND_WEIGHT = 0.24
    # Most future hand value exists with or without today's setup. Credit the
    # improvement caused specifically by visible Tomorrow text almost at full
    # future value, while slightly strengthening the existing late-game margin
    # objective so raw setup value cannot displace a close competitive win.
    TOMORROW_RESERVE_BONUS_WEIGHT = 0.80
    PENULTIMATE_MARGIN_WEIGHT = 2.5
    FINAL_MARGIN_WEIGHT = 6.5

    def _copy_game(self, game: Game) -> Game:
        return game.copy_for_simulation()

    def _planning_copy(
        self,
        game: Game,
        player_index: int | None = None,
    ) -> Game:
        """Create a reproducible sample that is independent of hidden state."""
        if player_index is None:
            raise ValueError("Galaxybrain planning requires a player index")

        simulation = self._copy_game(game)
        seed = self.rng.getrandbits(64)
        simulation.rng = random.Random(seed)
        planning_rng = random.Random(seed)
        simulation.sample_daily_conditions(player_index, planning_rng)

        hidden_counts = {
            index: len(player.hand)
            for index, player in enumerate(simulation.players)
            if index != player_index
        }
        unknown_cards = list(simulation.trunk)
        for index in hidden_counts:
            unknown_cards.extend(simulation.players[index].hand)
        unknown_cards.sort(key=lambda card: card.instance_id)
        planning_rng.shuffle(unknown_cards)

        cursor = 0
        for index, count in hidden_counts.items():
            simulation.players[index].hand = unknown_cards[cursor : cursor + count]
            cursor += count
        simulation.trunk = unknown_cards[cursor:]

        simulation.discard.sort(key=lambda card: card.instance_id)
        planning_rng.shuffle(simulation.discard)
        for ai in simulation.ais:
            if isinstance(ai, GalaxybrainAI):
                ai._evaluation_depth = self._evaluation_depth + 1
            elif isinstance(ai, MegamindAI):
                ai._evaluation_depth = self._evaluation_depth + 1
        return simulation

    def _suitcase_pick_value(
        self,
        root: Game,
        player_index: int,
        suitcase_index: int,
        *,
        extra_pick_cost: int = 0,
    ) -> float:
        """Value a pick with an exact bounded plan for the resulting hand."""
        simulation = self._copy_game(root)
        simulation.players[player_index].energy -= extra_pick_cost
        simulation.pick_suitcase_cards(
            player_index, (simulation.suitcase[suitcase_index],)
        )

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

        playable = simulation.playable_hand_indices(player_index)
        if playable:
            _, planned_value = self._best_play(
                simulation,
                player_index,
                playable,
                depth=2,
                beam_width=3,
            )
            value = max(value, planned_value)
        return value
