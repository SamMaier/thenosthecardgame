import unittest

from thenos.ais.hydrogen import _hydrogen_copy_game
from thenos.cards import make_card
from tests.helpers import empty_game


class ForcedFamilyFunTests(unittest.TestCase):
    def test_next_board_game_is_halved_and_later_board_games_are_not(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("forced-family-fun"), make_card("azul"), make_card("azul")]
        )

        card = game.play_card(0, 0)
        first_board_game = game.play_card(0, 0)
        later_board_game = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Board Game", "Indoors"}))
        self.assertEqual(game.energy_cost(0, later_board_game), 3)
        self.assertEqual(player.energy, 1)
        self.assertEqual(first_board_game.definition.cost, 3)

    def test_per_step_zero_clamp_prevents_halving_negative_cost_and_matches_sims(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 15
        game.suitcase = [make_card("nap") for _ in range(4)]
        game.trunk = [make_card("nap") for _ in range(4)]
        player.hand.extend(
            [
                make_card("bend-the-rules"),
                make_card("forced-family-fun"),
                make_card("beaver-burgers"),
                make_card("medical-advice"),
            ]
        )
        for _ in range(4):
            game.play_card(0, 0)

        target = make_card("tres-fute")
        player.hand = [target]
        self.assertEqual(game.energy_cost(0, target), 2)

        for sim_copy in (_hydrogen_copy_game(game),):
            self.assertEqual(sim_copy.energy_cost(0, sim_copy.players[0].hand[0]), 2)
            if hasattr(sim_copy, "cards_energy_costs"):
                self.assertEqual(
                    sim_copy.cards_energy_costs(0, sim_copy.players[0].hand), [2]
                )
            if hasattr(sim_copy, "hand_playable_with_costs"):
                self.assertEqual(sim_copy.hand_playable_with_costs(0), [(0, 2)])


if __name__ == "__main__":
    unittest.main()
