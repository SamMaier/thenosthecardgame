import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BendTheRulesTests(unittest.TestCase):
    def test_board_games_after_cost_one_less(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("bend-the-rules"), make_card("azul"), make_card("biography")]
        )

        card = game.play_card(0, 0)
        board_game = game.play_card(0, 0)
        non_board_game = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.energy_cost(0, non_board_game), 1)
        self.assertEqual(player.energy, 3)
        self.assertEqual(board_game.definition.cost, 3)


if __name__ == "__main__":
    unittest.main()
