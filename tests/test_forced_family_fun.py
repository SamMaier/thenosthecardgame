import unittest

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
        self.assertEqual(card.definition.tags, frozenset({"Event", "Indoors"}))
        self.assertEqual(game.energy_cost(0, later_board_game), 3)
        self.assertEqual(player.energy, 1)
        self.assertEqual(first_board_game.definition.cost, 3)


if __name__ == "__main__":
    unittest.main()
