import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CheapWhiteTests(unittest.TestCase):
    def test_cost_tags_and_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("cheap-white"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(player.energy, 5)
        self.assertEqual(game.card_fun(0, card), 3)


if __name__ == "__main__":
    unittest.main()
