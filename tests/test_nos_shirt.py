import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NosShirtTests(unittest.TestCase):
    def test_printed_values_and_scores_four_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("nos-shirt"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Nos Shirt")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 4)


if __name__ == "__main__":
    unittest.main()
