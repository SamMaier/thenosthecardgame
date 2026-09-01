import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class HighEndWhiteTests(unittest.TestCase):
    def test_food_cards_after_score_one_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("high-end-white"), make_card("cheap-white")])

        card = game.play_card(0, 0)
        food = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, food), 4)


if __name__ == "__main__":
    unittest.main()
