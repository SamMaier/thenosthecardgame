import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class HighEndRedTests(unittest.TestCase):
    def test_food_cards_after_score_two_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 8
        player.hand.extend(
            [make_card("high-end-red"), make_card("cheap-white"), make_card("cheap-red")]
        )

        card = game.play_card(0, 0)
        first_food = game.play_card(0, 0)
        second_food = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertEqual(game.card_fun(0, first_food), 5)
        self.assertEqual(game.card_fun(0, second_food), 4)


if __name__ == "__main__":
    unittest.main()
