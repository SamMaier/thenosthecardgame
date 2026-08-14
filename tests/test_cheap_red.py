import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CheapRedTests(unittest.TestCase):
    def test_next_food_scores_one_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("cheap-red"), make_card("biography"), make_card("cheap-white")]
        )

        card = game.play_card(0, 0)
        non_food = game.play_card(0, 0)
        food = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, non_food), 2)
        self.assertEqual(game.card_fun(0, food), 4)


if __name__ == "__main__":
    unittest.main()
