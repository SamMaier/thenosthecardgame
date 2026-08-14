import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class IceWineTests(unittest.TestCase):
    def test_food_cards_before_score_three_extra_fun_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 14
        player.hand.extend(
            [
                make_card("cheap-white"),
                make_card("waterski"),
                make_card("ice-wine"),
                make_card("cheap-white"),
            ]
        )

        food_before = game.play_card(0, 0)
        other_before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        food_after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.card_fun(0, food_before), 6)
        self.assertEqual(game.card_fun(0, other_before), 6)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, food_after), 3)


if __name__ == "__main__":
    unittest.main()
