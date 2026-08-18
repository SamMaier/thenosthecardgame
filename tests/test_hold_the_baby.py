import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class HoldTheBabyTests(unittest.TestCase):
    def test_later_cards_cost_one_more_and_score_two_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("biography"),
                make_card("hold-the-baby"),
                make_card("johnny-appleseed"),
                make_card("waterski"),
            ]
        )

        before = game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Hold the Baby")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.energy_cost(0, player.hand[0]), 2)
        self.assertEqual(game.energy_cost(0, player.hand[1]), 6)

        social = game.play_card(0, 0)
        exercise = game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(game.card_fun(0, before), 2)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, social), 3)
        self.assertEqual(game.card_fun(0, exercise), 8)


if __name__ == "__main__":
    unittest.main()
