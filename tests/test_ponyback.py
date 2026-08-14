import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PonybackTests(unittest.TestCase):
    def test_third_outdoors_card_after_scores_three_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 11
        player.hand.extend(
            [make_card("ponyback"), make_card("dock-fishing"), make_card("canoe"), make_card("waterski")]
        )

        card = game.play_card(0, 0)
        first = game.play_card(0, 0)
        second = game.play_card(0, 0)
        third = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(game.card_fun(0, first), 2)
        self.assertEqual(game.card_fun(0, second), 2)
        self.assertEqual(game.card_fun(0, third), 9)


if __name__ == "__main__":
    unittest.main()
