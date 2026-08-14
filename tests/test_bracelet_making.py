import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BraceletMakingTests(unittest.TestCase):
    def test_next_social_scores_four_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("bracelet-making"), make_card("johnny-appleseed")])

        card = game.play_card(0, 0)
        social = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertEqual(game.card_fun(0, social), 5)


if __name__ == "__main__":
    unittest.main()
