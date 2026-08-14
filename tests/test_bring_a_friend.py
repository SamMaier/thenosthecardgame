import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BringAFriendTests(unittest.TestCase):
    def test_cards_after_cost_double_and_score_double(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("bring-a-friend"), make_card("cheap-white")])

        card = game.play_card(0, 0)
        self.assertEqual(game.energy_cost(0, player.hand[0]), 4)
        after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, after), 6)


if __name__ == "__main__":
    unittest.main()
