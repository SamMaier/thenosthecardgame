import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BugSprayTests(unittest.TestCase):
    def test_outdoors_cards_after_score_one_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("bug-spray"), make_card("dock-fishing")])

        card = game.play_card(0, 0)
        outdoors = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, outdoors), 3)


if __name__ == "__main__":
    unittest.main()
