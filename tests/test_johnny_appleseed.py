import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class JohnnyAppleseedTests(unittest.TestCase):
    def test_next_event_scores_two_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("johnny-appleseed"), make_card("work-call")])

        card = game.play_card(0, 0)
        event = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertEqual(game.card_fun(0, event), -2)


if __name__ == "__main__":
    unittest.main()
