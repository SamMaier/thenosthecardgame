import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NosBookTests(unittest.TestCase):
    def test_social_and_event_before_and_after_score_bonus_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("johnny-appleseed"), make_card("nos-book"), make_card("work-call")]
        )

        before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(card.title, "Nos Book")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(game.card_fun(0, before), 2)
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, after), -1)


if __name__ == "__main__":
    unittest.main()
