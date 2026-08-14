import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WorkCallTests(unittest.TestCase):
    def test_negative_fun_and_immediate_next_card_double(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("work-call"), make_card("biography"), make_card("biography")]
        )

        card = game.play_card(0, 0)
        next_card = game.play_card(0, 0)
        later_card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Event", "Indoors"}))
        self.assertEqual(game.card_fun(0, card), -4)
        self.assertEqual(game.card_fun(0, next_card), 4)
        self.assertEqual(game.card_fun(0, later_card), 2)


if __name__ == "__main__":
    unittest.main()
