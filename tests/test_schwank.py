import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SchwankTests(unittest.TestCase):
    def test_third_exercise_after_scores_double_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 15
        player.hand.extend(
            [make_card("schwank"), make_card("waterski"), make_card("canoe"), make_card("kneeboard")]
        )

        card = game.play_card(0, 0)
        first = game.play_card(0, 0)
        second = game.play_card(0, 0)
        third = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.card_fun(0, first), 6)
        self.assertEqual(game.card_fun(0, second), 2)
        self.assertEqual(game.card_fun(0, third), 10)


if __name__ == "__main__":
    unittest.main()
