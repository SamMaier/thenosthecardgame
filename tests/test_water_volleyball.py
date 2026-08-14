import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WaterVolleyballTests(unittest.TestCase):
    def test_next_exercise_scores_four_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 9
        player.hand.extend([make_card("water-volleyball"), make_card("waterski")])

        card = game.play_card(0, 0)
        exercise = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, exercise), 10)


if __name__ == "__main__":
    unittest.main()
