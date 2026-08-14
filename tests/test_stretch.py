import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class StretchTests(unittest.TestCase):
    def test_exercise_cards_after_score_bonus_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("stretch"), make_card("waterski")])

        card = game.play_card(0, 0)
        exercise = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.tags, frozenset({"Exercise"}))
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, exercise), 7)


if __name__ == "__main__":
    unittest.main()
