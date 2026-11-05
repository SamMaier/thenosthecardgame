import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class MovieTests(unittest.TestCase):
    def test_indoors_cards_after_score_two_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("movie"), make_card("euchre")])

        card = game.play_card(0, 0)
        indoors = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Indoors"}))
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertEqual(game.card_fun(0, indoors), 4)


if __name__ == "__main__":
    unittest.main()
