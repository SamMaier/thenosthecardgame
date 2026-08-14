import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CharcuterieTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("charcuterie")

        self.assertEqual(card.title, "Charcuterie")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_one_energy_after_payment_and_scores_two_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("charcuterie"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertTrue(card.markers["_gave_energy"])


if __name__ == "__main__":
    unittest.main()
