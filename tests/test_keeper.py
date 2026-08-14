import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class KeeperTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("keeper")

        self.assertEqual(card.title, "Keeper")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_pays_two_energy_gains_one_and_scores_two_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("keeper"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertTrue(card.markers["_gave_energy"])


if __name__ == "__main__":
    unittest.main()
