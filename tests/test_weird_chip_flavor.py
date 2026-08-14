import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WeirdChipFlavorTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("weird-chip-flavor")

        self.assertEqual(card.title, "Weird Chip Flavor")
        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, -2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_two_energy_and_scores_minus_two_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.append(make_card("weird-chip-flavor"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 2)
        self.assertEqual(game.card_fun(0, card), -2)


if __name__ == "__main__":
    unittest.main()
