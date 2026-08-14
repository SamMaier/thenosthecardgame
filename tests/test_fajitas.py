import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FajitasTests(unittest.TestCase):
    def test_costs_three_then_gains_four_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fajitas"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(player.energy, 8)
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
