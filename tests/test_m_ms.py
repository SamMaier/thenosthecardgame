import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class MAndMsTests(unittest.TestCase):
    def test_cost_tags_fun_and_energy_gain(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.append(make_card("m-ms"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(player.energy, 1)
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
