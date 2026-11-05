import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WaterskiTests(unittest.TestCase):
    def test_costs_five_energy_and_scores_six_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("waterski"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(player.energy, 2)
        self.assertEqual(game.card_fun(0, card), 6)


if __name__ == "__main__":
    unittest.main()
