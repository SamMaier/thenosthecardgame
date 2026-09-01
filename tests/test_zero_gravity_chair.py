import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ZeroGravityChairTests(unittest.TestCase):
    def test_relax_cards_after_cost_one_less(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("zero-gravity-chair"), make_card("biography"), make_card("cheap-white")]
        )

        card = game.play_card(0, 0)
        relax = game.play_card(0, 0)
        non_relax = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(game.energy_cost(0, non_relax), 2)
        self.assertEqual(player.energy, 4)
        self.assertEqual(relax.definition.cost, 1)


if __name__ == "__main__":
    unittest.main()
