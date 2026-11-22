import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class RouladenTests(unittest.TestCase):
    def test_all_cards_after_cost_one_less_and_cost_cannot_go_negative(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("rouladen"), make_card("cheap-white"), make_card("tres-fute")]
        )

        card = game.play_card(0, 0)
        after = game.play_card(0, 0)
        zero_cost_after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.energy_cost(0, zero_cost_after), 0)
        self.assertEqual(player.energy, 2)
        self.assertEqual(after.definition.cost, 2)


if __name__ == "__main__":
    unittest.main()
