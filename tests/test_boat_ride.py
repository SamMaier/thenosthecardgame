import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BoatRideTests(unittest.TestCase):
    def test_next_item_costs_zero_and_later_items_cost_normally(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("boat-ride"), make_card("biography"), make_card("nos-shirt"), make_card("nos-shirt")]
        )

        card = game.play_card(0, 0)
        non_item = game.play_card(0, 0)
        first_item = game.play_card(0, 0)
        later_item = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Event", "Outdoors"}))
        self.assertEqual(non_item.definition.tags, frozenset({"Relax"}))
        self.assertEqual(first_item.definition.cost, 1)
        self.assertEqual(later_item.definition.cost, 1)
        self.assertEqual(player.energy, 3)


if __name__ == "__main__":
    unittest.main()
