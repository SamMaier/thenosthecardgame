import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class DishwashingTests(unittest.TestCase):
    def test_printed_values_and_tomorrow_food_discount(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("dishwashing"))

        dishwashing = game.play_card(0, 0)

        self.assertEqual(dishwashing.definition.cost, 1)
        self.assertEqual(dishwashing.definition.base_fun, 0)
        self.assertEqual(dishwashing.definition.tags, frozenset({"Event"}))
        self.assertEqual(player.energy, 6)

        player.hand.extend(
            [make_card("cheap-white"), make_card("waterski"), make_card("m-ms")]
        )
        self.assertEqual(game.energy_cost(0, player.hand[0]), 2)

        game.end_day()

        self.assertEqual(len(player.tomorrow_cards), 1)
        player.energy = 7
        self.assertEqual(game.energy_cost(0, player.hand[0]), 1)
        self.assertEqual(game.energy_cost(0, player.hand[1]), 5)
        self.assertEqual(game.energy_cost(0, player.hand[2]), 0)


if __name__ == "__main__":
    unittest.main()
