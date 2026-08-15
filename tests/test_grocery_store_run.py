import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FixedTrunkOrderAI(RandomAI):
    def __init__(self, order):
        super().__init__(random.Random(0))
        self.order = order

    def order_cards_for_trunk(self, game, player_index, cards):
        return self.order


class GroceryStoreRunTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("grocery-store-run")

        self.assertEqual(card.title, "Grocery Store Run")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food", "Event"}))

    def test_acquires_food_and_returns_other_cards_in_chosen_order(self) -> None:
        game = empty_game()
        game.ais[0] = FixedTrunkOrderAI((2, 0, 1))
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("grocery-store-run"))

        first_food = make_card("fajitas")
        first_other = make_card("biography")
        second_food = make_card("cheap-white")
        second_other = make_card("waterski")
        third_other = make_card("thriller-book")
        # The Trunk's top is the end of its internal list.
        game.trunk = [
            third_other,
            second_other,
            second_food,
            first_other,
            first_food,
        ]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertEqual(player.hand, [first_food, second_food])
        self.assertEqual(player.acquired_cards[first_food.title], 1)
        self.assertEqual(player.acquired_cards[second_food.title], 1)
        self.assertEqual(game.stats.card_acquisitions[first_food.title], 1)
        self.assertEqual(game.stats.card_acquisitions[second_food.title], 1)
        self.assertEqual(
            game.trunk,
            [second_other, first_other, third_other],
        )
        self.assertIs(game._draw_from_trunk(), third_other)
        self.assertIs(game._draw_from_trunk(), first_other)
        self.assertIs(game._draw_from_trunk(), second_other)

    def test_acquires_all_five_when_every_revealed_card_is_food(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("grocery-store-run"))
        foods = [make_card("fajitas") for _ in range(5)]
        game.trunk = list(reversed(foods))

        game.play_card(0, 0)

        self.assertEqual(player.hand, foods)
        self.assertEqual(len(game.trunk), 0)
        self.assertEqual(player.acquired_cards["Fajitas"], 5)
        self.assertEqual(game.stats.card_acquisitions["Fajitas"], 5)

    def test_rejects_an_invalid_trunk_order(self) -> None:
        game = empty_game()
        game.ais[0] = FixedTrunkOrderAI((0, 0, 1, 2, 3))
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("grocery-store-run"))
        game.trunk = [make_card("biography") for _ in range(5)]

        with self.assertRaisesRegex(ValueError, "every Trunk-order index"):
            game.play_card(0, 0)


if __name__ == "__main__":
    unittest.main()
