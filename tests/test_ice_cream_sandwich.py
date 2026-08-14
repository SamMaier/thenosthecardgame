import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class IceCreamSandwichTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("ice-cream-sandwich")

        self.assertEqual(card.title, "Ice Cream Sandwich")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_two_energy_after_a_previous_food_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("cheap-white"), make_card("ice-cream-sandwich")]
        )

        game.play_card(0, 0)
        sandwich = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertTrue(sandwich.markers["_gave_energy"])
        self.assertEqual(game.card_fun(0, sandwich), 1)

    def test_does_not_gain_energy_without_a_previous_food_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("biography"), make_card("ice-cream-sandwich")]
        )

        game.play_card(0, 0)
        sandwich = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertNotIn("_gave_energy", sandwich.markers)

    def test_later_food_card_does_not_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("ice-cream-sandwich"), make_card("cheap-white")]
        )

        sandwich = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertNotIn("_gave_energy", sandwich.markers)

    def test_active_tomorrow_food_does_not_count_as_previous(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        tomorrow_food = make_card("cheap-white")
        tomorrow_food.is_tomorrow = True
        player.tomorrow_cards.append(tomorrow_food)
        player.hand.append(make_card("ice-cream-sandwich"))

        sandwich = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertNotIn("_gave_energy", sandwich.markers)


if __name__ == "__main__":
    unittest.main()
