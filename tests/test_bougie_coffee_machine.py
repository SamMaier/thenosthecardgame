import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BougieCoffeeMachineTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("bougie-coffee-machine")

        self.assertEqual(card.title, "Bougie Coffee Machine")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_plays_drawn_food_cards_for_zero_energy_and_keeps_other_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        machine = make_card("bougie-coffee-machine")
        player.hand.append(machine)

        first_food = make_card("fajitas")
        other_card = make_card("biography")
        second_food = make_card("cheap-white")
        # The Trunk's top is the end of its internal list.
        game.trunk = [second_food, other_card, first_food]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 7)
        self.assertEqual(player.played_today, [machine, first_food, second_food])
        self.assertEqual(player.hand, [other_card])
        self.assertEqual(game.stats.card_plays["Fajitas"], 1)
        self.assertEqual(game.stats.card_plays["Cheap White"], 1)
        self.assertEqual(game.stats.card_acquisitions["Biography"], 1)
        self.assertEqual(game.stats.card_acquisitions["Fajitas"], 0)
        self.assertEqual(game.trunk, [])

    def test_restricted_drawn_food_is_played_ignoring_its_restriction(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        machine = make_card("bougie-coffee-machine")
        player.hand.append(machine)

        restricted_food = make_card("morning-coffee")
        other_card = make_card("biography")
        food = make_card("fajitas")
        game.trunk = [food, other_card, restricted_food]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 10)
        self.assertEqual(player.hand, [other_card])
        self.assertEqual(
            player.played_today,
            [machine, restricted_food, food],
        )
        self.assertEqual(game.stats.card_plays["Morning Coffee"], 1)
        self.assertEqual(game.stats.card_acquisitions["Morning Coffee"], 0)


if __name__ == "__main__":
    unittest.main()
