import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CampCrossroadsDessertTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("camp-crossroads-dessert")

        self.assertEqual(card.title, "Camp Crossroads Dessert")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_one_energy_per_previous_exercise_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 20
        player.hand.extend(
            [
                make_card("waterski"),
                make_card("biography"),
                make_card("canoe"),
                make_card("camp-crossroads-dessert"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        dessert = game.play_card(0, 0)

        self.assertEqual(player.energy, 12)
        self.assertEqual(game.card_fun(0, dessert), 0)

    def test_only_previous_exercise_cards_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("camp-crossroads-dessert"), make_card("waterski")]
        )

        dessert = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(game.card_fun(0, dessert), 0)


if __name__ == "__main__":
    unittest.main()
