import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CalzonesTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("calzones")

        self.assertEqual(card.title, "Calzones")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_three_energy_and_scores_two_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("calzones"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 7)
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertTrue(card.markers["_gave_energy"])

    def test_tomorrow_exercise_cost_increase(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("calzones"))

        calzones = game.play_card(0, 0)
        player.hand.extend([make_card("waterski"), make_card("biography")])

        self.assertEqual(game.energy_cost(0, player.hand[0]), 5)
        self.assertEqual(game.energy_cost(0, player.hand[1]), 1)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [calzones])
        self.assertTrue(calzones.is_tomorrow)

        game.start_day()
        self.assertEqual(game.energy_cost(0, player.hand[0]), 6)
        self.assertEqual(game.energy_cost(0, player.hand[1]), 1)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [])
        self.assertFalse(calzones.is_tomorrow)


if __name__ == "__main__":
    unittest.main()
