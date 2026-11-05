import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class MorningBikeTests(unittest.TestCase):
    def test_printed_values_and_first_card_restriction(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("biography"), make_card("morning-bike")])

        game.play_card(0, 0)

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

        card = player.hand[0]
        self.assertEqual(card.title, "Morning Bike")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))

    def test_first_play_cost_and_tomorrow_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("morning-bike"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 0)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()
        self.assertEqual(player.energy, 12)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [])
        self.assertFalse(card.is_tomorrow)


if __name__ == "__main__":
    unittest.main()
