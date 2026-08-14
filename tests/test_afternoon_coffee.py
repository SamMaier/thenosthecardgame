import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class AfternoonCoffeeTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("afternoon-coffee")

        self.assertEqual(card.title, "Afternoon Coffee")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_cannot_be_played_as_one_of_first_two_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("afternoon-coffee"))

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

        player.played_today.append(make_card("biography"))
        self.assertNotIn(0, game.playable_hand_indices(0))

    def test_is_playable_as_third_card_and_gains_three_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("biography"),
                make_card("biography"),
                make_card("afternoon-coffee"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        self.assertIn(0, game.playable_hand_indices(0))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 7)
        self.assertTrue(card.markers["_gave_energy"])
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
