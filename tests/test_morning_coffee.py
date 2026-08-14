import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class MorningCoffeeTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("morning-coffee")

        self.assertEqual(card.title, "Morning Coffee")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_must_be_first_card_and_gains_three_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("biography"), make_card("morning-coffee")])

        game.play_card(0, 0)

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)
        self.assertEqual(player.energy, 6)

        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("morning-coffee"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 9)
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
