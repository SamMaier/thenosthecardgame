import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WurstchenTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("wurstchen")

        self.assertEqual(card.title, "Würstchen")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_one_energy_per_card_remaining_in_hand(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("wurstchen"), make_card("biography"), make_card("fajitas")]
        )

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertEqual(len(player.hand), 2)
        self.assertTrue(card.markers["_gave_energy"])

    def test_gains_no_energy_with_an_empty_hand(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.append(make_card("wurstchen"))

        game.play_card(0, 0)

        self.assertEqual(player.energy, 0)


if __name__ == "__main__":
    unittest.main()
