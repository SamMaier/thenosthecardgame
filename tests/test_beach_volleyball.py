import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BeachVolleyballTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("beach-volleyball")

        self.assertEqual(card.title, "Beach Volleyball")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_next_indoors_card_gains_two_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.extend(
            [make_card("beach-volleyball"), make_card("nap")]
        )

        card = game.play_card(0, 0)
        indoors = game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertEqual(game.card_fun(0, card), 4)
        self.assertEqual(indoors.tags, frozenset({"Relax", "Indoors"}))

    def test_non_indoors_next_card_does_not_trigger_later_indoors_card(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.extend(
            [
                make_card("beach-volleyball"),
                make_card("biography"),
                make_card("nap"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 1)


if __name__ == "__main__":
    unittest.main()
