import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CanoeTests(unittest.TestCase):
    def test_next_item_gets_its_written_energy_cost_as_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("canoe"), make_card("cheap-white"), make_card("nos-shirt")]
        )

        card = game.play_card(0, 0)
        non_item = game.play_card(0, 0)
        item = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, non_item), 3)
        self.assertEqual(game.card_fun(0, item), 1)

    def test_written_cost_ignores_a_later_energy_modifier(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("rouladen"),
                make_card("canoe"),
                make_card("nos-shirt"),
            ]
        )

        game.play_card(0, 0)
        canoe = game.play_card(0, 0)
        item = player.hand[0]

        self.assertEqual(game.energy_cost(0, item), 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, canoe), 2)
        self.assertEqual(game.card_fun(0, item), 1)


if __name__ == "__main__":
    unittest.main()
