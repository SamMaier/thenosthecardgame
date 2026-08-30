import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FancyCraftTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("fancy-craft")

        self.assertEqual(card.title, "Fancy Craft")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_next_item_scores_twice_its_written_energy_cost(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("fancy-craft"),
                make_card("biography"),
                make_card("ponyback"),
            ]
        )

        fancy_craft = game.play_card(0, 0)
        non_item = game.play_card(0, 0)
        item = game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(game.card_fun(0, fancy_craft), 0)
        self.assertEqual(game.card_fun(0, non_item), 2)
        self.assertEqual(game.card_fun(0, item), 4)


if __name__ == "__main__":
    unittest.main()
