import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FixedTrunkOrderAI(RandomAI):
    def __init__(self, order):
        super().__init__(random.Random(0))
        self.order = order

    def order_cards_for_trunk(self, game, player_index, cards):
        return self.order


class FishingBoatTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("fishing-boat")

        self.assertEqual(card.title, "Fishing Boat")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_adds_first_outdoors_card_and_reorders_non_outdoors_cards(self) -> None:
        game = empty_game()
        game.ais[0] = FixedTrunkOrderAI((1, 0))
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fishing-boat"))

        first_other = make_card("biography")
        second_other = make_card("thriller-book")
        outdoors = make_card("waterski")
        game.trunk = [outdoors, second_other, first_other]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertEqual(player.hand, [outdoors])
        self.assertEqual(game.trunk, [first_other, second_other])
        self.assertEqual(player.acquired_cards[outdoors.title], 1)
        self.assertEqual(game.stats.card_acquisitions[outdoors.title], 1)
        self.assertIs(game._draw_from_trunk(), second_other)
        self.assertIs(game._draw_from_trunk(), first_other)

    def test_does_not_reorder_or_return_cards_when_top_card_is_outdoors(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fishing-boat"))
        outdoors = make_card("dock-fishing")
        game.trunk = [outdoors]

        game.play_card(0, 0)

        self.assertEqual(player.hand, [outdoors])
        self.assertEqual(game.trunk, [])


if __name__ == "__main__":
    unittest.main()
