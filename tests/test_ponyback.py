import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PonybackTests(unittest.TestCase):
    def test_tomorrow_outdoors_cards_cost_one_less(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("ponyback"))

        card = game.play_card(0, 0)
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

        game.end_day()
        game.start_day()
        player.hand.extend([make_card("waterski"), make_card("biography")])

        self.assertEqual(game.energy_cost(0, player.hand[0]), 4)
        self.assertEqual(game.energy_cost(0, player.hand[1]), 1)


if __name__ == "__main__":
    unittest.main()
