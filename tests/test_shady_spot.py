import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ShadySpotTests(unittest.TestCase):
    def test_next_outdoors_card_costs_half_rounded_down(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("shady-spot"), make_card("waterski"), make_card("dock-fishing")]
        )

        card = game.play_card(0, 0)
        outdoors = game.play_card(0, 0)
        later_outdoors = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))
        self.assertEqual(game.energy_cost(0, later_outdoors), 1)
        self.assertEqual(player.energy, 2)
        self.assertEqual(outdoors.definition.cost, 5)


if __name__ == "__main__":
    unittest.main()
