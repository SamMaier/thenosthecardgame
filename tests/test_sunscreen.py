import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SunscreenTests(unittest.TestCase):
    def test_outdoors_cards_after_cost_one_less(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 9
        player.hand.extend(
            [make_card("sunscreen"), make_card("waterski"), make_card("biography")]
        )

        card = game.play_card(0, 0)
        outdoors = game.play_card(0, 0)
        indoors = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(game.energy_cost(0, indoors), 1)
        self.assertEqual(player.energy, 1)
        self.assertEqual(outdoors.definition.cost, 5)


if __name__ == "__main__":
    unittest.main()
