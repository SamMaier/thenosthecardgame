import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ShadySpotTests(unittest.TestCase):
    def test_outdoors_cards_after_cost_one_less(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 9
        player.hand.extend(
            [make_card("shady-spot"), make_card("waterski"), make_card("biography")]
        )

        card = game.play_card(0, 0)
        outdoors = game.play_card(0, 0)
        indoors = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))
        self.assertEqual(game.energy_cost(0, indoors), 1)
        self.assertEqual(player.energy, 0)
        self.assertEqual(outdoors.definition.cost, 5)


if __name__ == "__main__":
    unittest.main()
