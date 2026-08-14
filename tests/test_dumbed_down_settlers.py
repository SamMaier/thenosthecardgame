import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class DumbedDownSettlersTests(unittest.TestCase):
    def test_cost_tags_and_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("dumbed-down-settlers"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 4)


if __name__ == "__main__":
    unittest.main()
