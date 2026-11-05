import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class AdventureRaceTests(unittest.TestCase):
    def test_cost_tags_and_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.append(make_card("adventure-race"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 12)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 20)


if __name__ == "__main__":
    unittest.main()
