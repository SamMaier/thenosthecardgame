import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NonFictionBookTests(unittest.TestCase):
    def test_costs_one_energy_and_scores_two_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("non-fiction-book"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Non-Fiction Book")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))
        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 2)


if __name__ == "__main__":
    unittest.main()
