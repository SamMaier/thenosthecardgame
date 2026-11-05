import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BeaverBurgerTests(unittest.TestCase):
    def test_all_cards_after_cost_one_more(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("beaver-burger"), make_card("tres-fute"), make_card("biography")]
        )

        card = game.play_card(0, 0)
        zero_cost_card = game.play_card(0, 0)
        later_card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))
        self.assertEqual(game.energy_cost(0, later_card), 2)
        self.assertEqual(player.energy, 1)
        self.assertEqual(zero_cost_card.definition.cost, 0)


if __name__ == "__main__":
    unittest.main()
