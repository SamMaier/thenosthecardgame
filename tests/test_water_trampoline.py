import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WaterTrampolineTests(unittest.TestCase):
    def test_only_immediate_next_relax_card_is_doubled(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("water-trampoline"), make_card("cheap-white"), make_card("biography")]
        )

        card = game.play_card(0, 0)
        non_relax = game.play_card(0, 0)
        later_relax = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertEqual(game.card_fun(0, non_relax), 3)
        self.assertEqual(game.card_fun(0, later_relax), 2)


if __name__ == "__main__":
    unittest.main()
