import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CliffClimbingTests(unittest.TestCase):
    def test_outdoors_cards_before_score_bonus_fun_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 17
        player.hand.extend(
            [
                make_card("waterski"),
                make_card("biography"),
                make_card("cliff-climbing"),
                make_card("waterski"),
            ]
        )

        outdoors_before = game.play_card(0, 0)
        other_before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        outdoors_after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(game.card_fun(0, outdoors_before), 8)
        self.assertEqual(game.card_fun(0, other_before), 2)
        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, outdoors_after), 6)


if __name__ == "__main__":
    unittest.main()
