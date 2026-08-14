import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WakeboardTests(unittest.TestCase):
    def test_printed_values_and_other_cards_before_and_after(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("biography"),
                make_card("wakeboard"),
                make_card("waterski"),
            ]
        )

        before = game.play_card(0, 0)
        wakeboard = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(wakeboard.title, "Wakeboard")
        self.assertEqual(wakeboard.definition.cost, 6)
        self.assertEqual(wakeboard.definition.base_fun, 10)
        self.assertEqual(
            wakeboard.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(game.card_fun(0, wakeboard), 10)
        self.assertEqual(game.card_fun(0, before), 1)
        self.assertEqual(game.card_fun(0, after), 1)

    def test_written_cost_is_used_before_later_fun_modifiers(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("wakeboard"),
                make_card("bring-a-friend"),
                make_card("cheap-white"),
            ]
        )

        wakeboard = game.play_card(0, 0)
        game.play_card(0, 0)
        other = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, wakeboard), 10)
        self.assertEqual(game.card_fun(0, other), 2)


if __name__ == "__main__":
    unittest.main()
