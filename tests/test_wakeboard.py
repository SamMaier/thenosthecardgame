import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WakeboardTests(unittest.TestCase):
    def test_printed_values_and_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 8
        player.hand.append(make_card("wakeboard"))
        wakeboard = game.play_card(0, 0)

        self.assertEqual(wakeboard.title, "Wakeboard")
        self.assertEqual(wakeboard.definition.cost, 8)
        self.assertEqual(wakeboard.definition.base_fun, 10)
        self.assertEqual(
            wakeboard.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )
        self.assertEqual(game.card_fun(0, wakeboard), 10)
        self.assertEqual(player.energy, 0)

    def test_other_cards_do_not_reduce_wakeboard_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("wakeboard"),
                make_card("biography"),
            ]
        )

        wakeboard = game.play_card(0, 0)
        game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, wakeboard), 10)


if __name__ == "__main__":
    unittest.main()
