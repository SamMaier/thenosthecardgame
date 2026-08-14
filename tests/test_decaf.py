import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class DecafTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("decaf")

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_scores_two_fun_per_leftover_energy_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("decaf"))

        decaf = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, decaf), 8)

        game.end_day()

        self.assertEqual(player.fun, 8)

    def test_scores_zero_when_no_energy_is_left(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("decaf"))

        decaf = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, decaf), 0)


if __name__ == "__main__":
    unittest.main()
