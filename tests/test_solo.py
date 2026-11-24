import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SoloTests(unittest.TestCase):
    def test_cost_tags_and_base_fun(self) -> None:
        card = make_card("solo")

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_rescores_card_after_end_day_when_leading_or_tied_today(
        self,
    ) -> None:
        # Tied for today's Fun (despite lagging in prior cumulative Fun), plus Work Call doubles Solo.
        game = empty_game()
        p0 = game.players[0]
        p1 = game.players[1]
        p0.fun = 0
        p1.fun = 20
        p0.energy = 10
        p1.energy = 10
        p0.hand.extend([make_card("work-call"), make_card("solo")])
        p1.hand.append(make_card("biography"))

        game.play_card(0, 0)
        solo = game.play_card(0, 0)
        game.play_card(1, 0)  # scores 2 today

        # Work Call (-4) + Solo (3*2=6) = 2 Fun today, tying p1's 2 Fun.
        self.assertEqual(game.card_fun(0, solo), 6)
        game.end_day()

        # Solo rescores its 6 Fun -> 2 + 6 = 8.
        self.assertEqual(p0.fun, 8)

    def test_does_not_rescore_when_behind_an_opponent_today(self) -> None:
        game = empty_game()
        p0 = game.players[0]
        p1 = game.players[1]
        p0.energy = 7
        p1.energy = 7
        p0.hand.append(make_card("solo"))
        p1.hand.append(make_card("waterski"))

        game.play_card(0, 0)  # 3 today
        game.play_card(1, 0)  # 6 today
        game.end_day()

        self.assertEqual(p0.fun, 3)

    def test_counts_fit_to_print_end_day_fun_in_comparison(self) -> None:
        game = empty_game()
        p0 = game.players[0]
        p1 = game.players[1]
        p0.energy = 10
        p1.energy = 7
        p0.hand.extend([make_card("solo"), make_card("fit-to-print")])
        p1.hand.append(make_card("waterski"))

        game.play_card(0, 0)
        game.play_card(0, 0)  # 3 + 5 = 8 today (most cards played: 2 > 1)
        game.play_card(1, 0)  # 6 today
        game.end_day()

        # Solo rescores 3 -> 3 + 5 + 3 = 11.
        self.assertEqual(p0.fun, 11)


if __name__ == "__main__":
    unittest.main()
