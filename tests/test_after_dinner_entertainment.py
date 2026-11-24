import unittest

from thenos.ais.hydrogen import _hydrogen_copy_game
from thenos.cards import make_card
from tests.helpers import empty_game


class AfterDinnerEntertainmentTests(unittest.TestCase):
    def test_without_earlier_food_next_exercise_is_not_doubled_and_no_discount(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("after-dinner-entertainment"),
                make_card("waterski"),
                make_card("waterski"),
            ]
        )

        card = game.play_card(0, 0)
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.energy_cost(0, player.hand[0]), 5)

        first_ex = game.play_card(0, 0)
        self.assertEqual(player.energy, 3)
        self.assertEqual(game.card_fun(0, first_ex), 6)

    def test_earlier_food_doubles_only_next_exercise_and_matches_fast_sim(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 20
        player.hand.extend(
            [
                make_card("stretch"),
                make_card("cheap-white"),
                make_card("after-dinner-entertainment"),
                make_card("biography"),
                make_card("waterski"),
                make_card("waterski"),
            ]
        )

        for _ in range(4):
            game.play_card(0, 0)
        first_ex = game.play_card(0, 0)
        later_ex = game.play_card(0, 0)

        # Stretch adds +2 to Exercise; ADE doubles (6 + 2) -> 16 on first Exercise only.
        self.assertEqual(game.card_fun(0, first_ex), 16)
        self.assertEqual(game.card_fun(0, later_ex), 6 + 2)

        hsim = _hydrogen_copy_game(game)
        self.assertEqual(
            hsim.score_played_today_fast(0)[0],
            sum(game.card_fun(0, c) for c in player.played_today),
        )

    def test_has_no_tomorrow_effect_and_food_after_ade_does_not_arm_it(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("after-dinner-entertainment"),
                make_card("cheap-white"),
                make_card("waterski"),
            ]
        )

        ade = game.play_card(0, 0)
        game.play_card(0, 0)
        ex = game.play_card(0, 0)
        self.assertEqual(game.card_fun(0, ex), 6)
        game.end_day()
        self.assertNotIn(ade, player.tomorrow_cards)


if __name__ == "__main__":
    unittest.main()
