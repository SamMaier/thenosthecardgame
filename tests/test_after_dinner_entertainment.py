import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class AfterDinnerEntertainmentTests(unittest.TestCase):
    def test_next_social_costs_two_less_and_scores_two_more(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("after-dinner-entertainment"),
                make_card("johnny-appleseed"),
                make_card("johnny-appleseed"),
            ]
        )

        card = game.play_card(0, 0)
        next_social = game.play_card(0, 0)
        later_social = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.energy_cost(0, later_social), 1)
        self.assertEqual(player.energy, 4)
        self.assertEqual(next_social.definition.cost, 1)
        self.assertEqual(game.card_fun(0, next_social), 3)
        self.assertEqual(game.card_fun(0, later_social), 1)

    def test_has_no_tomorrow_effect(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("after-dinner-entertainment"))

        card = game.play_card(0, 0)
        game.end_day()
        self.assertNotIn(card, player.tomorrow_cards)

    def test_unplayed_social_card_in_hand_does_not_get_fun_bonus(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        unplayed_social = make_card("johnny-appleseed")
        player.hand.extend([make_card("after-dinner-entertainment"), unplayed_social])

        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, unplayed_social), 1)


if __name__ == "__main__":
    unittest.main()
