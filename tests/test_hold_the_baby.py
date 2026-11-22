import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class HoldTheBabyTests(unittest.TestCase):
    def test_later_cards_cost_one_more_and_score_three_extra_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 12
        player.hand.extend(
            [
                make_card("biography"),
                make_card("hold-the-baby"),
                make_card("johnny-appleseed"),
                make_card("waterski"),
            ]
        )

        before = game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Hold the Baby")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.energy_cost(0, player.hand[0]), 2)
        self.assertEqual(game.energy_cost(0, player.hand[1]), 6)

        social = game.play_card(0, 0)
        exercise = game.play_card(0, 0)

        self.assertEqual(player.energy, 1)
        self.assertEqual(game.card_fun(0, before), 2)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, social), 4)
        self.assertEqual(game.card_fun(0, exercise), 9)

    def test_free_play_after_hold_the_baby_ignores_cost_penalty_and_scores_bonus(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.extend(
            [make_card("hold-the-baby"), make_card("assorted-cutlery")]
        )
        trunk_card = make_card("waterski")
        game.trunk = [trunk_card]

        game.play_card(0, 0)  # Hold the Baby (cost 2 -> energy 2)
        cutlery = game.play_card(0, 0)  # Assorted Cutlery (cost 1 + 1 = 2 -> energy 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, cutlery), 3)
        self.assertEqual(game.card_fun(0, trunk_card), 9)


if __name__ == "__main__":
    unittest.main()
