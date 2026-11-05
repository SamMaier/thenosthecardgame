import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FamilyBaseballGameTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("family-baseball-game")

        self.assertEqual(card.title, "Family Baseball Game")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Social", "Outdoors"}),
        )

    def test_subtracts_one_fun_for_relax_and_board_game_cards_played_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("azul"),
                make_card("family-baseball-game"),
                make_card("splendor"),
                make_card("biography"),
                make_card("cheap-white"),
            ]
        )

        board_game_before = game.play_card(0, 0)
        family_baseball_game = game.play_card(0, 0)
        board_game_after = game.play_card(0, 0)
        relax_after = game.play_card(0, 0)
        other_tag = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, board_game_before), 3)
        self.assertEqual(game.card_fun(0, family_baseball_game), 5)
        self.assertEqual(game.card_fun(0, board_game_after), 1)
        self.assertEqual(game.card_fun(0, relax_after), 1)
        self.assertEqual(game.card_fun(0, other_tag), 3)

        game.end_day()

        self.assertEqual(player.fun, 13)


if __name__ == "__main__":
    unittest.main()
