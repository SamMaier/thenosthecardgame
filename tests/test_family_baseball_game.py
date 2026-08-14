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
            frozenset({"Exercise", "Event", "Outdoors"}),
        )

    def test_scores_zero_for_all_board_games_played_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 8
        player.hand.extend(
            [
                make_card("azul"),
                make_card("family-baseball-game"),
                make_card("splendor"),
                make_card("biography"),
            ]
        )

        board_game_before = game.play_card(0, 0)
        family_baseball_game = game.play_card(0, 0)
        board_game_after = game.play_card(0, 0)
        non_board_game = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, board_game_before), 0)
        self.assertEqual(game.card_fun(0, family_baseball_game), 5)
        self.assertEqual(game.card_fun(0, board_game_after), 0)
        self.assertEqual(game.card_fun(0, non_board_game), 2)

        game.end_day()

        self.assertEqual(player.fun, 7)


if __name__ == "__main__":
    unittest.main()
