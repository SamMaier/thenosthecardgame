import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class CarcassonneTests(unittest.TestCase):
    def test_cost_tags_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        player.hand.append(make_card("carcassonne"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 1)

    def test_scores_bonus_when_directly_between_two_board_games(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.extend(
            [make_card("tres-fute"), make_card("carcassonne"), make_card("splendor")]
        )

        game.play_card(0, 0)
        card = game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 5)

    def test_no_bonus_without_two_immediate_board_game_neighbors(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.extend(
            [
                make_card("tres-fute"),
                make_card("carcassonne"),
                make_card("biography"),
                make_card("splendor"),
            ]
        )

        game.play_card(0, 0)
        card = game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 1)

    def test_no_bonus_at_the_start_or_end_of_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.extend(
            [
                make_card("carcassonne"),
                make_card("tres-fute"),
                make_card("splendor"),
                make_card("carcassonne"),
            ]
        )

        first = game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        last = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, first), 1)
        self.assertEqual(game.card_fun(0, last), 1)


if __name__ == "__main__":
    unittest.main()
