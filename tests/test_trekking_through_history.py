import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TrekkingThroughHistoryTests(unittest.TestCase):
    def test_board_games_before_and_after_score_bonus_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("splendor"), make_card("trekking-through-history"), make_card("tres-fute")]
        )

        before = game.play_card(0, 0)
        card = game.play_card(0, 0)
        after = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(game.card_fun(0, before), 3)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, after), 2)


if __name__ == "__main__":
    unittest.main()
