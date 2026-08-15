import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SkiOnCousinsShouldersTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("ski-on-cousins-shoulders")

        self.assertEqual(card.title, "Ski on Cousin's Shoulders")
        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_scores_bonus_when_lower_than_at_least_half_of_opponents(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.append(make_card("ski-on-cousins-shoulders"))
        card = game.play_card(0, 0)

        player.fun = 10
        game.players[1].fun = 11
        game.players[2].fun = 12
        game.players[3].fun = 10

        self.assertEqual(game.card_fun(0, card), 8)

    def test_bonus_requires_strictly_lower_than_at_least_half(self) -> None:
        for opponent_scores in ((11, 10, 9), (10, 10, 9)):
            with self.subTest(opponent_scores=opponent_scores):
                game = empty_game()
                player = game.players[0]
                player.energy = 5
                player.hand.append(make_card("ski-on-cousins-shoulders"))
                card = game.play_card(0, 0)

                player.fun = 10
                for opponent, score in zip(game.players[1:], opponent_scores):
                    opponent.fun = score

                self.assertEqual(game.card_fun(0, card), 3)

    def test_end_day_comparison_uses_scores_before_any_card_scoring(self) -> None:
        game = empty_game()
        for player in game.players[:3]:
            player.played_today.append(make_card("biography"))
        game.players[3].played_today.append(
            make_card("ski-on-cousins-shoulders")
        )

        game.end_day()

        self.assertEqual([player.fun for player in game.players], [2, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
