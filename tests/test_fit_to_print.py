import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FitToPrintTests(unittest.TestCase):
    def test_cost_tags_and_printed_fun(self) -> None:
        card = make_card("fit-to-print")

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Board Game", "Indoors"}),
        )
        self.assertEqual(card.definition.base_fun, 0)

    def test_scores_bonus_when_played_most_cards_including_ties(self) -> None:
        for opponent_count in (1, 2):
            with self.subTest(opponent_count=opponent_count):
                game = empty_game()
                player = game.players[0]
                player.energy = 7
                player.hand.append(make_card("fit-to-print"))
                game.play_card(0, 0)
                player.played_today.append(make_card("fajitas"))

                for opponent in game.players[1:]:
                    opponent.played_today.extend(
                        make_card("fajitas") for _ in range(opponent_count)
                    )

                game.end_day()

                self.assertEqual(player.fun, 4)

    def test_does_not_score_bonus_when_any_opponent_is_ahead(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fit-to-print"))
        card = game.play_card(0, 0)
        player.played_today.append(make_card("fajitas"))

        game.players[1].played_today.extend(
            make_card("fajitas") for _ in range(3)
        )
        game.players[2].played_today.append(make_card("fajitas"))

        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
