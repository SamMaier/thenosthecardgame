import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FitToPrintTests(unittest.TestCase):
    def test_cost_tags_and_printed_fun(self) -> None:
        card = make_card("fit-to-print")

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Board Game", "Indoors"}),
        )
        self.assertEqual(card.definition.base_fun, 1)

    def test_scores_bonus_when_played_more_cards_than_every_opponent(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("fit-to-print"))
        card = game.play_card(0, 0)
        player.played_today.append(make_card("fajitas"))

        for opponent in game.players[1:]:
            opponent.played_today.append(make_card("fajitas"))

        game.end_day()

        self.assertEqual(player.fun, 5)

    def test_does_not_score_bonus_when_an_opponent_is_tied_or_ahead(self) -> None:
        for opponent_count in (2, 3):
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

                card = next(
                    played_card
                    for played_card in player.played_today
                    if played_card.title == "Fit to Print"
                )
                self.assertEqual(game.card_fun(0, card), 1)


if __name__ == "__main__":
    unittest.main()
