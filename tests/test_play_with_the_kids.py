import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PlayWithTheKidsTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("play-with-the-kids")

        self.assertEqual(card.title, "Play With the Kids")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(card.definition.tags, frozenset({"Exercise"}))

    def test_scores_bonus_when_played_most_cards_including_ties(self) -> None:
        for opponent_count in (1, 2):
            with self.subTest(opponent_count=opponent_count):
                game = empty_game()
                player = game.players[0]
                player.energy = 7
                player.hand.append(make_card("play-with-the-kids"))
                card = game.play_card(0, 0)
                self.assertEqual(player.energy, 3)
                player.played_today.append(make_card("fajitas"))

                for opponent in game.players[1:]:
                    opponent.played_today.extend(
                        make_card("fajitas") for _ in range(opponent_count)
                    )

                self.assertEqual(game.card_fun(0, card), 7)
                game.end_day()

                self.assertEqual(player.fun, 7)

    def test_does_not_score_bonus_when_opponent_played_more_even_with_larger_hand(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("play-with-the-kids"))
        card = game.play_card(0, 0)
        player.played_today.append(make_card("fajitas"))
        player.hand.extend(make_card("fajitas") for _ in range(5))

        game.players[1].played_today.extend(make_card("fajitas") for _ in range(3))
        game.players[2].played_today.append(make_card("fajitas"))

        self.assertEqual(game.card_fun(0, card), 3)

    def test_fast_sim_scoring_parity(self) -> None:
        from thenos.ais.carbon import _carbon_copy_game
        from thenos.ais.hydrogen import _hydrogen_copy_game

        for opp_played, expected_total in ((1, 7), (2, 7), (3, 3)):
            with self.subTest(opp_played=opp_played):
                game = empty_game()
                player = game.players[0]
                player.energy = 7
                player.hand.append(make_card("play-with-the-kids"))
                game.play_card(0, 0)
                player.played_today.append(make_card("nap"))
                game.players[1].played_today.extend(
                    make_card("nap") for _ in range(opp_played)
                )
                exact = sum(game.card_fun(0, c) for c in player.played_today)
                self.assertEqual(exact, expected_total)
                self.assertEqual(
                    _hydrogen_copy_game(game).score_played_today_fast(0)[0], exact
                )
                csim = _carbon_copy_game(game)
                csim.end_day()
                self.assertEqual(csim.players[0].fun, exact)


if __name__ == "__main__":
    unittest.main()
