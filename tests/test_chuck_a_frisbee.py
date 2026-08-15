import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ChuckAFrisbeeTests(unittest.TestCase):
    def test_costs_three_energy_and_scores_three_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("chuck-a-frisbee"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 3)

    def test_returns_to_hand_after_scoring_instead_of_discarding(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        card = make_card("chuck-a-frisbee")
        player.hand.append(card)
        game.play_card(0, 0)

        game.end_day()

        self.assertEqual(player.fun, 3)
        self.assertIn(card, player.hand)
        self.assertNotIn(card, game.discard)
        self.assertEqual(player.played_today, [])

    def test_return_waits_until_every_players_scoring_is_complete(self) -> None:
        for fit_player_index in (0, 1):
            with self.subTest(fit_player_index=fit_player_index):
                game = empty_game()
                chuck_player_index = 1 - fit_player_index
                game.players[fit_player_index].played_today.append(
                    make_card("fit-to-print")
                )
                game.players[chuck_player_index].played_today.append(
                    make_card("chuck-a-frisbee")
                )

                game.end_day()

                self.assertEqual(game.players[fit_player_index].fun, 1)


if __name__ == "__main__":
    unittest.main()
