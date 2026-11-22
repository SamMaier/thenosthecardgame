import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SkiOnCousinsShouldersTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("ski-on-cousins-shoulders")

        self.assertEqual(card.title, "Ski on Shoulders")
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
        player.fun = 10
        game.players[1].fun = 11
        game.players[2].fun = 12
        game.players[3].fun = 10
        player.hand.append(make_card("ski-on-cousins-shoulders"))
        card = game.play_card(0, 0)

        self.assertTrue(card.markers.get("energy_cube"))
        self.assertEqual(game.card_fun(0, card), 8)

    def test_bonus_requires_strictly_lower_than_at_least_half(self) -> None:
        for opponent_scores in ((11, 10, 9), (10, 10, 9)):
            with self.subTest(opponent_scores=opponent_scores):
                game = empty_game()
                player = game.players[0]
                player.energy = 5
                player.fun = 10
                for opponent, score in zip(game.players[1:], opponent_scores):
                    opponent.fun = score
                player.hand.append(make_card("ski-on-cousins-shoulders"))
                card = game.play_card(0, 0)

                self.assertNotIn("energy_cube", card.markers)
                self.assertEqual(game.card_fun(0, card), 3)

    def test_evaluates_on_play_even_if_scores_change_before_end_of_day(self) -> None:
        game = empty_game()
        for p in game.players:
            p.energy = 7
        game.players[3].hand.append(make_card("ski-on-cousins-shoulders"))
        ski = game.play_card(3, 0)

        # Opponents score immediately afterward via Wingspan and Non-Fiction Book at end of day,
        # which does not retroactively trigger Ski on Shoulders.
        for opponent in game.players[:3]:
            opponent.hand.extend([make_card("wingspan"), make_card("biography")])
            game.play_card(game.players.index(opponent), 0)
            game.play_card(game.players.index(opponent), 0)

        self.assertNotIn("energy_cube", ski.markers)
        self.assertEqual(game.card_fun(3, ski), 3)
        game.end_day()
        self.assertEqual([p.fun for p in game.players], [4, 4, 4, 3])

    def test_immediate_fun_before_play_counts_toward_comparison(self) -> None:
        game = empty_game()
        for p in game.players:
            p.energy = 7
        for opponent in game.players[:2]:
            opponent.hand.append(make_card("wingspan"))
            game.play_card(game.players.index(opponent), 0)
        game.players[3].hand.append(make_card("ski-on-cousins-shoulders"))
        ski = game.play_card(3, 0)

        self.assertTrue(ski.markers.get("energy_cube"))
        self.assertEqual(game.card_fun(3, ski), 8)


if __name__ == "__main__":
    unittest.main()
