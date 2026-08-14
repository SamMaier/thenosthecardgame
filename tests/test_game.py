import unittest
from collections import Counter

from thenos.cards import CARD_REGISTRY, create_default_deck
from thenos.game import DAYS_PER_GAME, Game, fractional_wins


class GameTests(unittest.TestCase):
    def test_default_deck_has_fifty_copies_of_each_implemented_card(self) -> None:
        counts = Counter(card.title for card in create_default_deck())
        self.assertEqual(len(counts), len(CARD_REGISTRY))
        self.assertTrue(all(count == 50 for count in counts.values()))

    def test_complete_seeded_game_runs_six_days(self) -> None:
        game = Game.default(seed=12345)

        result = game.run()

        self.assertEqual(result.days_played, DAYS_PER_GAME)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 80)
        self.assertAlmostEqual(sum(result.win_shares), 1.0)
        self.assertTrue(all(score >= 0 for score in result.scores))

    def test_tied_players_split_one_win(self) -> None:
        self.assertEqual(fractional_wins((10, 10, 4, 2)), (0.5, 0.5, 0.0, 0.0))
        self.assertEqual(fractional_wins((7, 7, 7, 7)), (0.25, 0.25, 0.25, 0.25))

    def test_seeded_games_are_reproducible(self) -> None:
        first = Game.default(seed=77).run()
        second = Game.default(seed=77).run()
        self.assertEqual(first, second)

    def test_unpack_costs_fun_discards_and_refills_four_cards(self) -> None:
        game = Game.default(seed=8)
        game.setup()
        original_ids = {card.instance_id for card in game.suitcase}

        game.unpack(0)

        self.assertEqual(game.players[0].fun, -1)
        self.assertEqual(len(game.suitcase), 4)
        self.assertTrue(original_ids.isdisjoint(card.instance_id for card in game.suitcase))
        self.assertEqual(
            original_ids,
            {card.instance_id for card in game.discard},
        )


if __name__ == "__main__":
    unittest.main()
