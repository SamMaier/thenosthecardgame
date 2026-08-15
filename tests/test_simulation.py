import unittest

from thenos.cards import CARD_REGISTRY
from thenos.simulation import (
    simulate_genius_vs_planner,
    simulate_games,
    simulate_greedy_vs_random,
    simulate_planner_vs_greedy,
)


class SimulationTests(unittest.TestCase):
    def test_batch_collects_pick_play_and_win_statistics(self) -> None:
        report = simulate_games(10, seed=2026)

        self.assertEqual(report.games, 10)
        self.assertEqual(set(report.cards), {card.title for card in CARD_REGISTRY.values()})
        self.assertGreaterEqual(sum(card.picks for card in report.cards.values()), 750)
        for card in report.cards.values():
            self.assertGreater(card.offers, 0)
            self.assertGreater(card.picks, 0)
            self.assertGreaterEqual(card.offers, card.picks)
            self.assertGreaterEqual(card.acquisitions, card.picks)
            self.assertGreaterEqual(card.plays, 0)
            self.assertLessEqual(card.play_rate, 1.0)
            self.assertGreaterEqual(card.win_rate_when_picked, 0.0)
            self.assertLessEqual(card.win_rate_when_picked, 1.0)
            self.assertGreaterEqual(card.win_rate_when_acquired, 0.0)
            self.assertLessEqual(card.win_rate_when_acquired, 1.0)

    def test_matchup_aggregates_named_ai_results(self) -> None:
        report = simulate_greedy_vs_random(1, seed=99)

        self.assertEqual(report.ais["Greedy"].games, 1)
        self.assertEqual(report.ais["Random"].games, 3)
        self.assertAlmostEqual(
            report.ais["Greedy"].win_credit
            + report.ais["Random"].win_credit,
            1.0,
        )

    def test_parallel_batch_matches_serial_batch_exactly(self) -> None:
        serial = simulate_games(4, seed=314159, workers=1)
        parallel = simulate_games(4, seed=314159, workers=2)

        self.assertEqual(parallel, serial)

    def test_planner_matchup_aggregates_three_greedy_opponents(self) -> None:
        report = simulate_planner_vs_greedy(1, seed=101)

        self.assertEqual(report.ais["Planner"].games, 1)
        self.assertEqual(report.ais["Greedy"].games, 3)

    def test_genius_matchup_aggregates_three_planner_opponents(self) -> None:
        report = simulate_genius_vs_planner(1, seed=102)

        self.assertEqual(report.ais["Genius"].games, 1)
        self.assertEqual(report.ais["Planner"].games, 3)

    def test_worker_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            simulate_games(1, workers=0)


if __name__ == "__main__":
    unittest.main()
