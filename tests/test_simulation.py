import unittest
from csv import DictReader
from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import call, patch

from thenos.cards import CARD_REGISTRY
from thenos.simulation import (
    CardStatistics,
    SimulationReport,
    _GameOutcome,
    _PlayerOutcome,
    _merge_outcome,
    simulate_genius_vs_planner,
    simulate_games,
    simulate_greedy_vs_random,
    simulate_planner_vs_greedy,
    write_report_csv,
)


class SimulationTests(unittest.TestCase):
    @staticmethod
    def _empty_outcome() -> _GameOutcome:
        return _GameOutcome(
            free_pick_offers=Counter(),
            free_picks=Counter(),
            suitcase_offers=Counter(),
            suitcase_picks=Counter(),
            card_acquisitions=Counter(),
            card_plays=Counter(),
            card_plays_without_acquisition=Counter(),
            players=(),
        )

    def test_progress_prints_every_eight_games_for_short_runs(self) -> None:
        with (
            patch("thenos.simulation._run_game", return_value=self._empty_outcome()),
            patch("thenos.simulation.print") as print_mock,
        ):
            simulate_games(16, seed=2026, workers=1)

        self.assertEqual(
            print_mock.call_args_list,
            [
                call("Run 8 complete", flush=True),
                call("Run 16 complete", flush=True),
            ],
        )

    def test_progress_prints_every_64_games_for_long_runs(self) -> None:
        with (
            patch("thenos.simulation._run_game", return_value=self._empty_outcome()),
            patch("thenos.simulation.print") as print_mock,
        ):
            simulate_games(128, seed=2026, workers=1)

        self.assertEqual(
            print_mock.call_args_list,
            [
                call("Run 64 complete", flush=True),
                call("Run 128 complete", flush=True),
            ],
        )

    def test_batch_collects_pick_play_and_win_statistics(self) -> None:
        report = simulate_games(10, seed=2026)

        self.assertEqual(report.games, 10)
        self.assertEqual(set(report.cards), {card.title for card in CARD_REGISTRY.values()})
        self.assertGreaterEqual(sum(card.picks for card in report.cards.values()), 750)
        self.assertEqual(
            sum(card.free_pick_offers for card in report.cards.values()),
            4 * sum(card.free_picks for card in report.cards.values()),
        )
        for card in report.cards.values():
            self.assertGreater(card.offers, 0)
            self.assertGreater(card.picks, 0)
            self.assertGreaterEqual(card.offers, card.picks)
            self.assertGreaterEqual(card.free_pick_offers, card.free_picks)
            self.assertGreaterEqual(card.free_pick_rate, 0.0)
            self.assertLessEqual(card.free_pick_rate, 1.0)
            self.assertGreaterEqual(card.acquisitions, card.picks)
            self.assertGreaterEqual(card.plays, 0)
            self.assertLessEqual(card.play_rate, 1.0)
            self.assertGreaterEqual(card.win_rate_when_picked, 0.0)
            self.assertLessEqual(card.win_rate_when_picked, 1.0)
            self.assertGreaterEqual(card.win_rate_when_acquired, 0.0)
            self.assertLessEqual(card.win_rate_when_acquired, 1.0)

    def test_card_outcomes_use_player_game_exposure_and_fun_difference(self) -> None:
        report = SimulationReport(
            games=1,
            cards={"Biography": CardStatistics(), "Fajitas": CardStatistics()},
        )
        outcome = _GameOutcome(
            free_pick_offers=Counter({"Biography": 2, "Fajitas": 1}),
            free_picks=Counter({"Biography": 1}),
            suitcase_offers=Counter(),
            suitcase_picks=Counter(),
            card_acquisitions=Counter({"Biography": 2, "Fajitas": 1}),
            card_plays=Counter(),
            card_plays_without_acquisition=Counter(),
            players=(
                _PlayerOutcome(
                    "Player 1",
                    "Test AI",
                    10,
                    1.0,
                    Counter(),
                    Counter({"Biography": 2}),
                ),
                _PlayerOutcome(
                    "Player 2",
                    "Test AI",
                    4,
                    0.0,
                    Counter(),
                    Counter({"Fajitas": 1}),
                ),
            ),
        )

        _merge_outcome(report, outcome)

        biography = report.cards["Biography"]
        self.assertEqual(biography.acquisitions, 2)
        self.assertEqual(biography.player_games_with_card, 1)
        self.assertEqual(biography.free_pick_rate, 0.5)
        self.assertEqual(biography.win_rate, 1.0)
        self.assertEqual(biography.fun_added, 6.0)
        self.assertEqual(report.cards["Fajitas"].fun_added, -6.0)

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

    def test_csv_report_preserves_metrics_denominators_and_metadata(self) -> None:
        report = SimulationReport(
            games=1,
            cards={
                "Biography": CardStatistics(
                    free_pick_offers=4,
                    free_picks=1,
                    acquisitions=2,
                    plays=1,
                    win_credit_when_acquired=0.5,
                    player_games_with_card=2,
                    fun_total_with_card=18,
                    player_games_without_card=2,
                    fun_total_without_card=10,
                )
            },
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.csv"
            written = write_report_csv(
                report,
                output,
                metadata={"seed": 123, "workers": 16},
            )
            with written.open(encoding="utf-8", newline="") as stream:
                rows = list(DictReader(stream))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["seed"], "123")
        self.assertEqual(row["workers"], "16")
        self.assertEqual(row["card"], "Biography")
        self.assertEqual(row["free_pick_offers"], "4")
        self.assertEqual(row["free_picks"], "1")
        self.assertEqual(row["player_games_with_card"], "2")
        self.assertEqual(row["player_games_without_card"], "2")
        self.assertEqual(row["fun_total_with_card"], "18")
        self.assertEqual(row["fun_total_without_card"], "10")


if __name__ == "__main__":
    unittest.main()
