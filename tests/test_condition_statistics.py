import csv
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game
from thenos.simulation import (
    SimulationReport, CardStatistics, simulate_games,
    write_report_csv, condition_report_path,
)


class ConditionStatisticsTests(unittest.TestCase):
    def test_daily_delta_includes_all_phases_and_excludes_previous_days(self):
        game = Game.default(123, daily_conditions=True)
        game.setup()
        for player in game.players:
            player.fun = 100
        original_start = game.start_day

        def start():
            original_start()
            game.players[0].fun += 4

        def draw():
            game.players[0].fun += 1

        def play():
            game.players[0].fun -= 2

        def end():
            game.players[0].fun += 7

        with patch.object(game, "start_day", side_effect=start), patch.object(game, "draw_phase", side_effect=draw), patch.object(game, "playing_phase", side_effect=play), patch.object(game, "end_day", side_effect=end):
            game.run_day()
            game.run_day()
        self.assertEqual(sum(game.stats.condition_days.values()), 2)
        self.assertEqual(sum(game.stats.condition_fun.values()), 20)
        self.assertEqual(set(game.stats.condition_fun.values()), {10})

    def test_exact_weighted_average_and_difference_with_missing_observations(self):
        report = SimulationReport(
            games=1, daily_conditions=True,
            condition_days=Counter({"Rainy Day": 2, "Normal day": 1}),
            condition_fun=Counter({"Rainy Day": -8, "Normal day": 20}),
        )
        rows = {row['condition']: row for row in report.condition_rows()}
        self.assertEqual(len(rows), len(DAILY_CONDITIONS))
        rainy = rows['Rainy Day']
        self.assertEqual(rainy['days'], 2)
        self.assertEqual(rainy['player_days'], 8)
        self.assertEqual(rainy['average_daily_fun'], -1)
        self.assertEqual(rainy['overall_player_days'], 12)
        self.assertEqual(rainy['overall_average_daily_fun'], 1)
        self.assertEqual(rainy['fun_difference'], -2)
        self.assertEqual(rows['Normal day']['fun_difference'], 4)
        self.assertIsNone(rows['Beautiful Day']['average_daily_fun'])
        self.assertIsNone(rows['Beautiful Day']['fun_difference'])

    def test_complete_batch_reconciles_daily_and_final_totals_and_processes(self):
        serial = simulate_games(4, seed=20260905, workers=1, daily_conditions=True)
        parallel = simulate_games(4, seed=20260905, workers=2, daily_conditions=True)
        self.assertEqual(serial, parallel)
        self.assertEqual(sum(serial.condition_days.values()), 24)
        self.assertEqual(sum(serial.condition_fun.values()), sum(serial.score_totals.values()))

    def test_disabled_has_no_condition_statistics(self):
        report = simulate_games(1, seed=234, workers=1)
        self.assertEqual(report.condition_rows(), [])
        self.assertEqual(report.condition_days, Counter())
        self.assertEqual(report.condition_fun, Counter())

    def test_planning_copy_statistics_are_independent(self):
        game = Game.default(345, daily_conditions=True)
        game.setup()
        game.run_day()
        clone = game.copy_for_simulation()
        self.assertEqual(clone.stats.condition_fun, game.stats.condition_fun)
        clone.stats.condition_fun['Rainy Day'] += 100
        clone.stats.condition_days['Rainy Day'] += 1
        self.assertNotEqual(clone.stats.condition_fun, game.stats.condition_fun)
        self.assertNotEqual(clone.stats.condition_days, game.stats.condition_days)

    def test_csv_companion_preserves_counts_totals_metadata_and_missing_values(self):
        report = SimulationReport(
            games=1, cards={"Azul": CardStatistics()}, daily_conditions=True,
            condition_days=Counter({"Rainy Day": 1}),
            condition_fun=Counter({"Rainy Day": 12}),
        )
        with TemporaryDirectory() as directory:
            output = write_report_csv(report, Path(directory) / "run.csv", metadata={"seed": 123})
            with condition_report_path(output).open() as source:
                rows = {r['condition']: r for r in csv.DictReader(source)}
            with output.open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 1)
        self.assertEqual(rows['Rainy Day']['seed'], '123')
        self.assertEqual(rows['Rainy Day']['player_days'], '4')
        self.assertEqual(rows['Rainy Day']['fun_total'], '12')
        self.assertEqual(rows['Rainy Day']['average_daily_fun'], '3.0')
        self.assertEqual(rows['Beautiful Day']['fun_difference'], '')

    def test_disabled_csv_does_not_create_companion(self):
        with TemporaryDirectory() as directory:
            output = write_report_csv(SimulationReport(1, cards={"Azul": CardStatistics()}), Path(directory) / "run.csv")
            self.assertFalse(condition_report_path(output).exists())
