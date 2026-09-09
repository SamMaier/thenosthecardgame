import csv
import random
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from thenos.ais import RandomAI
from thenos.cards.base import CardDefinition, CardInstance
from thenos.characters import CHARACTERS
from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game
from thenos.simulation import (
    CardStatistics,
    SimulationReport,
    character_report_path,
    simulate_four_galaxybrain,
    simulate_games,
    write_report_csv,
)


def character(title):
    return next(item for item in CHARACTERS if item.title == title)


def empty_game():
    return Game(
        [],
        [RandomAI(random.Random(index)) for index in range(4)],
        random.Random(1),
    )


class CharacterTests(unittest.TestCase):
    def test_catalog_matches_tsv(self):
        path = Path(__file__).resolve().parents[1] / "characters.tsv"
        with path.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        self.assertEqual(
            [(item.title, item.effect) for item in CHARACTERS],
            [(row["Character"], row["Effect"]) for row in rows],
        )

    def test_disabled_is_default_and_does_not_consume_randomness(self):
        implicit = Game.default(1001)
        explicit = Game.default(1001, characters=False)
        self.assertEqual(implicit.run(), explicit.run())
        self.assertEqual(implicit.rng.getstate(), explicit.rng.getstate())
        self.assertTrue(all(player.character is None for player in implicit.players))

    def test_four_distinct_characters_are_reproducibly_assigned_at_setup(self):
        games = [Game.default(1002, characters=True) for _ in range(2)]
        assignments = []
        for game in games:
            self.assertTrue(all(player.character is None for player in game.players))
            game.setup()
            assigned = tuple(player.character for player in game.players)
            self.assertEqual(len(set(assigned)), 4)
            self.assertTrue(set(assigned).issubset(CHARACTERS))
            assignments.append(assigned)
        self.assertEqual(assignments[0], assignments[1])

    def test_tag_characters_add_fun_and_child_zeroes_food(self):
        cases = (
            ("Foodie", "Food"),
            ("Gamer", "Board Game"),
            ("Athlete", "Exercise"),
            ("Extrovert", "Social"),
            ("Introvert", "Relax"),
        )
        for title, tag in cases:
            with self.subTest(character=title):
                game = empty_game()
                game.players[0].character = character(title)
                target = CardInstance(
                    1,
                    CardDefinition("target", "Target", frozenset({tag}), 1, 4),
                )
                self.assertEqual(game.card_fun(0, target), 5)

        child_game = empty_game()
        child_game.players[0].character = character("Child")
        food = CardInstance(
            2,
            CardDefinition("food", "Food", frozenset({"Food"}), 1, 9),
        )
        board_game = CardInstance(
            3,
            CardDefinition("board", "Board", frozenset({"Board Game"}), 1, 4),
        )
        self.assertEqual(child_game.card_fun(0, food), 0)
        self.assertEqual(child_game.card_fun(0, board_game), 4)

    def test_child_food_stays_zero_after_ordinary_card_modifiers(self):
        game = empty_game()
        game.players[0].character = character("Child")
        food = CardInstance(
            1,
            CardDefinition("food", "Food", frozenset({"Food"}), 1, 9),
        )
        source = CardInstance(
            2,
            CardDefinition("source", "Source", frozenset(), 1, 0),
        )
        source.effective_behavior.modify_fun = lambda *args: 99
        game.players[0].played_today = [source, food]
        self.assertEqual(game.card_fun(0, food), 0)

    def test_daily_energy_and_fun_bonuses_persist(self):
        game = empty_game()
        game.players[0].character = character("Old Fogey")
        game.players[1].character = character("Child")
        game.start_day()
        self.assertEqual([player.energy for player in game.players], [6, 8, 7, 7])
        self.assertEqual(game.players[0].fun, 2)
        game.start_day()
        self.assertEqual([player.energy for player in game.players], [6, 8, 7, 7])
        self.assertEqual(game.players[0].fun, 4)

    def test_character_and_daily_condition_starting_effects_combine(self):
        game = empty_game()
        game.daily_conditions = True
        sickness = next(
            item for item in DAILY_CONDITIONS
            if item.title == "Sickness Spreading"
        )
        game._condition_deck = [sickness]
        game.players[0].character = character("Old Fogey")
        game.players[1].character = character("Child")
        game.start_day()
        self.assertEqual([player.energy for player in game.players], [5, 7, 6, 6])
        self.assertEqual(game.players[0].fun, 2)

    def test_planner_starts_with_six_cards_and_others_start_with_three(self):
        game = Game.default(1009, characters=True)
        with patch("thenos.game.random.Random.sample") as sample:
            sample.return_value = [
                character("Foodie"),
                character("Planner"),
                character("Child"),
                character("Athlete"),
            ]
            game.setup()
        self.assertEqual([len(player.hand) for player in game.players], [3, 6, 3, 3])

    def test_planner_has_only_three_daily_suitcase_picks(self):
        game = empty_game()
        game.players[1].character = character("Planner")
        with patch.object(game, "pick_from_suitcase") as pick:
            game.draw_phase()
        self.assertEqual(pick.call_count, 12)
        self.assertEqual(
            [call.args[0] for call in pick.call_args_list],
            [0, 1, 2, 3] * 3,
        )

    def test_gear_guy_item_cards_cost_one_less_energy(self):
        game = empty_game()
        game.players[0].character = character("Gear Guy")
        item = CardInstance(
            1,
            CardDefinition("item", "Item", frozenset({"Item"}), 3, 4),
        )
        free_item = CardInstance(
            2,
            CardDefinition("free-item", "Free Item", frozenset({"Item"}), 0, 1),
        )
        non_item = CardInstance(
            3,
            CardDefinition("social", "Social", frozenset({"Social"}), 3, 1),
        )
        self.assertEqual(game.energy_cost(0, item), 2)
        self.assertEqual(game.energy_cost(0, free_item), 0)
        self.assertEqual(game.energy_cost(0, non_item), 3)
        self.assertEqual(game.card_fun(0, item), 4)

    def test_copy_preserves_characters_independently(self):
        game = Game.default(1003, characters=True)
        game.setup()
        clone = game.copy_for_simulation()
        self.assertEqual(
            [player.character for player in clone.players],
            [player.character for player in game.players],
        )
        replacement = next(
            item for item in CHARACTERS if item != game.players[0].character
        )
        clone.players[0].character = replacement
        self.assertNotEqual(clone.players[0].character, game.players[0].character)

    def test_serial_parallel_parity_with_characters(self):
        self.assertEqual(
            simulate_games(4, seed=1004, workers=1, characters=True),
            simulate_games(4, seed=1004, workers=2, characters=True),
        )

    def test_serial_parallel_parity_with_conditions_and_characters(self):
        self.assertEqual(
            simulate_games(
                4,
                seed=1008,
                workers=1,
                daily_conditions=True,
                characters=True,
            ),
            simulate_games(
                4,
                seed=1008,
                workers=2,
                daily_conditions=True,
                characters=True,
            ),
        )

    def test_galaxybrain_serial_parallel_parity_with_characters(self):
        self.assertEqual(
            simulate_four_galaxybrain(4, seed=1005, workers=1, characters=True),
            simulate_four_galaxybrain(4, seed=1005, workers=2, characters=True),
        )

    def test_cli_flag_reaches_every_mode_and_defaults_off(self):
        from thenos.__main__ import main
        import sys

        modes = {
            "": "simulate_games",
            "--four-galaxybrain": "simulate_four_galaxybrain",
            "--greedy-vs-random": "simulate_greedy_vs_random",
            "--planner-vs-greedy": "simulate_planner_vs_greedy",
            "--galaxybrain-vs-planner": "simulate_galaxybrain_vs_planner",
        }
        report = simulate_games(1, seed=1006, workers=1)
        for mode, function in modes.items():
            for enabled in (False, True):
                args = ["thenos", "4", "--output", "unused.csv"]
                if mode:
                    args.append(mode)
                if enabled:
                    args.append("--characters")
                with (
                    patch.object(sys, "argv", args),
                    patch("thenos.__main__." + function, return_value=report) as run,
                    patch("thenos.__main__._code_revision", return_value="test"),
                    patch("thenos.__main__.write_report_csv", return_value=Path("unused.csv")),
                    patch("builtins.print"),
                ):
                    main()
                self.assertEqual(run.call_args.kwargs["characters"], enabled)


class CharacterStatisticsTests(unittest.TestCase):
    def test_rows_include_all_characters_and_exact_denominators(self):
        report = SimulationReport(
            games=1,
            characters=True,
            character_games=Counter({"Foodie": 2, "Child": 1}),
            character_fun=Counter({"Foodie": 20, "Child": 4}),
            character_win_credit=Counter({"Foodie": 1.5}),
        )
        rows = {row["character"]: row for row in report.character_rows()}
        self.assertEqual(len(rows), len(CHARACTERS))
        self.assertEqual(rows["Foodie"]["player_games"], 2)
        self.assertEqual(rows["Foodie"]["average_fun"], 10)
        self.assertEqual(rows["Foodie"]["win_rate"], 0.75)
        self.assertEqual(rows["Foodie"]["fun_difference"], 2)
        self.assertIsNone(rows["Gamer"]["average_fun"])
        self.assertIsNone(rows["Gamer"]["win_rate"])

    def test_batch_reconciles_character_player_games(self):
        report = simulate_games(4, seed=1007, workers=1, characters=True)
        self.assertEqual(sum(report.character_games.values()), 16)
        self.assertEqual(sum(report.character_fun.values()), sum(report.score_totals.values()))
        self.assertAlmostEqual(sum(report.character_win_credit.values()), 4.0)

    def test_csv_writes_character_companion_and_metadata(self):
        report = SimulationReport(
            games=1,
            cards={"Azul": CardStatistics()},
            characters=True,
            character_games=Counter({"Foodie": 1}),
            character_fun=Counter({"Foodie": 12}),
            character_win_credit=Counter({"Foodie": 1.0}),
        )
        with TemporaryDirectory() as directory:
            output = write_report_csv(
                report,
                Path(directory) / "run.csv",
                metadata={"seed": 123},
            )
            with character_report_path(output).open() as source:
                rows = {row["character"]: row for row in csv.DictReader(source)}
        self.assertEqual(rows["Foodie"]["characters"], "True")
        self.assertEqual(rows["Foodie"]["seed"], "123")
        self.assertEqual(rows["Foodie"]["player_games"], "1")
        self.assertEqual(rows["Foodie"]["win_rate"], "1.0")
        self.assertEqual(rows["Gamer"]["average_fun"], "")

    def test_disabled_has_no_rows_or_companion(self):
        report = SimulationReport(1, cards={"Azul": CardStatistics()})
        self.assertEqual(report.character_rows(), [])
        with TemporaryDirectory() as directory:
            output = write_report_csv(report, Path(directory) / "run.csv")
            self.assertFalse(character_report_path(output).exists())


if __name__ == "__main__":
    unittest.main()
