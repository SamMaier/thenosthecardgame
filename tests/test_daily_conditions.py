import csv
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from thenos.ais import GalaxybrainAI, RandomAI
from thenos.ais.greedy import GreedyAI
from thenos.ais.planner import PlannerAI
from thenos.ais.megamind import MegamindAI
from thenos.ais.genius import GeniusAI
from thenos.cards.base import CardDefinition, CardInstance
from thenos.cards.catalog import CARD_REGISTRY, create_default_deck, make_card
from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game
from thenos.simulation import simulate_games, simulate_four_galaxybrain, write_report_csv


def condition(title):
    return next(card for card in DAILY_CONDITIONS if card.title == title)


def game_with(title):
    game = Game([], [RandomAI(random.Random(i)) for i in range(4)], random.Random(1), daily_conditions=True)
    game.daily_condition = condition(title)
    return game


class DailyConditionTests(unittest.TestCase):
    def test_catalog_matches_tsv(self):
        with (Path(__file__).resolve().parents[1] / "daily_conditions.tsv").open() as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        self.assertEqual([(c.title, c.effect) for c in DAILY_CONDITIONS],
                         [(r["Daily Condition Card Name"], r["Effect"]) for r in rows])

    def test_main_catalog_matches_csv(self):
        with (Path(__file__).resolve().parents[1] / "cards.csv").open(encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
        actual = {c.title: (c.tags, c.cost) for c in CARD_REGISTRY.values()}
        self.assertEqual(actual, {r["Title"]: (frozenset(t.strip() for t in r["Tags"].split(",")), int(r["Cost"])) for r in rows})

    def test_disabled_is_default_and_does_not_consume_condition_randomness(self):
        implicit, explicit = Game.default(901), Game.default(901, daily_conditions=False)
        self.assertEqual(implicit.run(), explicit.run())
        self.assertEqual(implicit.rng.getstate(), explicit.rng.getstate())
        self.assertEqual(implicit.revealed_conditions, [])
        self.assertIsNone(implicit.daily_condition)
        self.assertNotIn("Read the Radar", [c.title for c in create_default_deck()])

    def test_six_unique_reproducible_reveals_before_draw_phase(self):
        games = [Game.default(902, daily_conditions=True) for _ in range(2)]
        for game in games:
            game.setup()
            for day in range(6):
                original_draw = game.draw_phase
                def draw_after_reveal():
                    self.assertEqual(len(game.revealed_conditions), day + 1)
                    self.assertIsNotNone(game.daily_condition)
                    original_draw()
                with patch.object(game, "draw_phase", side_effect=draw_after_reveal) as draw:
                    game.run_day()
                draw.assert_called_once()
                self.assertEqual(len(game.revealed_conditions), day + 1)
            self.assertEqual(len(set(game.revealed_conditions)), 6)
            self.assertEqual(len(game._condition_deck), 2)
        self.assertEqual(games[0].revealed_conditions, games[1].revealed_conditions)
        self.assertEqual(games[0].result(), games[1].result())

    def test_all_fun_modifiers_and_unaffected_tags(self):
        cases = [("Rainy Day", "Outdoors", -1), ("Beautiful Day", "Outdoors", 1),
                 ("Plumbing Issue", "Indoors", -1), ("New Arrivals", "Social", 1),
                 ("Everyone Booked", "Social", -1), ("Normal day", "Social", 0)]
        for title, tag, delta in cases:
            with self.subTest(title=title):
                game = game_with(title)
                target = CardInstance(1, CardDefinition("test", "Test", frozenset({tag}), 2, 0))
                game.players[0].played_today = [target]
                self.assertEqual(game.card_fun(0, target), delta)
                self.assertEqual(game.card_fun(0, make_card("azul")), 4)
                target.is_tomorrow = True
                self.assertEqual(game.card_fun(0, target), 0)

    def test_fun_condition_precedes_multiplication(self):
        game = game_with("Beautiful Day")
        target = make_card("waterski")
        game.players[0].played_today = [make_card("work-call"), target]
        self.assertEqual(game.card_fun(0, target), 14)

    def test_free_effect_play_keeps_condition_fun_without_charging_energy(self):
        game = game_with("Beautiful Day")
        target = make_card("waterski")
        game.players[0].energy = 0
        game.play_card_for_effect(0, target)
        self.assertEqual(game.players[0].energy, 0)
        self.assertEqual(game.card_fun(0, target), 7)

    def test_copied_effect_uses_copier_tags_for_condition(self):
        game = game_with("Beautiful Day")
        copier = make_card("wedding-anniversary")
        game.players[0].played_today = [copier]
        game.copy_card_effect(0, make_card("waterski"), copier, pay_source_cost=False)
        self.assertEqual(game.card_fun(0, copier), 6)

    def test_energy_condition_precedes_halving_and_clamps_at_zero(self):
        game = game_with("Brutally Hot Day")
        game.players[0].played_today = [make_card("forced-family-fun")]
        self.assertEqual(game.energy_cost(0, make_card("scrabble")), 2)
        self.assertEqual(game.energy_cost(0, make_card("stay-up-late")), 0)
        both = CardInstance(1, CardDefinition("both", "Both", frozenset({"Indoors", "Outdoors"}), 3))
        self.assertEqual(game.energy_cost(0, both), 3)

    def test_sickness_before_tomorrow_and_resets_next_day(self):
        game = game_with("Normal day")
        game._condition_deck = [condition("Normal day"), condition("Sickness Spreading")]
        card = make_card("zumba")
        card.is_tomorrow = True
        game.players[0].tomorrow_cards = [card]
        game.start_day()
        self.assertEqual([p.energy for p in game.players], [10, 6, 6, 6])
        game.end_day()
        game.start_day()
        self.assertEqual([p.energy for p in game.players], [7] * 4)

    def test_copy_is_independent_and_preserves_conditions(self):
        game = Game.default(904, daily_conditions=True)
        game.setup()
        game.start_day()
        game.arrange_daily_conditions(0)
        clone = game.copy_for_simulation()
        self.assertEqual(clone.daily_condition, game.daily_condition)
        self.assertEqual(clone.known_daily_conditions(0), game.known_daily_conditions(0))
        clone.start_day()
        self.assertEqual(len(game.revealed_conditions), 1)
        self.assertEqual(len(clone.revealed_conditions), 2)

    def test_future_estimate_does_not_carry_current_weather(self):
        game = game_with("Sickness Spreading")
        self.assertEqual(game.prepare_condition_forecast(0), 7)
        self.assertIsNone(game.daily_condition)
        game._condition_knowledge[0] = (condition("Sickness Spreading"),)
        self.assertEqual(game.prepare_condition_forecast(0), 6)

    def test_serial_parallel_parity_with_conditions(self):
        self.assertEqual(simulate_games(4, seed=905, workers=1, daily_conditions=True),
                         simulate_games(4, seed=905, workers=2, daily_conditions=True))

    def test_galaxybrain_serial_parallel_parity_with_conditions(self):
        self.assertEqual(simulate_four_galaxybrain(4, seed=907, workers=1, daily_conditions=True),
                         simulate_four_galaxybrain(4, seed=907, workers=2, daily_conditions=True))

    def test_cli_flag_reaches_every_mode_and_defaults_off(self):
        from thenos.__main__ import main
        import sys
        modes = {"": "simulate_games", "--four-galaxybrain": "simulate_four_galaxybrain",
                 "--greedy-vs-random": "simulate_greedy_vs_random",
                 "--planner-vs-greedy": "simulate_planner_vs_greedy",
                 "--galaxybrain-vs-planner": "simulate_galaxybrain_vs_planner"}
        report = simulate_games(1, seed=908, workers=1)
        for mode, function in modes.items():
            for enabled in (False, True):
                args = ["thenos", "4", "--output", "unused.csv"]
                if mode:
                    args.append(mode)
                if enabled:
                    args.append("--daily-conditions")
                with patch.object(sys, "argv", args), patch("thenos.__main__." + function, return_value=report) as run, patch("thenos.__main__._code_revision", return_value="test"), patch("thenos.__main__.write_report_csv", return_value=Path("unused.csv")), patch("builtins.print"):
                    main()
                self.assertEqual(run.call_args.kwargs["daily_conditions"], enabled)

    def test_csv_preserves_mode_for_programmatic_runs(self):
        from tempfile import TemporaryDirectory
        report = simulate_games(1, seed=909, workers=1, daily_conditions=True)
        with TemporaryDirectory() as directory:
            output = write_report_csv(report, Path(directory) / "report.csv")
            with output.open() as source:
                rows = list(csv.DictReader(source))
        self.assertEqual({r["daily_conditions"] for r in rows}, {"True"})

    def test_all_policies_accept_new_decisions(self):
        game = game_with("Normal day")
        game.players[0].hand = [make_card("waterski")]
        for cls in (RandomAI, GreedyAI, PlannerAI, GeniusAI, MegamindAI, GalaxybrainAI):
            with self.subTest(policy=cls.__name__):
                ai = cls(random.Random(0))
                self.assertIn(ai.choose_tag(game, 0, ("Food", "Relax")), ("Food", "Relax"))
                self.assertEqual(sorted(ai.order_daily_conditions(game, 0, DAILY_CONDITIONS[:3])), [0, 1, 2])

    def test_galaxybrain_samples_hidden_order_but_keeps_private_knowledge(self):
        game = Game.default(906, daily_conditions=True)
        game.setup()
        game.start_day()
        game.arrange_daily_conditions(0)
        game.ais[0] = GalaxybrainAI(random.Random(9))
        other = game.copy_for_simulation()
        other._condition_deck.reverse()
        first = game.ais[0]._planning_copy(game, 0)
        second = other.ais[0]._planning_copy(other, 0)
        self.assertEqual(first._condition_deck, second._condition_deck)
        self.assertEqual(tuple(reversed(first._condition_deck[-3:])), game.known_daily_conditions(0))
        self.assertEqual(first.known_daily_conditions(1), ())
