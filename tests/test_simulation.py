import unittest
from csv import DictReader
from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

from thenos.ais import RandomAI
from thenos.cards import CARD_REGISTRY
from thenos.simulation import (
    AIStatistics,
    CardStatistics,
    Competitor,
    SimulationReport,
    _GameOutcome,
    _PlayerOutcome,
    _merge_outcome,
    simulate_four_osmium,
    simulate_four_osmiumbrain,
    simulate_games,
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

    def test_parallel_progress_counts_futures_as_they_complete(self) -> None:
        futures = [Mock() for _ in range(16)]
        for future in futures:
            future.result.return_value = self._empty_outcome()
        executor = MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = None
        executor.submit.side_effect = futures

        with (
            patch(
                "thenos.simulation.ProcessPoolExecutor",
                return_value=executor,
            ),
            patch(
                "thenos.simulation.as_completed",
                return_value=tuple(reversed(futures)),
            ),
            patch("thenos.simulation.print") as print_mock,
        ):
            simulate_games(16, seed=2026, workers=2)

        self.assertEqual(
            print_mock.call_args_list,
            [
                call("Run 8 complete", flush=True),
                call("Run 16 complete", flush=True),
            ],
        )

    def test_batch_collects_pick_play_and_win_statistics(self) -> None:
        report = simulate_games(16, seed=2026)

        self.assertEqual(report.games, 16)
        self.assertEqual(set(report.cards), {card.title for card in CARD_REGISTRY.values()})
        self.assertGreaterEqual(sum(card.picks for card in report.cards.values()), 750)
        self.assertEqual(
            sum(card.free_pick_offers for card in report.cards.values()),
            4 * sum(card.free_picks for card in report.cards.values()),
        )
        for title, card in report.cards.items():
            self.assertGreater(card.offers, 0)
            self.assertGreater(card.picks, 0)
            self.assertGreaterEqual(card.offers, card.picks)
            self.assertGreaterEqual(card.free_pick_offers, card.free_picks)
            self.assertGreaterEqual(card.free_pick_rate, 0.0)
            self.assertLessEqual(card.free_pick_rate, 1.0)
            self.assertGreaterEqual(card.acquisitions, card.picks)
            self.assertGreaterEqual(card.plays, 0)
            if title in {"Chuck a Frisbee", "Wedding Anniversary", "Leftovers", "Nos Shirt"}:
                self.assertLessEqual(card.play_rate, 6.0)
            else:
                self.assertLessEqual(card.play_rate, 1.0)
            self.assertGreaterEqual(card.win_rate_when_picked, 0.0)
            self.assertLessEqual(card.win_rate_when_picked, 1.0)
            self.assertGreaterEqual(card.win_rate_when_acquired, 0.0)
            self.assertLessEqual(card.win_rate_when_acquired, 1.0)
            self.assertGreater(card.player_games_first_seen, 0)
            self.assertGreaterEqual(card.first_seen_win_rate, 0.0)
            self.assertLessEqual(card.first_seen_win_rate, 1.0)

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

    def test_first_seen_win_rate_tracks_first_offered_player_without_double_weighting(
        self,
    ) -> None:
        report = SimulationReport(
            games=1,
            cards={"Unseen": CardStatistics()},
        )
        outcome = _GameOutcome(
            free_pick_offers=Counter(),
            free_picks=Counter(),
            suitcase_offers=Counter(),
            suitcase_picks=Counter(),
            card_acquisitions=Counter(),
            card_plays=Counter(),
            card_plays_without_acquisition=Counter(),
            players=(
                _PlayerOutcome(
                    "Player 1",
                    "Test AI",
                    8,
                    0.0,
                    Counter(),
                    Counter(),
                ),
                _PlayerOutcome(
                    "Player 2",
                    "Test AI",
                    12,
                    0.5,
                    Counter(),
                    Counter(),
                    Counter({"Sleep In": 2}),
                ),
                _PlayerOutcome(
                    "Player 3",
                    "Test AI",
                    6,
                    0.0,
                    Counter({"Sleep In": 1}),
                    Counter({"Sleep In": 1}),
                ),
                _PlayerOutcome(
                    "Player 4",
                    "Test AI",
                    12,
                    0.5,
                    Counter(),
                    Counter(),
                ),
            ),
        )

        _merge_outcome(report, outcome)

        sleep_in = report.cards["Sleep In"]
        self.assertEqual(sleep_in.win_rate, 0.0)
        self.assertEqual(sleep_in.player_games_with_card, 1)
        self.assertEqual(sleep_in.player_games_without_card, 3)
        self.assertEqual(sleep_in.player_games_first_seen, 1)
        self.assertEqual(sleep_in.win_credit_when_first_seen, 0.5)
        self.assertEqual(sleep_in.first_seen_win_rate, 0.5)
        self.assertEqual(report.cards["Unseen"].first_seen_win_rate, 0.0)

    def test_matchup_aggregates_named_ai_results(self) -> None:
        report = simulate_games(
            1,
            seed=99,
            competitors=(
                Competitor("Alpha", RandomAI),
                Competitor("Beta", RandomAI),
                Competitor("Beta", RandomAI),
                Competitor("Beta", RandomAI),
            ),
            workers=1,
        )

        self.assertEqual(report.ais["Alpha"].games, 1)
        self.assertEqual(report.ais["Beta"].games, 3)
        self.assertAlmostEqual(
            report.ais["Alpha"].win_credit + report.ais["Beta"].win_credit,
            1.0,
        )

    def test_parallel_batch_matches_serial_batch_exactly(self) -> None:
        serial = simulate_games(4, seed=314159, workers=1)
        parallel = simulate_games(4, seed=314159, workers=2)

        self.assertEqual(parallel, serial)

    def test_four_osmium_and_cli_mode(self) -> None:
        import io
        import sys
        from contextlib import redirect_stderr
        from thenos.__main__ import main

        self.assertIs(simulate_four_osmiumbrain, simulate_four_osmium)

        report_four = simulate_four_osmium(1, seed=109, workers=1)
        self.assertEqual(report_four.ais["Osmium"].games, 4)
        self.assertAlmostEqual(report_four.ais["Osmium"].win_credit, 1.0)

        with patch.object(sys, "argv", ["thenos", "4", "--four-osmium"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

        fake_report = simulate_games(1, seed=110, workers=1)
        with (
            patch.object(
                sys,
                "argv",
                ["thenos", "4", "--four-osmiumbrain", "--output", "unused.csv"],
            ),
            patch(
                "thenos.__main__.simulate_four_osmium",
                return_value=fake_report,
            ) as run_mock,
            patch("thenos.__main__._code_revision", return_value="test"),
            patch(
                "thenos.__main__.write_report_csv",
                return_value=Path("unused.csv"),
            ) as write_mock,
            patch("builtins.print"),
        ):
            main()
        run_mock.assert_called_once()
        metadata = write_mock.call_args.kwargs["metadata"]
        self.assertEqual(metadata["run_mode"], "four-osmium")
        self.assertEqual(metadata["competitors"], "Osmium,Osmium,Osmium,Osmium")
        self.assertTrue(metadata["rotate_seats"])

    def test_no_arg_cli_defaults_to_four_osmium_10000_runs_32_workers_and_sequential_date_output(
        self,
    ) -> None:
        from datetime import date
        import sys
        from thenos.__main__ import main

        fake_report = simulate_games(1, seed=111, workers=1)
        fake_report.ais["Osmium"] = AIStatistics(
            games=4,
            score_total=120,
            win_credit=1.0,
            outright_wins=1,
        )

        class _FixedDate(date):
            current = date(2026, 11, 20)

            @classmethod
            def today(cls) -> date:
                return cls.current

        with TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            with (
                patch.object(sys, "argv", ["thenos"]),
                patch("thenos.__main__.DEFAULT_RESULTS_DIR", results_dir),
                patch("thenos.__main__.date", _FixedDate),
                patch(
                    "thenos.__main__.simulate_four_osmium",
                    return_value=fake_report,
                ) as run_mock,
                patch("thenos.__main__._code_revision", return_value="test"),
                patch(
                    "thenos.__main__.write_report_csv",
                    side_effect=lambda _r, out, **_kw: Path(out),
                ) as write_mock,
                patch("builtins.print"),
            ):
                main()
            run_mock.assert_called_once_with(
                10000,
                20261120,
                workers=32,
                daily_conditions=False,
                characters=False,
            )
            self.assertEqual(
                write_mock.call_args.args[1],
                results_dir / "four-osmium-1-20261120.csv",
            )
            metadata = write_mock.call_args.kwargs["metadata"]
            self.assertEqual(metadata["seed"], 20261120)
            self.assertEqual(metadata["run_mode"], "four-osmium")
            self.assertEqual(metadata["competitors"], "Osmium,Osmium,Osmium,Osmium")
            self.assertEqual(metadata["workers"], 32)
            self.assertTrue(metadata["rotate_seats"])

            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "four-osmium-1-20261120.csv").write_text("", encoding="utf-8")

            # Second run on same date increments both run_number and seed.
            with (
                patch.object(sys, "argv", ["thenos"]),
                patch("thenos.__main__.DEFAULT_RESULTS_DIR", results_dir),
                patch("thenos.__main__.date", _FixedDate),
                patch(
                    "thenos.__main__.simulate_four_osmium",
                    return_value=fake_report,
                ) as run_mock,
                patch("thenos.__main__._code_revision", return_value="test"),
                patch(
                    "thenos.__main__.write_report_csv",
                    side_effect=lambda _r, out, **_kw: Path(out),
                ) as write_mock,
                patch("builtins.print"),
            ):
                main()
            run_mock.assert_called_once_with(
                10000,
                20261121,
                workers=32,
                daily_conditions=False,
                characters=False,
            )
            self.assertEqual(
                write_mock.call_args.args[1],
                results_dir / "four-osmium-2-20261121.csv",
            )

            (results_dir / "four-osmium-2-20261121.csv").write_text("", encoding="utf-8")
            (results_dir / "four-osmium-10000-seed-20261122.csv").write_text(
                "", encoding="utf-8"
            )
            _FixedDate.current = date(2026, 11, 21)

            # Next day skips already-used seeds (20261121 and legacy 20261122)
            # without letting legacy 10000-game filename inflate run_number.
            with (
                patch.object(sys, "argv", ["thenos"]),
                patch("thenos.__main__.DEFAULT_RESULTS_DIR", results_dir),
                patch("thenos.__main__.date", _FixedDate),
                patch(
                    "thenos.__main__.simulate_four_osmium",
                    return_value=fake_report,
                ) as run_mock,
                patch("thenos.__main__._code_revision", return_value="test"),
                patch(
                    "thenos.__main__.write_report_csv",
                    side_effect=lambda _r, out, **_kw: Path(out),
                ) as write_mock,
                patch("builtins.print"),
            ):
                main()
            run_mock.assert_called_once_with(
                10000,
                20261123,
                workers=32,
                daily_conditions=False,
                characters=False,
            )
            self.assertEqual(
                write_mock.call_args.args[1],
                results_dir / "four-osmium-3-20261123.csv",
            )

            # Sidecar files and four-osmium-seed-<seed>.csv are also respected via main([]).
            (results_dir / "four-osmium-3-20261123.characters.csv").write_text(
                "", encoding="utf-8"
            )
            (results_dir / "four-osmium-seed-20261124.csv").write_text(
                "", encoding="utf-8"
            )
            with (
                patch("thenos.__main__.DEFAULT_RESULTS_DIR", results_dir),
                patch("thenos.__main__.date", _FixedDate),
                patch(
                    "thenos.__main__.simulate_four_osmium",
                    return_value=fake_report,
                ) as run_mock,
                patch("thenos.__main__._code_revision", return_value="test"),
                patch(
                    "thenos.__main__.write_report_csv",
                    side_effect=lambda _r, out, **_kw: Path(out),
                ) as write_mock,
                patch("builtins.print"),
            ):
                main([])
            run_mock.assert_called_once_with(
                10000,
                20261125,
                workers=32,
                daily_conditions=False,
                characters=False,
            )
            self.assertEqual(
                write_mock.call_args.args[1],
                results_dir / "four-osmium-4-20261125.csv",
            )

            file_instead_of_dir = Path(directory) / "not_a_dir"
            file_instead_of_dir.write_text("x", encoding="utf-8")
            with (
                patch("thenos.__main__.DEFAULT_RESULTS_DIR", file_instead_of_dir),
                self.assertRaises(NotADirectoryError),
            ):
                main([])

    def test_cli_default_mode_with_args_and_four_random_flag(self) -> None:
        import io
        import sys
        from contextlib import redirect_stderr
        from thenos.__main__ import main

        with patch.object(sys, "argv", ["thenos", "4"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

        fake_report = simulate_games(1, seed=112, workers=1)
        with (
            patch.object(sys, "argv", ["thenos", "4", "--output", "unused.csv"]),
            patch(
                "thenos.__main__.simulate_four_osmium",
                return_value=fake_report,
            ) as osmium_mock,
            patch("thenos.__main__._code_revision", return_value="test"),
            patch(
                "thenos.__main__.write_report_csv",
                return_value=Path("unused.csv"),
            ) as write_mock,
            patch("builtins.print"),
        ):
            main()
        osmium_mock.assert_called_once_with(
            4,
            None,
            workers=32,
            daily_conditions=False,
            characters=False,
        )
        self.assertEqual(write_mock.call_args.kwargs["metadata"]["run_mode"], "four-osmium")

        printed_lines: list[str] = []
        with (
            patch(
                "thenos.__main__.simulate_games",
                return_value=fake_report,
            ) as random_no_out_mock,
            patch("thenos.__main__._code_revision", return_value="test"),
            patch(
                "builtins.print",
                side_effect=lambda *vals, **_kw: printed_lines.append(
                    " ".join(str(v) for v in vals)
                ),
            ),
        ):
            main(["4", "--four-random"])
        random_no_out_mock.assert_called_once_with(
            4,
            None,
            workers=32,
            daily_conditions=False,
            characters=False,
        )
        self.assertFalse(any(line.startswith("AI ") for line in printed_lines))

        with (
            patch.object(
                sys,
                "argv",
                ["thenos", "4", "--four-random", "--output", "unused.csv"],
            ),
            patch(
                "thenos.__main__.simulate_games",
                return_value=fake_report,
            ) as random_mock,
            patch("thenos.__main__._code_revision", return_value="test"),
            patch(
                "thenos.__main__.write_report_csv",
                return_value=Path("unused.csv"),
            ) as write_mock,
            patch("builtins.print"),
        ):
            main()
        random_mock.assert_called_once_with(
            4,
            None,
            workers=32,
            daily_conditions=False,
            characters=False,
        )
        metadata = write_mock.call_args.kwargs["metadata"]
        self.assertEqual(metadata["run_mode"], "four-random")
        self.assertEqual(metadata["competitors"], "Random,Random,Random,Random")
        self.assertFalse(metadata["rotate_seats"])

    def test_worker_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            simulate_games(1, workers=0)

    def test_csv_report_writes_only_requested_card_columns(self) -> None:
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
                    win_credit_when_first_seen=0.75,
                    player_games_first_seen=2,
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
                reader = DictReader(stream)
                fieldnames = reader.fieldnames
                rows = list(reader)

        self.assertEqual(
            fieldnames,
            [
                "card",
                "free_pick_rate",
                "win_rate",
                "fun_added",
                "play_rate",
                "win_credit_when_acquired",
                "first_seen_win_rate",
            ],
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["card"], "Biography")
        self.assertEqual(row["free_pick_rate"], "0.25")
        self.assertEqual(row["win_rate"], "0.25")
        self.assertEqual(row["fun_added"], "4.0")
        self.assertEqual(row["play_rate"], "0.5")
        self.assertEqual(row["win_credit_when_acquired"], "0.5")
        self.assertEqual(row["first_seen_win_rate"], "0.375")


if __name__ == "__main__":
    unittest.main()
