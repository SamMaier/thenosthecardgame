import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pull_cards_csv import (
    DEFAULT_SHEET_URL,
    IMPLEMENT_PROMPT,
    _validate_agy_cwd,
    clean_and_validate_cards_csv,
    detect_card_renames,
    extract_spreadsheet_id,
    fetch_tab_csv,
    main,
    parse_sheet_tabs,
    photo_stem,
    rename_card_files,
    resolve_tab_gid,
    run_antigravity_session,
    summarize_changes,
)


class PullCardsCsvTests(unittest.TestCase):
    def test_extract_spreadsheet_id(self) -> None:
        self.assertEqual(
            extract_spreadsheet_id(DEFAULT_SHEET_URL),
            "17hAe2onKFPaKInl2ehIONyngefBoCh5r3twGXVujTtE",
        )
        with self.assertRaises(ValueError):
            extract_spreadsheet_id("https://example.com/not-a-sheet")
        with self.assertRaises(ValueError):
            extract_spreadsheet_id(
                "https://docs.google.com/spreadsheets/d/e/2PACX-1vTest/pubhtml"
            )

    def test_parse_sheet_tabs_handles_escapes_surrogates_and_anchoring(self) -> None:
        many_props = ", ".join(f'k{i}: "v{i}"' for i in range(35))
        html = (
            "var items = [];"
            f"items.push({{{many_props}, nested: {{x: 1}}}});"
            'items.push({sheet_name: "Wrong", name: "Cards", pageUrl: "});\\x3d1", parent_gid: "9", gid: "0"});'
            'items.push({gid: "259747421", name: "Quote \\"Tab\\" \\u0026 \\uD83C\\uDCCF"});'
        )
        self.assertEqual(
            parse_sheet_tabs(html),
            {"Cards": "0", 'Quote "Tab" & \U0001f0cf': "259747421"},
        )

    def test_clean_and_validate_strips_blank_summary_and_unnamed_columns(self) -> None:
        raw = (
            "Title,Tags,Cost,Effect,Flavor text,\r\n"
            "7 WONDERS,BOARDGAME,2,(fun)(fun)(fun),Flavor,scratch note\r\n"
            'MULTILINE,"FOOD, SOCIAL",3,"(fun)\nPick (suitcase)",,\r\n'
            ",,,,,\r\n"
            ",,2.565217391,,,\r\n"
        )
        rows, formatted = clean_and_validate_cards_csv(raw)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            formatted,
            "Title,Tags,Cost,Effect,Flavor text\r\n"
            "7 WONDERS,BOARDGAME,2,(fun)(fun)(fun),Flavor\r\n"
            'MULTILINE,"FOOD, SOCIAL",3,"(fun)\nPick (suitcase)",',
        )

    def test_validation_rejects_malformed_rows_and_headers(self) -> None:
        bad_cases = [
            "",
            "Wrong,Tags,Cost,Effect,Flavor text\r\nA,FOOD,1,x,y",
            "Title,Tags,Cost,Effect,Flavor text,NamedExtra\r\nA,FOOD,1,x,y,z",
            "Title,Tags,Cost,Effect,Flavor text\r\nA,FOOD,1",
            "Title,Tags,Cost,Effect,Flavor text\r\n,FOOD,1,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\n,,1,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\n,,,,stray flavor",
            "Title,Tags,Cost,Effect,Flavor text\r\n A,FOOD,1,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\nA,,1,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\nA,FOOD,2.5,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\nA,FOOD,+2,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\nA,FOOD,-1,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\nA,FOOD,1,x,y\r\na,FOOD,2,x,y",
            "Title,Tags,Cost,Effect,Flavor text\r\n,,,,",
        ]
        for raw in bad_cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    clean_and_validate_cards_csv(raw)

    def test_fetch_tab_csv_strips_bom_and_rejects_html_responses(self) -> None:
        with patch(
            "scripts.pull_cards_csv._fetch_bytes",
            return_value="\ufeffTitle,Tags,Cost,Effect,Flavor text".encode("utf-8"),
        ):
            self.assertEqual(
                fetch_tab_csv("sheet123", "0"),
                "Title,Tags,Cost,Effect,Flavor text",
            )
        for html in (b"<!DOCTYPE html><html>", b"  <!doctype html><html>Sign in</html>"):
            with patch("scripts.pull_cards_csv._fetch_bytes", return_value=html):
                with self.assertRaises(RuntimeError):
                    fetch_tab_csv("sheet123", "0")

    def test_resolve_tab_gid_errors_on_missing_or_undiscovered_tabs(self) -> None:
        with patch("scripts.pull_cards_csv._fetch_bytes", return_value=b"<html></html>"):
            with self.assertRaises(RuntimeError):
                resolve_tab_gid("sheet123", "Cards")

        html = b'items.push({name: "Other", gid: "99"});'
        with patch("scripts.pull_cards_csv._fetch_bytes", return_value=html):
            with self.assertRaises(ValueError):
                resolve_tab_gid("sheet123", "Cards")

    def test_detect_card_renames_priority_and_short_title_safety(self) -> None:
        old_rows = [
            ["5 10 15", "BOARDGAME", "3", "(fun)(fun)\nPick (suitcase)", "old"],
            ["CIVILIZATION THE BOARD GAME", "BOARDGAME, INDOORS", "10", "(fun)", "x"],
            # Renamed + rebalanced, while an unrelated new FOOD card shares old effect!
            ["BEAVER BURGER", "FOOD", "3", "(fun)(fun)(fun)", "x"],
            # Short 4-char title should NOT false-match another 4-char title with diff effect.
            ["SOLO", "BOARDGAME", "3", "(fun)(fun)", "x"],
        ]
        new_rows = [
            ["DOMINION", "BOARDGAME", "3", "(fun)(fun)\nPick (suitcase)", "new"],
            ["CIVILIZATION THE BOARDGAME", "BOARDGAME, INDOORS", "10", "(fun)(fun)", "x"],
            ["BEAVER BURGERS", "FOOD", "3", "(fun)(fun)(fun)(fun)", "x"],
            ["FRUIT SALAD", "FOOD", "2", "(fun)(fun)(fun)", "x"],
            ["POLO", "BOARDGAME", "4", "(energy)(energy)", "x"],
        ]
        self.assertEqual(
            detect_card_renames(old_rows, new_rows),
            [
                ("5 10 15", "DOMINION"),
                ("CIVILIZATION THE BOARD GAME", "CIVILIZATION THE BOARDGAME"),
                ("BEAVER BURGER", "BEAVER BURGERS"),
            ],
        )
        self.assertEqual(photo_stem('WHAT/WHY: "A*B"?'), "whatwhy_ab")

    def test_rename_card_files_scoping_swaps_collisions_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            photos = repo / "design" / "photos"
            tests_dir = repo / "tests"
            engine_cards = repo / "thenos" / "cards"
            photos.mkdir(parents=True)
            tests_dir.mkdir(parents=True)
            engine_cards.mkdir(parents=True)

            (photos / "5_10_15.JPG").write_bytes(b"img")
            (photos / "COPY.jpg").write_bytes(b"placeholder")
            (tests_dir / "test_5_10_15.py").write_text("card test", encoding="utf-8")
            # Engine modules and protected non-card tests must NEVER be renamed.
            (engine_cards / "food.py").write_text("engine", encoding="utf-8")
            (tests_dir / "test_game.py").write_text("engine test", encoding="utf-8")
            (tests_dir / "test_beryllium_ai.py").write_text("ai test", encoding="utf-8")

            # Dry run reports renames without modifying files.
            planned = rename_card_files(
                repo,
                [
                    ("5 10 15", "DOMINION"),
                    ("FOOD", "SNACKS"),
                    ("GAME", "MATCH"),
                    ("BERYLLIUM AI", "CARBON AI"),
                    ("COPY", "PASTE"),
                ],
                dry_run=True,
            )
            self.assertEqual(len(planned), 2)
            self.assertTrue((photos / "5_10_15.JPG").is_file())
            self.assertTrue((tests_dir / "test_5_10_15.py").is_file())

            # Real run renames only the card photo & card test.
            moved = rename_card_files(
                repo,
                [
                    ("5 10 15", "DOMINION"),
                    ("FOOD", "SNACKS"),
                    ("GAME", "MATCH"),
                    ("BERYLLIUM AI", "CARBON AI"),
                    ("COPY", "PASTE"),
                ],
            )
            self.assertEqual(len(moved), 2)
            self.assertTrue((photos / "dominion.JPG").is_file())
            self.assertTrue((tests_dir / "test_dominion.py").is_file())
            self.assertTrue((engine_cards / "food.py").is_file())
            self.assertTrue((tests_dir / "test_game.py").is_file())
            self.assertTrue((tests_dir / "test_beryllium_ai.py").is_file())
            self.assertTrue((photos / "COPY.jpg").is_file())

            # Swap A <-> B works cleanly via staging.
            (tests_dir / "test_alpha.py").write_text("alpha", encoding="utf-8")
            (tests_dir / "test_beta.py").write_text("beta", encoding="utf-8")
            rename_card_files(repo, [("ALPHA", "BETA"), ("BETA", "ALPHA")])
            self.assertEqual((tests_dir / "test_alpha.py").read_text(), "beta")
            self.assertEqual((tests_dir / "test_beta.py").read_text(), "alpha")

            # Existing destination raises FileExistsError.
            (tests_dir / "test_old_card.py").write_text("old", encoding="utf-8")
            (tests_dir / "test_new_card.py").write_text("new", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                rename_card_files(repo, [("OLD CARD", "NEW CARD")])
            self.assertTrue((tests_dir / "test_old_card.py").is_file())

            # Duplicate sources mapping to the same destination raises FileExistsError.
            (photos / "dup-card.jpg").write_bytes(b"1")
            (photos / "dup_card.jpg").write_bytes(b"2")
            with self.assertRaises(FileExistsError):
                rename_card_files(repo, [("DUP CARD", "TARGET CARD")])

    def test_main_ordering_flags_and_antigravity_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "cards.csv"
            (repo / "tests").mkdir()
            (repo / "design" / "photos").mkdir(parents=True)
            (repo / "tests" / "test_5_10_15.py").write_text("pass", encoding="utf-8")
            (repo / "design" / "photos" / "5_10_15.jpg").write_bytes(b"jpg")

            initial_csv = (
                b"Title,Tags,Cost,Effect,Flavor text\r\n"
                b'5 10 15,BOARDGAME,3,"(fun)(fun)\nPick (suitcase)",old\r\n'
                b"SAME CARD,RELAX,2,(fun)(fun),same\r\n"
                b"MOD CARD,SOCIAL,3,(fun),before"
            )
            out.write_bytes(initial_csv)
            html = b'items.push({name: "Cards", gid: "42"});'
            exported = (
                "Title,Tags,Cost,Effect,Flavor text\r\n"
                'DOMINION,BOARDGAME,3,"(fun)(fun)\nPick (suitcase)",new\r\n'
                "SAME CARD,RELAX,2,(fun)(fun),same\r\n"
                "MOD CARD,SOCIAL,4,(fun),after\r\n"
                ",,,,\r\n"
                ",,3.0,,"
            ).encode("utf-8")

            def fake_fetch(url: str, *, timeout: float = 30) -> bytes:
                if url.endswith("/htmlview"):
                    return html
                self.assertIn("gid=42", url)
                return exported

            # If a rename collides, cards.csv MUST NOT be overwritten yet!
            collision = repo / "tests" / "test_dominion.py"
            collision.write_text("existing", encoding="utf-8")
            with (
                patch("scripts.pull_cards_csv._fetch_bytes", side_effect=fake_fetch),
                self.assertRaises(FileExistsError),
            ):
                main(["--agy-bin", "fake-agy", "--output", str(out)])
            self.assertEqual(out.read_bytes(), initial_csv)
            collision.unlink()

            # --dry-run validates and previews without writing or running agy.
            dry_out = io.StringIO()
            with (
                patch("scripts.pull_cards_csv._fetch_bytes", side_effect=fake_fetch),
                patch("scripts.pull_cards_csv.subprocess.run") as mock_dry_run,
                contextlib.redirect_stdout(dry_out),
            ):
                rc = main(["--dry-run", "--output", str(out)])
            self.assertEqual(rc, 0)
            mock_dry_run.assert_not_called()
            self.assertEqual(out.read_bytes(), initial_csv)
            self.assertIn("Would rename file:", dry_out.getvalue())

            # Full run renames files, writes cards.csv, and invokes agy.
            stdout = io.StringIO()
            with (
                patch("scripts.pull_cards_csv._fetch_bytes", side_effect=fake_fetch),
                patch(
                    "scripts.pull_cards_csv.subprocess.run",
                    return_value=subprocess.CompletedProcess(["agy"], 0),
                ) as mock_run,
                contextlib.redirect_stdout(stdout),
            ):
                rc = main(["--agy-bin", "fake-agy", "--output", str(out)])

            self.assertEqual(rc, 0)
            output_text = stdout.getvalue()
            self.assertIn("R 5 10 15 -> DOMINION", output_text)
            self.assertIn("~ MOD CARD", output_text)
            self.assertTrue((repo / "tests" / "test_dominion.py").is_file())
            self.assertTrue((repo / "design" / "photos" / "dominion.jpg").is_file())
            mock_run.assert_called_once_with(
                [
                    "fake-agy",
                    "--dangerously-skip-permissions",
                    "--print-timeout",
                    "0s",
                    "-p",
                    IMPLEMENT_PROMPT,
                ],
                cwd=out.resolve().parent,
                check=False,
            )

            # Second run with no changes does not re-launch agy unless --force-implement.
            stdout_same = io.StringIO()
            with (
                patch("scripts.pull_cards_csv._fetch_bytes", return_value=exported),
                patch("scripts.pull_cards_csv.subprocess.run") as mock_run_same,
                contextlib.redirect_stdout(stdout_same),
            ):
                rc = main(["--gid", "42", "--agy-bin", "fake-agy", "--output", str(out)])
            self.assertEqual(rc, 0)
            mock_run_same.assert_not_called()
            self.assertIn("No card changes.", stdout_same.getvalue())

            with (
                patch("scripts.pull_cards_csv._fetch_bytes", return_value=exported),
                patch(
                    "scripts.pull_cards_csv.subprocess.run",
                    return_value=subprocess.CompletedProcess(["agy"], 0),
                ) as mock_force,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc = main(
                    [
                        "--gid",
                        "42",
                        "--force-implement",
                        "--agy-bin",
                        "fake-agy",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(rc, 0)
            mock_force.assert_called_once()

    def test_agy_errors_and_wsl_cwd_guard(self) -> None:
        with patch("scripts.pull_cards_csv.os.name", "posix"):
            with self.assertRaises(RuntimeError):
                _validate_agy_cwd("/mnt/c/agy.exe", Path("/tmp/not_windows_mount"))

        with (
            patch(
                "scripts.pull_cards_csv.subprocess.run",
                return_value=subprocess.CompletedProcess(["agy"], 2),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                run_antigravity_session(
                    Path("/mnt/c/Users/Test/Repo"), agy_bin="fake-agy"
                )

    def test_main_recovers_when_existing_csv_is_corrupt_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cards.csv"
            out.write_bytes(b"Title,Tags,Cost,Effect,Flavor text\r\n\xff\xfeCorrupt")
            exported = (
                "Title,Tags,Cost,Effect,Flavor text\r\n"
                "7 WONDERS,BOARDGAME,2,(fun)(fun)(fun),Flavor"
            ).encode("utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("scripts.pull_cards_csv._fetch_bytes", return_value=exported),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                rc = main(["--gid", "0", "--no-implement", "--output", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn("Warning:", stderr.getvalue())
            self.assertEqual(out.read_bytes(), exported)

    def test_summarize_changes(self) -> None:
        added, removed, modified = summarize_changes(
            [["A", "FOOD", "1", "x", ""], ["B", "FOOD", "2", "x", ""]],
            [["B", "FOOD", "3", "x", ""], ["C", "FOOD", "1", "x", ""]],
        )
        self.assertEqual(added, ["C"])
        self.assertEqual(removed, ["A"])
        self.assertEqual(modified, ["B"])


if __name__ == "__main__":
    unittest.main()
