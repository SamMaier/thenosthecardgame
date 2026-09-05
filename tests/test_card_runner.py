import unittest
from pathlib import Path

from scripts.implement_cards import (
    GROUP_SPECS,
    MODEL,
    REASONING_EFFORT,
    build_jobs,
    build_prompt,
    codex_command,
    normalized_title,
    pending_cards,
    read_card_rows,
    select_jobs,
    slugify,
)


REPO = Path(__file__).resolve().parents[1]


class CardRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = read_card_rows(REPO)
        cls.jobs = build_jobs(cls.rows)

    def test_builds_five_groups_then_ninety_four_single_card_jobs(self) -> None:
        self.assertEqual(len(self.jobs), 99)
        self.assertEqual(
            [len(job.cards) for job in self.jobs[:5]],
            [len(spec[2]) for spec in GROUP_SPECS],
        )
        self.assertTrue(all(len(job.cards) == 1 for job in self.jobs[5:]))

    def test_sorted_boundaries_match_expected_cards(self) -> None:
        self.assertEqual(self.jobs[0].cards[0].title, "Tres Fute")
        self.assertEqual(self.jobs[0].cards[-1].title, "Dock Fishing")
        self.assertEqual(self.jobs[1].cards[0].title, "M&Ms")
        self.assertEqual(self.jobs[4].cards[-1].title, "After Dinner Entertainment")
        self.assertEqual(self.jobs[5].cards[0].title, "The Crew")
        self.assertEqual(self.jobs[2].cards[-1].title, "Hold the Baby")

    def test_existing_cards_are_removed_from_pending_group(self) -> None:
        pending = pending_cards(
            self.jobs[0], {"Biography": "biography", "Waterski": "waterski"}
        )
        self.assertEqual(len(pending), 17)
        self.assertNotIn("Biography", {card.title for card in pending})

    def test_implemented_card_is_skipped_despite_title_formatting(self) -> None:
        unique_card = self.jobs[5]
        registry = {"THE-CREW!": "unrelated-slug"}

        self.assertEqual(pending_cards(unique_card, registry), ())

    def test_implemented_card_is_skipped_by_registered_slug(self) -> None:
        unique_card = self.jobs[5]
        registry = {"A display title that differs": "the-crew"}

        self.assertEqual(pending_cards(unique_card, registry), ())

    def test_can_select_group_or_unique_card(self) -> None:
        self.assertEqual(select_jobs(self.jobs, "pure-energy", None)[0].key, "pure-energy")
        selected = select_jobs(self.jobs, None, "The Crew")
        self.assertEqual(selected[0].cards[0].title, "The Crew")

    def test_prompt_forbids_commits_and_includes_exact_card_data(self) -> None:
        job = self.jobs[5]
        prompt = build_prompt(job, job.cards, "Use clarification X.")
        self.assertIn("Do not create a Git commit", prompt)
        self.assertIn("Title: The Crew", prompt)
        self.assertIn("Use clarification X.", prompt)

    def test_codex_command_uses_luna_high_workspace_write(self) -> None:
        command = codex_command("codex", REPO)
        self.assertIn(MODEL, command)
        self.assertIn(f'model_reasoning_effort="{REASONING_EFFORT}"', command)
        self.assertIn("workspace-write", command)
        self.assertEqual(command[-1], "-")

    def test_slugify_handles_punctuation_and_accents(self) -> None:
        self.assertEqual(slugify("M&Ms"), "m-ms")
        self.assertEqual(slugify("Pudding Chômeur"), "pudding-chomeur")

    def test_normalized_title_ignores_case_punctuation_and_accents(self) -> None:
        self.assertEqual(normalized_title("Pudding Chômeur"), "puddingchomeur")
        self.assertEqual(normalized_title("THE-CREW!"), "thecrew")


if __name__ == "__main__":
    unittest.main()
