#!/usr/bin/env python3
"""Implement cards in ordered batches with non-interactive Codex CLI runs.

The runner is deliberately fail-closed: it requires a clean Git worktree,
verifies registration and focused test files, runs the entire test suite, and
only then creates one commit and advances to the next batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "high"
EXPECTED_CARD_COUNT = 153

# These cards share implementation patterns and can be handled in batches.
# Titles are explicit so reordering cards.csv cannot change batch membership.
GROUP_SPECS = (
    (
        "pure-fun",
        "pure Fun cards",
        (
            "Tres Fute",
            "Splendor",
            "7 Wonders",
            "Dumbed Down Settlers",
            "Azul",
            "Agricola",
            "Terra Mystica",
            "Risk",
            "Civilization the Board Game",
            "Kneeboard",
            "Waterski",
            "Spikeball",
            "Screamer Battle",
            "Adventure Race",
            "Cheap White",
            "Chalk Art",
            "Biography",
            "Thriller Book",
            "Dock Fishing",
        ),
    ),
    ("pure-energy", "pure Energy cards", ("M&Ms", "Fajitas", "Nap")),
    (
        "future-fun",
        "cards adding Fun to cards played afterward",
        (
            "Trekking Through History",
            "Euchre",
            "Work Call",
            "Stretch",
            "Canoe",
            "Water Trampoline",
            "Water Volleyball",
            "Cheap Red",
            "Schwank",
            "High-End Red",
            "High-End White",
            "Nos Shirt",
            "Epic Playlist",
            "Ponyback",
            "Bug Spray",
            "Prime Picnic Table",
            "Nos Book",
            "Sweet Lawn Chair",
            "Bracelet Making",
            "Movie",
            "Johnny Appleseed",
            "Bring a Friend",
            "Doxology",
            "Long Distance Visitors",
            "Hold the Baby",
        ),
    ),
    (
        "previous-fun",
        "cards adding Fun to cards played before",
        ("Outdoor Movie", "Cliff Climbing", "Ice Wine", "Evening on the Dock"),
    ),
    (
        "future-energy",
        "cards changing Energy costs for future cards today",
        (
            "Forced Family Fun",
            "Boat Ride",
            "Treat Cereal",
            "Beaver Burger",
            "Rouladen",
            "Zero Gravity Chair",
            "Sunscreen",
            "Shady Spot",
            "Bend the Rules",
            "Medical Advice",
            "After Dinner Entertainment",
        ),
    ),
)


class RunnerError(RuntimeError):
    """A condition that must stop automation without committing."""


@dataclass(frozen=True, slots=True)
class CardRow:
    title: str
    tags: str
    cost: str
    effect: str


@dataclass(frozen=True, slots=True)
class Job:
    key: str
    label: str
    cards: tuple[CardRow, ...]


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    pieces: list[str] = []
    previous_was_separator = False
    for character in normalized.lower():
        if character.isalnum():
            pieces.append(character)
            previous_was_separator = False
        elif pieces and not previous_was_separator:
            pieces.append("-")
            previous_was_separator = True
    return "".join(pieces).strip("-")


def normalized_title(value: str) -> str:
    """Return a punctuation-, accent-, and case-insensitive title key."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in normalized.casefold() if character.isalnum()
    )


def read_card_rows(repo: Path) -> list[CardRow]:
    path = repo / "cards.csv"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = [
                CardRow(row["Title"], row["Tags"], row["Cost"], row["Effect"])
                for row in csv.DictReader(source)
            ]
    except (OSError, KeyError, csv.Error) as error:
        raise RunnerError(f"Could not read {path}: {error}") from error
    if len(rows) != EXPECTED_CARD_COUNT:
        raise RunnerError(
            f"Expected {EXPECTED_CARD_COUNT} cards in cards.csv, found {len(rows)}. "
            "Update the batch boundaries before running automation."
        )
    return rows


def build_jobs(rows: Sequence[CardRow]) -> list[Job]:
    jobs: list[Job] = []
    rows_by_title = {normalized_title(row.title): row for row in rows}
    grouped_titles: set[str] = set()
    for key, label, titles in GROUP_SPECS:
        cards: list[CardRow] = []
        for title in titles:
            title_key = normalized_title(title)
            try:
                card = rows_by_title[title_key]
            except KeyError as error:
                raise RunnerError(f"Batch {key} is missing card: {title}") from error
            if title_key in grouped_titles:
                raise RunnerError(f"Card appears in multiple batches: {title}")
            grouped_titles.add(title_key)
            cards.append(card)
        jobs.append(Job(key, label, tuple(cards)))

    for card in rows:
        if normalized_title(card.title) in grouped_titles:
            continue
        jobs.append(
            Job(
                key=f"unique-{slugify(card.title)}",
                label=f"unique card: {card.title}",
                cards=(card,),
            )
        )
    return jobs


def run_command(
    arguments: Sequence[str],
    repo: Path,
    *,
    capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=repo,
        input=input_text,
        text=True,
        capture_output=capture,
        check=False,
    )


def git(repo: Path, *arguments: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(("git", *arguments), repo, capture=capture)


def require_success(result: subprocess.CompletedProcess[str], description: str) -> str:
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "no output").strip()
        raise RunnerError(f"{description} failed ({result.returncode}):\n{details}")
    return result.stdout.strip()


def current_head(repo: Path) -> str:
    return require_success(git(repo, "rev-parse", "HEAD"), "Reading Git HEAD")


def ensure_git_preconditions(repo: Path) -> None:
    require_success(git(repo, "rev-parse", "--show-toplevel"), "Finding Git repository")
    status = require_success(
        git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "Reading Git status",
    )
    if status:
        raise RunnerError(
            "The Git worktree is not clean. Commit or intentionally remove/stash these "
            f"changes before running automation:\n{status}"
        )
    branch = require_success(git(repo, "branch", "--show-current"), "Reading Git branch")
    if not branch:
        raise RunnerError("Refusing to run on a detached HEAD")
    for key in ("user.name", "user.email"):
        value = git(repo, "config", "--get", key)
        if value.returncode != 0 or not value.stdout.strip():
            raise RunnerError(
                f"Git {key} is not configured in WSL. Configure it before automation."
            )


def read_registry(repo: Path) -> dict[str, str]:
    program = textwrap.dedent(
        """
        import json
        from thenos.cards.catalog import CARD_REGISTRY
        print(json.dumps({card.title: card.slug for card in CARD_REGISTRY.values()}))
        """
    )
    result = run_command((sys.executable, "-B", "-c", program), repo, capture=True)
    output = require_success(result, "Loading CARD_REGISTRY")
    try:
        registry = json.loads(output)
    except json.JSONDecodeError as error:
        raise RunnerError(f"CARD_REGISTRY produced invalid JSON: {error}") from error
    if not isinstance(registry, dict) or not all(
        isinstance(title, str) and isinstance(slug, str)
        for title, slug in registry.items()
    ):
        raise RunnerError("CARD_REGISTRY output had an unexpected shape")
    return registry


def select_jobs(
    jobs: Sequence[Job], batch: str | None, card_title: str | None
) -> list[Job]:
    if batch:
        selected = [job for job in jobs if job.key == batch]
        if not selected:
            raise RunnerError(f"Unknown batch: {batch}")
        return selected
    if card_title:
        selected = [
            job
            for job in jobs
            if len(job.cards) == 1 and job.cards[0].title.casefold() == card_title.casefold()
        ]
        if not selected:
            raise RunnerError(f"Unique card not found: {card_title}")
        return selected
    return list(jobs)


def pending_cards(job: Job, registry: dict[str, str]) -> tuple[CardRow, ...]:
    implemented_titles = {normalized_title(title) for title in registry}
    implemented_slugs = set(registry.values())
    return tuple(
        card
        for card in job.cards
        if normalized_title(card.title) not in implemented_titles
        and slugify(card.title) not in implemented_slugs
    )


def format_card(card: CardRow) -> str:
    effect = card.effect.replace("\n", "\n    ")
    return textwrap.dedent(
        f"""
        - Title: {card.title}
          Tags: {card.tags}
          Cost: {card.cost}
          Effect:
            {effect}
        """
    ).strip()


def build_prompt(
    job: Job, cards: Sequence[CardRow], extra_instructions: str = ""
) -> str:
    card_details = "\n\n".join(format_card(card) for card in cards)
    extra = ""
    if extra_instructions.strip():
        extra = f"\n\nAdditional user clarifications:\n{extra_instructions.strip()}"
    return textwrap.dedent(
        f"""
        Implement this The Nos card batch: {job.label}.

        First read AGENTS.md, rules.md, and the existing engine and tests. Implement only
        the target cards below. Follow AGENTS.md exactly. Add each CardDefinition to the
        registry, add generic engine hooks only where the rules require them, and create a
        focused tests/test_<card_slug_with_hyphens_changed_to_underscores>.py for every
        target card. Run the complete unit test suite before finishing.

        Do not edit cards.csv or rules.md. Do not create a Git commit; the outer runner owns
        verification and committing. Do not guess at ambiguous rules. If a behavior cannot
        be determined from the repository and the additional clarifications, stop and explain
        the ambiguity without claiming the card is implemented.

        Target cards:

        {card_details}{extra}
        """
    ).strip()


def codex_command(codex_binary: str, repo: Path) -> list[str]:
    return [
        codex_binary,
        "--ask-for-approval",
        "never",
        "exec",
        "--ephemeral",
        "--color",
        "never",
        "--model",
        MODEL,
        "--config",
        f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--sandbox",
        "workspace-write",
        "--cd",
        str(repo),
        "-",
    ]


def verify_implementation(repo: Path, cards: Sequence[CardRow]) -> dict[str, str]:
    registry = read_registry(repo)
    missing = [card.title for card in cards if card.title not in registry]
    if missing:
        raise RunnerError(
            "Codex exited without registering every requested card: " + ", ".join(missing)
        )

    missing_tests: list[str] = []
    for card in cards:
        registered_slug = registry[card.title]
        expected = repo / "tests" / f"test_{registered_slug.replace('-', '_')}.py"
        if not expected.is_file():
            missing_tests.append(str(expected.relative_to(repo)))
    if missing_tests:
        raise RunnerError(
            "Every card needs its focused test file. Missing: " + ", ".join(missing_tests)
        )
    return registry


def run_tests(repo: Path) -> None:
    print("\nRunning complete unit test suite...", flush=True)
    result = run_command(
        (sys.executable, "-B", "-m", "unittest", "discover", "-v"),
        repo,
        capture=False,
    )
    if result.returncode != 0:
        raise RunnerError(
            f"Tests failed with exit code {result.returncode}. Changes remain uncommitted."
        )


def commit_job(repo: Path, job: Job, cards: Sequence[CardRow]) -> str:
    require_success(git(repo, "add", "-A"), "Staging verified card changes")
    staged = git(repo, "diff", "--cached", "--quiet", capture=True)
    if staged.returncode == 0:
        raise RunnerError("Codex made no changes, so there is nothing to commit")
    if staged.returncode != 1:
        raise RunnerError("Could not inspect staged changes")

    if len(cards) == 1:
        message = f"card: implement {cards[0].title}"
    else:
        message = f"cards: implement {job.key} batch"
    result = git(repo, "commit", "-m", message)
    require_success(result, "Creating Git commit")
    return current_head(repo)


def process_job(
    repo: Path,
    job: Job,
    cards: Sequence[CardRow],
    codex_binary: str,
    extra_instructions: str,
) -> None:
    ensure_git_preconditions(repo)
    head_before = current_head(repo)
    titles = ", ".join(card.title for card in cards)
    print(f"\n=== {job.key}: {titles} ===", flush=True)

    prompt = build_prompt(job, cards, extra_instructions)
    result = run_command(
        codex_command(codex_binary, repo),
        repo,
        capture=False,
        input_text=prompt,
    )
    if result.returncode != 0:
        raise RunnerError(
            f"Codex failed with exit code {result.returncode}. Changes remain uncommitted."
        )
    if current_head(repo) != head_before:
        raise RunnerError("Codex created a commit even though the runner owns commits")

    verify_implementation(repo, cards)
    status = require_success(
        git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "Inspecting Codex changes",
    )
    if not status:
        raise RunnerError("Codex made no changes for this pending batch")
    run_tests(repo)
    new_head = commit_job(repo, job, cards)
    print(f"Committed {new_head[:12]} for {job.key}", flush=True)


def read_extra_instructions(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RunnerError(f"Could not read clarification file {path}: {error}") from error


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Implement sorted cards with verified Codex CLI batches"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--batch",
        choices=[spec[0] for spec in GROUP_SPECS],
        help="Run only one of the five grouped tiers",
    )
    selection.add_argument(
        "--card",
        help="Run only one exact card title from the unique tier",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Stop successfully after this many Codex jobs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List jobs and implementation status without changing anything",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending jobs without calling Codex or Git commit",
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        help="UTF-8 file of user clarifications appended to each Codex prompt",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex CLI executable name or path (default: codex)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to this script's parent repository)",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        if args.max_jobs is not None and args.max_jobs < 1:
            raise RunnerError("--max-jobs must be positive")
        repo = args.repo.resolve()
        rows = read_card_rows(repo)
        jobs = select_jobs(build_jobs(rows), args.batch, args.card)
        registry = read_registry(repo)

        pending_jobs: list[tuple[Job, tuple[CardRow, ...]]] = []
        for job in jobs:
            pending = pending_cards(job, registry)
            state = "done" if not pending else f"pending {len(pending)}/{len(job.cards)}"
            if args.list or args.dry_run:
                print(f"{job.key:<45} {state}")
            if pending:
                pending_jobs.append((job, pending))

        if args.list or args.dry_run:
            print(f"\nPending Codex jobs: {len(pending_jobs)}")
            return 0
        if not pending_jobs:
            print("All selected cards are already implemented.")
            return 0

        if shutil.which(args.codex_bin) is None:
            raise RunnerError(
                f"Codex CLI executable not found: {args.codex_bin}. Install it in WSL first."
            )
        ensure_git_preconditions(repo)
        extra_instructions = read_extra_instructions(args.instructions)

        completed = 0
        for job, pending in pending_jobs:
            process_job(repo, job, pending, args.codex_bin, extra_instructions)
            completed += 1
            if args.max_jobs is not None and completed >= args.max_jobs:
                break
        print(f"\nCompleted and committed {completed} Codex job(s).")
        return 0
    except RunnerError as error:
        print(f"\nSTOPPED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSTOPPED: interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
