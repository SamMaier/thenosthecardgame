#!/usr/bin/env python3
"""Pull the Cards tab from Google Sheets, rename card files, and sync code."""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import http.client
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "17hAe2onKFPaKInl2ehIONyngefBoCh5r3twGXVujTtE/edit?usp=sharing"
)
DEFAULT_TAB_NAME = "Cards"
DEFAULT_OUTPUT = ROOT / "cards.csv"
EXPECTED_HEADERS = ("Title", "Tags", "Cost", "Effect", "Flavor text")
IMPLEMENT_PROMPT = (
    "cards.csv was just updated, please fix all updated card implementations"
)
USER_AGENT = "Mozilla/5.0 (compatible; TheNosCardSync/1.0)"
TIMEOUT_SECONDS = 30

PROTECTED_TEST_FILES = frozenset(
    {
        "__init__.py",
        "helpers.py",
        "test_card_runner.py",
        "test_characters.py",
        "test_condition_statistics.py",
        "test_daily_conditions.py",
        "test_game.py",
        "test_pull_cards_csv.py",
        "test_simulation.py",
    }
)
PROTECTED_PHOTO_FILES = frozenset(
    {
        "copy.jpg",
        "__genericbg.jpg",
        "_genericfood.jpg",
        "_genericwine.jpg",
    }
)

_SPREADSHEET_ID_RE = re.compile(r"/spreadsheets/d/(?!e/)([a-zA-Z0-9_-]+)")
_ITEMS_PUSH_RE = re.compile(
    r"""items\.push\(\{((?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^"'}])*)\}\);""",
    re.DOTALL,
)
_JS_NAME_RE = re.compile(r'(?:^|[{,\s])name:\s*"((?:\\.|[^"\\])*)"')
_JS_GID_RE = re.compile(r'(?:^|[{,\s])gid:\s*"(-?\d+)"')
_JS_ESCAPE_RE = re.compile(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)")
_INVALID_PATH_CHARS_RE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')

_SIMPLE_JS_ESCAPES = {
    '"': '"',
    "'": "'",
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def extract_spreadsheet_id(url: str) -> str:
    """Extract the Google Spreadsheet document ID from a sharing/edit URL."""
    match = _SPREADSHEET_ID_RE.search(url)
    if match is None:
        raise ValueError(f"Could not extract spreadsheet ID from URL: {url!r}")
    return match.group(1)


def _unescape_js_string(raw: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith("x") and len(token) == 3:
            return chr(int(token[1:], 16))
        if token.startswith("u") and len(token) == 5:
            return chr(int(token[1:], 16))
        return _SIMPLE_JS_ESCAPES.get(token, token)

    unescaped = _JS_ESCAPE_RE.sub(_replace, raw)
    return unescaped.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def parse_sheet_tabs(htmlview_text: str) -> dict[str, str]:
    """Return a mapping of worksheet tab name -> gid from /htmlview markup."""
    tabs: dict[str, str] = {}
    for body in _ITEMS_PUSH_RE.findall(htmlview_text):
        name_match = _JS_NAME_RE.search(body)
        gid_match = _JS_GID_RE.search(body)
        if name_match is not None and gid_match is not None:
            name = _unescape_js_string(name_match.group(1))
            tabs[name] = gid_match.group(1)
    return tabs


def _fetch_bytes(url: str, *, timeout: float = TIMEOUT_SECONDS) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as error:
        raise RuntimeError(f"Failed to fetch {url}: {error}") from error


def resolve_tab_gid(
    spreadsheet_id: str,
    tab_name: str,
    *,
    timeout: float = TIMEOUT_SECONDS,
) -> str:
    """Resolve a worksheet tab name to its numeric gid."""
    htmlview_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/htmlview"
    html = _fetch_bytes(htmlview_url, timeout=timeout).decode("utf-8", "replace")
    tabs = parse_sheet_tabs(html)
    if not tabs:
        raise RuntimeError(
            f"Could not discover worksheet tabs from {htmlview_url}; "
            "verify the sheet is publicly viewable or pass --gid explicitly."
        )
    if tab_name not in tabs:
        available = ", ".join(repr(name) for name in tabs)
        raise ValueError(
            f"Tab {tab_name!r} not found in spreadsheet (available: {available})"
        )
    return tabs[tab_name]


def fetch_tab_csv(
    spreadsheet_id: str,
    gid: str,
    *,
    timeout: float = TIMEOUT_SECONDS,
) -> str:
    """Download the raw CSV export for a specific worksheet gid."""
    query = urllib.parse.urlencode({"format": "csv", "gid": gid})
    export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?{query}"
    text = _fetch_bytes(export_url, timeout=timeout).decode("utf-8-sig")
    stripped_prefix = text.lstrip()[:15].casefold()
    if stripped_prefix.startswith(("<!doctype", "<html")):
        raise RuntimeError(
            f"Received HTML instead of CSV from {export_url}; "
            "verify the spreadsheet is shared publicly."
        )
    return text


def _title_key(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    return "".join(ch for ch in normalized.casefold() if ch.isalnum())


def slugify_title(title: str) -> str:
    """Convert a card title to a hyphenated ASCII slug."""
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
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


def photo_stem(title: str) -> str:
    """Match the card photo filename convention used in scripts/add_card_images.py."""
    stem = unicodedata.normalize("NFKD", title)
    stem = stem.encode("ascii", "ignore").decode("ascii").lower()
    stem = _INVALID_PATH_CHARS_RE.sub("", stem)
    return stem.replace(" ", "_").strip(".")


def _python_stem(title: str) -> str:
    return slugify_title(title).replace("-", "_")


def _apostrophe_stripped_python_stem(title: str) -> str:
    stripped = title.replace("'", "").replace("\u2019", "")
    return slugify_title(stripped).replace("-", "_")


def clean_and_validate_cards_csv(raw_csv: str) -> tuple[list[list[str]], str]:
    """Validate exported sheet rows, strip blank/summary rows, and format CSV."""
    reader = csv.reader(io.StringIO(raw_csv, newline=""))
    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError("Exported CSV is empty") from error

    col_count = len(EXPECTED_HEADERS)
    if tuple(header[:col_count]) != EXPECTED_HEADERS:
        raise ValueError(
            f"Unexpected CSV header {header[:col_count]!r}; "
            f"expected {list(EXPECTED_HEADERS)!r}"
        )
    if any(cell.strip() for cell in header[col_count:]):
        raise ValueError(f"Unexpected extra header columns: {header[col_count:]!r}")

    card_rows: list[list[str]] = []
    seen_titles: set[str] = set()

    for row_number, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < col_count:
            raise ValueError(
                f"Row {row_number} has {len(row)} columns; expected at least {col_count}"
            )
        cells = row[:col_count]
        title, tags, cost, effect, flavor = cells

        if not title.strip():
            # Trailing summary formulas (e.g. average Cost) have an empty Title,
            # Tags, Effect, and Flavor text; reject rows missing only a Title.
            if tags.strip() or effect.strip() or flavor.strip():
                raise ValueError(
                    f"Row {row_number} has an empty Title with card data: {cells!r}"
                )
            continue

        if title != title.strip():
            raise ValueError(
                f"Row {row_number} Title has leading/trailing whitespace: {title!r}"
            )
        if not tags.strip():
            raise ValueError(f"Row {row_number} ({title!r}) has empty Tags")
        if not (cost == cost.strip() and cost.isascii() and cost.isdigit()):
            raise ValueError(
                f"Row {row_number} ({title!r}) has invalid Cost: {cost!r}"
            )

        key = _title_key(title)
        if not key or key in seen_titles:
            raise ValueError(
                f"Duplicate or invalid card Title on row {row_number}: {title!r}"
            )

        seen_titles.add(key)
        card_rows.append(cells)

    if not card_rows:
        raise ValueError("No card rows found in exported CSV")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(EXPECTED_HEADERS)
    writer.writerows(card_rows)
    formatted = buffer.getvalue().removesuffix("\r\n")
    return card_rows, formatted


def _tag_set(tags_cell: str) -> frozenset[str]:
    return frozenset(t.strip().upper() for t in tags_cell.split(",") if t.strip())


def detect_card_renames(
    previous_rows: Sequence[Sequence[str]],
    new_rows: Sequence[Sequence[str]],
) -> list[tuple[str, str]]:
    """Pair removed and added card rows that represent card renames."""
    col_count = len(EXPECTED_HEADERS)
    norm_prev = [(list(row) + [""] * col_count)[:col_count] for row in previous_rows]
    norm_new = [(list(row) + [""] * col_count)[:col_count] for row in new_rows]

    old_by_title = {row[0]: tuple(row) for row in norm_prev}
    new_by_title = {row[0]: tuple(row) for row in norm_new}

    unmatched_old = [tuple(row) for row in norm_prev if row[0] not in new_by_title]
    unmatched_new = [tuple(row) for row in norm_new if row[0] not in old_by_title]
    paired: dict[str, str] = {}

    def _pair_by_key(key_fn) -> None:
        nonlocal unmatched_old, unmatched_new
        old_groups: dict[object, list[tuple[str, ...]]] = {}
        new_groups: dict[object, list[tuple[str, ...]]] = {}
        for row in unmatched_old:
            old_groups.setdefault(key_fn(row), []).append(row)
        for row in unmatched_new:
            new_groups.setdefault(key_fn(row), []).append(row)

        matched_old_titles: set[str] = set()
        matched_new_titles: set[str] = set()
        for key, old_candidates in old_groups.items():
            new_candidates = new_groups.get(key, [])
            if len(old_candidates) == 1 and len(new_candidates) == 1:
                old_title = old_candidates[0][0]
                new_title = new_candidates[0][0]
                paired[old_title] = new_title
                matched_old_titles.add(old_title)
                matched_new_titles.add(new_title)

        unmatched_old = [r for r in unmatched_old if r[0] not in matched_old_titles]
        unmatched_new = [r for r in unmatched_new if r[0] not in matched_new_titles]

    # 1. Punctuation/spacing/accent/case-only title edits.
    _pair_by_key(lambda r: _title_key(r[0]))

    # 2. High title similarity with overlapping tags (runs before generic effect
    # matching so e.g. "BEAVER BURGER" -> "BEAVER BURGERS" is not hijacked by an
    # unrelated new card sharing a common effect like "(fun)(fun)(fun)").
    best_for_old: dict[str, tuple[float, str]] = {}
    tied_old: set[str] = set()
    best_for_new: dict[str, tuple[float, str]] = {}
    tied_new: set[str] = set()
    for old_row in unmatched_old:
        old_key = _title_key(old_row[0])
        if len(old_key) < 5:
            continue
        old_tags = _tag_set(old_row[1])
        for new_row in unmatched_new:
            new_key = _title_key(new_row[0])
            if len(new_key) < 5 or not (old_tags & _tag_set(new_row[1])):
                continue
            ratio = difflib.SequenceMatcher(None, old_key, new_key).ratio()
            if ratio < 0.75:
                continue
            prev_old = best_for_old.get(old_row[0])
            if prev_old is None or ratio > prev_old[0]:
                best_for_old[old_row[0]] = (ratio, new_row[0])
                tied_old.discard(old_row[0])
            elif ratio == prev_old[0]:
                tied_old.add(old_row[0])

            prev_new = best_for_new.get(new_row[0])
            if prev_new is None or ratio > prev_new[0]:
                best_for_new[new_row[0]] = (ratio, old_row[0])
                tied_new.discard(new_row[0])
            elif ratio == prev_new[0]:
                tied_new.add(new_row[0])

    sim_old_matched: set[str] = set()
    sim_new_matched: set[str] = set()
    for old_title, (_ratio, new_title) in best_for_old.items():
        if old_title in tied_old or new_title in tied_new:
            continue
        if best_for_new.get(new_title, (0.0, ""))[1] == old_title:
            paired[old_title] = new_title
            sim_old_matched.add(old_title)
            sim_new_matched.add(new_title)
    unmatched_old = [r for r in unmatched_old if r[0] not in sim_old_matched]
    unmatched_new = [r for r in unmatched_new if r[0] not in sim_new_matched]

    # 3. Identical (Tags, Cost, Effect) — e.g. retheming a card in place.
    _pair_by_key(lambda r: (_tag_set(r[1]), r[2].strip(), r[3].strip()))
    # 4. Identical (Tags, Effect) when Cost also changed.
    _pair_by_key(lambda r: (_tag_set(r[1]), r[3].strip()))

    return [
        (row[0], paired[row[0]])
        for row in norm_prev
        if row[0] in paired
    ]


def summarize_changes(
    previous_rows: Sequence[Sequence[str]],
    new_rows: Sequence[Sequence[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Return (added, removed, modified) card titles between two row lists."""
    col_count = len(EXPECTED_HEADERS)
    old_by_title = {row[0]: tuple(row[:col_count]) for row in previous_rows}
    new_by_title = {row[0]: tuple(row[:col_count]) for row in new_rows}

    added = [title for title in new_by_title if title not in old_by_title]
    removed = [title for title in old_by_title if title not in new_by_title]
    modified = [
        title
        for title, new_row in new_by_title.items()
        if title in old_by_title and old_by_title[title] != new_row
    ]
    return added, removed, modified


def _iter_card_candidate_files(repo: Path) -> list[tuple[Path, str]]:
    """Yield (path, kind) for scoped card test files and card photo assets."""
    candidates: list[tuple[Path, str]] = []

    tests_dir = repo / "tests"
    if tests_dir.is_dir():
        for path in sorted(tests_dir.iterdir()):
            if not path.is_file() or path.suffix.casefold() != ".py":
                continue
            name_lower = path.name.casefold()
            if (
                name_lower in PROTECTED_TEST_FILES
                or not name_lower.startswith("test_")
                or name_lower.endswith("_ai.py")
            ):
                continue
            candidates.append((path, "test"))

    photos_dir = repo / "design" / "photos"
    if photos_dir.is_dir():
        for path in sorted(photos_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.name.casefold() in PROTECTED_PHOTO_FILES:
                continue
            candidates.append((path, "photo"))

    return candidates


def _execute_staged_renames(planned: Sequence[tuple[Path, Path]]) -> None:
    staged: list[tuple[Path, Path, Path]] = []
    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in planned:
            fd, temp_name = tempfile.mkstemp(
                dir=source.parent,
                prefix=f".rename.{source.name}.",
                suffix=".tmp",
            )
            os.close(fd)
            temp_path = Path(temp_name)
            temp_path.unlink()
            source.rename(temp_path)
            staged.append((source, temp_path, destination))

        for source, temp_path, destination in staged:
            temp_path.rename(destination)
            completed.append((source, destination))
    except BaseException:
        # Undo Phase-2 moves in reverse order back to temp paths, then restore
        # all staged temp paths back to their original source locations.
        staged_by_src = {src: tmp for src, tmp, _dst in staged}
        for source, destination in reversed(completed):
            temp_path = staged_by_src[source]
            if destination.exists() and not temp_path.exists():
                with contextlib.suppress(OSError):
                    destination.rename(temp_path)
        for source, temp_path, _destination in reversed(staged):
            if temp_path.exists() and not source.exists():
                with contextlib.suppress(OSError):
                    temp_path.rename(source)
        raise


def rename_card_files(
    repo: Path,
    renames: Sequence[tuple[str, str]],
    *,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Find card test/photo files named after old titles and rename them."""
    if not renames:
        return []

    test_stem_map: dict[str, str] = {}
    photo_stem_map: dict[str, str] = {}
    for old_title, new_title in renames:
        new_py = _python_stem(new_title)
        if new_py:
            new_test = f"test_{new_py}"
            for old_py in (
                _python_stem(old_title),
                _apostrophe_stripped_python_stem(old_title),
            ):
                if old_py:
                    test_stem_map[f"test_{old_py}".casefold()] = new_test

        new_photo = photo_stem(new_title)
        if new_photo:
            for old_asset in (
                photo_stem(old_title),
                slugify_title(old_title),
                _python_stem(old_title),
            ):
                if old_asset:
                    photo_stem_map.setdefault(old_asset.casefold(), new_photo)

    planned: list[tuple[Path, Path]] = []
    seen_destinations: set[tuple[str, str]] = set()
    for source, kind in _iter_card_candidate_files(repo):
        stem_key = source.stem.casefold()
        mapping = test_stem_map if kind == "test" else photo_stem_map
        new_stem = mapping.get(stem_key)
        if new_stem is None:
            continue
        destination = source.with_name(f"{new_stem}{source.suffix}")
        if destination.name == source.name:
            continue
        dest_key = (str(destination.parent.resolve()).casefold(), destination.name.casefold())
        if dest_key in seen_destinations:
            raise FileExistsError(
                f"Multiple card files map to the same destination: {destination}"
            )
        seen_destinations.add(dest_key)
        planned.append((source, destination))

    moving_sources = [source for source, _ in planned]
    for source, destination in planned:
        if destination.exists() and not any(
            os.path.samefile(destination, moving) for moving in moving_sources
        ):
            raise FileExistsError(
                f"Cannot rename {source} -> {destination}: destination already exists"
            )

    if not dry_run and planned:
        _execute_staged_renames(planned)

    return planned


def find_agy_binary(repo: Path) -> str:
    """Locate the Antigravity CLI (`agy`) binary across Windows and WSL."""
    env_override = os.environ.get("AGY_BIN", "").strip()
    if env_override:
        return env_override

    for name in ("agy", "agy.exe"):
        found = shutil.which(name)
        if found:
            return found

    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "agy" / "bin" / "agy.exe")
    candidates.append(Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe")
    candidates.append(Path.home() / ".local" / "bin" / "agy")

    resolved_parts = repo.resolve().parts
    if (
        len(resolved_parts) >= 5
        and resolved_parts[:3] == ("/", "mnt", "c")
        and resolved_parts[3].casefold() == "users"
    ):
        win_user = resolved_parts[4]
        candidates.append(
            Path("/mnt/c/Users") / win_user / "AppData" / "Local" / "agy" / "bin" / "agy.exe"
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise RuntimeError(
        "Could not find the Antigravity CLI ('agy'). Install it from "
        "https://antigravity.google/docs/cli or pass --agy-bin / --no-implement."
    )


def _validate_agy_cwd(binary: str, repo: Path) -> None:
    if os.name == "nt" or not binary.casefold().endswith(".exe"):
        return
    parts = repo.resolve().parts
    if not (
        len(parts) >= 3
        and parts[:2] == ("/", "mnt")
        and len(parts[2]) == 1
        and parts[2].isalpha()
    ):
        raise RuntimeError(
            f"Cannot run Windows binary {binary!r} with non-Windows working "
            f"directory {repo}; install Linux 'agy' or move repo under /mnt/<drive>."
        )


def run_antigravity_session(
    repo: Path,
    prompt: str = IMPLEMENT_PROMPT,
    *,
    agy_bin: str | None = None,
) -> None:
    """Launch a non-interactive Antigravity CLI session to update card code."""
    binary = agy_bin.strip() if agy_bin else find_agy_binary(repo)
    _validate_agy_cwd(binary, repo)
    command = [
        binary,
        "--dangerously-skip-permissions",
        "--print-timeout",
        "0s",
        "-p",
        prompt,
    ]
    print(f"Launching Antigravity session in {repo} ...", flush=True)
    try:
        result = subprocess.run(command, cwd=repo, check=False)
    except OSError as error:
        raise RuntimeError(
            f"Failed to launch Antigravity CLI ({binary}): {error}"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(
            f"Antigravity session failed with exit code {result.returncode} "
            "(re-run with --force-implement to retry implementation)"
        )


def _read_existing_rows(path: Path) -> list[list[str]] | None:
    if not path.is_file():
        return None
    col_count = len(EXPECTED_HEADERS)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return [
                (row + [""] * col_count)[:col_count]
                for row in reader
                if any(cell.strip() for cell in row) and row[0].strip()
            ]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        print(
            f"Warning: could not parse existing {path} for diff ({error})",
            file=sys.stderr,
        )
        return None


def write_atomic_bytes(path: Path, data: bytes) -> None:
    """Atomically replace path with data while preserving existing file mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            shutil.copymode(path, temp_path)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(temp_path, 0o666 & ~umask)
        temp_path.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.chmod(temp_path, 0o600)
        temp_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pull the Cards tab from Google Sheets, rename card-named files, "
            "and launch an Antigravity session to update card implementations."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SHEET_URL,
        help="Google Sheets URL (default: project card spreadsheet).",
    )
    parser.add_argument(
        "--tab",
        default=DEFAULT_TAB_NAME,
        help=f"Worksheet tab name to export (default: {DEFAULT_TAB_NAME!r}).",
    )
    parser.add_argument(
        "--gid",
        default=None,
        help="Optional explicit worksheet gid (skips tab-name lookup).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository root for renaming files and running agy (default: output parent).",
    )
    parser.add_argument(
        "--agy-bin",
        default=None,
        help="Optional path to the 'agy' CLI executable.",
    )
    parser.add_argument(
        "--no-implement",
        action="store_true",
        help="Skip launching the Antigravity session after updating cards.csv.",
    )
    parser.add_argument(
        "--force-implement",
        action="store_true",
        help="Launch the Antigravity session even if cards.csv had no diff.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, validate, and report changes/renames without writing or running agy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output.resolve()
    repo_root = args.repo.resolve() if args.repo is not None else output_path.parent

    spreadsheet_id = extract_spreadsheet_id(args.url)
    gid = args.gid if args.gid is not None else resolve_tab_gid(spreadsheet_id, args.tab)
    raw_csv = fetch_tab_csv(spreadsheet_id, gid)
    card_rows, formatted_csv = clean_and_validate_cards_csv(raw_csv)

    previous_rows = _read_existing_rows(output_path)
    added: list[str] = []
    removed: list[str] = []
    modified: list[str] = []
    has_changes = False
    card_renames: list[tuple[str, str]] = []
    if previous_rows is not None:
        added, removed, modified = summarize_changes(previous_rows, card_rows)
        has_changes = bool(added or removed or modified)
        card_renames = detect_card_renames(previous_rows, card_rows)

    should_implement = (
        not args.dry_run
        and not args.no_implement
        and (has_changes or args.force_implement)
    )
    resolved_agy_bin: str | None = None
    if should_implement:
        resolved_agy_bin = (
            args.agy_bin.strip() if args.agy_bin else find_agy_binary(repo_root)
        )
        _validate_agy_cwd(resolved_agy_bin, repo_root)

    # Perform/validate file renames BEFORE overwriting cards.csv so a collision
    # leaves cards.csv untouched for a clean retry.
    file_renames = rename_card_files(
        repo_root,
        card_renames,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        try:
            write_atomic_bytes(output_path, formatted_csv.encode("utf-8"))
        except BaseException:
            if file_renames:
                with contextlib.suppress(OSError):
                    _execute_staged_renames(
                        [(dst, src) for src, dst in reversed(file_renames)]
                    )
            raise

    action = "Validated" if args.dry_run else "Wrote"
    print(f"{action} {len(card_rows)} cards -> {output_path}", flush=True)

    if previous_rows is not None:
        renamed_old = {old for old, _ in card_renames}
        renamed_new = {new for _, new in card_renames}
        if not has_changes:
            print("No card changes.", flush=True)
        else:
            for old_title, new_title in card_renames:
                print(f"  R {old_title} -> {new_title}", flush=True)
            for title in added:
                if title not in renamed_new:
                    print(f"  + {title}", flush=True)
            for title in removed:
                if title not in renamed_old:
                    print(f"  - {title}", flush=True)
            for title in modified:
                print(f"  ~ {title}", flush=True)

    for source, destination in file_renames:
        rel_src = source.relative_to(repo_root)
        rel_dst = destination.relative_to(repo_root)
        prefix = "Would rename" if args.dry_run else "Renamed"
        print(f"  {prefix} file: {rel_src} -> {rel_dst}", flush=True)

    if should_implement:
        run_antigravity_session(
            repo_root,
            IMPLEMENT_PROMPT,
            agy_bin=resolved_agy_bin,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
