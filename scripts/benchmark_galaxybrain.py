"""Compare Galaxybrain with deprecated Megamind and retain the results."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from thenos.ais import GalaxybrainAI
from thenos.ais.megamind import MegamindAI
from thenos.simulation import AIStatistics, Competitor, simulate_games


TIMING_SEED = 20260921
STRENGTH_SEED = 20260922


def code_revision() -> str:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return f"{revision}+dirty" if dirty and revision != "unknown" else revision


def statistics(stats: AIStatistics) -> dict[str, int | float]:
    return {
        "player_games": stats.games,
        "score_total": stats.score_total,
        "average_fun": stats.average_score,
        "win_credit": stats.win_credit,
        "win_rate_per_appearance": stats.win_rate,
        "outright_wins": stats.outright_wins,
        "shared_wins": stats.shared_wins,
    }


def timed_four_policy(
    name: str,
    factory: type[GalaxybrainAI] | type[MegamindAI],
    *,
    workers: int,
    daily_conditions: bool = False,
) -> tuple[float, AIStatistics]:
    started = time.perf_counter()
    report = simulate_games(
        16,
        seed=TIMING_SEED,
        competitors=tuple(Competitor(name, factory) for _ in range(4)),
        rotate_seats=True,
        workers=workers,
        daily_conditions=daily_conditions,
    )
    return time.perf_counter() - started, report.ais[name]


def write_result(result: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / f"galaxybrain-validation-{timestamp}.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--daily-conditions", action="store_true")
    args = parser.parse_args()

    result: dict[str, object] = {
        "metadata": {
            "code_revision": code_revision(),
            "python": platform.python_version(),
            "workers": args.workers,
            "rotate_seats": True,
            "daily_conditions": args.daily_conditions,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    }

    galaxy_seconds, galaxy_timing_stats = timed_four_policy(
        "Galaxybrain", GalaxybrainAI, workers=args.workers, daily_conditions=args.daily_conditions
    )
    mega_seconds, mega_timing_stats = timed_four_policy(
        "Megamind", MegamindAI, workers=args.workers, daily_conditions=args.daily_conditions
    )
    timing_passed = galaxy_seconds < mega_seconds
    result["timing"] = {
        "games_per_policy": 16,
        "seed": TIMING_SEED,
        "galaxybrain_elapsed_seconds": galaxy_seconds,
        "megamind_elapsed_seconds": mega_seconds,
        "galaxybrain": statistics(galaxy_timing_stats),
        "megamind": statistics(mega_timing_stats),
        "passed": timing_passed,
    }
    print(
        f"Timing: Galaxybrain {galaxy_seconds:.3f}s; "
        f"Megamind {mega_seconds:.3f}s; passed={timing_passed}",
        flush=True,
    )

    if timing_passed:
        competitors = (
            Competitor("Galaxybrain", GalaxybrainAI),
            Competitor("Megamind", MegamindAI),
            Competitor("Galaxybrain", GalaxybrainAI),
            Competitor("Megamind", MegamindAI),
        )
        started = time.perf_counter()
        report = simulate_games(
            100,
            seed=STRENGTH_SEED,
            competitors=competitors,
            rotate_seats=True,
            workers=args.workers, daily_conditions=args.daily_conditions,
        )
        strength_seconds = time.perf_counter() - started
        galaxy = report.ais["Galaxybrain"]
        mega = report.ais["Megamind"]
        strength_passed = (
            galaxy.average_score > mega.average_score
            and galaxy.win_credit > 50.0
        )
        result["strength"] = {
            "games": 100,
            "seed": STRENGTH_SEED,
            "elapsed_seconds": strength_seconds,
            "competitors": [
                "Galaxybrain",
                "Megamind",
                "Galaxybrain",
                "Megamind",
            ],
            "galaxybrain": statistics(galaxy),
            "megamind": statistics(mega),
            "passed": strength_passed,
        }
        print(
            f"Strength: Galaxybrain Fun {galaxy.average_score:.3f}, "
            f"win credit {galaxy.win_credit:.3f}; Megamind Fun "
            f"{mega.average_score:.3f}, win credit {mega.win_credit:.3f}; "
            f"passed={strength_passed}",
            flush=True,
        )
    else:
        strength_passed = False
        result["strength"] = {"skipped": "timing gate failed"}

    output = write_result(result, args.output_dir)
    print(f"Result: {output.resolve()}")
    return 0 if timing_passed and strength_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
