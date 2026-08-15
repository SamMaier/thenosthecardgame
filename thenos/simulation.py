"""Run batches of games and aggregate balance statistics."""

from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

from thenos.ais import GeniusAI, GreedyAI, PlannerAI, PlayerAI, RandomAI
from thenos.cards.catalog import create_default_deck
from thenos.game import Game, PLAYER_COUNT


AIFactory = Callable[[random.Random], PlayerAI]


@dataclass(frozen=True, slots=True)
class Competitor:
    """A named AI factory occupying one seat in every simulated game."""

    name: str
    factory: AIFactory


@dataclass(slots=True)
class AIStatistics:
    games: int = 0
    score_total: int = 0
    win_credit: float = 0.0
    outright_wins: int = 0
    shared_wins: int = 0

    @property
    def average_score(self) -> float:
        return self.score_total / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        """Fractional win rate, with tied winners splitting one win."""
        return self.win_credit / self.games if self.games else 0.0


@dataclass(slots=True)
class CardStatistics:
    offers: int = 0
    picks: int = 0
    acquisitions: int = 0
    plays: int = 0
    plays_without_acquisition: int = 0
    win_credit_when_picked: float = 0.0
    win_credit_when_acquired: float = 0.0

    @property
    def pick_rate(self) -> float:
        return self.picks / self.offers if self.offers else 0.0

    @property
    def play_rate(self) -> float:
        denominator = self.acquisitions + self.plays_without_acquisition
        return self.plays / denominator if denominator else 0.0

    @property
    def win_rate_when_picked(self) -> float:
        return self.win_credit_when_picked / self.picks if self.picks else 0.0

    @property
    def win_rate_when_acquired(self) -> float:
        return (
            self.win_credit_when_acquired / self.acquisitions
            if self.acquisitions
            else 0.0
        )


@dataclass(slots=True)
class SimulationReport:
    games: int
    cards: dict[str, CardStatistics] = field(default_factory=dict)
    score_totals: Counter[str] = field(default_factory=Counter)
    ais: dict[str, AIStatistics] = field(default_factory=dict)

    def rows(self) -> list[dict[str, int | float | str]]:
        return [
            {
                "card": title,
                "offers": stats.offers,
                "picks": stats.picks,
                "pick_rate": stats.pick_rate,
                "acquisitions": stats.acquisitions,
                "plays": stats.plays,
                "play_rate": stats.play_rate,
                "win_rate_when_picked": stats.win_rate_when_picked,
                "win_rate_when_acquired": stats.win_rate_when_acquired,
            }
            for title, stats in sorted(self.cards.items())
        ]


@dataclass(frozen=True, slots=True)
class _GameJob:
    seed: int
    competitors: tuple[Competitor, ...]


@dataclass(frozen=True, slots=True)
class _PlayerOutcome:
    player_name: str
    competitor_name: str
    fun: int
    win_share: float
    picked_cards: Counter[str]
    acquired_cards: Counter[str]


@dataclass(frozen=True, slots=True)
class _GameOutcome:
    suitcase_offers: Counter[str]
    suitcase_picks: Counter[str]
    card_acquisitions: Counter[str]
    card_plays: Counter[str]
    card_plays_without_acquisition: Counter[str]
    players: tuple[_PlayerOutcome, ...]


def _run_game(job: _GameJob) -> _GameOutcome:
    """Run one self-contained game; safe to call in a worker process."""
    game_rng = random.Random(job.seed)
    ais = [
        competitor.factory(random.Random(game_rng.getrandbits(64)))
        for competitor in job.competitors
    ]
    game = Game(create_default_deck(), ais, game_rng)
    result = game.run()
    return _GameOutcome(
        suitcase_offers=game.stats.suitcase_offers,
        suitcase_picks=game.stats.suitcase_picks,
        card_acquisitions=game.stats.card_acquisitions,
        card_plays=game.stats.card_plays,
        card_plays_without_acquisition=(
            game.stats.card_plays_without_acquisition
        ),
        players=tuple(
            _PlayerOutcome(
                player_name=player.name,
                competitor_name=competitor.name,
                fun=player.fun,
                win_share=win_share,
                picked_cards=player.picked_cards,
                acquired_cards=player.acquired_cards,
            )
            for player, win_share, competitor in zip(
                game.players,
                result.win_shares,
                job.competitors,
                strict=True,
            )
        ),
    )


def _merge_outcome(report: SimulationReport, outcome: _GameOutcome) -> None:
    titles = set(outcome.suitcase_offers)
    titles.update(outcome.suitcase_picks)
    titles.update(outcome.card_acquisitions)
    titles.update(outcome.card_plays)
    titles.update(outcome.card_plays_without_acquisition)
    for title in titles:
        stats = report.cards.setdefault(title, CardStatistics())
        stats.offers += outcome.suitcase_offers[title]
        stats.picks += outcome.suitcase_picks[title]
        stats.acquisitions += outcome.card_acquisitions[title]
        stats.plays += outcome.card_plays[title]
        stats.plays_without_acquisition += (
            outcome.card_plays_without_acquisition[title]
        )

    for player in outcome.players:
        for title, copies_picked in player.picked_cards.items():
            report.cards[title].win_credit_when_picked += (
                player.win_share * copies_picked
            )
        for title, copies_acquired in player.acquired_cards.items():
            report.cards[title].win_credit_when_acquired += (
                player.win_share * copies_acquired
            )

        ai_stats = report.ais.setdefault(player.competitor_name, AIStatistics())
        ai_stats.games += 1
        ai_stats.score_total += player.fun
        ai_stats.win_credit += player.win_share
        ai_stats.outright_wins += player.win_share == 1.0
        ai_stats.shared_wins += 0.0 < player.win_share < 1.0
        report.score_totals[player.player_name] += player.fun


def simulate_games(
    games: int,
    seed: int | None = None,
    competitors: Sequence[Competitor] | None = None,
    *,
    rotate_seats: bool = False,
    workers: int = 1,
) -> SimulationReport:
    """Run games and aggregate card and AI results.

    With no competitors supplied this preserves the original four-random-AI
    simulation. Named duplicate competitors are aggregated, which makes a
    three-Random-versus-one-Greedy report concise. Optional seat rotation
    removes fixed first-seat bias from head-to-head experiments. Set ``workers``
    above one to distribute games across processes and use multiple CPU cores.
    Parallel competitor factories must be picklable (normally a top-level class
    or function).
    """
    if games < 1:
        raise ValueError("games must be positive")
    if competitors is None:
        competitors = tuple(
            Competitor(f"Player {index + 1}", RandomAI)
            for index in range(PLAYER_COUNT)
        )
    else:
        competitors = tuple(competitors)
    if len(competitors) != PLAYER_COUNT:
        raise ValueError(f"The Nos requires exactly {PLAYER_COUNT} competitors")
    if workers < 1:
        raise ValueError("workers must be positive")

    master_rng = random.Random(seed)
    report = SimulationReport(games=games)
    jobs: list[_GameJob] = []
    for game_number in range(games):
        if rotate_seats:
            offset = game_number % PLAYER_COUNT
            seated_competitors = (
                competitors[offset:] + competitors[:offset]
            )
        else:
            seated_competitors = competitors
        jobs.append(
            _GameJob(master_rng.getrandbits(64), seated_competitors)
        )

    if workers == 1:
        outcomes = map(_run_game, jobs)
        for outcome in outcomes:
            _merge_outcome(report, outcome)
    else:
        chunksize = max(1, games // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outcomes = executor.map(_run_game, jobs, chunksize=chunksize)
            for outcome in outcomes:
                _merge_outcome(report, outcome)

    return report


def simulate_greedy_vs_random(
    games: int,
    seed: int | None = None,
    *,
    workers: int = 1,
) -> SimulationReport:
    """Run a seat-balanced match of one Greedy AI against three Random AIs."""
    return simulate_games(
        games,
        seed,
        (
            Competitor("Greedy", GreedyAI),
            Competitor("Random", RandomAI),
            Competitor("Random", RandomAI),
            Competitor("Random", RandomAI),
        ),
        rotate_seats=True,
        workers=workers,
    )


def simulate_planner_vs_greedy(
    games: int,
    seed: int | None = None,
    *,
    workers: int = 1,
) -> SimulationReport:
    """Run a seat-balanced match of one Planner against three Greedy AIs."""
    return simulate_games(
        games,
        seed,
        (
            Competitor("Planner", PlannerAI),
            Competitor("Greedy", GreedyAI),
            Competitor("Greedy", GreedyAI),
            Competitor("Greedy", GreedyAI),
        ),
        rotate_seats=True,
        workers=workers,
    )


def simulate_genius_vs_planner(
    games: int,
    seed: int | None = None,
    *,
    workers: int = 1,
) -> SimulationReport:
    """Run a seat-balanced match of one Genius against three Planners."""
    return simulate_games(
        games,
        seed,
        (
            Competitor("Genius", GeniusAI),
            Competitor("Planner", PlannerAI),
            Competitor("Planner", PlannerAI),
            Competitor("Planner", PlannerAI),
        ),
        rotate_seats=True,
        workers=workers,
    )
