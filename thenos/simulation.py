"""Run batches of games and aggregate balance statistics."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from thenos.game import Game


@dataclass(slots=True)
class CardStatistics:
    offers: int = 0
    picks: int = 0
    acquisitions: int = 0
    plays: int = 0
    win_credit_when_picked: float = 0.0
    win_credit_when_acquired: float = 0.0

    @property
    def pick_rate(self) -> float:
        return self.picks / self.offers if self.offers else 0.0

    @property
    def play_rate(self) -> float:
        return self.plays / self.acquisitions if self.acquisitions else 0.0

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


def simulate_games(games: int, seed: int | None = None) -> SimulationReport:
    if games < 1:
        raise ValueError("games must be positive")
    master_rng = random.Random(seed)
    report = SimulationReport(games=games)

    for _ in range(games):
        game = Game.default(seed=master_rng.getrandbits(64))
        result = game.run()

        titles = set(game.stats.suitcase_offers)
        titles.update(game.stats.suitcase_picks)
        titles.update(game.stats.card_acquisitions)
        titles.update(game.stats.card_plays)
        for title in titles:
            stats = report.cards.setdefault(title, CardStatistics())
            stats.offers += game.stats.suitcase_offers[title]
            stats.picks += game.stats.suitcase_picks[title]
            stats.acquisitions += game.stats.card_acquisitions[title]
            stats.plays += game.stats.card_plays[title]

        for player, win_share in zip(game.players, result.win_shares, strict=True):
            for title, copies_picked in player.picked_cards.items():
                report.cards[title].win_credit_when_picked += win_share * copies_picked
            for title, copies_acquired in player.acquired_cards.items():
                report.cards[title].win_credit_when_acquired += (
                    win_share * copies_acquired
                )

        for player in game.players:
            report.score_totals[player.name] += player.fun

    return report
