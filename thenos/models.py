"""Mutable game state objects."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from thenos.cards.base import CardInstance


@dataclass(slots=True)
class PlayerState:
    name: str
    hand: list[CardInstance] = field(default_factory=list)
    played_today: list[CardInstance] = field(default_factory=list)
    tomorrow_cards: list[CardInstance] = field(default_factory=list)
    energy: int = 0
    fun: int = 0
    asleep: bool = False
    picked_cards: Counter[str] = field(default_factory=Counter)
    acquired_cards: Counter[str] = field(default_factory=Counter)

    @property
    def visible_cards(self) -> list[CardInstance]:
        # Returning a new list prevents callers from mutating either zone.
        return [*self.tomorrow_cards, *self.played_today]


@dataclass(slots=True)
class GameStats:
    suitcase_offers: Counter[str] = field(default_factory=Counter)
    suitcase_picks: Counter[str] = field(default_factory=Counter)
    card_acquisitions: Counter[str] = field(default_factory=Counter)
    card_plays: Counter[str] = field(default_factory=Counter)
