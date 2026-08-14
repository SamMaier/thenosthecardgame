"""Extension points used by card implementations.

Behaviors should be stateless. Per-copy state belongs on ``CardInstance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class CardBehavior:
    """Default behavior for a card with no special rules text."""

    has_tomorrow_action = False

    def can_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        return True

    def modify_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_cost: int,
    ) -> int:
        """Modify ``target`` while ``source`` is visible."""
        return current_cost

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        pass

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        """Resolve the Tomorrow action of an active card."""
        pass

    def allows_extra_suitcase_pick(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        """Whether an active Tomorrow card offers an extra pick this turn."""
        return False

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.definition.base_fun

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        """Modify ``target`` while ``source`` is visible."""
        return current_fun


@dataclass(frozen=True, slots=True)
class CardDefinition:
    slug: str
    title: str
    tags: frozenset[str]
    cost: int
    base_fun: int = 0
    behavior: CardBehavior = field(default_factory=CardBehavior)


@dataclass(slots=True)
class CardInstance:
    """A physical card copy and any state attached to that copy."""

    instance_id: int
    definition: CardDefinition
    is_tomorrow: bool = False
    markers: dict[str, object] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.definition.title
