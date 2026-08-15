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

    def modify_own_energy_cost(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
        current_cost: int,
    ) -> int:
        """Modify this card's own cost before visible sources are applied."""
        return current_cost

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        pass

    def allows_extra_card_plays(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        """Whether this card lets its player continue playing this turn."""
        return False

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        """Resolve the Tomorrow action of an active card."""
        pass

    def on_card_play(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        played_card: CardInstance,
    ) -> None:
        """Resolve an active card's reaction to one of its player's plays."""
        pass

    def allows_energy_gain(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
        source: CardInstance | None,
    ) -> bool:
        """Whether this visible card permits an Energy gain for its player."""
        return True

    def on_card_acquire(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        acquired_card: CardInstance,
    ) -> None:
        """Resolve an active card's reaction to one of its owner's acquisitions."""
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
        return card.effective_base_fun

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

    def on_score(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        """Resolve this card's effect immediately after it scores."""
        pass

    def on_end_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        """Resolve an effect timed after Fun scoring and before cleanup."""
        pass


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

    @property
    def effective_definition(self) -> CardDefinition:
        """Return the definition currently supplying this copy's behavior."""
        copied_definition = self.markers.get("_copied_definition")
        if isinstance(copied_definition, CardDefinition):
            return copied_definition
        return self.definition

    @property
    def effective_behavior(self) -> CardBehavior:
        return self.effective_definition.behavior

    @property
    def effective_cost(self) -> int:
        return self.effective_definition.cost

    @property
    def effective_base_fun(self) -> int:
        return self.effective_definition.base_fun

    @property
    def tags(self) -> frozenset[str]:
        """The card's printed tags, which copied effects do not replace."""
        return self.definition.tags
