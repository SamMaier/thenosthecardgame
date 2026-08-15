"""Social card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.fun_effects import (
    FunForAllCardsBeforeBehavior,
    _is_after,
    _today_position,
)

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class DatesFirstNozBehavior(CardBehavior):
    """Pick from the Suitcase after later plays, then boost Tomorrow's plays."""

    has_tomorrow_action = True

    def on_card_play(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        played_card: CardInstance,
    ) -> None:
        if _is_after(player, source, played_card):
            game.pick_from_suitcase(game.players.index(player))

    def modify_fun(
        self,
        game: Game,
        player: PlayerState,
        source: CardInstance,
        target: CardInstance,
        current_fun: int,
    ) -> int:
        if source.is_tomorrow and _today_position(player, target) is not None:
            return current_fun + 1
        return current_fun


class TellAStoryBehavior(CardBehavior):
    """Score a bonus if this player played an Event earlier today."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        card_position = next(
            (
                position
                for position, played_card in enumerate(player.played_today)
                if played_card is card
            ),
            None,
        )
        if card_position is None:
            return card.effective_base_fun

        has_previous_event = any(
            "Event" in played_card.tags
            for played_card in player.played_today[:card_position]
        )
        return card.effective_base_fun + (3 if has_previous_event else 0)


class NewNozBookEntryBehavior(CardBehavior):
    """Score a bonus when at least five cards were played earlier today."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        card_position = _today_position(player, card)
        if card_position is None:
            return card.effective_base_fun

        bonus = 4 if card_position >= 5 else 0
        return card.effective_base_fun + bonus


class CampfireBehavior(CardBehavior):
    """Score one Fun for each active Tomorrow card this player has."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        return card.effective_base_fun + len(player.tomorrow_cards)


class ScoutTheOtherCottagesBehavior(FunForAllCardsBeforeBehavior):
    """Mark the current Suitcase and acquire survivors after all scoring."""

    def __init__(self) -> None:
        super().__init__(-1)

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        owner_marker = f"_scout_energy_cube_{card.instance_id}"
        for suitcase_card in game.suitcase:
            suitcase_card.markers["energy_cube"] = True
            suitcase_card.markers[owner_marker] = True

    def on_end_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        owner_marker = f"_scout_energy_cube_{card.instance_id}"
        targets = tuple(
            suitcase_card
            for suitcase_card in game.suitcase
            if suitcase_card.markers.get(owner_marker)
        )
        if targets:
            game.acquire_suitcase_cards(game.players.index(player), targets)


class EveningChatBehavior(CardBehavior):
    """Go to bed immediately and reduce the next day's starting Energy."""

    has_tomorrow_action = True

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        game.go_to_bed(game.players.index(player))

    def on_start_day(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        player.energy -= 1


class BeachConversationBehavior(CardBehavior):
    """Gain Energy when the immediately preceding play was Outdoors."""

    def on_play(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> None:
        if (
            len(player.played_today) >= 2
            and "Outdoors" in player.played_today[-2].tags
        ):
            game.gain_energy(player, 2, card)


TELL_A_STORY = CardDefinition(
    slug="tell-a-story",
    title="Tell a Story",
    tags=frozenset({"Social"}),
    cost=3,
    base_fun=2,
    behavior=TellAStoryBehavior(),
)

NEW_NOZ_BOOK_ENTRY = CardDefinition(
    slug="new-noz-book-entry",
    title="New Noz Book Entry",
    tags=frozenset({"Social", "Event"}),
    cost=2,
    base_fun=1,
    behavior=NewNozBookEntryBehavior(),
)

DATES_FIRST_NOZ = CardDefinition(
    slug="dates-first-noz",
    title="Date's First Noz",
    tags=frozenset({"Social"}),
    cost=7,
    behavior=DatesFirstNozBehavior(),
)

CAMPFIRE = CardDefinition(
    slug="campfire",
    title="Campfire",
    tags=frozenset({"Social", "Event", "Outdoors"}),
    cost=1,
    behavior=CampfireBehavior(),
)

SCOUT_THE_OTHER_COTTAGES = CardDefinition(
    slug="scout-the-other-cottages",
    title="Scout the Other Cottages",
    tags=frozenset({"Social", "Indoors"}),
    cost=2,
    behavior=ScoutTheOtherCottagesBehavior(),
)

EVENING_CHAT = CardDefinition(
    slug="evening-chat",
    title="Evening Chat",
    tags=frozenset({"Social", "Indoors"}),
    cost=3,
    base_fun=5,
    behavior=EveningChatBehavior(),
)

BEACH_CONVERSATION = CardDefinition(
    slug="beach-conversation",
    title="Beach Conversation",
    tags=frozenset({"Social", "Outdoors"}),
    cost=2,
    base_fun=2,
    behavior=BeachConversationBehavior(),
)
