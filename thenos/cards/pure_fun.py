"""Pure Fun card definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenos.cards.base import CardBehavior, CardDefinition, CardInstance

if TYPE_CHECKING:
    from thenos.game import Game
    from thenos.models import PlayerState


class NewfangledTresFuteBehavior(CardBehavior):
    """Score one Fun for each distinct tag played by this player today."""

    def fun_value(
        self,
        game: Game,
        player: PlayerState,
        card: CardInstance,
    ) -> int:
        unique_tags = {
            tag
            for played_card in player.played_today
            for tag in played_card.tags
        }
        return card.effective_base_fun + len(unique_tags)


TRES_FUTE = CardDefinition(
    slug="tres-fute",
    title="Tres Fute",
    tags=frozenset({"Board Game"}),
    cost=0,
    base_fun=1,
)

NEWFANGLED_TRES_FUTE = CardDefinition(
    slug="newfangled-tres-fute",
    title="Newfangled Tres Fute",
    tags=frozenset({"Board Game"}),
    cost=4,
    base_fun=1,
    behavior=NewfangledTresFuteBehavior(),
)

SPLENDOR = CardDefinition(
    slug="splendor",
    title="Splendor",
    tags=frozenset({"Board Game"}),
    cost=1,
    base_fun=2,
)

SEVEN_WONDERS = CardDefinition(
    slug="7-wonders",
    title="7 Wonders",
    tags=frozenset({"Board Game"}),
    cost=2,
    base_fun=3,
)

DUMBED_DOWN_SETTLERS = CardDefinition(
    slug="dumbed-down-settlers",
    title="Dumbed Down Settlers",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=4,
)

AZUL = CardDefinition(
    slug="azul",
    title="Azul",
    tags=frozenset({"Board Game"}),
    cost=3,
    base_fun=4,
)

AGRICOLA = CardDefinition(
    slug="agricola",
    title="Agricola",
    tags=frozenset({"Board Game"}),
    cost=5,
    base_fun=6,
)

TERRA_MYSTICA = CardDefinition(
    slug="terra-mystica",
    title="Terra Mystica",
    tags=frozenset({"Board Game"}),
    cost=6,
    base_fun=7,
)

RISK = CardDefinition(
    slug="risk",
    title="Risk",
    tags=frozenset({"Board Game"}),
    cost=7,
    base_fun=8,
)

CIVILIZATION_THE_BOARD_GAME = CardDefinition(
    slug="civilization-the-board-game",
    title="Civilization the Board Game",
    tags=frozenset({"Board Game", "Indoors"}),
    cost=10,
    base_fun=14,
)

KNEEBOARD = CardDefinition(
    slug="kneeboard",
    title="Kneeboard",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=4,
    base_fun=5,
)

SPIKEBALL = CardDefinition(
    slug="spikeball",
    title="Spikeball",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=6,
    base_fun=7,
)

SCREAMER_BATTLE = CardDefinition(
    slug="screamer-battle",
    title="Screamer Battle",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=7,
    base_fun=8,
)

ADVENTURE_RACE = CardDefinition(
    slug="adventure-race",
    title="Adventure Race",
    tags=frozenset({"Exercise", "Outdoors"}),
    cost=12,
    base_fun=20,
)

CHEAP_WHITE = CardDefinition(
    slug="cheap-white",
    title="Cheap White",
    tags=frozenset({"Food"}),
    cost=2,
    base_fun=3,
)

CHALK_ART = CardDefinition(
    slug="chalk-art",
    title="Chalk Art",
    tags=frozenset({"Relax"}),
    cost=0,
    base_fun=1,
)

THRILLER_BOOK = CardDefinition(
    slug="thriller-book",
    title="Thriller Book",
    tags=frozenset({"Relax"}),
    cost=2,
    base_fun=3,
)

DOCK_FISHING = CardDefinition(
    slug="dock-fishing",
    title="Dock Fishing",
    tags=frozenset({"Relax", "Outdoors"}),
    cost=1,
    base_fun=2,
)


PURE_FUN_CARDS = (
    NEWFANGLED_TRES_FUTE,
    TRES_FUTE,
    SPLENDOR,
    SEVEN_WONDERS,
    DUMBED_DOWN_SETTLERS,
    AZUL,
    AGRICOLA,
    TERRA_MYSTICA,
    RISK,
    CIVILIZATION_THE_BOARD_GAME,
    KNEEBOARD,
    SPIKEBALL,
    SCREAMER_BATTLE,
    ADVENTURE_RACE,
    CHEAP_WHITE,
    CHALK_ART,
    THRILLER_BOOK,
    DOCK_FISHING,
)
