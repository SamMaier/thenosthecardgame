"""Central registry for every implemented card."""

from __future__ import annotations

from itertools import count

from thenos.cards.base import CardDefinition, CardInstance
from thenos.cards.copy_effects import LAST_YEARS_SHORTS, WEDDING_ANNIVERSARY
from thenos.cards.basic import (
    BIOGRAPHY,
    CARCASSONNE,
    EPIC_DUELS,
    EPIC_PRANK,
    EUCHRE_TOURNAMENT,
    EUCHRE_TOURNAMENT_AWARDS_CEREMONY,
    FAJITAS,
    FIVE_TEN_FIFTEEN,
    FIT_TO_PRINT,
    MAKE_PLANS,
    PUERTO_RICO,
    SAN_JUAN,
    SCRABBLE,
    SETTLERS_CITIES_AND_KNIGHTS,
    SOLO,
    THE_CREW,
    WINGSPAN,
    WATERSKI,
)
from thenos.cards.events import (
    CHRISTMAS_NAME_DRAW,
    PHOTO_SHOOT,
    SING_SONG,
    STAY_UP_LATE,
)
from thenos.cards.food import FOOD_CARDS
from thenos.cards.items import (
    ASSORTED_CUTLERY,
    BOOBY_PRIZE,
    BOUGIE_COFFEE_MACHINE,
    FANCY_FLOATIE,
    FISHING_BOAT,
    SKI_BOAT,
)
from thenos.cards.exercise import (
    CHUCK_A_FRISBEE,
    KAYAK,
    MORNING_BIKE,
    MORNING_RUN,
    MORNING_WALK,
    NAVY_SEALING,
    PADDLEBOARD,
    PADDLEBOAT,
    PLAY_WITH_THE_KIDS,
    SKI_ON_COUSINS_SHOULDERS,
    SLALOM_START,
    THROW_A_BASEBALL,
    WAKESURF,
    WRESTLE_THE_KIDS,
    ZUMBA,
)
from thenos.cards.energy_effects import ENERGY_EFFECT_CARDS
from thenos.cards.fun_effects import FUN_EFFECT_CARDS
from thenos.cards.pure_energy import PURE_ENERGY_CARDS
from thenos.cards.pure_fun import PURE_FUN_CARDS
from thenos.cards.relax import (
    CHEESY_PHONE_GAME,
    CLASSIC_BOOK,
    COLOURING,
    EARLY_BEDTIME,
    FANCY_CRAFT,
    FISHING_EVENING,
    FISHING_MORNING,
    SUNRISE,
    SLEEP_IN,
    FLOATING,
    PAINT,
    PAINT_ROCKS,
    TANNING,
)
from thenos.cards.social import (
    CAMPFIRE,
    DATES_FIRST_NOZ,
    NEW_NOZ_BOOK_ENTRY,
    TELL_A_STORY,
)


CARD_REGISTRY: dict[str, CardDefinition] = {
    card.slug: card
    for card in (
        BIOGRAPHY,
        CARCASSONNE,
        EPIC_DUELS,
        EPIC_PRANK,
        EUCHRE_TOURNAMENT,
        EUCHRE_TOURNAMENT_AWARDS_CEREMONY,
        FAJITAS,
        FIVE_TEN_FIFTEEN,
        FIT_TO_PRINT,
        MAKE_PLANS,
        PUERTO_RICO,
        SAN_JUAN,
        SCRABBLE,
        NEW_NOZ_BOOK_ENTRY,
        SETTLERS_CITIES_AND_KNIGHTS,
        SOLO,
        THE_CREW,
        WINGSPAN,
        WATERSKI,
        KAYAK,
        WEDDING_ANNIVERSARY,
        LAST_YEARS_SHORTS,
        SING_SONG,
        CHRISTMAS_NAME_DRAW,
        STAY_UP_LATE,
        PHOTO_SHOOT,
        DATES_FIRST_NOZ,
        TELL_A_STORY,
        CAMPFIRE,
        *FOOD_CARDS,
        BOOBY_PRIZE,
        ASSORTED_CUTLERY,
        BOUGIE_COFFEE_MACHINE,
        FANCY_FLOATIE,
        FISHING_BOAT,
        SKI_BOAT,
        EARLY_BEDTIME,
        CHEESY_PHONE_GAME,
        FANCY_CRAFT,
        CLASSIC_BOOK,
        SUNRISE,
        FISHING_EVENING,
        FISHING_MORNING,
        TANNING,
        SLEEP_IN,
        FLOATING,
        ZUMBA,
        PLAY_WITH_THE_KIDS,
        THROW_A_BASEBALL,
        WAKESURF,
        SLALOM_START,
        SKI_ON_COUSINS_SHOULDERS,
        WRESTLE_THE_KIDS,
        MORNING_BIKE,
        MORNING_RUN,
        MORNING_WALK,
        CHUCK_A_FRISBEE,
        PADDLEBOARD,
        NAVY_SEALING,
        PADDLEBOAT,
        *ENERGY_EFFECT_CARDS,
        *FUN_EFFECT_CARDS,
        *PURE_FUN_CARDS,
        *PURE_ENERGY_CARDS,
        COLOURING,
        PAINT,
        PAINT_ROCKS,
    )
}

_instance_ids = count()


def make_card(slug: str) -> CardInstance:
    """Create one physical copy of a registered card."""
    try:
        definition = CARD_REGISTRY[slug]
    except KeyError as error:
        raise KeyError(f"Card is not implemented: {slug}") from error
    return CardInstance(next(_instance_ids), definition)


def create_default_deck(copies_per_card: int = 50) -> list[CardInstance]:
    if copies_per_card < 1:
        raise ValueError("copies_per_card must be positive")
    return [
        make_card(slug)
        for slug in CARD_REGISTRY
        for _ in range(copies_per_card)
    ]
