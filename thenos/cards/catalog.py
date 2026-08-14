"""Central registry for every implemented card."""

from __future__ import annotations

from itertools import count

from thenos.cards.base import CardDefinition, CardInstance
from thenos.cards.basic import BIOGRAPHY, FAJITAS, WATERSKI


CARD_REGISTRY: dict[str, CardDefinition] = {
    card.slug: card for card in (BIOGRAPHY, FAJITAS, WATERSKI)
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

