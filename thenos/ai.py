"""Backward-compatible imports for the AI package.

New code should import policies from :mod:`thenos.ais` or their individual
modules. Existing card tests and downstream users can continue importing here.
"""

from thenos.ais import (
    GreedyAI,
    MegamindAI,
    PlannerAI,
    PlayerAI,
    RandomAI,
)

__all__ = [
    "GreedyAI",
    "MegamindAI",
    "PlannerAI",
    "PlayerAI",
    "RandomAI",
]
