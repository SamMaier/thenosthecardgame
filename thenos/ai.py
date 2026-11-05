"""Backward-compatible imports for the standard AI policies.

New code should import policies from :mod:`thenos.ais`. Deprecated policies
remain available only from their individual modules.
"""

from thenos.ais import (
    ArgonBrain,
    ArgonbrainAI,
    GalaxybrainAI,
    GreedyAI,
    PlannerAI,
    PlayerAI,
    RandomAI,
)

__all__ = [
    "ArgonBrain",
    "ArgonbrainAI",
    "GalaxybrainAI",
    "GreedyAI",
    "PlannerAI",
    "PlayerAI",
    "RandomAI",
]
