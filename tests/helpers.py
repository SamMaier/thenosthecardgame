from __future__ import annotations

import random

from thenos.ai import RandomAI
from thenos.game import Game, PLAYER_COUNT


def empty_game() -> Game:
    rng = random.Random(0)
    return Game([], [RandomAI(rng) for _ in range(PLAYER_COUNT)], rng)

