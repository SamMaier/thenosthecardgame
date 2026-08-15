# The Nos card-game simulator

A headless Python rules engine for AI balance testing. The default deck contains
one copy of every implemented card.

Run the tests:

```console
python -m unittest discover -v
```

Run a reproducible batch simulation:

```console
python -m thenos 1000 --seed 1
```

Run the standard seat-balanced AI benchmark:

```console
python -m thenos 1000 --seed 1 --greedy-vs-random --workers 8
```

Games are distributed across worker processes so CPU-bound AI lookahead can use
all eight cores. Seeded output is identical for one or multiple workers.

The report separates Suitcase pick rate from acquisition-based play and win
rates, so cards dealt in opening hands are represented correctly. Matchup mode
also reports each named AI's average score and fractional win rate. See
`AGENTS.md` for the competitive-AI development workflow.
