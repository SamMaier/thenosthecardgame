# The Nos card-game simulator

A headless Python rules engine for AI balance testing. The default deck contains
one copy of every implemented card.

Run the tests:

```console
python -m unittest discover -v
```

Run a reproducible batch simulation:

```console
python -m thenos 1000 --seed 1 --output results/random-1000-seed-1.csv
```

Run the standard four-Genius card-data simulation. Four-Genius mode requires an
output path so a long run cannot finish without persisting its report:

```console
python -m thenos 4096 --seed 20260817 --four-genius --workers 16 --output results/four-genius-4096-seed-20260817.csv
```

Run the standard seat-balanced AI benchmark:

```console
python -m thenos 1000 --seed 1 --greedy-vs-random --workers 16
```

Games are distributed across worker processes so CPU-bound AI lookahead can use
all sixteen cores. Seeded output is identical for one or multiple workers.

The default card table reports free pick rate, win rate after acquiring a card,
and the difference between average final Fun with and without that card. Free
pick rate only includes unconstrained Suitcase choices; card-effect bulk picks
do not count as free choices. Matchup mode reports each named AI's average score
and fractional win rate. See `AGENTS.md` for the competitive-AI development
workflow.
