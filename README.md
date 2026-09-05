# The Nos card-game simulator

A headless Python rules engine for AI balance testing. The default deck contains
one copy of every ordinary card; Read the Radar is included only with Daily Conditions.

Run the tests:

```console
python -m unittest discover -v
```

Run a reproducible batch simulation:

```console
python -m thenos 1000 --seed 1 --output results/random-1000-seed-1.csv
```

Run the standard four-Galaxybrain card-data simulation. Four-Galaxybrain mode
requires an output path so a long run cannot finish without persisting its
report:

```console
python -m thenos 4096 --seed 20260817 --four-galaxybrain --workers 16 --output results/four-galaxybrain-4096-seed-20260817.csv
```

Run the standard seat-balanced AI benchmark:

```console
python -m thenos 1000 --seed 1 --greedy-vs-random --workers 16
```

`GalaxybrainAI` is the preferred competitive built-in policy and is available
from `thenos.ais`. To compare it explicitly with deprecated Megamind, run the
fixed speed and two-versus-two acceptance benchmark with:

```console
python -m scripts.benchmark_galaxybrain --workers 16
```

Games are distributed across worker processes so CPU-bound AI lookahead can use
all sixteen cores. Seeded output is identical for one or multiple workers.

The older `MegamindAI` and `GeniusAI` policies are deprecated and are not used
by standard simulation modes. They remain available for explicit comparisons
via `from thenos.ais.megamind import MegamindAI` and
`from thenos.ais.genius import GeniusAI`; pass either as a `Competitor` factory
to `simulate_games` when needed.

The default card table reports free pick rate, win rate after acquiring a card,
and the difference between average final Fun with and without that card. Free
pick rate only includes unconstrained Suitcase choices; card-effect bulk picks
do not count as free choices. Matchup mode reports each named AI's average score
and fractional win rate. See `AGENTS.md` for the competitive-AI development
workflow.

## Optional Daily Conditions

Daily Conditions are disabled by default. Add `--daily-conditions` to any
simulation mode to enable them, for example:

```powershell
wsl.exe python3 -m thenos 8 --seed 20260904 --workers 16 --four-galaxybrain --daily-conditions --output results/daily-conditions-smoke.csv
```

The separate eight-card deck is shuffled once per game. One condition is revealed
before drawing each day, without replacement, and applies to all players for that
day before ordinary card modifiers (including Tomorrow modifiers). Conditions
never enter the Trunk, Suitcase, hands, or acquisition statistics.

Read the Radar is included in the Trunk only with this option. It privately orders
the next three conditions, or all remaining conditions if fewer than three remain.
AIs retain their own known order; another player's private reorder invalidates
knowledge of the affected positions. Future planning uses known upcoming conditions
and otherwise a neutral estimate, without reading the hidden condition order.
Potato Pancakes and Poker are ordinary cards available in both modes.

The Python API accepts `daily_conditions=True` in `Game.default`,
`create_default_deck`, `Game`, and all `simulate_*` entry points. CSV metadata
records `daily_conditions`; reports retain a zero-observation Read the Radar row
when it is disabled. The benchmark script also accepts `--daily-conditions`.

Enabled simulations also print a Daily Conditions table. `Avg Fun` is the average
net Fun gained per player during a whole day with that condition, including
start-of-day, drawing, immediate play, and end-of-day effects. `Difference` is
that average minus the overall average across all player-days in the same run.
For example, `-0.75` means players scored 0.75 less Fun per day than the run's
overall daily average. This is an observed association, not a causal estimate;
day number, hands, Tomorrow effects, and Read the Radar can influence it.

With `--output results/run.csv`, the condition table is automatically saved to
`results/run.daily_conditions.csv` before the console tables print. It includes
all eight conditions, occurrence counts, player-day counts, Fun totals, and the
baseline's counts and totals. Unobserved averages are blank in CSV and `n/a` in
the console. Programmatic consumers can use `report.condition_rows()`; calling
`write_report_csv` saves both reports. Runs without `--daily-conditions` do not
collect or print condition statistics.
