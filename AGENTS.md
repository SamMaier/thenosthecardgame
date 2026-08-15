# The Nos Simulator: Building Competitive AIs

This repository is a headless Python simulator for four AI players. The card
catalog and rules engine are complete enough to serve as the competition
environment; current development should prioritize player policies, fair
matchups, reproducible experiments, and useful strategy measurements.

`cards.csv` is the source of truth for card wording and `rules.md` is the source
of truth for general rules. An AI may use any information exposed by the game
state, but it must not inspect hidden card order or another AI's private state.

## Project map

- `thenos/ais/interface.py`: the stable `PlayerAI` protocol. Every AI must
  implement all decisions in this interface.
- `thenos/ais/random_ai.py`: the baseline random policy.
- `thenos/ais/greedy.py`: one-step, end-of-day score maximization.
- `thenos/ai.py`: compatibility imports for older callers; put no new AI logic
  here.
- `thenos/simulation.py`: named competitors, seat rotation, batch execution,
  and card/AI statistics.
- `thenos/game.py`: rules, zones, turn order, delegated decisions, and scoring.
- `thenos/cards/`: card definitions and behavior hooks an AI may need to model.
- `tests/test_greedy_ai.py`: focused policy-decision examples.
- `tests/test_simulation.py`: batch and matchup reporting tests.

## Workflow for adding an AI

1. State the policy's objective and information horizon. Be explicit about
   whether it is myopic, searches future plays, models opponents, or values
   cards for later days.
2. Create one module under `thenos/ais/`, named after the strategy. Implement
   `PlayerAI` directly or subclass a policy only when inherited fallback
   choices are intentional and documented.
3. Keep strategy in the AI. Do not add card-name checks or policy-specific
   branches to `Game`; add a generic observation or decision interface only
   when the existing protocol cannot express the strategy.
4. Use only observable state. In particular, do not read the order of
   `game.trunk`, random-generator state, or opponents' hands to gain an unfair
   advantage. Simulation code may control seeds but policies may not exploit
   them.
5. Make stochastic policies accept an injected `random.Random`. Break equally
   valued choices without positional bias, and ensure a seeded matchup is
   reproducible. Keep policy RNGs independent from the game RNG so AI tie
   breaking cannot change future deck shuffles.
6. Add focused `tests/test_<strategy>_ai.py` cases for its distinctive choices,
   tie handling, non-mutation during lookahead, and any stopping rule. Also add
   a whole-game smoke test through the competition runner.
7. Benchmark against named baselines with seat rotation. Report game count,
   seed, fractional win rate, and average score; do not present a tiny batch as
   a stable strength estimate.
8. Run `python -m unittest discover -v`, then run at least one seeded default
   batch and the relevant seeded matchup. A policy change is incomplete until
   all three complete successfully.

### Windows Python command

The project requires Python 3.11 or newer. On Windows, if the native
`python` command is unavailable or `py` resolves to an older interpreter, run
the checks through the persistent WSL installation instead:

```powershell
wsl.exe python3 -m unittest discover -v
wsl.exe python3 -m thenos 200 --seed 1 --workers 8
wsl.exe python3 -m thenos 200 --seed 1 --genius-vs-planner --workers 8
```

Do not use a Python 3.8 interpreter for this repository; the engine relies on
Python 3.11 features such as slotted dataclasses. Keep the WSL prefix on both
test and simulation commands so results are reproducible across Codex chats.

## AI-versus-AI evaluation workflow

Every new policy should be evaluated as a named competitor against the policies
it is intended to beat. Use the competition runner rather than calling
`Game.run()` in an ad-hoc loop:

```python
from thenos.ais import GeniusAI, PlannerAI
from thenos.simulation import Competitor, simulate_games

report = simulate_games(
    200,
    seed=20260816,
    competitors=(
        Competitor("Genius", GeniusAI),
        Competitor("Planner", PlannerAI),
        Competitor("Planner", PlannerAI),
        Competitor("Planner", PlannerAI),
    ),
    rotate_seats=True,
    workers=8,
)
```

Use the dedicated CLI matchup when one exists, for example:
`python -m thenos 200 --seed 20260816 --genius-vs-planner --workers 8`.
`workers` uses independent processes through `ProcessPoolExecutor`; it is not
threaded execution. Keep the game count divisible by four so every competitor
gets every seat equally often.

For each matchup:

- Use distinct seeds for tuning and final evaluation. Record the exact seed,
  game count, competitor composition, seat rotation, and worker count.
- Report both each named AI's average score and fractional win rate. In a
  one-versus-three matchup, the three repeated opponents are aggregated, so
  their displayed win rate is per opponent appearance; also state the combined
  opponent win credit when it clarifies the result.
- Treat fractional wins as one win split evenly among tied winners. Do not
  count a tie as a full win for every tied AI.
- Use at least 100--200 games for a headline comparison when runtime permits;
  label smaller smoke batches as preliminary and do not call them stable
  strength estimates.
- Keep tuning batches separate from the final seed. A policy that reaches a
  target on one convenient seed must be checked on a fresh seat-balanced batch.
- Compare against relevant baselines and against copies of the strongest
  available policy, not only against RandomAI. A policy's objective is its
  matchup win rate, but average score and card statistics help diagnose why it
  wins or loses.

## Competition conventions

- Exactly four AIs play each game. Use `Competitor` entries and
  `simulate_games(..., competitors=..., rotate_seats=True)` for general
  experiments.
- Repeated competitor names are aggregated. This is useful for one policy
  against three copies of a baseline.
- `simulate_greedy_vs_random(games, seed)` is the standard first benchmark and
  rotates seats automatically. Pass `workers=8` to use eight CPU cores; this is
  process-based because Python threads do not speed up CPU-bound game search.
- Win rate uses fractional wins: tied winners divide one win equally. A
  four-player random baseline should approach 25% per seat over a large sample.
- Rotate seats unless measuring a deliberate seat effect. Multiples of four
  games give every policy equal exposure to each starting seat.
- Compare average score alongside win rate. Record the exact seed and number of
  games so results can be reproduced.
- Keep training/tuning games separate from the final evaluation seeds when a
  policy has learned parameters.

## AI interface and evaluation notes

The engine delegates Suitcase picks, plays, targets, discards, optional actions,
Energy spending, and Trunk ordering through `PlayerAI`. Engine methods validate
all returned indices and choices. New optional game decisions must be added to
the protocol and every built-in policy, with validation in `Game`.

Lookahead must not mutate the live game. A policy can evaluate copied state, a
purpose-built immutable observation, or a tested reversible simulation. Bound
search depth and branching explicitly so batch simulations remain practical.
Document approximations for decisions whose context is not fully represented by
the generic interface.

Only visible cards affect Energy cost and scoring. Visibility order is active
Tomorrow cards first, then today's cards from left to right. Use
`Game.energy_cost`, `Game.playable_hand_indices`, and `Game.card_fun` rather than
duplicating those rules in an AI. Card effects that add cards, alter markers,
or score immediately are reasons to prefer engine-backed evaluation over raw
printed Fun.

## Engine invariants to preserve

- Four players, six days, seven starting cards, seven Energy per day, three
  Suitcase selections per player per day, and four visible Suitcase slots.
- Taking a Suitcase card refills that position immediately.
- The random baseline never Unpacks or voluntarily goes to bed, uniformly
  chooses Suitcase cards, and uniformly plays legal affordable cards until none
  remain.
- Energy cost cannot be negative; hand size is unlimited.
- Played cards are discarded after scoring unless they move to Tomorrow. Active
  Tomorrow cards are discarded after their active day.
- Tied winners split one win. Card reports distinguish Suitcase-picked copies
  from all acquired copies.
- Daily Conditions are not modeled. Ask before adding that system.

When rules-engine work is genuinely required for an AI, keep it generic and
retain the card implementation workflow in the existing tests and card modules:
behaviors remain stateless, per-copy state stays in `CardInstance.markers`, and
all rules and card tests must continue to pass.
