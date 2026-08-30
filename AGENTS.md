# The Nos Simulator: Card Data Simulations

This repository is a headless Python simulator for four AI players. The rules
engine and card catalog primarily support reproducible card-strength studies.
Current work should prioritize trustworthy card metrics, long four-Genius
batches, reproducible seeds, and clear reporting over adding more policies.

`cards.csv` is the source of truth for card wording and `rules.md` is the source
of truth for general rules. Daily Conditions are not modeled; ask before adding
that system.

## Project map

- `thenos/simulation.py`: competitors, process-based batch execution, seat
  rotation, card statistics, and `SimulationReport.rows()`.
- `thenos/game.py` and `thenos/models.py`: rules and the low-level statistics
  hooks for free choices, acquisitions, plays, scores, and wins.
- `thenos/ais/genius.py`: the preferred policy for meaningful card-data runs.
- `thenos/ais/random_ai.py`: a plumbing baseline, not a card-ranking policy.
- `thenos/ais/interface.py`: the stable `PlayerAI` decision protocol.
- `thenos/cards/`: stateless card behaviors; per-copy state belongs in
  `CardInstance.markers`.
- `tests/test_simulation.py` and `tests/test_game.py`: metric semantics,
  reproducibility, process parity, and engine integration.

## The three primary card metrics

Every card report must include these columns. Keep their definitions stable so
results from different seeds and code revisions remain comparable.

### Free pick rate

Measure only unrestricted choices presented through `Game.pick_from_suitcase`.
For each choice, every visible physical Suitcase card receives one offer: the
selected card receives `1.0` and every unselected card receives `0.0`. Aggregate
by title as:

```text
free pick rate = free picks / free pick offers
```

Thus a card always taken by the second player who sees it approaches `0.5`.
Duplicate titles count once per visible physical copy. Bulk, automatic, or
restricted card-effect pickups that call `pick_suitcase_cards` directly are not
free choices and must not enter this metric.

Always retain the offer denominator. A displayed `0.0` can mean either many
rejections or no observations, which are not equivalent.

### Win rate

Use one observation per player-game in which that player had the card added to
their hand by any mechanism, including the starting deal, Suitcase picks, and
card effects. Acquiring multiple copies in the same player-game must not
double-weight the final result. Cards played directly from another zone without
entering the hand are not acquisitions.

Use the engine's fractional win credit: outright wins are `1.0`, losses are
`0.0`, and tied winners divide one win. Report:

```text
win rate = total win credit / player-games with card
```

### Fun added

Compare final scores for player-games that acquired the card with player-games
that did not:

```text
fun added = mean final Fun with card - mean final Fun without card
```

This is an observational association, not a causal card value. Strong AIs may
select cards that fit already-good hands, cards can be acquired together, and
rare-card estimates can be noisy. Preserve both with-card and without-card
sample counts and do not rank cards from tiny denominators.

## Standard four-Genius card-data run

Use four copies of `GeniusAI` for card-strength data. The default
`python -m thenos` command uses four Random AIs; it is useful for fast plumbing
checks, but its free pick rates should converge toward 25% and are not strategic
rankings.

Use the competition runner rather than an ad-hoc `Game.run()` loop:

```python
from thenos.ais import GeniusAI
from thenos.simulation import Competitor, simulate_games

report = simulate_games(
    32,
    seed=20260816,
    competitors=tuple(Competitor("Genius", GeniusAI) for _ in range(4)),
    rotate_seats=True,
    workers=16,
)

print("Card | Free pick rate | Win rate | Fun added")
for row in report.rows():
    print(
        f"{row['card']} | {row['free_pick_rate']:.1%} | "
        f"{row['win_rate']:.1%} | {row['fun_added']:+.2f}"
    )
```

`workers` uses independent processes through `ProcessPoolExecutor`, not
threads. Genius lookahead is expensive: a 32-game, 16-worker run has taken
about nine minutes on the current development machine. A many-hundred-game run
may take hours, so run it only when requested and leave an ample command
timeout.

### Batch sizes and seeds

- Use 8 games for a quick four-Genius wiring smoke test.
- Use 32 games only as a preliminary metric check; do not call it stable.
- Use many hundreds of games for a serious card ranking.
- Keep game counts divisible by four and enable seat rotation.
- Use distinct seeds for tuning, smoke checks, and final evaluation.
- Record the exact seed, game count, competitor composition, seat rotation,
  worker count, code revision, and elapsed time.
- Never select or discard a result because its seed is inconvenient.

## Reporting and interpretation

The main output is one table containing every registered card and all three
primary metrics. For analysis, also retain these denominators and diagnostics:

- free-pick offers and free picks;
- player-games with and without the card;
- total acquisitions and plays;
- average final score and fractional win rate for each named policy.

Repeated competitor names are aggregated. With four Genius entries, the
`Genius` result contains four player appearances per game and should have about
a 25% per-appearance win rate. Tied winners must split one win rather than each
receiving a full win.

When identifying best and worst cards:

- show sample counts alongside rates;
- separate free-choice preference from outcome association;
- flag surprising disagreements between the metrics for investigation;
- avoid causal language for Fun added;
- label smoke results preliminary; and
- compare fresh seeds before making balance recommendations.

## Statistics implementation rules

- Record genuine free-choice offers at the point the AI is presented with the
  full Suitcase, before the selected card is removed or its slot refilled.
- Keep legacy Suitcase pickup counters separate from free-choice counters;
  automatic or multi-card effects still matter diagnostically.
- Route every card that enters a hand through `Game.give_card` so acquisitions
  from all mechanisms are captured consistently.
- Aggregate acquisition outcomes once per card title per player-game, not once
  per physical copy.
- Prepopulate reports from `CARD_REGISTRY` so cards with zero observations still
  appear in the table.
- Preserve exact equality between seeded serial and parallel reports. Policy
  RNGs must remain independent from the game RNG.
- Statistics gathering must not change game decisions, shuffle order, or live
  state observed by an AI.

Metric changes require focused tests for exact numerator and denominator
semantics, duplicate acquisitions, forced versus free picks, tied wins, and Fun
baselines. Also retain a whole-game smoke test and serial-versus-parallel parity
test.

## Required verification

The project requires Python 3.11 or newer. On this Windows workspace, use the
persistent WSL Python so results are reproducible across Codex chats:

```powershell
wsl.exe python3 -m unittest discover -v
wsl.exe python3 -m thenos 32 --seed 20260815 --workers 16
wsl.exe python3 -c "from thenos.ais import GeniusAI; from thenos.simulation import Competitor, simulate_games; r=simulate_games(8, seed=20260816, competitors=tuple(Competitor('Genius', GeniusAI) for _ in range(4)), rotate_seats=True, workers=16); print(len(r.cards), r.ais['Genius'])"
```

Do not use Python 3.8; the engine relies on Python 3.11 features such as slotted
dataclasses. For statistics changes, complete the full tests, a seeded Random
plumbing batch, and a seeded four-Genius smoke batch before considering the
work finished.

## Policy and engine constraints

An AI may use only observable game state. It must not inspect hidden Trunk
order, another player's hand, or RNG state. Lookahead must not mutate the live
game, and stochastic policies must accept an injected `random.Random` without
positional tie bias. Keep policy logic out of `Game`; add only generic,
validated observation or decision interfaces when required.

Preserve these core invariants:

- exactly four players, six days, four starting cards, seven Energy per day,
  three Suitcase selections per player per day, and four Suitcase slots;
- immediate refill of a selected Suitcase slot;
- nonnegative Energy costs and unlimited hand size;
- discard after scoring unless a card moves to Tomorrow, followed by discard
  after its active Tomorrow day; and
- one total win credit split across tied winners.

Only visible cards affect Energy cost and scoring modifiers. Active Tomorrow
cards apply only their Tomorrow text: they do not score printed Fun or trigger
non-Tomorrow effects. Today's cards score from left to right. Use generic
engine helpers such as `Game.energy_cost`, `Game.playable_hand_indices`, and
`Game.card_fun` instead of duplicating rules in policies or reports.
