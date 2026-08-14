# The Nos Simulator: Adding a Card

This repository is a headless Python simulator for four pre-programmed AIs. The
source of truth for card wording is `cards.csv`; `rules.md` is the source of
truth for general rules. Do not infer an ambiguous interaction. Ask the user
before encoding behavior that is not settled by those files.

Only Biography, Fajitas, and Waterski are currently implemented. The default
deck contains 50 copies of each registered card.

## Project map

- `thenos/cards/base.py`: `CardDefinition`, per-copy `CardInstance`, and all
  behavior hooks.
- `thenos/cards/basic.py`: current card definitions and behavior classes.
  Create more topic files as the catalog grows; do not put card-name checks in
  the game engine.
- `thenos/cards/catalog.py`: the central `CARD_REGISTRY` and deck factory.
- `thenos/game.py`: generic zones, turn order, drawing, play, scoring, discard,
  reshuffle, and Tomorrow lifecycle.
- `thenos/ai.py`: the AI protocol and baseline random AI.
- `thenos/simulation.py`: batch execution and balance statistics.
- `tests/`: card unit tests and whole-game tests.

## Workflow for implementing one card

1. Read the exact row in `cards.csv`, then read every relevant general rule in
   `rules.md`. Search existing card implementations for a reusable behavior.
2. List any unclear timing, targeting, copying, visibility, scoring, or AI
   decision. Ask the user instead of choosing an interpretation.
3. Add a `CardDefinition` with a lowercase hyphenated slug, exact display title,
   exact tags, printed Energy cost, and printed base Fun. Use a dedicated
   `CardBehavior` subclass for rules text.
4. Register the definition in `thenos/cards/catalog.py`. Registration
   automatically adds 50 copies to the default deck.
5. Add a focused `tests/test_<slug>.py`. Test printed cost, printed Fun, tags,
   restrictions, immediate effects, left-to-right interactions, and boundary
   cases that are present in the card text.
6. Run `python -m unittest discover -v`, then run a seeded batch with
   `python -m thenos 1000 --seed 1`. A change is incomplete until both pass.

## Behavior hooks

Behaviors are stateless. Store choices, targets, and energy-cube markers in the
`markers` dictionary of the affected `CardInstance`.

- `can_play`: restrictions checked before cost/payment.
- `modify_energy_cost`: a visible source modifies the current card's running
  Energy cost. The engine invokes sources left to right and clamps the final
  cost to zero.
- `on_play`: immediate effects after payment and after the card becomes visible.
- `on_start_day`: the action of an active Tomorrow card.
- `fun_value`: the target card's starting Fun value.
- `modify_fun`: a visible source modifies one target's running Fun value. The
  engine invokes visible sources left to right.
- `has_tomorrow_action`: set to `True` to move a newly played card into the
  Tomorrow zone at cleanup. Active Tomorrow cards are ordered before today's
  cards and are discarded at that day's cleanup.

When a new mechanic cannot be expressed by these hooks, add a generic hook or
engine concept named for the rule timing, not for a specific card.

## Engine invariants

- Exactly four players, six days, seven starting cards, seven Energy per day,
  three Suitcase picks per player per day, and four visible Suitcase slots.
- Taking a Suitcase card refills that same spot immediately.
- Use `Game.unpack` for Unpack actions and `Game.discard_card` for individual
  discards. The latter clears per-copy markers and Tomorrow state before the
  card can be recycled.
- Card effects that put a card into a hand must call `Game.give_card` so
  acquisition, play-rate, and win-rate statistics retain correct denominators.
- The baseline AI never Unpacks and never voluntarily goes to bed. It uniformly
  chooses a Suitcase card, and uniformly chooses among legal affordable cards
  until no such card remains.
- Energy cost cannot be negative. Hand size is unlimited.
- Only visible cards affect cost and scoring. Visibility order is active
  Tomorrow cards first, then cards played today from left to right.
- Played cards are discarded after scoring unless they move to Tomorrow. The
  discard pile is shuffled into a new Trunk when needed.
- Tied winners divide one win equally. Reports distinguish win rate for
  Suitcase-picked copies from win rate for all acquired copies (including the
  opening hand), and credit each copy with its owner's fractional win.
- Daily Conditions are not modeled. Although `rules.md` still mentions revealing
  one, the current card data provides no Daily Condition deck or supported
  interaction. Ask before adding that system.
