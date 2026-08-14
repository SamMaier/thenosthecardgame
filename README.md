# The Nos card-game simulator

A headless Python rules engine for AI balance testing. The initial scaffold
implements 50 copies each of Biography, Fajitas, and Waterski.

Run the tests:

```console
python -m unittest discover -v
```

Run a reproducible batch simulation:

```console
python -m thenos 1000 --seed 1
```

The report separates Suitcase pick rate from acquisition-based play and win
rates, so cards dealt in opening hands are represented correctly. See
`AGENTS.md` for the required workflow when implementing another card.
