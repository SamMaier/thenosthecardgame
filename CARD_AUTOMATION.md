# Codex card implementation runner

`scripts/implement_cards.py` processes `cards.csv` in its current complexity
order. The first five tiers are one Codex job and one Git commit each. Every
remaining unique card is one job and one commit.

The runner uses GPT-5.6 Luna with high reasoning, filters out cards already in
`CARD_REGISTRY`, and is safe to rerun after successful batches. For each job it:

1. requires a clean, attached Git worktree with commit identity configured;
2. invokes non-interactive Codex with workspace-only write access;
3. verifies every target is registered and has a focused test named from its
   slug, with hyphens changed to underscores;
4. runs the complete `unittest` suite;
5. commits on success, or exits without committing on any failure.

## WSL setup

Install Python and Git in Ubuntu/Debian WSL, then install Codex:

```bash
sudo apt update
sudo apt install -y python3 git curl
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Restart the shell if `codex` is not immediately on `PATH`, then run `codex`
once and complete ChatGPT sign-in.

Configure the Git identity used by automated commits:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Verify before a real run

From the repository root in WSL:

```bash
codex --version
python3 --version
git --version
python3 -B -m unittest discover -v
python3 scripts/implement_cards.py --list
python3 scripts/implement_cards.py --dry-run
```

The actual runner requires a clean tree. Commit the runner, sorted CSV, and any
other intended baseline changes first. Remove or ignore stale editor swap files.

For a cautious first call, process one job:

```bash
python3 scripts/implement_cards.py --max-jobs 1
```

Then process every remaining job until completion or the first failure:

```bash
python3 scripts/implement_cards.py
```

Useful targeted forms:

```bash
python3 scripts/implement_cards.py --batch pure-fun
python3 scripts/implement_cards.py --card "The Crew"
python3 scripts/implement_cards.py --card "The Crew" --instructions clarifications.txt
```

On failure, inspect the uncommitted changes and Codex output. Fix or intentionally
discard those changes before rerunning; the clean-tree guard prevents accidental
mixing with the next card.
