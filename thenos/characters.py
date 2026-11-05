"""Optional characters assigned to players for an entire game."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Character:
    title: str
    effect: str
    fun_tag: str = ""
    fun_delta: int = 0
    zero_fun_tag: str = ""
    daily_energy_delta: int = 0
    daily_fun_delta: int = 0
    starting_hand_delta: int = 0
    energy_tag: str = ""
    energy_delta: int = 0

    def modify_fun(self, tags: frozenset[str], value: int) -> int:
        if self.fun_tag in tags:
            value += self.fun_delta
        return value

    def finalize_fun(self, tags: frozenset[str], value: int) -> int:
        """Apply absolute scoring restrictions after ordinary modifiers."""
        return 0 if self.zero_fun_tag in tags else value

    def modify_energy_cost(self, tags: frozenset[str], value: int) -> int:
        if self.energy_tag in tags:
            value += self.energy_delta
        return value


CHARACTERS = (
    Character("Foodie", "+1 Fun from all Food", "Food", 1),
    Character("Gamer", "+1 Fun from all Board Games", "Board Game", 1),
    Character(
        "Old Fogey",
        "+2 Fun -1 Energy every day",
        daily_energy_delta=-1,
        daily_fun_delta=2,
    ),
    Character(
        "Child",
        "+1 Energy daily; All Food cards score 0 Fun.",
        zero_fun_tag="Food",
        daily_energy_delta=1,
    ),
    Character("Athlete", "+1 Fun from all Exercise cards.", "Exercise", 1),
    Character("Extrovert", "+1 Fun from all Social cards.", "Social", 1),
    Character("Introvert", "+1 Fun from all Relax cards.", "Relax", 1),
    Character(
        "Planner",
        "Start the game with 6 cards",
        starting_hand_delta=3,
    ),
    Character(
        "Gear Guy",
        "Item cards cost -1 Energy",
        energy_tag="Item",
        energy_delta=-1,
    ),
)
