"""The optional, separate Daily Condition deck (see daily_conditions.tsv)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DailyCondition:
    title: str
    effect: str
    fun_tag: str = ""
    fun_delta: int = 0
    starting_energy_delta: int = 0
    hot_day: bool = False

    def modify_fun(self, tags: frozenset[str], value: int) -> int:
        return value + (self.fun_delta if self.fun_tag in tags else 0)

    def modify_energy_cost(self, tags: frozenset[str], cost: int) -> int:
        if self.hot_day:
            cost += int("Outdoors" in tags) - int("Indoors" in tags)
        return cost


DAILY_CONDITIONS = (
    DailyCondition("Rainy Day", "Every Outdoors card played today scores -1 Fun.", "Outdoors", -1),
    DailyCondition("Beautiful Day", "Every Outdoors card played today scores +1 Fun.", "Outdoors", 1),
    DailyCondition("Plumbing Issue", "Every Indoors card played today scores -1 Fun.", "Indoors", -1),
    DailyCondition("Sickness Spreading", "Players start with -1 Energy today.", starting_energy_delta=-1),
    DailyCondition("New Arrivals", "Every Social card played today scores +1 Fun.", "Social", 1),
    DailyCondition("Brutally Hot Day", "Every Outdoors card played today costs +1 Energy, and every Indoors card costs -1 Energy.", hot_day=True),
    DailyCondition("Everyone Booked", "Every Social card played today scores -1 Fun.", "Social", -1),
    DailyCondition("Normal day", "No Effect."),
)
