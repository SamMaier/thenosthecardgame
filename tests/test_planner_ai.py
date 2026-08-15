import random
import unittest

from thenos.ais import PlannerAI, RandomAI
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.exercise import ZUMBA
from thenos.cards.fun_effects import WORK_CALL
from thenos.game import DAYS_PER_GAME, Game, PLAYER_COUNT
from thenos.simulation import simulate_planner_vs_greedy


def card(
    instance_id: int,
    title: str,
    *,
    cost: int = 0,
    fun: int = 0,
) -> CardInstance:
    return CardInstance(
        instance_id,
        CardDefinition(
            slug=title.lower().replace(" ", "-"),
            title=title,
            tags=frozenset(),
            cost=cost,
            base_fun=fun,
            behavior=CardBehavior(),
        ),
    )


def planner_game(seed: int = 7) -> Game:
    game_rng = random.Random(seed)
    return Game(
        [],
        [
            PlannerAI(random.Random(seed + 1)),
            *(RandomAI(random.Random(seed + 2 + index)) for index in range(3)),
        ],
        game_rng,
    )


class PlannerAITests(unittest.TestCase):
    def test_plans_setup_card_before_later_payoff(self) -> None:
        game = planner_game()
        player = game.players[0]
        player.energy = 7
        player.hand = [
            CardInstance(1, WORK_CALL),
            card(2, "Five Fun", cost=5, fun=5),
        ]

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1))

        self.assertEqual(choice, 0)
        self.assertEqual(player.energy, 7)
        self.assertEqual(len(player.hand), 2)

    def test_values_tomorrow_setup_before_final_day(self) -> None:
        game = planner_game()
        player = game.players[0]
        player.energy = 4
        player.hand = [
            CardInstance(1, ZUMBA),
            card(2, "Three Fun", cost=4, fun=3),
        ]

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1))

        self.assertEqual(choice, 0)

    def test_ignores_tomorrow_setup_on_final_day(self) -> None:
        game = planner_game()
        game.day = DAYS_PER_GAME
        player = game.players[0]
        player.energy = 4
        player.hand = [
            CardInstance(1, ZUMBA),
            card(2, "Three Fun", cost=4, fun=3),
        ]

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1))

        self.assertEqual(choice, 1)

    def test_seeded_choice_is_reproducible_and_does_not_mutate_game(self) -> None:
        games = [planner_game(19), planner_game(19)]
        for game in games:
            game.players[0].energy = 2
            game.players[0].hand = [
                card(1, "Tie A", cost=1, fun=1),
                card(2, "Tie B", cost=1, fun=1),
            ]

        choices = [game.ais[0].choose_card_to_play(game, 0, (0, 1)) for game in games]

        self.assertEqual(choices[0], choices[1])
        for game in games:
            self.assertEqual(game.players[0].energy, 2)
            self.assertEqual([card.title for card in game.players[0].hand], ["Tie A", "Tie B"])

    def test_whole_game_competition_smoke(self) -> None:
        report = simulate_planner_vs_greedy(1, seed=20260814)

        self.assertEqual(report.ais["Planner"].games, 1)
        self.assertEqual(report.ais["Greedy"].games, PLAYER_COUNT - 1)
        self.assertAlmostEqual(
            report.ais["Planner"].win_credit + report.ais["Greedy"].win_credit,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
