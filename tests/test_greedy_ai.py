import random
import unittest

from thenos.ais import GreedyAI, RandomAI
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.food import DECAF
from thenos.game import Game, PLAYER_COUNT


class ImmediateFunBehavior(CardBehavior):
    def on_play(self, game, player, card):
        player.fun += 5


def card(
    instance_id: int,
    title: str,
    *,
    cost: int = 0,
    fun: int = 0,
    behavior: CardBehavior | None = None,
) -> CardInstance:
    return CardInstance(
        instance_id,
        CardDefinition(
            slug=title.lower().replace(" ", "-"),
            title=title,
            tags=frozenset(),
            cost=cost,
            base_fun=fun,
            behavior=behavior or CardBehavior(),
        ),
    )


def greedy_game() -> Game:
    rng = random.Random(7)
    return Game(
        [],
        [GreedyAI(rng), *(RandomAI(rng) for _ in range(PLAYER_COUNT - 1))],
        rng,
    )


class GreedyAITests(unittest.TestCase):
    def test_play_choice_uses_resolved_score_not_only_printed_fun(self) -> None:
        game = greedy_game()
        player = game.players[0]
        player.energy = 7
        player.hand = [
            card(1, "Printed Four", fun=4),
            card(2, "Immediate Five", behavior=ImmediateFunBehavior()),
        ]

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1))

        self.assertEqual(choice, 1)
        self.assertEqual(player.fun, 0)
        self.assertEqual(len(player.hand), 2)

    def test_extra_play_stops_when_spending_energy_reduces_today_score(self) -> None:
        game = greedy_game()
        player = game.players[0]
        player.energy = 1
        player.played_today = [CardInstance(1, DECAF)]
        player.hand = [card(2, "No Fun", cost=1)]

        choice = game.ais[0].choose_extra_card_to_play(game, 0, (0,))

        self.assertIsNone(choice)

    def test_goes_to_bed_to_preserve_decaf_energy(self) -> None:
        game = greedy_game()
        player = game.players[0]
        player.energy = 4
        player.played_today = [CardInstance(1, DECAF)]
        player.hand = [card(2, "Three Fun", cost=2, fun=3)]

        goes_to_bed = game.ais[0].choose_to_go_to_bed(game, 0, (0,))

        self.assertTrue(goes_to_bed)
        self.assertEqual(player.energy, 4)
        self.assertEqual(len(player.hand), 1)

    def test_continuing_choice_is_cached_for_the_immediate_play(self) -> None:
        game = greedy_game()
        player = game.players[0]
        player.energy = 1
        player.hand = [card(1, "One Fun", fun=1)]

        goes_to_bed = game.ais[0].choose_to_go_to_bed(game, 0, (0,))
        choice = game.ais[0].choose_card_to_play(game, 0, (0,))

        self.assertFalse(goes_to_bed)
        self.assertEqual(choice, 0)

    def test_suitcase_choice_prefers_card_with_best_today_score(self) -> None:
        game = greedy_game()
        player = game.players[0]
        player.energy = 7
        game.suitcase = [
            card(1, "One", fun=1),
            card(2, "Five", fun=5),
        ]
        game.trunk = [card(3, "Refill A"), card(4, "Refill B")]

        choice = game.ais[0].choose_suitcase_card(
            game, 0, tuple(game.suitcase)
        )

        self.assertEqual(choice, 1)
        self.assertEqual(
            [candidate.title for candidate in game.suitcase],
            ["One", "Five"],
        )


if __name__ == "__main__":
    unittest.main()
