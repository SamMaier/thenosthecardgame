import copy
import random
import unittest

from thenos.ais import GeniusAI, PlannerAI, RandomAI
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.cards.fun_effects import WORK_CALL
from thenos.game import Game


def card(instance_id: int, title: str, *, cost: int, fun: int) -> CardInstance:
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


def genius_game(seed: int = 11) -> Game:
    return Game(
        [],
        [
            GeniusAI(random.Random(seed)),
            *(PlannerAI(random.Random(seed + index + 1)) for index in range(3)),
        ],
        random.Random(seed + 10),
    )


class GeniusAITests(unittest.TestCase):
    def test_finds_setup_payoff_sequence_without_mutating_game(self) -> None:
        game = genius_game()
        player = game.players[0]
        player.energy = 7
        player.hand = [
            CardInstance(1, WORK_CALL),
            card(2, "Five Fun", cost=5, fun=5),
            card(3, "One Fun", cost=1, fun=1),
        ]
        before_energy = player.energy
        before_hand = tuple(player.hand)

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1, 2))

        self.assertEqual(choice, 0)
        self.assertEqual(player.energy, before_energy)
        self.assertEqual(tuple(player.hand), before_hand)
        self.assertEqual(player.played_today, [])

    def test_seeded_equal_choices_are_reproducible(self) -> None:
        games = [genius_game(31), genius_game(31)]
        for game in games:
            game.players[0].energy = 2
            game.players[0].hand = [
                card(1, "Tie A", cost=1, fun=1),
                card(2, "Tie B", cost=1, fun=1),
            ]

        choices = [
            game.ais[0].choose_card_to_play(game, 0, (0, 1))
            for game in games
        ]

        self.assertEqual(choices[0], choices[1])

    def test_planning_ignores_hidden_order_and_opponent_cards(self) -> None:
        game = Game(
            create_default_deck(),
            [
                GeniusAI(random.Random(71)),
                *(PlannerAI(random.Random(72 + index)) for index in range(3)),
            ],
            random.Random(75),
        )
        game.setup()
        first = copy.deepcopy(game)
        second = copy.deepcopy(game)
        second.trunk.reverse()
        second.players[1].hand[0], second.trunk[0] = (
            second.trunk[0],
            second.players[1].hand[0],
        )

        first_sample = first.ais[0]._planning_copy(first)
        second_sample = second.ais[0]._planning_copy(second)

        self.assertEqual(
            [card.title for card in first_sample.trunk],
            [card.title for card in second_sample.trunk],
        )
        self.assertEqual(
            [
                [card.title for card in player.hand]
                for player in first_sample.players[1:]
            ],
            [
                [card.title for card in player.hand]
                for player in second_sample.players[1:]
            ],
        )

    def test_planning_preserves_acting_hand_with_four_geniuses(self) -> None:
        game = Game(
            create_default_deck(),
            [GeniusAI(random.Random(81 + index)) for index in range(4)],
            random.Random(85),
        )
        game.setup()
        acting_index = 2
        expected_hand_ids = [
            card.instance_id for card in game.players[acting_index].hand
        ]

        sample = game.ais[acting_index]._planning_copy(game)

        self.assertEqual(
            [card.instance_id for card in sample.players[acting_index].hand],
            expected_hand_ids,
        )


if __name__ == "__main__":
    unittest.main()
