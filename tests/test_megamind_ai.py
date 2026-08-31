import random
import unittest

from thenos.ais import MegamindAI, PlannerAI, RandomAI
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.cards.food import DECAF
from thenos.cards.fun_effects import WORK_CALL
from thenos.game import Game


def card(instance_id: int, title: str, cost: int, fun: int) -> CardInstance:
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


def megamind_game(seed: int = 11) -> Game:
    return Game(
        [],
        [
            MegamindAI(random.Random(seed)),
            *(PlannerAI(random.Random(seed + index + 1)) for index in range(3)),
        ],
        random.Random(seed + 10),
    )


class MegamindAITests(unittest.TestCase):
    def test_state_value_uses_one_game_copy_for_simple_state(self):
        class CopyCounter:
            copies = 0

            def __deepcopy__(self, memo):
                type(self).copies += 1
                copied = type(self)()
                memo[id(self)] = copied
                return copied

        game = megamind_game()
        game.day = 1
        physical_card = card(1, "Simple", 1, 2)
        physical_card.markers["copy_counter"] = CopyCounter()
        game.players[0].hand = [physical_card]

        game.ais[0]._state_value(game, 0)

        self.assertEqual(CopyCounter.copies, 1)

    def test_uses_generic_card_values_without_title_specific_priors(self):
        megamind = MegamindAI(random.Random(1))
        planner = PlannerAI(random.Random(1))

        for physical_card in create_default_deck():
            with self.subTest(card=physical_card.title):
                self.assertEqual(
                    megamind._card_value(physical_card),
                    planner._card_value(physical_card),
                )

    def test_goes_to_bed_to_preserve_end_day_energy_value(self):
        game = megamind_game()
        game.day = 6
        player = game.players[0]
        player.energy = 4
        player.played_today = [CardInstance(1, DECAF)]
        player.hand = [card(2, "Three Fun", 2, 3)]

        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))
        self.assertEqual(player.energy, 4)
        self.assertEqual(len(player.hand), 1)

    def test_uses_energy_gain_to_unlock_same_day_high_cost_play(self):
        game = megamind_game()
        megamind = game.ais[0]
        player = game.players[0]
        player.energy = 6
        player.hand = [
            make_card("weird-chip-flavor"),
            card(2, "Big Day", 8, 8),
        ]

        self.assertFalse(megamind.choose_to_go_to_bed(game, 0, (0,)))
        self.assertEqual(megamind.choose_card_to_play(game, 0, (0,)), 0)

    def test_does_not_value_unaffordable_card_as_future_play(self):
        game = megamind_game()
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("adventure-race")]

        self.assertEqual(game.playable_hand_indices(0), [])
        self.assertEqual(game.ais[0]._future_hand_value(game, 0), 0.0)

    def test_saves_energy_enabler_without_follow_up_play(self):
        game = megamind_game()
        game.players[0].energy = 0
        game.players[0].hand = [make_card("weird-chip-flavor")]

        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

    def test_ignores_tomorrow_setup_after_final_day(self):
        game = megamind_game()
        game.day = 6
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("dates-first-nos")]
        game.suitcase = [
            card(100 + index, f"Suitcase {index}", 2, 2)
            for index in range(4)
        ]
        game.trunk = [
            card(200 + index, f"Trunk {index}", 2, 2)
            for index in range(20)
        ]

        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

    def test_finds_setup_payoff_sequence_without_mutating_game(self):
        game = megamind_game()
        player = game.players[0]
        player.energy = 7
        player.hand = [
            CardInstance(1, WORK_CALL),
            card(2, "Five Fun", 5, 5),
            card(3, "One Fun", 1, 1),
        ]
        before_hand = tuple(player.hand)

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1, 2))

        self.assertEqual(choice, 0)
        self.assertEqual(player.energy, 7)
        self.assertEqual(tuple(player.hand), before_hand)
        self.assertEqual(player.played_today, [])

    def test_does_not_apply_one_shot_effect_to_every_remaining_card(self):
        game = megamind_game()
        player = game.players[0]
        player.energy = 6
        player.hand = [
            CardInstance(1, WORK_CALL),
            *(
                card(2 + index, f"Ordinary {index}", 1, 2)
                for index in range(4)
            ),
        ]

        choice = game.ais[0].choose_card_to_play(
            game, 0, tuple(range(len(player.hand)))
        )

        self.assertNotEqual(choice, 0)

    def test_planning_preserves_acting_hand_with_four_megaminds(self):
        game = Game(
            create_default_deck(),
            [MegamindAI(random.Random(81 + index)) for index in range(4)],
            random.Random(85),
        )
        game.setup()
        acting_index = 2
        expected = [
            physical.instance_id
            for physical in game.players[acting_index].hand
        ]

        sample = game.ais[acting_index]._planning_copy(game)

        self.assertEqual(
            [
                physical.instance_id
                for physical in sample.players[acting_index].hand
            ],
            expected,
        )

    def test_finds_best_generic_energy_allocation_without_mutating_game(self):
        game = Game(
            [],
            [
                MegamindAI(random.Random(1)),
                *(RandomAI(random.Random(2 + index)) for index in range(3)),
            ],
            random.Random(5),
        )
        player = game.players[0]
        player.energy = 7
        player.hand = [
            card(1, "Large", 7, 6),
            card(2, "Medium A", 3, 4),
            card(3, "Medium B", 4, 4),
        ]

        choice = game.ais[0].choose_card_to_play(game, 0, (0, 1, 2))

        self.assertIn(choice, (1, 2))
        self.assertEqual(player.energy, 7)
        self.assertEqual(len(player.hand), 3)

    def test_seeded_equal_choices_are_reproducible(self):
        choices = []
        for _ in range(2):
            game = Game(
                [],
                [
                    MegamindAI(random.Random(11)),
                    *(RandomAI(random.Random(20 + index)) for index in range(3)),
                ],
                random.Random(30),
            )
            game.players[0].energy = 2
            game.players[0].hand = [
                card(1, "Equal A", 1, 1),
                card(2, "Equal B", 1, 1),
            ]
            choices.append(game.ais[0].choose_card_to_play(game, 0, (0, 1)))
        self.assertEqual(choices[0], choices[1])


if __name__ == "__main__":
    unittest.main()
