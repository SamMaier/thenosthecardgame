import copy
import random
import unittest

from thenos.ais import GeniusAI, PlannerAI, RandomAI
from thenos.cards import make_card
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.cards.food import DECAF
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
    def test_state_value_uses_one_game_copy_for_simple_state(self) -> None:
        class CopyCounter:
            copies = 0

            def __deepcopy__(self, memo):
                type(self).copies += 1
                copied = type(self)()
                memo[id(self)] = copied
                return copied

        game = genius_game()
        game.day = 1
        player = game.players[0]
        player.energy = 7
        physical_card = card(1, "Simple", cost=1, fun=2)
        physical_card.markers["copy_counter"] = CopyCounter()
        player.hand = [physical_card]

        game.ais[0]._state_value(game, 0)

        self.assertEqual(CopyCounter.copies, 1)

    def test_uses_generic_card_values_without_title_specific_priors(self) -> None:
        genius = GeniusAI(random.Random(1))
        planner = PlannerAI(random.Random(1))

        for physical_card in create_default_deck():
            with self.subTest(card=physical_card.title):
                self.assertEqual(
                    genius._card_value(physical_card),
                    planner._card_value(physical_card),
                )

    def test_goes_to_bed_to_preserve_decaf_energy(self) -> None:
        game = genius_game()
        game.day = 6
        player = game.players[0]
        player.energy = 4
        player.played_today = [CardInstance(1, DECAF)]
        player.hand = [card(2, "Three Fun", cost=2, fun=3)]

        goes_to_bed = game.ais[0].choose_to_go_to_bed(game, 0, (0,))

        self.assertTrue(goes_to_bed)
        self.assertEqual(player.energy, 4)
        self.assertEqual(len(player.hand), 1)

    def test_scores_relative_play_bonus_against_observable_opponent_capacity(
        self,
    ) -> None:
        game = genius_game()
        genius = game.ais[0]
        player = game.players[0]
        player.played_today = [
            make_card("fit-to-print"),
            make_card("fajitas"),
        ]
        for opponent in game.players[1:]:
            opponent.energy = 7
            opponent.hand = [make_card("fajitas") for _ in range(4)]

        # The opponents have not played yet, but their public hand sizes and
        # Energy make later plays plausible. Genius must not score the bonus
        # from the partial position as though the day ended immediately.
        self.assertEqual(genius._state_value(game, 0), 1.0)

        for opponent in game.players[1:]:
            opponent.asleep = True
        self.assertEqual(genius._state_value(game, 0), 5.0)

    def test_uses_energy_gain_to_unlock_a_same_day_high_cost_play(self) -> None:
        game = genius_game()
        genius = game.ais[0]
        genius.SEARCH_DEPTH = 1
        player = game.players[0]
        player.energy = 6
        player.hand = [
            make_card("weird-chip-flavor"),
            card(2, "Big Day", cost=8, fun=8),
        ]

        # Weird Chip Flavor is the only legal first play. Its +2 Energy
        # makes the otherwise unaffordable card worth playing immediately.
        self.assertFalse(genius.choose_to_go_to_bed(game, 0, (0,)))
        self.assertEqual(genius.choose_card_to_play(game, 0, (0,)), 0)

    def test_saves_an_energy_enabler_without_a_follow_up_play(self) -> None:
        game = genius_game()
        player = game.players[0]
        player.energy = 0
        player.hand = [make_card("weird-chip-flavor")]

        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

    def test_counts_next_day_picks_when_evaluating_tomorrow_setup(self) -> None:
        game = genius_game()
        game.day = 1
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("dates-first-nos")]
        game.suitcase = [
            card(100 + index, f"Suitcase {index}", cost=2, fun=2)
            for index in range(4)
        ]
        game.trunk = [
            card(200 + index, f"Trunk {index}", cost=2, fun=2)
            for index in range(20)
        ]

        # The printed card itself scores nothing, but its Tomorrow modifier
        # will apply to the three cards Genius is guaranteed to acquire.
        self.assertFalse(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

    def test_does_not_project_suitcase_picks_after_the_final_day(self) -> None:
        game = genius_game()
        game.day = 6
        player = game.players[0]
        player.energy = 7
        player.hand = [make_card("dates-first-nos")]
        game.suitcase = [
            card(100 + index, f"Suitcase {index}", cost=2, fun=2)
            for index in range(4)
        ]
        game.trunk = [
            card(200 + index, f"Trunk {index}", cost=2, fun=2)
            for index in range(20)
        ]

        self.assertTrue(game.ais[0].choose_to_go_to_bed(game, 0, (0,)))

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
