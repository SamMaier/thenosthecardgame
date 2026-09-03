import copy
import random
import unittest

from thenos.ais import GalaxybrainAI, RandomAI
from thenos.cards.base import CardBehavior, CardDefinition, CardInstance
from thenos.cards.catalog import create_default_deck
from thenos.game import Game


def card(
    instance_id: int,
    title: str,
    *,
    cost: int = 1,
    fun: int = 1,
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


def galaxy_game(seed: int = 11) -> Game:
    return Game(
        [],
        [
            GalaxybrainAI(random.Random(seed)),
            *(RandomAI(random.Random(seed + index + 1)) for index in range(3)),
        ],
        random.Random(seed + 10),
    )


class GalaxybrainAITests(unittest.TestCase):
    def test_credits_tomorrow_hand_benefit_before_final_day(self):
        class TomorrowFunBehavior(CardBehavior):
            has_tomorrow_action = True

            def modify_tomorrow_fun(
                self, game, player, source, target, current_fun
            ):
                return current_fun + 2

        game = galaxy_game(seed=17)
        game.day = 2
        player = game.players[0]
        player.played_today = [
            CardInstance(
                1,
                CardDefinition(
                    slug="tomorrow-source",
                    title="Tomorrow source",
                    tags=frozenset(),
                    cost=1,
                    base_fun=1,
                    behavior=TomorrowFunBehavior(),
                ),
            )
        ]
        player.hand = [
            card(2, "Future A", cost=1, fun=1),
            card(3, "Future B", cost=1, fun=1),
        ]

        galaxybrain_value = game.ais[0]._state_value(game, 0)
        game.ais[0].TOMORROW_RESERVE_BONUS_WEIGHT = 0.0
        ordinary_value = game.ais[0]._state_value(game, 0)

        self.assertAlmostEqual(
            galaxybrain_value - ordinary_value,
            2 * 2 * 0.80 * 0.50,
        )

    def test_credits_tomorrow_hand_benefit_on_day_five(self):
        class TomorrowFunBehavior(CardBehavior):
            has_tomorrow_action = True

            def modify_tomorrow_fun(
                self, game, player, source, target, current_fun
            ):
                return current_fun + 5

        game = galaxy_game(seed=18)
        game.day = 5
        player = game.players[0]
        player.played_today = [
            CardInstance(
                1,
                CardDefinition(
                    slug="tomorrow-source",
                    title="Tomorrow source",
                    tags=frozenset(),
                    cost=1,
                    base_fun=1,
                    behavior=TomorrowFunBehavior(),
                ),
            )
        ]
        player.hand = [card(2, "Future", cost=1, fun=1)]

        galaxybrain_value = game.ais[0]._state_value(game, 0)
        game.ais[0].TOMORROW_RESERVE_BONUS_WEIGHT = 0.0

        self.assertGreater(
            galaxybrain_value,
            game.ais[0]._state_value(game, 0),
        )

    def test_fast_copy_matches_deepcopy_and_preserves_marker_references(self):
        game = Game.default(seed=101)
        game.setup()
        source, target = game.suitcase[:2]
        source.markers["target"] = target
        target.markers["energy_cube"] = True

        expected = copy.deepcopy(game)
        actual = game.copy_for_simulation()

        self.assertEqual(
            [card.instance_id for card in actual.trunk],
            [card.instance_id for card in expected.trunk],
        )
        self.assertEqual(
            [card.instance_id for card in actual.suitcase],
            [card.instance_id for card in expected.suitcase],
        )
        self.assertIs(
            actual.suitcase[0].markers["target"],
            actual.suitcase[1],
        )
        self.assertIsNot(actual.suitcase[0], source)
        self.assertIs(actual.suitcase[0].definition, source.definition)
        self.assertEqual(actual.rng.getstate(), expected.rng.getstate())

        actual.suitcase[1].markers["energy_cube"] = False
        self.assertTrue(target.markers["energy_cube"])

    def test_planning_is_independent_of_hidden_order_and_allocation(self):
        def prepared_game(reverse: bool) -> Game:
            game = galaxy_game(seed=31)
            game.day = 2
            game.players[0].energy = 7
            game.players[0].hand = [card(1, "Owned", cost=2, fun=2)]
            game.suitcase = [
                card(10, "Choice A", cost=3, fun=4),
                card(11, "Choice B", cost=4, fun=5),
                card(12, "Choice C", cost=1, fun=1),
                card(13, "Choice D", cost=6, fun=6),
            ]
            unknown = [
                card(100 + index, f"Unknown {index}", cost=2, fun=index % 4)
                for index in range(24)
            ]
            if reverse:
                unknown.reverse()
            cursor = 0
            for player in game.players[1:]:
                player.hand = unknown[cursor : cursor + 3]
                cursor += 3
            game.trunk = unknown[cursor:]
            return game

        first = prepared_game(False)
        second = prepared_game(True)
        first_before = (
            tuple(card.instance_id for card in first.trunk),
            tuple(card.instance_id for card in first.players[0].hand),
        )

        first_choice = first.ais[0].choose_suitcase_card(
            first, 0, tuple(first.suitcase)
        )
        second_choice = second.ais[0].choose_suitcase_card(
            second, 0, tuple(second.suitcase)
        )

        self.assertEqual(first_choice, second_choice)
        self.assertEqual(
            first_before,
            (
                tuple(card.instance_id for card in first.trunk),
                tuple(card.instance_id for card in first.players[0].hand),
            ),
        )

    def test_titles_and_slugs_do_not_change_generic_play_choice(self):
        def choice(titles: tuple[str, str]) -> int:
            game = galaxy_game(seed=41)
            game.day = 3
            game.players[0].energy = 4
            game.players[0].hand = [
                card(1, titles[0], cost=2, fun=3),
                card(2, titles[1], cost=3, fun=2),
            ]
            return game.ais[0].choose_card_to_play(game, 0, (0, 1))

        self.assertEqual(choice(("Alpha", "Beta")), choice(("Zed", "Omega")))

    def test_seeded_four_galaxybrain_setup_preserves_acting_hand(self):
        game = Game(
            create_default_deck(),
            [GalaxybrainAI(random.Random(81 + index)) for index in range(4)],
            random.Random(85),
        )
        game.setup()
        expected = [card.instance_id for card in game.players[2].hand]

        sample = game.ais[2]._planning_copy(game, 2)

        self.assertEqual(
            [card.instance_id for card in sample.players[2].hand],
            expected,
        )


if __name__ == "__main__":
    unittest.main()
