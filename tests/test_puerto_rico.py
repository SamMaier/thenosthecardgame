import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetSuitcaseAI(RandomAI):
    def __init__(self, target_index: int, rng, pick_index: int = 0) -> None:
        super().__init__(rng)
        self.target_index = target_index
        self.pick_index = pick_index

    def choose_suitcase_target(self, game, player_index, suitcase):
        return self.target_index

    def choose_suitcase_card(self, game, player_index, suitcase):
        return self.pick_index


class PuertoRicoTests(unittest.TestCase):
    def test_printed_values_tags_and_base_fun(self) -> None:
        card = make_card("puerto-rico")

        self.assertEqual(card.title, "Puerto Rico")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_marks_a_suitcase_card_without_taking_it_and_scores_bonus(self) -> None:
        game = empty_game()
        game.ais[0] = TargetSuitcaseAI(1, game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("puerto-rico"))
        target = make_card("biography")
        game.suitcase = [
            make_card("fajitas"),
            target,
            make_card("solo"),
            make_card("nap"),
        ]

        card = game.play_card(0, 0)

        self.assertIn(target, game.suitcase)
        self.assertEqual(sum(player.picked_cards.values()), 0)
        self.assertTrue(target.markers["energy_cube"])
        self.assertNotIn("energy_cube", card.markers)
        self.assertIs(card.markers["suitcase_target"], target)
        self.assertEqual(game.card_fun(0, card), 6)

        game.end_day()

        self.assertEqual(player.fun, 6)

    def test_taking_or_discarding_target_removes_bonus(self) -> None:
        for action in ("take", "discard"):
            with self.subTest(action=action):
                game = empty_game()
                game.ais[0] = TargetSuitcaseAI(1, game.rng, pick_index=1)
                player = game.players[0]
                player.energy = 7
                player.hand.append(make_card("puerto-rico"))
                target = make_card("biography")
                game.suitcase = [
                    make_card("fajitas"),
                    target,
                    make_card("solo"),
                    make_card("nap"),
                ]
                card = game.play_card(0, 0)

                if action == "take":
                    game.trunk = [make_card("biography")]
                    game.pick_from_suitcase(0)
                    self.assertIn(target, player.hand)
                else:
                    game.trunk = [make_card("biography") for _ in range(4)]
                    game.unpack(0)
                    self.assertIn(target, game.discard)

                self.assertEqual(game.card_fun(0, card), 3)

    def test_discarded_target_does_not_regain_bonus_if_recycled(self) -> None:
        game = empty_game()
        game.ais[0] = TargetSuitcaseAI(1, game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("puerto-rico"))
        target = make_card("biography")
        game.suitcase = [
            make_card("fajitas"),
            target,
            make_card("solo"),
            make_card("nap"),
        ]
        game.trunk = []

        card = game.play_card(0, 0)
        game.unpack(0)

        self.assertIn(target, game.suitcase)
        self.assertNotIn("energy_cube", target.markers)
        self.assertEqual(game.card_fun(0, card), 3)

    def test_must_be_one_of_first_two_cards_played_today(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [make_card("biography"), make_card("biography"), make_card("puerto-rico")]
        )
        game.suitcase = [make_card("fajitas") for _ in range(4)]

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertNotIn(0, game.playable_hand_indices(0))
        with self.assertRaisesRegex(ValueError, "cannot legally be played"):
            game.play_card(0, 0)

    def test_can_be_played_as_the_second_card_today(self) -> None:
        game = empty_game()
        game.ais[0] = TargetSuitcaseAI(0, game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("biography"), make_card("puerto-rico")])
        game.suitcase = [make_card("fajitas") for _ in range(4)]

        game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Puerto Rico")
        self.assertEqual(len(player.played_today), 2)


if __name__ == "__main__":
    unittest.main()
