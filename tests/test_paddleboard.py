import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class PaddleboardAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class PaddleboardTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("paddleboard")

        self.assertEqual(card.title, "Paddleboard")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_discards_entire_hand_and_picks_same_number(self) -> None:
        game = empty_game()
        game.ais[0] = PaddleboardAI(game.rng)
        player = game.players[0]
        player.energy = 7
        paddleboard = make_card("paddleboard")
        discarded_cards = [
            make_card("biography"),
            make_card("waterski"),
            make_card("fajitas"),
        ]
        for discarded_card in discarded_cards:
            discarded_card.markers["test"] = True
        player.hand = [paddleboard, *discarded_cards]

        first_pick = make_card("cheap-white")
        replacement_after_first_pick = make_card("chalk-art")
        replacement_after_second_pick = make_card("nap")
        replacement_after_third_pick = make_card("solo")
        game.suitcase = [
            first_pick,
            make_card("biography"),
            make_card("waterski"),
            make_card("fajitas"),
        ]
        game.trunk = [
            replacement_after_third_pick,
            replacement_after_second_pick,
            replacement_after_first_pick,
        ]

        played = game.play_card(0, 0)

        self.assertIs(played, paddleboard)
        self.assertEqual(player.energy, 3)
        self.assertEqual(
            player.hand,
            [first_pick, replacement_after_first_pick, replacement_after_second_pick],
        )
        self.assertEqual(
            {id(card) for card in game.discard},
            {id(card) for card in discarded_cards},
        )
        self.assertTrue(all(card.markers == {} for card in discarded_cards))
        self.assertEqual(game.suitcase[0], replacement_after_third_pick)
        self.assertEqual(player.picked_cards["Cheap White"], 1)
        self.assertEqual(player.picked_cards["Chalk Art"], 1)
        self.assertEqual(player.picked_cards["Nap"], 1)
        self.assertEqual(game.card_fun(0, paddleboard), 5)

    def test_empty_hand_discards_and_picks_nothing(self) -> None:
        game = empty_game()
        game.ais[0] = PaddleboardAI(game.rng)
        player = game.players[0]
        player.energy = 4
        paddleboard = make_card("paddleboard")
        player.hand.append(paddleboard)
        suitcase = [make_card("biography") for _ in range(4)]
        game.suitcase = suitcase.copy()

        game.play_card(0, 0)

        self.assertEqual(player.hand, [])
        self.assertEqual(game.discard, [])
        self.assertEqual(game.suitcase, suitcase)
        self.assertEqual(game.card_fun(0, paddleboard), 5)


if __name__ == "__main__":
    unittest.main()
