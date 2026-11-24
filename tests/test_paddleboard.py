import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PaddleboardTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("paddleboard")

        self.assertEqual(card.title, "Paddleboard")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 7)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_scores_minus_one_per_hand_card_when_played_only(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        paddleboard = make_card("paddleboard")
        player.hand = [
            paddleboard,
            make_card("biography"),
            make_card("waterski"),
            make_card("fajitas"),
            make_card("nap"),
        ]

        played = game.play_card(0, 0)

        self.assertIs(played, paddleboard)
        self.assertEqual(player.energy, 7)
        self.assertEqual(played.markers["energy_cubes"], 4)
        self.assertEqual(game.card_fun(0, paddleboard), 3)

        # Emptying the hand afterward does not alter the locked-in penalty.
        player.hand.clear()
        self.assertEqual(game.card_fun(0, paddleboard), 3)

    def test_negative_fun_when_many_cards_in_hand_and_full_seven_on_empty_hand(
        self,
    ) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        big_pb = make_card("paddleboard")
        player.hand = [big_pb, *(make_card("nap") for _ in range(9))]

        game.play_card(0, 0)
        self.assertEqual(big_pb.markers["energy_cubes"], 9)
        self.assertEqual(game.card_fun(0, big_pb), -2)

        g_empty = empty_game()
        p_empty = g_empty.players[0]
        p_empty.energy = 3
        empty_pb = make_card("paddleboard")
        p_empty.hand.append(empty_pb)

        g_empty.play_card(0, 0)
        self.assertEqual(empty_pb.markers["energy_cubes"], 0)
        self.assertEqual(g_empty.card_fun(0, empty_pb), 7)


if __name__ == "__main__":
    unittest.main()
