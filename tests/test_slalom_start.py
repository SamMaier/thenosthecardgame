import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SlalomStartTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("slalom-start")

        self.assertEqual(card.title, "Slalom Start")
        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_scores_bonus_with_exactly_one_card_in_hand_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.extend([make_card("slalom-start"), make_card("biography")])

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 8)
        game.end_day()

        self.assertEqual(player.fun, 8)

    def test_scores_only_base_fun_with_zero_or_two_cards_in_hand(self) -> None:
        for hand_size in (0, 2):
            with self.subTest(hand_size=hand_size):
                game = empty_game()
                player = game.players[0]
                player.energy = 5
                player.hand.append(make_card("slalom-start"))
                player.hand.extend(
                    make_card("biography") for _ in range(hand_size)
                )

                card = game.play_card(0, 0)

                self.assertEqual(game.card_fun(0, card), 2)


if __name__ == "__main__":
    unittest.main()
