import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WrestleTheKidsTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("wrestle-the-kids")

        self.assertEqual(card.title, "Wrestle the Kids")
        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Exercise"}))

    def test_scores_one_fun_per_card_remaining_in_hand_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("wrestle-the-kids"),
                make_card("biography"),
                make_card("fajitas"),
            ]
        )

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 2)
        self.assertEqual(game.card_fun(0, card), 3)

        game.end_day()

        self.assertEqual(player.fun, 3)

    def test_scores_only_base_fun_when_hand_is_empty(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.append(make_card("wrestle-the-kids"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 1)


if __name__ == "__main__":
    unittest.main()
