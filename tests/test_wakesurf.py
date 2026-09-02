import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class WakesurfTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("wakesurf")

        self.assertEqual(card.title, "Wakesurf")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_scores_bonus_when_hand_is_empty_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("wakesurf"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 6)
        game.end_day()

        self.assertEqual(player.fun, 6)

    def test_scores_only_base_fun_when_hand_is_not_empty_at_end_of_day(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.extend([make_card("wakesurf"), make_card("biography")])

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 1)
        game.end_day()

        self.assertEqual(player.fun, 1)


if __name__ == "__main__":
    unittest.main()
