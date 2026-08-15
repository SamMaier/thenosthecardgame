import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PaintRocksTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("paint-rocks")

        self.assertEqual(card.title, "Paint Rocks")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))

    def test_scores_bonus_after_two_previous_outdoors_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 6
        player.hand.extend(
            [
                make_card("dock-fishing"),
                make_card("canoe"),
                make_card("paint-rocks"),
            ]
        )

        first = game.play_card(0, 0)
        second = game.play_card(0, 0)
        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, first), 2)
        self.assertEqual(game.card_fun(0, second), 2)
        self.assertEqual(game.card_fun(0, card), 4)

    def test_needs_two_previous_outdoors_cards_and_later_cards_do_not_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend(
            [
                make_card("paint-rocks"),
                make_card("dock-fishing"),
                make_card("canoe"),
            ]
        )

        card = game.play_card(0, 0)
        first = game.play_card(0, 0)
        second = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 2)
        self.assertEqual(game.card_fun(0, first), 2)
        self.assertEqual(game.card_fun(0, second), 2)


if __name__ == "__main__":
    unittest.main()
