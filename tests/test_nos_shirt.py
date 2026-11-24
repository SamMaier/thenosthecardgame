import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class NosShirtTests(unittest.TestCase):
    def test_printed_values_and_scores_one_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("nos-shirt"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Nos Shirt")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(player.energy, 6)
        self.assertEqual(game.card_fun(0, card), 1)

    def test_returns_to_hand_only_when_scoring_more_than_seven_fun_today(
        self,
    ) -> None:
        # Exact boundary 7 today (despite high cumulative score) -> discarded.
        g_seven = empty_game()
        p7 = g_seven.players[0]
        p7.fun = 25
        p7.energy = 7
        p7.hand.extend([make_card("nos-shirt"), make_card("waterski")])
        shirt7 = g_seven.play_card(0, 0)
        g_seven.play_card(0, 0)  # 1 + 6 = 7 Fun today
        g_seven.end_day()
        self.assertNotIn(shirt7, p7.hand)
        self.assertIn(shirt7, g_seven.discard)

        # Boundary 8 today -> returns to hand cleanly.
        g_eight = empty_game()
        p8 = g_eight.players[0]
        p8.energy = 7
        p8.hand.extend(
            [
                make_card("nos-shirt"),
                make_card("waterski"),
                make_card("tres-fute"),
            ]
        )
        shirt8 = g_eight.play_card(0, 0)
        g_eight.play_card(0, 0)
        g_eight.play_card(0, 0)  # 1 + 6 + 1 = 8 Fun today
        g_eight.end_day()
        self.assertIn(shirt8, p8.hand)
        self.assertNotIn(shirt8, g_eight.discard)
        self.assertEqual(shirt8.markers, {})
        self.assertFalse(shirt8.is_tomorrow)

    def test_wedding_anniversary_copying_nos_shirt_returns_as_printed_card(
        self,
    ) -> None:
        game = empty_game()
        p0 = game.players[0]
        p0.energy = 10
        opp_shirt = make_card("nos-shirt")
        game.players[1].played_today.append(opp_shirt)
        p0.hand.extend(
            [
                make_card("wedding-anniversary"),
                make_card("waterski"),
                make_card("tres-fute"),
            ]
        )
        wedding = game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)  # 1 + 6 + 1 = 8 today
        game.end_day()

        self.assertIn(wedding, p0.hand)
        self.assertEqual(wedding.markers, {})
        self.assertIs(wedding.effective_behavior, wedding.definition.behavior)
        self.assertEqual(wedding.effective_cost, 0)


if __name__ == "__main__":
    unittest.main()
