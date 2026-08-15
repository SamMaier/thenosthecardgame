import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class AunterviewTests(unittest.TestCase):
    def test_printed_values_and_tomorrow_social_bonus(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("aunterview"))

        aunterview = game.play_card(0, 0)

        self.assertEqual(aunterview.title, "Aunterview")
        self.assertEqual(aunterview.definition.cost, 2)
        self.assertEqual(aunterview.definition.base_fun, 0)
        self.assertEqual(aunterview.definition.tags, frozenset({"Social"}))
        self.assertEqual(player.energy, 5)

        player.hand.extend([make_card("johnny-appleseed"), make_card("biography")])
        social = game.play_card(0, 0)
        relax = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, social), 1)
        self.assertEqual(game.card_fun(0, relax), 2)

        game.end_day()
        self.assertEqual(player.tomorrow_cards, [aunterview])
        self.assertTrue(aunterview.is_tomorrow)

        game.start_day()
        player.hand.extend([make_card("johnny-appleseed"), make_card("biography")])
        tomorrow_social = game.play_card(0, 0)
        tomorrow_relax = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, aunterview), 1)
        self.assertEqual(game.card_fun(0, tomorrow_social), 2)
        self.assertEqual(game.card_fun(0, tomorrow_relax), 2)

        game.end_day()
        self.assertEqual(player.fun, 8)
        self.assertEqual(player.tomorrow_cards, [])
        self.assertFalse(aunterview.is_tomorrow)


if __name__ == "__main__":
    unittest.main()
