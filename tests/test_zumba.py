import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ZumbaTests(unittest.TestCase):
    def test_printed_values_and_tomorrow_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.append(make_card("zumba"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Zumba")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Exercise"}))
        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 1)

        game.end_day()

        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()

        self.assertEqual(player.energy, 11)

        game.end_day()

        self.assertEqual(player.tomorrow_cards, [])
        self.assertFalse(card.is_tomorrow)


if __name__ == "__main__":
    unittest.main()
