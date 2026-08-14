import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class StayUpLateTests(unittest.TestCase):
    def test_immediate_energy_gain_and_tomorrow_penalty(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.append(make_card("stay-up-late"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Event", "Indoors"}))
        self.assertEqual(player.energy, 2)
        self.assertEqual(game.card_fun(0, card), 0)

        game.end_day()

        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()

        self.assertEqual(player.energy, 5)


if __name__ == "__main__":
    unittest.main()
