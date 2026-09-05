import random
import unittest
from unittest.mock import Mock

from thenos.ais import RandomAI
from thenos.cards.catalog import make_card
from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game


class ReadTheRadarTests(unittest.TestCase):
    def game(self):
        game = Game([], [RandomAI(random.Random(i)) for i in range(4)], daily_conditions=True)
        game.players[0].energy = 7
        game.give_card(0, make_card("read-the-radar"))
        return game

    def test_play_reorders_three_privately_without_acquiring_conditions(self):
        game = self.game()
        original = tuple(reversed(game._condition_deck[-3:]))
        game.ais[0].order_daily_conditions = Mock(return_value=(2, 0, 1))
        game.play_card(0, 0)
        ordered = (original[2], original[0], original[1])
        self.assertEqual(game.known_daily_conditions(0), ordered)
        self.assertEqual(game.known_daily_conditions(1), (None,) * 3)
        self.assertEqual(tuple(reversed(game._condition_deck[-3:])), ordered)
        self.assertEqual(game.players[0].energy, 6)
        self.assertEqual(dict(game.stats.card_acquisitions), {"Read the Radar": 1})
        game.start_day()
        self.assertEqual(game.daily_condition, ordered[0])
        self.assertEqual(game.known_daily_conditions(0), ordered[1:])

    def test_later_private_reorder_invalidates_previous_knowledge(self):
        game = self.game()
        game.arrange_daily_conditions(0)
        game.arrange_daily_conditions(1)
        self.assertEqual(game.known_daily_conditions(0), (None,) * 3)
        self.assertTrue(all(c is not None for c in game.known_daily_conditions(1)))

    def test_final_day_uses_remaining_two_and_invalid_order_is_rejected(self):
        game = self.game()
        game._condition_deck = list(DAILY_CONDITIONS[:2])
        game.ais[0].order_daily_conditions = Mock(return_value=(1, 0))
        game.arrange_daily_conditions(0)
        self.assertEqual(game.known_daily_conditions(0), DAILY_CONDITIONS[:2])
        before = game._condition_deck.copy()
        game.ais[0].order_daily_conditions = Mock(return_value=(0, 0))
        with self.assertRaises(ValueError):
            game.arrange_daily_conditions(0)
        self.assertEqual(game._condition_deck, before)

    def test_disabled_effect_is_safe_for_custom_decks(self):
        game = Game.default(1)
        game.ais[0].order_daily_conditions = Mock(side_effect=AssertionError)
        game.arrange_daily_conditions(0)
        self.assertEqual(game.known_daily_conditions(0), ())
