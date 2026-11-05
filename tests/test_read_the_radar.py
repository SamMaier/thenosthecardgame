import random
import unittest
from unittest.mock import Mock

from thenos.ais import RandomAI
from thenos.cards.catalog import make_card
from thenos.daily_conditions import DAILY_CONDITIONS
from thenos.game import Game
from tests.helpers import empty_game


class ReadTheRadarTests(unittest.TestCase):
    def game(self):
        game = Game([], [RandomAI(random.Random(i)) for i in range(4)], daily_conditions=True)
        game.players[0].energy = 7
        game.give_card(0, make_card("read-the-radar"))
        return game

    def test_printed_values_and_tags(self) -> None:
        card = make_card("read-the-radar")

        self.assertEqual(card.title, "Read the Radar")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_play_reveals_top_five_trunk_cards_and_returns_them_in_chosen_order(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        game.give_card(0, make_card("read-the-radar"))

        cards = [
            make_card("fajitas"),
            make_card("biography"),
            make_card("cheap-white"),
            make_card("waterski"),
            make_card("thriller-book"),
        ]
        # Top of Trunk is the end of the internal list; reveal_from_trunk(5) pops from end.
        bottom_card = make_card("solo")
        game.trunk = [bottom_card, *reversed(cards)]
        game.ais[0].order_cards_for_trunk = Mock(return_value=(4, 1, 3, 0, 2))

        game.play_card(0, 0)

        self.assertEqual(player.energy, 6)
        self.assertEqual(dict(game.stats.card_acquisitions), {"Read the Radar": 1})
        expected_top_to_bottom = [cards[4], cards[1], cards[3], cards[0], cards[2]]
        self.assertEqual(
            [game._draw_from_trunk() for _ in range(5)],
            expected_top_to_bottom,
        )
        self.assertIs(game._draw_from_trunk(), bottom_card)

    def test_arrange_daily_conditions_reorders_three_privately(self):
        game = self.game()
        original = tuple(reversed(game._condition_deck[-3:]))
        game.ais[0].order_daily_conditions = Mock(return_value=(2, 0, 1))
        game.arrange_daily_conditions(0)
        ordered = (original[2], original[0], original[1])
        self.assertEqual(game.known_daily_conditions(0), ordered)
        self.assertEqual(game.known_daily_conditions(1), (None,) * 3)
        self.assertEqual(tuple(reversed(game._condition_deck[-3:])), ordered)
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


if __name__ == "__main__":
    unittest.main()
