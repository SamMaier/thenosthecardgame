import unittest
from unittest.mock import Mock
from tests.helpers import empty_game
from thenos.cards.catalog import make_card


class PokerTests(unittest.TestCase):
    def test_prediction_precedes_reveal_bonus_is_retained_until_scoring(self):
        for tag, expected in (("Food", 5), ("Relax", 1)):
            with self.subTest(tag=tag):
                game = empty_game()
                game.players[0].energy = 7
                revealed = make_card("fajitas")
                game.trunk = [revealed]
                def choose(game_arg, index, tags):
                    self.assertEqual(game_arg.trunk, [revealed])
                    return tag
                game.ais[0].choose_tag = choose
                poker = make_card("poker")
                game.give_card(0, poker)
                game.play_card(0, 0)
                self.assertEqual(game.card_fun(0, poker), expected)
                self.assertEqual(game.discard, [revealed])
                self.assertEqual(game.stats.card_acquisitions["Fajitas"], 0)
                self.assertEqual(game.players[0].energy, 5)
                game.end_day()
                self.assertEqual(game.players[0].fun, expected)
                self.assertEqual(poker.markers, {})

    def test_invalid_tag_rejected_before_reveal(self):
        game = empty_game()
        game.trunk = [make_card("fajitas")]
        game.ais[0].choose_tag = Mock(return_value="Invalid")
        poker = make_card("poker")
        with self.assertRaises(ValueError):
            poker.effective_behavior.on_play(game, game.players[0], poker)
        self.assertEqual(len(game.trunk), 1)
