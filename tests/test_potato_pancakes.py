import unittest
from tests.helpers import empty_game
from thenos.cards.catalog import make_card


class PotatoPancakesTests(unittest.TestCase):
    def test_only_previous_visible_relax_cards_give_energy(self):
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.played_today = [make_card("nap"), make_card("azul"), make_card("classic-book")]
        tomorrow = make_card("floating")
        tomorrow.is_tomorrow = True
        player.tomorrow_cards = [tomorrow]
        game.give_card(0, make_card("potato-pancakes"))
        game.play_card(0, 0)
        self.assertEqual(player.energy, 9)

    def test_no_previous_relax_cards(self):
        game = empty_game()
        game.players[0].energy = 7
        game.give_card(0, make_card("potato-pancakes"))
        game.play_card(0, 0)
        self.assertEqual(game.players[0].energy, 5)
