import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class NosBookAI(RandomAI):
    def __init__(self, target, rng):
        super().__init__(rng)
        self.target = target

    def choose_card_target(self, game, player_index, eligible_cards):
        return eligible_cards.index(self.target)


class NosBookTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("nos-book")

        self.assertEqual(card.title, "Nos Book")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_chooses_previous_card_today_and_adds_three_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        biography = make_card("biography")
        cheap_white = make_card("cheap-white")
        nos_book = make_card("nos-book")
        after_card = make_card("splendor")
        player.hand.extend([biography, cheap_white, nos_book, after_card])
        game.ais[0] = NosBookAI(cheap_white, random.Random(0))

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertTrue(cheap_white.markers.get("energy_cube"))
        self.assertIs(nos_book.markers.get("target_card"), cheap_white)
        self.assertEqual(game.card_fun(0, biography), 2)
        self.assertEqual(game.card_fun(0, cheap_white), 6)
        self.assertEqual(game.card_fun(0, nos_book), 0)
        self.assertEqual(game.card_fun(0, after_card), 2)

    def test_playing_as_first_card_today_does_nothing(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        nos_book = make_card("nos-book")
        biography = make_card("biography")
        player.hand.extend([nos_book, biography])

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertNotIn("target_card", nos_book.markers)
        self.assertEqual(game.card_fun(0, nos_book), 0)
        self.assertEqual(game.card_fun(0, biography), 2)


if __name__ == "__main__":
    unittest.main()
