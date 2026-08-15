import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ColouringTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("colouring")

        self.assertEqual(card.title, "Colouring")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))

    def test_draws_one_card_into_hand(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("colouring"))
        drawn_card = make_card("biography")
        game.trunk = [drawn_card]

        colouring = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(player.hand, [drawn_card])
        self.assertEqual(game.trunk, [])
        self.assertEqual(game.card_fun(0, colouring), 2)
        self.assertEqual(player.acquired_cards[drawn_card.title], 1)
        self.assertEqual(game.stats.card_acquisitions[drawn_card.title], 1)

    def test_draw_reshuffles_discard_when_trunk_is_empty(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("colouring"))
        drawn_card = make_card("biography")
        game.discard = [drawn_card]

        game.play_card(0, 0)

        self.assertEqual(player.hand, [drawn_card])
        self.assertEqual(game.trunk, [])
        self.assertEqual(game.discard, [])


if __name__ == "__main__":
    unittest.main()
