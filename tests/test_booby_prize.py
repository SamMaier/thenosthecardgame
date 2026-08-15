import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class BoobyPrizeTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("booby-prize")

        self.assertEqual(card.title, "Booby Prize")
        self.assertEqual(card.definition.cost, 0)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_draws_one_card_into_hand_without_spending_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        booby_prize = make_card("booby-prize")
        drawn_card = make_card("biography")
        player.hand.append(booby_prize)
        game.trunk = [drawn_card]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 7)
        self.assertEqual(player.hand, [drawn_card])
        self.assertEqual(game.trunk, [])
        self.assertEqual(player.acquired_cards[drawn_card.title], 1)
        self.assertEqual(game.stats.card_acquisitions[drawn_card.title], 1)

    def test_draw_reshuffles_discard_when_trunk_is_empty(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 1
        player.hand.append(make_card("booby-prize"))
        drawn_card = make_card("biography")
        game.discard = [drawn_card]

        game.play_card(0, 0)

        self.assertEqual(player.hand, [drawn_card])
        self.assertEqual(game.trunk, [])
        self.assertEqual(game.discard, [])


if __name__ == "__main__":
    unittest.main()
