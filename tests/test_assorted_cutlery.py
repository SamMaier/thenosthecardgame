import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class AssortedCutleryTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("assorted-cutlery")

        self.assertEqual(card.title, "Assorted Cutlery")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))

    def test_plays_the_top_trunk_card_for_zero_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("assorted-cutlery"))
        played_card = make_card("fajitas")
        game.trunk = [played_card]

        assorted_cutlery = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(player.played_today, [assorted_cutlery, played_card])
        self.assertEqual(game.trunk, [])
        self.assertEqual(game.stats.card_plays[played_card.title], 1)
        self.assertEqual(player.acquired_cards[played_card.title], 0)

    def test_restricted_trunk_card_returns_to_hand_without_cost(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("assorted-cutlery"))
        restricted_card = make_card("morning-coffee")
        game.trunk = [restricted_card]

        game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(player.played_today[0].title, "Assorted Cutlery")
        self.assertEqual(player.hand, [restricted_card])
        self.assertEqual(game.stats.card_plays[restricted_card.title], 0)
        self.assertEqual(player.acquired_cards[restricted_card.title], 1)


if __name__ == "__main__":
    unittest.main()
