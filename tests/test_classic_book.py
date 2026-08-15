import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ClassicBookTests(unittest.TestCase):
    def test_printed_values_and_cost_without_previous_relax_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 2
        card = make_card("classic-book")
        player.hand.append(card)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))
        self.assertEqual(game.energy_cost(0, card), 2)

        game.play_card(0, 0)

        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 2)

    def test_previous_relax_cards_reduce_cost_and_non_relax_cards_do_not(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend(
            [
                make_card("biography"),
                make_card("tres-fute"),
                make_card("classic-book"),
            ]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        card = player.hand[0]

        self.assertEqual(game.energy_cost(0, card), 1)
        game.play_card(0, 0)
        self.assertEqual(player.energy, 5)

    def test_cost_cannot_go_below_zero_and_tomorrow_relax_cards_do_not_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        tomorrow_card = make_card("biography")
        tomorrow_card.is_tomorrow = True
        player.tomorrow_cards.append(tomorrow_card)
        card = make_card("classic-book")
        player.hand.append(card)

        self.assertEqual(game.energy_cost(0, card), 2)

        for _ in range(3):
            player.played_today.append(make_card("biography"))

        self.assertEqual(game.energy_cost(0, card), 0)


if __name__ == "__main__":
    unittest.main()
