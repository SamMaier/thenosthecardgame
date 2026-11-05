import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class FloatingTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("floating")

        self.assertEqual(card.title, "Floating")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Relax", "Outdoors"}))

    def test_gains_three_energy_after_paying_cost(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("floating"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 9)
        self.assertEqual(game.card_fun(0, card), 0)

    def test_starts_next_day_with_one_less_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("floating"))

        card = game.play_card(0, 0)
        game.end_day()

        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()

        self.assertEqual(player.energy, 6)


if __name__ == "__main__":
    unittest.main()
