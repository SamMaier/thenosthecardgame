import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SteakTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("steak")

        self.assertEqual(card.title, "Steak")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_three_energy_after_a_written_cost_of_five(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 20
        player.hand.extend(
            [make_card("rouladen"), make_card("agricola"), make_card("steak")]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        steak = game.play_card(0, 0)

        # Rouladen discounts Agricola to four Energy, but Agricola's written
        # cost remains five and therefore triggers Steak.
        self.assertEqual(player.energy, 14)
        self.assertEqual(game.card_fun(0, steak), 1)

    def test_four_cost_card_does_not_trigger(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("kneeboard"), make_card("steak")])

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 4)

    def test_active_tomorrow_card_does_not_count_as_previous(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        tomorrow_card = make_card("risk")
        tomorrow_card.is_tomorrow = True
        player.tomorrow_cards.append(tomorrow_card)
        player.hand.append(make_card("steak"))

        game.play_card(0, 0)

        self.assertEqual(player.energy, 5)


if __name__ == "__main__":
    unittest.main()
