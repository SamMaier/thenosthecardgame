import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ButterTartsTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("butter-tarts")

        self.assertEqual(card.title, "Butter Tarts")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_doubles_energy_remaining_after_payment(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.append(make_card("butter-tarts"))

        game.play_card(0, 0)

        self.assertEqual(player.energy, 12)

    def test_does_not_create_energy_when_payment_leaves_none(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        player.hand.append(make_card("butter-tarts"))

        game.play_card(0, 0)

        self.assertEqual(player.energy, 0)

    def test_sleep_in_blocks_the_energy_increase(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("sleep-in"), make_card("butter-tarts")])

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 5)

    def test_counts_as_giving_energy_for_bubly(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 10
        player.hand.extend([make_card("butter-tarts"), make_card("bubly")])

        butter = game.play_card(0, 0)
        bubly = game.play_card(0, 0)

        self.assertTrue(butter.markers["_gave_energy"])
        self.assertEqual(game.card_fun(0, bubly), 2)


if __name__ == "__main__":
    unittest.main()
