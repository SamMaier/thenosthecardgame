import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class DoritosTests(unittest.TestCase):
    def test_gains_two_energy_after_one_previous_food(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.extend([make_card("m-ms"), make_card("doritos")])

        game.play_card(0, 0)
        doritos = game.play_card(0, 0)

        self.assertEqual(doritos.definition.cost, 0)
        self.assertEqual(doritos.definition.base_fun, 0)
        self.assertEqual(doritos.definition.tags, frozenset({"Food"}))
        self.assertEqual(player.energy, 3)

    def test_gains_two_energy_only_once_after_multiple_previous_foods(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.extend(
            [make_card("m-ms"), make_card("m-ms"), make_card("doritos")]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        doritos = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, doritos), 0)

    def test_non_food_cards_do_not_count(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 0
        player.hand.extend(
            [make_card("nap"), make_card("tres-fute"), make_card("doritos")]
        )

        game.play_card(0, 0)
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.energy, 1)


if __name__ == "__main__":
    unittest.main()
