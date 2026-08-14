import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PaddleboatTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("paddleboat")

        self.assertEqual(card.title, "Paddleboat")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_picked_cards_after_play_add_two_fun_each(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        paddleboat = make_card("paddleboat")
        player.hand.append(paddleboat)
        first_pick = make_card("biography")
        second_pick = make_card("waterski")
        game.suitcase = [
            first_pick,
            second_pick,
            make_card("fajitas"),
            make_card("nap"),
        ]
        game.trunk = [
            make_card("cheap-white"),
            make_card("chalk-art"),
        ]

        game.play_card(0, 0)
        game.pick_suitcase_cards(0, (first_pick, second_pick))

        self.assertEqual(paddleboat.markers["energy_cubes"], 2)
        self.assertEqual(game.card_fun(0, paddleboat), 4)

    def test_drawn_or_picked_card_before_play_does_not_trigger(self) -> None:
        game = empty_game()
        player = game.players[0]
        paddleboat = make_card("paddleboat")
        player.hand.append(paddleboat)

        game.give_card(0, make_card("biography"))

        self.assertNotIn("energy_cubes", paddleboat.markers)

    def test_card_acquired_after_play_triggers_even_when_not_picked(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        paddleboat = make_card("paddleboat")
        player.hand.append(paddleboat)

        game.play_card(0, 0)
        game.give_card(0, make_card("biography"))

        self.assertEqual(paddleboat.markers["energy_cubes"], 1)
        self.assertEqual(game.card_fun(0, paddleboat), 2)


if __name__ == "__main__":
    unittest.main()
