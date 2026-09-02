import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PaddleboatTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("paddleboat")

        self.assertEqual(card.title, "Paddleboat")
        self.assertEqual(card.definition.cost, 3)
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

    def test_non_pick_or_draw_acquisition_does_not_trigger(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        paddleboat = make_card("paddleboat")
        player.hand.append(paddleboat)

        game.play_card(0, 0)
        game.give_card(0, make_card("biography"))

        self.assertNotIn("energy_cubes", paddleboat.markers)
        self.assertEqual(game.card_fun(0, paddleboat), 0)

    def test_drawn_card_after_play_adds_two_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        paddleboat = make_card("paddleboat")
        player.hand.append(paddleboat)
        game.trunk = [make_card("biography")]

        game.play_card(0, 0)
        drawn_card = game.draw_from_trunk(0, 1)[0]
        game.give_card(0, drawn_card)

        self.assertEqual(paddleboat.markers["energy_cubes"], 1)
        self.assertEqual(game.card_fun(0, paddleboat), 2)

    def test_card_played_from_trunk_does_not_trigger(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        paddleboat = make_card("paddleboat")
        player.hand.extend([paddleboat, make_card("assorted-cutlery")])
        game.trunk = [make_card("sunrise")]

        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(player.hand, [])
        self.assertEqual(
            [card.title for card in player.played_today],
            ["Paddleboat", "Assorted Cutlery", "Sunrise"],
        )
        self.assertNotIn("energy_cubes", paddleboat.markers)
        self.assertEqual(game.card_fun(0, paddleboat), 0)


if __name__ == "__main__":
    unittest.main()
