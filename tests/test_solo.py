import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class SoloTests(unittest.TestCase):
    def test_cost_tags_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("solo"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 3)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 5)
        self.assertTrue(card.markers["energy_cube"])

    def test_no_bonus_if_any_opponent_played_a_board_game_previously_today(self) -> None:
        game = empty_game()
        opponent = game.players[1]
        opponent.energy = 1
        opponent.hand.append(make_card("tres-fute"))
        game.play_card(1, 0)

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("solo"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 3)
        self.assertNotIn("energy_cube", card.markers)

    def test_non_board_games_by_opponents_do_not_prevent_bonus(self) -> None:
        game = empty_game()
        opponent = game.players[1]
        opponent.energy = 1
        opponent.hand.append(make_card("biography"))
        game.play_card(1, 0)

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("solo"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 5)
        self.assertTrue(card.markers["energy_cube"])

    def test_active_tomorrow_board_games_do_not_count_as_previously_played(self) -> None:
        game = empty_game()
        tomorrow_card = make_card("tres-fute")
        tomorrow_card.is_tomorrow = True
        game.players[1].tomorrow_cards.append(tomorrow_card)

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("solo"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 5)
        self.assertTrue(card.markers["energy_cube"])


if __name__ == "__main__":
    unittest.main()
