import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class TheCrewTests(unittest.TestCase):
    def test_cost_tags_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 1
        player.hand.append(make_card("the-crew"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(player.energy, 0)
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertNotIn("energy_cube", card.markers)

    def test_bonus_requires_two_of_three_opponents_to_play_board_games(self) -> None:
        game = empty_game()
        for opponent_index in (1, 2):
            opponent = game.players[opponent_index]
            opponent.energy = 1
            opponent.hand.append(make_card("tres-fute"))
            game.play_card(opponent_index, 0)

        player = game.players[0]
        player.energy = 1
        player.hand.append(make_card("the-crew"))

        card = game.play_card(0, 0)

        self.assertTrue(card.markers["energy_cube"])
        self.assertEqual(game.card_fun(0, card), 3)

    def test_one_opponent_or_non_board_game_does_not_qualify(self) -> None:
        game = empty_game()
        opponent = game.players[1]
        opponent.energy = 1
        opponent.hand.append(make_card("tres-fute"))
        game.play_card(1, 0)

        other_opponent = game.players[2]
        other_opponent.energy = 1
        other_opponent.hand.append(make_card("biography"))
        game.play_card(2, 0)

        player = game.players[0]
        player.energy = 1
        player.hand.append(make_card("the-crew"))

        card = game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, card), 1)
        self.assertNotIn("energy_cube", card.markers)


if __name__ == "__main__":
    unittest.main()
