import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetPlayerAI(RandomAI):
    def __init__(self, target_player_index, rng):
        super().__init__(rng)
        self.target_player_index = target_player_index
        self.eligible_player_indices = None

    def choose_player(self, game, player_index, eligible_player_indices):
        self.eligible_player_indices = tuple(eligible_player_indices)
        if self.target_player_index in eligible_player_indices:
            return self.target_player_index
        return eligible_player_indices[0]


class SanJuanTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("san-juan"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(player.energy, 4)

    def test_selects_a_player_with_two_or_fewer_cards_and_scores_bonus(self) -> None:
        game = empty_game()
        ai = TargetPlayerAI(1, game.rng)
        game.ais[0] = ai
        target = game.players[1]
        target.played_today.extend([make_card("biography"), make_card("biography")])

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("san-juan"))

        card = game.play_card(0, 0)

        self.assertEqual(ai.eligible_player_indices, (0, 1, 2, 3))
        self.assertTrue(card.markers["energy_cube"])
        self.assertEqual(card.markers["target_player_index"], 1)
        self.assertEqual(game.card_fun(0, card), 5)

    def test_target_later_playing_a_board_game_removes_bonus(self) -> None:
        game = empty_game()
        ai = TargetPlayerAI(1, game.rng)
        game.ais[0] = ai

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("san-juan"))
        card = game.play_card(0, 0)

        target = game.players[1]
        target.energy = 0
        target.hand.append(make_card("tres-fute"))
        game.play_card(1, 0)

        self.assertEqual(game.card_fun(0, card), 2)

    def test_players_with_more_than_two_cards_are_not_eligible(self) -> None:
        game = empty_game()
        ai = TargetPlayerAI(1, game.rng)
        game.ais[0] = ai
        game.players[1].played_today.extend(
            [make_card("biography") for _ in range(3)]
        )

        player = game.players[0]
        player.energy = 3
        player.hand.append(make_card("san-juan"))

        card = game.play_card(0, 0)

        self.assertNotIn(1, ai.eligible_player_indices)
        self.assertNotEqual(card.markers["target_player_index"], 1)


if __name__ == "__main__":
    unittest.main()
