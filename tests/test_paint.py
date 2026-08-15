import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class RecordingFirstPlayableAI(RandomAI):
    def __init__(self, rng, player_index, play_log):
        super().__init__(rng)
        self.player_index = player_index
        self.play_log = play_log

    def choose_card_to_play(self, game, player_index, playable_hand_indices):
        self.play_log.append(self.player_index)
        return playable_hand_indices[0]


class PaintTests(unittest.TestCase):
    def test_printed_values_and_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("paint"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Paint")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(card.definition.tags, frozenset({"Relax"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 4)

    def test_skips_only_the_players_next_turn(self) -> None:
        game = empty_game()
        play_log = []
        for player_index in range(4):
            game.ais[player_index] = RecordingFirstPlayableAI(
                game.rng, player_index, play_log
            )

        game.players[0].energy = 7
        game.players[0].hand.extend([make_card("paint"), make_card("biography")])
        for player in game.players[1:]:
            player.energy = 7
            player.hand.extend([make_card("biography"), make_card("biography")])

        game.playing_phase()

        self.assertEqual(play_log[:8], [0, 1, 2, 3, 1, 2, 3, 0])
        self.assertEqual(
            [card.title for card in game.players[0].played_today],
            ["Paint", "Biography"],
        )
        self.assertEqual(game.players[0].skipped_turns, 0)


if __name__ == "__main__":
    unittest.main()
