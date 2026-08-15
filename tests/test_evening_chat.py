import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstPlayableAI(RandomAI):
    def choose_card_to_play(self, game, player_index, playable_hand_indices):
        return playable_hand_indices[0]


class EveningChatTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("evening-chat")

        self.assertEqual(card.title, "Evening Chat")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Social", "Indoors"}),
        )

    def test_immediately_goes_to_bed_and_scores_five_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("evening-chat"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 4)
        self.assertTrue(player.asleep)
        self.assertEqual(game.card_fun(0, card), 5)
        with self.assertRaisesRegex(ValueError, "already gone to bed"):
            game.play_card(0, 0)

    def test_playing_phase_does_not_give_the_player_another_turn(self) -> None:
        game = empty_game()
        game.ais[0] = FirstPlayableAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("evening-chat"), make_card("biography")])

        game.playing_phase()

        self.assertEqual(
            [card.title for card in player.played_today],
            ["Evening Chat"],
        )
        self.assertEqual([card.title for card in player.hand], ["Biography"])
        self.assertEqual(game.starting_player, 0)

    def test_tomorrow_reduces_next_day_starting_energy(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("evening-chat"))

        card = game.play_card(0, 0)
        game.end_day()

        self.assertEqual(player.tomorrow_cards, [card])
        self.assertTrue(card.is_tomorrow)

        game.start_day()

        self.assertEqual(player.energy, 6)


if __name__ == "__main__":
    unittest.main()
