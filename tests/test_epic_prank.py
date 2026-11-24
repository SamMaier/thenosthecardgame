import unittest

from thenos.ai import RandomAI
from thenos.ais.carbon import _carbon_copy_game
from thenos.ais.hydrogen import _hydrogen_copy_game
from thenos.cards import make_card
from tests.helpers import empty_game


class RecordingTargetPlayerAI(RandomAI):
    def __init__(self, target_player_index: int, rng) -> None:
        super().__init__(rng)
        self.target_player_index = target_player_index
        self.seen_eligible = None

    def choose_player(self, game, player_index, eligible_player_indices):
        self.seen_eligible = tuple(eligible_player_indices)
        return self.target_player_index


class EpicPrankTests(unittest.TestCase):
    def test_printed_values_and_choose_player_opponents(self) -> None:
        game = empty_game()
        ai = RecordingTargetPlayerAI(2, game.rng)
        game.ais[0] = ai
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("epic-prank"))

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Epic Prank")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertTrue(card.effective_behavior.has_tomorrow_action)
        self.assertEqual(player.energy, 4)
        self.assertEqual(ai.seen_eligible, (1, 2, 3))
        self.assertEqual(card.markers["target_player_index"], 2)
        self.assertEqual(game.card_fun(0, card), 0)

    def test_discards_when_scoring_less_today_than_target_and_keeps_on_tie(
        self,
    ) -> None:
        # Strictly less today's Fun than target -> discarded at END OF DAY.
        g_lose = empty_game()
        g_lose.ais[0] = RecordingTargetPlayerAI(1, g_lose.rng)
        g_lose.players[0].energy = 7
        g_lose.players[1].energy = 7
        g_lose.players[0].hand.extend(
            [make_card("epic-prank"), make_card("biography")]
        )
        g_lose.players[1].hand.append(make_card("waterski"))
        prank_lose = g_lose.play_card(0, 0)
        g_lose.play_card(0, 0)  # 2 today Fun
        g_lose.play_card(1, 0)  # 6 today Fun
        g_lose.end_day()
        self.assertNotIn(prank_lose, g_lose.players[0].tomorrow_cards)
        self.assertIn(prank_lose, g_lose.discard)

        # Tied today's Fun (prior-day Fun difference is ignored!) -> survives.
        g_tie = empty_game()
        g_tie.ais[0] = RecordingTargetPlayerAI(1, g_tie.rng)
        g_tie.players[0].fun = 0
        g_tie.players[1].fun = 20
        g_tie.players[0].energy = 7
        g_tie.players[1].energy = 7
        g_tie.players[0].hand.extend(
            [make_card("epic-prank"), make_card("biography")]
        )
        g_tie.players[1].hand.append(make_card("biography"))
        prank_tie = g_tie.play_card(0, 0)
        g_tie.play_card(0, 0)  # 2 Fun today
        g_tie.play_card(1, 0)  # 2 Fun today
        g_tie.end_day()
        self.assertIn(prank_tie, g_tie.players[0].tomorrow_cards)

    def test_tomorrow_doubles_first_card_only_and_matches_fast_sims(
        self,
    ) -> None:
        game = empty_game()
        game.ais[0] = RecordingTargetPlayerAI(1, game.rng)
        p0 = game.players[0]
        p0.energy = 10
        p0.hand.extend([make_card("epic-prank"), make_card("biography")])
        game.play_card(0, 0)
        game.play_card(0, 0)
        game.end_day()
        game.start_day()

        p0.energy = 12
        # Add a Tomorrow additive bonus (+2 Exercise); left-to-right tomorrow order:
        # [Epic Prank (*2), Teach Kid (+2)] -> 6 * 2 + 2 = 14.
        tom_ex = make_card("teach-kid-to-ski")
        tom_ex.is_tomorrow = True
        p0.tomorrow_cards.append(tom_ex)
        first = make_card("waterski")
        second = make_card("biography")
        p0.hand.extend([first, second])
        game.play_card(0, 0)
        game.play_card(0, 0)

        self.assertEqual(game.card_fun(0, first), 14)
        self.assertEqual(game.card_fun(0, second), 2)

        hsim = _hydrogen_copy_game(game)
        self.assertEqual(hsim.score_played_today_fast(0)[0], 16)

        csim = _carbon_copy_game(game)
        csim.end_day()
        self.assertEqual(csim.players[0].fun, p0.fun + 16)

        # Reverse Tomorrow order [Teach Kid (+2), Epic Prank (*2)] -> (6 + 2) * 2 = 16.
        p0.tomorrow_cards.reverse()
        self.assertEqual(game.card_fun(0, first), 16)
        self.assertEqual(
            _hydrogen_copy_game(game).score_played_today_fast(0)[0], 18
        )


if __name__ == "__main__":
    unittest.main()
