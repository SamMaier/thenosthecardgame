import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class LastDiscardAI(RandomAI):
    def choose_card_to_discard(self, game, player_index, hand):
        return len(hand) - 1


class EuchreTournamentTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("euchre-tournament")

        self.assertEqual(card.title, "Euchre Tournament")
        self.assertEqual(card.definition.cost, 7)
        self.assertEqual(card.definition.base_fun, 5)
        self.assertEqual(card.definition.tags, frozenset({"Board Game", "Indoors"}))

    def test_draws_four_then_discards_four_from_the_resulting_hand(self) -> None:
        game = empty_game()
        game.ais[0] = LastDiscardAI(game.rng)
        player = game.players[0]
        player.energy = 7
        tournament = make_card("euchre-tournament")
        kept = make_card("biography")
        player.hand.extend([tournament, kept])
        drawn = [make_card("nap"), make_card("solo"), make_card("fajitas"), make_card("waterski")]
        game.trunk = list(drawn)

        card = game.play_card(0, 0)

        self.assertEqual(player.hand, [kept])
        self.assertEqual(len(game.discard), 4)
        self.assertCountEqual(game.discard, drawn)
        self.assertEqual(sum(player.acquired_cards.values()), 4)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 0)
        self.assertEqual(game.card_fun(0, card), 5)
        self.assertEqual(player.energy, 0)


if __name__ == "__main__":
    unittest.main()
