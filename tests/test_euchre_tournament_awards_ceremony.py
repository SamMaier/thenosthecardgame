import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class EuchreTournamentAwardsCeremonyTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("euchre-tournament-awards-ceremony")

        self.assertEqual(card.title, "Euchre Tournament Awards Ceremony")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Event", "Outdoors"}),
        )

    def test_picks_three_cards_sequentially_with_immediate_refills(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("euchre-tournament-awards-ceremony"))

        first_pick = make_card("biography")
        second_pick = make_card("waterski")
        third_pick = make_card("fajitas")
        third_replacement = make_card("cheap-white")
        game.suitcase = [
            first_pick,
            make_card("solo"),
            make_card("nap"),
            make_card("wingspan"),
        ]
        game.trunk = [third_replacement, third_pick, second_pick]

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertIn(first_pick, player.hand)
        self.assertIn(second_pick, player.hand)
        self.assertIn(third_pick, player.hand)
        self.assertIs(game.suitcase[0], third_replacement)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(player.picked_cards.values()), 3)
        self.assertEqual(sum(player.acquired_cards.values()), 3)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 3)
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
