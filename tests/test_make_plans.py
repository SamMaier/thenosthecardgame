import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class MakePlansTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("make-plans")

        self.assertEqual(card.title, "Make Plans")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))

    def test_picks_two_cards_sequentially_with_immediate_refills(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("make-plans"))

        first_pick = make_card("biography")
        second_pick = make_card("waterski")
        second_replacement = make_card("fajitas")
        game.suitcase = [
            first_pick,
            make_card("solo"),
            make_card("nap"),
            make_card("wingspan"),
        ]
        game.trunk = [second_replacement, second_pick]

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertIn(first_pick, player.hand)
        self.assertIn(second_pick, player.hand)
        self.assertIs(game.suitcase[0], second_replacement)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(player.picked_cards.values()), 2)
        self.assertEqual(sum(player.acquired_cards.values()), 2)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 2)
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
