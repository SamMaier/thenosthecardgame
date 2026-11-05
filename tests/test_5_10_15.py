import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class FiveTenFifteenTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("5-10-15"))
        game.suitcase = [make_card("biography") for _ in range(4)]
        game.trunk = [make_card("biography")]

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "5 10 15")
        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 2)

    def test_picks_one_suitcase_card_and_refills_its_slot(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("5-10-15"))
        picked = make_card("waterski")
        refill = make_card("biography")
        game.suitcase = [picked, make_card("fajitas"), make_card("solo"), make_card("nap")]
        game.trunk = [refill]

        game.play_card(0, 0)

        self.assertIn(picked, player.hand)
        self.assertIs(game.suitcase[0], refill)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(player.picked_cards["Waterski"], 1)
        self.assertEqual(player.acquired_cards["Waterski"], 1)
        self.assertEqual(game.stats.suitcase_picks["Waterski"], 1)
        self.assertEqual(sum(game.stats.suitcase_offers.values()), 4)


if __name__ == "__main__":
    unittest.main()
