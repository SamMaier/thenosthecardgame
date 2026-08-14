import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class FirstSuitcaseAI(RandomAI):
    def choose_suitcase_card(self, game, player_index, suitcase):
        return 0


class ThrowABaseballTests(unittest.TestCase):
    def test_printed_values_and_tags(self) -> None:
        card = make_card("throw-a-baseball")

        self.assertEqual(card.title, "Throw a Baseball")
        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.base_fun, 1)
        self.assertEqual(
            card.definition.tags,
            frozenset({"Exercise", "Outdoors"}),
        )

    def test_picks_one_suitcase_card_and_refills_its_slot(self) -> None:
        game = empty_game()
        game.ais[0] = FirstSuitcaseAI(game.rng)
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("throw-a-baseball"))

        picked = make_card("biography")
        refill = make_card("waterski")
        game.suitcase = [picked, make_card("fajitas"), make_card("solo"), make_card("nap")]
        game.trunk = [refill]

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 5)
        self.assertEqual(game.card_fun(0, card), 1)
        self.assertIn(picked, player.hand)
        self.assertIs(game.suitcase[0], refill)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(player.picked_cards["Biography"], 1)
        self.assertEqual(player.acquired_cards["Biography"], 1)
        self.assertEqual(game.stats.suitcase_picks["Biography"], 1)


if __name__ == "__main__":
    unittest.main()
