import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetDiscardAI(RandomAI):
    def __init__(self, discard_index: int, rng) -> None:
        super().__init__(rng)
        self.discard_index = discard_index

    def choose_card_to_discard(self, game, player_index, hand):
        return self.discard_index


class SettlersCitiesAndKnightsTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("settlers-cities-and-knights")

        self.assertEqual(card.title, "Settlers (Cities and Knights)")
        self.assertEqual(card.definition.cost, 5)
        self.assertEqual(card.definition.base_fun, 4)
        self.assertEqual(card.definition.tags, frozenset({"Board Game"}))

    def test_discards_one_hand_card_for_four_fun(self) -> None:
        game = empty_game()
        game.ais[0] = TargetDiscardAI(0, game.rng)
        player = game.players[0]
        player.energy = 7
        discarded = make_card("biography")
        retained = make_card("nap")
        player.hand.extend(
            [make_card("settlers-cities-and-knights"), discarded, retained]
        )

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 2)
        self.assertEqual(player.hand, [retained])
        self.assertIn(discarded, game.discard)
        self.assertTrue(card.markers["discarded_card"])
        self.assertEqual(game.card_fun(0, card), 8)

    def test_without_a_card_in_hand_scores_only_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 5
        player.hand.append(make_card("settlers-cities-and-knights"))

        card = game.play_card(0, 0)

        self.assertEqual(player.hand, [])
        self.assertEqual(game.discard, [])
        self.assertNotIn("discarded_card", card.markers)
        self.assertEqual(game.card_fun(0, card), 4)


if __name__ == "__main__":
    unittest.main()
