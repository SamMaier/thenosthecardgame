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


class PuddingChomeurTests(unittest.TestCase):
    def test_printed_values(self) -> None:
        card = make_card("pudding-chomeur")

        self.assertEqual(card.title, "Pudding Chômeur")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.base_fun, 0)
        self.assertEqual(card.definition.tags, frozenset({"Food"}))

    def test_gains_three_energy_then_discards_a_hand_card(self) -> None:
        game = empty_game()
        game.ais[0] = TargetDiscardAI(1, game.rng)
        player = game.players[0]
        player.energy = 7
        pudding = make_card("pudding-chomeur")
        retained = make_card("biography")
        discarded = make_card("nap")
        player.hand = [pudding, retained, discarded]

        played = game.play_card(0, 0)

        self.assertEqual(player.energy, 9)
        self.assertEqual(player.hand, [retained])
        self.assertIn(discarded, game.discard)
        self.assertIs(played, pudding)
        self.assertTrue(played.markers["_gave_energy"])

    def test_does_not_discard_when_no_hand_card_remains(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 1
        player.hand.append(make_card("pudding-chomeur"))

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertEqual(player.hand, [])
        self.assertEqual(game.discard, [])
        self.assertEqual(game.card_fun(0, card), 0)


if __name__ == "__main__":
    unittest.main()
