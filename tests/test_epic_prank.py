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


class EpicPrankTests(unittest.TestCase):
    def test_printed_values_and_base_fun(self) -> None:
        card = make_card("epic-prank")

        self.assertEqual(card.title, "Epic Prank")
        self.assertEqual(card.definition.cost, 4)
        self.assertEqual(card.definition.base_fun, 2)
        self.assertEqual(card.definition.tags, frozenset({"Event"}))

    def test_discards_selected_item_for_five_fun(self) -> None:
        game = empty_game()
        game.ais[0] = TargetDiscardAI(0, game.rng)
        player = game.players[0]
        player.energy = 7
        item = make_card("nos-shirt")
        retained = make_card("biography")
        player.hand.extend([make_card("epic-prank"), item, retained])

        card = game.play_card(0, 0)

        self.assertEqual(player.energy, 3)
        self.assertEqual(player.hand, [retained])
        self.assertIn(item, game.discard)
        self.assertIs(card.markers["discarded_item"], item)
        self.assertEqual(game.card_fun(0, card), 7)

    def test_without_an_item_in_hand_scores_only_base_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 4
        retained = make_card("biography")
        player.hand.extend([make_card("epic-prank"), retained])

        card = game.play_card(0, 0)

        self.assertEqual(player.hand, [retained])
        self.assertEqual(game.discard, [])
        self.assertNotIn("discarded_item", card.markers)
        self.assertEqual(game.card_fun(0, card), 2)


if __name__ == "__main__":
    unittest.main()
