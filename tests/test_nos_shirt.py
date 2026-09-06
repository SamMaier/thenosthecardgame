import random
import unittest

from thenos.ai import RandomAI
from thenos.cards import make_card
from tests.helpers import empty_game


class TargetCopyAI(RandomAI):
    def choose_card_to_copy(self, game, player_index, eligible_cards):
        return next(
            index for index, card in enumerate(eligible_cards)
            if card.title == "Booby Prize"
        )


class NosShirtTests(unittest.TestCase):
    def test_copies_any_players_active_item_without_its_cost_or_tags(self) -> None:
        game = empty_game()
        game.ais[0] = TargetCopyAI(random.Random(0))
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("nos-shirt"))
        target = make_card("booby-prize")
        game.players[1].played_today.append(target)
        drawn = make_card("biography")
        game.trunk.append(drawn)

        card = game.play_card(0, 0)

        self.assertEqual(card.title, "Nos Shirt")
        self.assertEqual(card.definition.cost, 1)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(player.energy, 6)
        self.assertIs(card.effective_behavior, target.effective_behavior)
        self.assertIn(drawn, player.hand)
        self.assertTrue(target.markers["energy_cube"])


if __name__ == "__main__":
    unittest.main()
