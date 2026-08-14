import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class PrimePicnicTableTests(unittest.TestCase):
    def test_event_cards_after_cost_one_less_and_score_one_more(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("prime-picnic-table"), make_card("work-call")])

        card = game.play_card(0, 0)
        self.assertEqual(game.energy_cost(0, player.hand[0]), 1)
        event = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 3)
        self.assertEqual(card.definition.tags, frozenset({"Item"}))
        self.assertEqual(player.energy, 3)
        self.assertEqual(game.card_fun(0, event), -3)


if __name__ == "__main__":
    unittest.main()
