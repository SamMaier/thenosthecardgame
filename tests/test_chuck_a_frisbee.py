import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class ChuckAFrisbeeTests(unittest.TestCase):
    def test_costs_three_energy_and_scores_three_fun(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.append(make_card("chuck-a-frisbee"))

        card = game.play_card(0, 0)

        self.assertEqual(card.definition.tags, frozenset({"Exercise", "Outdoors"}))
        self.assertEqual(player.energy, 4)
        self.assertEqual(game.card_fun(0, card), 3)

    def test_returns_to_hand_after_scoring_instead_of_discarding(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        card = make_card("chuck-a-frisbee")
        player.hand.append(card)
        game.play_card(0, 0)

        game.end_day()

        self.assertEqual(player.fun, 3)
        self.assertIn(card, player.hand)
        self.assertNotIn(card, game.discard)
        self.assertEqual(player.played_today, [])


if __name__ == "__main__":
    unittest.main()
