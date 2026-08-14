import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class LongDistanceVisitorsTests(unittest.TestCase):
    def test_picks_three_refilling_cards_and_bonuses_later_social_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 7
        player.hand.extend([make_card("long-distance-visitors"), make_card("johnny-appleseed")])
        game.suitcase = [make_card("cheap-white") for _ in range(4)]
        game.trunk = [make_card("biography") for _ in range(3)]

        card = game.play_card(0, 0)
        social = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 6)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, social), 2)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(sum(player.picked_cards.values()), 3)
        self.assertEqual(sum(game.stats.suitcase_picks.values()), 3)


if __name__ == "__main__":
    unittest.main()
