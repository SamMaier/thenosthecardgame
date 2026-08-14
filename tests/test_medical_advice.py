import unittest

from thenos.cards import make_card
from tests.helpers import empty_game


class MedicalAdviceTests(unittest.TestCase):
    def test_picks_three_penalizes_previous_and_taxes_later_cards(self) -> None:
        game = empty_game()
        player = game.players[0]
        player.energy = 8
        player.hand.extend(
            [make_card("biography"), make_card("medical-advice"), make_card("cheap-white")]
        )
        game.suitcase = [make_card("waterski") for _ in range(4)]
        game.trunk = [make_card("biography") for _ in range(3)]

        previous = game.play_card(0, 0)
        card = game.play_card(0, 0)
        later = game.play_card(0, 0)

        self.assertEqual(card.definition.cost, 2)
        self.assertEqual(card.definition.tags, frozenset({"Social"}))
        self.assertEqual(game.card_fun(0, previous), 1)
        self.assertEqual(game.card_fun(0, card), 0)
        self.assertEqual(game.card_fun(0, later), 3)
        self.assertEqual(sum(player.picked_cards.values()), 3)
        self.assertEqual(len(game.suitcase), 4)
        self.assertEqual(player.energy, 2)


if __name__ == "__main__":
    unittest.main()
